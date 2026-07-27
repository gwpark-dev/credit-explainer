# -*- coding: utf-8 -*-
"""신용사면 시뮬레이션: 연체기록 마스킹 → 성능 저하 측정 → 사면 flag 복원
설계:
- 사면 대상: 현재연체자 중 '전액 상환' 가정 집단(무작위 p%). 실제 사면 구조(상환 완료자 기록 삭제) 반영.
- 마스킹: 과거3/6/12개월내연체여부, 현재연체여부 → 0 (train/test 동일 적용 = 사면 후 세상)
- flag 복원: '사면이력여부' 변수 추가 (CB 내부 flag 보존 시나리오, 9주차 토의의 방어책 #1)
- 모형: 기존 실습과제 파이프라인 그대로 (파생 16개 + EasyEnsemble 20×HGB)
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

seed = 42
rng = np.random.RandomState(seed)

df = pd.read_csv('/mnt/user-data/uploads/Desktop--연세대학교/5학기/인공지능과신용평가모델/과제/data/AI_Play_DB.csv', encoding='cp949')
TARGET_COL = '향후12개월내연체여부'
DELIN_COLS = ['과거3개월내연체여부','과거6개월내연체여부','과거12개월내연체여부','현재연체여부']

# ---------- 원 노트북과 동일한 전처리/파생/모형 ----------
def split_xy(d):
    future_cols = [c for c in d.columns if c.startswith('향후') and c.endswith('연체여부')]
    leak = [c for c in future_cols if c != TARGET_COL]
    X = d.drop(columns=['차주번호'] + leak + [TARGET_COL], errors='ignore').copy()
    return X, d[TARGET_COL].astype(int)

def safe_log1p(a):
    return np.log1p(np.clip(np.asarray(a, dtype=float), 0, None))

def make_preprocess(X):
    bin_cols = [c for c in X.columns if '여부' in c]
    num_cols = [c for c in X.columns if c not in bin_cols]
    log_cols = [c for c in num_cols if any(t in c for t in ['잔액','사용액','보험료'])]
    lin_cols = [c for c in num_cols if c not in log_cols]
    log_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('log1p', FunctionTransformer(safe_log1p)), ('scaler', RobustScaler())])
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    bin_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
    return ColumnTransformer([('lognum', log_pipe, log_cols), ('num', num_pipe, lin_cols),
                              ('bin', bin_pipe, bin_cols)], remainder='drop')

def add_features(X):
    X = X.copy(); eps = 1e-9
    X['총대출잔액'] = X[['신용대출잔액','장기카드대출잔액','단기카드대출잔액','주택담보대출잔액','주택외담보대출잔액','기타대출잔액']].sum(axis=1)
    X['총대출건수'] = X[['신용대출건수','장기카드대출건수','단기카드대출건수','주택담보대출건수','주택외담보대출건수','기타대출건수']].sum(axis=1)
    X['건당평균대출잔액'] = X['총대출잔액'] / (X['총대출건수'] + eps)
    fin_total = X[['1금융권대출잔액','2금융권대출잔액','3금융권대출잔액']].sum(axis=1)
    X['2금융권잔액비중'] = X['2금융권대출잔액'] / (fin_total + eps)
    X['3금융권잔액비중'] = X['3금융권대출잔액'] / (fin_total + eps)
    X['비1금융권잔액비중'] = (X['2금융권대출잔액'] + X['3금융권대출잔액']) / (fin_total + eps)
    X['신용성대출비중'] = (X['신용대출잔액'] + X['장기카드대출잔액'] + X['단기카드대출잔액']) / (X['총대출잔액'] + eps)
    rate_pairs = [('신용대출잔액','신용대출금리'),('장기카드대출잔액','장기카드대출금리'),('단기카드대출잔액','단기카드대출금리')]
    X['잔액가중평균금리'] = sum(X[b]*X[r] for b, r in rate_pairs) / (sum(X[b] for b, _ in rate_pairs) + eps)
    X['최고금리'] = X[['신용대출금리','장기카드대출금리','단기카드대출금리']].max(axis=1)
    X['카드사용액합계'] = X['월평균신용카드사용액'] + X['월평균체크카드사용액']
    X['신용카드사용비중'] = X['월평균신용카드사용액'] / (X['카드사용액합계'] + eps)
    X['카드사용액대비총부채'] = X['총대출잔액'] / (X['카드사용액합계'] + eps)
    X['기관당평균대출건수'] = X['총대출건수'] / (X['대출기관수'] + eps)
    X['보험료합계'] = X[['월납입보험료','연금보험료','운전자보험료','종신보험료','질병보험료']].sum(axis=1)
    X['보험료대비부채'] = X['총대출잔액'] / (X['보험료합계'] + eps)
    X['연체이력강도'] = X[DELIN_COLS].sum(axis=1)
    return X

def undersample(X, y, ratio=1, seed=0):
    r = np.random.RandomState(seed)
    pos = np.where(y.values == 1)[0]; neg = np.where(y.values == 0)[0]
    neg_s = r.choice(neg, size=min(len(neg), len(pos)*ratio), replace=False)
    idx = np.concatenate([pos, neg_s]); r.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]

class EasyEnsembleHGB:
    def __init__(self, n_bags=20, ratio=1, seed=seed):
        self.n_bags, self.ratio, self.seed = n_bags, ratio, seed
    def fit(self, X, y):
        self.pipes_ = []
        for b in range(self.n_bags):
            Xs, ys = undersample(X, y, self.ratio, seed=self.seed + b)
            pipe = Pipeline([('pre', make_preprocess(X)),
                             ('clf', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                                    max_leaf_nodes=31, random_state=self.seed + b))])
            pipe.fit(Xs, ys); self.pipes_.append(pipe)
        return self
    def predict_proba1(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.pipes_], axis=0)

def ks_stat(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    return float(np.max(tpr - fpr))

# ---------- 사면 시나리오 생성 ----------
cur_delin_idx = df.index[df['현재연체여부'] == 1].to_numpy()  # 2,729명
print(f'현재연체자: {len(cur_delin_idx)}명, 이들의 향후12연체율: {df.loc[cur_delin_idx, TARGET_COL].mean():.3%}')

def make_scenario(df, amnesty_rate, with_flag=False, rng_seed=7):
    """amnesty_rate 비율의 현재연체자를 '전액상환→사면'으로 가정, 연체기록 전체 삭제"""
    d = df.copy()
    r = np.random.RandomState(rng_seed)
    n_amn = int(round(len(cur_delin_idx) * amnesty_rate))
    amn_idx = r.choice(cur_delin_idx, size=n_amn, replace=False)
    d.loc[amn_idx, DELIN_COLS] = 0
    if with_flag:
        d['사면이력여부'] = 0
        d.loc[amn_idx, '사면이력여부'] = 1
    return d, amn_idx

def run(d, label):
    X, y = split_xy(d)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xf_tr, Xf_te = add_features(X_tr), add_features(X_te)
    t0 = time.time()
    m = EasyEnsembleHGB().fit(Xf_tr, y_tr)
    p = m.predict_proba1(Xf_te)
    auc = roc_auc_score(y_te, p); ks = ks_stat(y_te, p)
    print(f'[{label}] AUC={auc:.4f}  KS={ks:.4f}  ({time.time()-t0:.0f}s)')
    return {'label': label, 'AUC': auc, 'KS': ks, 'test_index': X_te.index.to_numpy(), 'probs': p, 'y_te': y_te.to_numpy()}

results = []
# 1) 베이스라인 (사면 전 세상)
base = run(df, '베이스라인(사면 전)')
results.append(base)

# 2) 사면율 dose-response: 50%, 100%
amn_indices = {}
for rate in [0.5, 1.0]:
    d, amn_idx = make_scenario(df, rate)
    amn_indices[rate] = amn_idx
    results.append(run(d, f'사면 {int(rate*100)}% (기록 삭제)'))

# 3) 사면 100% + flag 보존
d_flag, amn_idx_f = make_scenario(df, 1.0, with_flag=True)
results.append(run(d_flag, '사면 100% + 사면flag 보존'))

# ---------- 사면 대상 그룹 심층 분석 (사면 100% vs 베이스라인) ----------
full_100 = next(r for r in results if r['label'] == '사면 100% (기록 삭제)')
flag_100 = next(r for r in results if r['label'] == '사면 100% + 사면flag 보존')

def subgroup_analysis(res, amn_set, name):
    te_idx = res['test_index']; probs = res['probs']; y_te = res['y_te']
    in_amn = np.isin(te_idx, list(amn_set))
    out = {}
    out['n_amn_test'] = int(in_amn.sum())
    out['amn_bad_rate'] = float(y_te[in_amn].mean())
    out['amn_mean_pd'] = float(probs[in_amn].mean())
    out['clean_mean_pd'] = float(probs[~in_amn].mean())
    # 사면 대상자 중 실제 미래 불량자의 예측 PD
    bad_amn = in_amn & (y_te == 1)
    out['n_bad_amn_test'] = int(bad_amn.sum())
    out['bad_amn_mean_pd'] = float(probs[bad_amn].mean()) if bad_amn.sum() else None
    # 전체 홀드아웃에서 미래불량 사면자의 위험순위 percentile (높을수록 위험하게 봄)
    ranks = pd.Series(probs).rank(pct=True).to_numpy()
    out['bad_amn_mean_risk_pctl'] = float(ranks[bad_amn].mean()) if bad_amn.sum() else None
    print(f'--- {name} ---')
    for k, v in out.items(): print(f'  {k}: {v}')
    return out

amn_set = set(amn_indices[1.0])
sa_base = subgroup_analysis(base, amn_set, '베이스라인에서의 (미래)사면대상 그룹')
sa_full = subgroup_analysis(full_100, amn_set, '사면 100% 후 그룹')
sa_flag = subgroup_analysis(flag_100, set(amn_idx_f), '사면 100%+flag 후 그룹')

summary = pd.DataFrame([{'시나리오': r['label'], 'AUC': round(r['AUC'],4), 'KS': round(r['KS'],4)} for r in results])
print(); print(summary.to_string(index=False))
summary.to_csv('/home/claude/amnesty_results.csv', index=False)
json.dump({'base': sa_base, 'full': sa_full, 'flag': sa_flag}, open('/home/claude/amnesty_subgroup.json','w'), ensure_ascii=False, indent=1)
