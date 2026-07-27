# -*- coding: utf-8 -*-
"""실습과제 신용평가 파이프라인 모듈.

experiments/sim_amnesty.py(검증된 코드)에서 포팅 — 로직 변경 금지.
Platt 보정은 원 노트북 "AI & Data 활용 안내서(AI Play DB)_성능개선_박건우.ipynb" 셀 16 포팅.
기준 수치: 홀드아웃 AUC 0.9247, KS 0.7355 (seed 42, test_size 0.2 층화분할).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

seed = 42

TARGET_COL = '향후12개월내연체여부'
DELIN_COLS = ['과거3개월내연체여부', '과거6개월내연체여부', '과거12개월내연체여부', '현재연체여부']


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
    log_cols = [c for c in num_cols if any(t in c for t in ['잔액', '사용액', '보험료'])]
    lin_cols = [c for c in num_cols if c not in log_cols]
    log_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('log1p', FunctionTransformer(safe_log1p)), ('scaler', RobustScaler())])
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    bin_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
    return ColumnTransformer([('lognum', log_pipe, log_cols), ('num', num_pipe, lin_cols),
                              ('bin', bin_pipe, bin_cols)], remainder='drop')


def add_features(X):
    X = X.copy(); eps = 1e-9
    X['총대출잔액'] = X[['신용대출잔액', '장기카드대출잔액', '단기카드대출잔액', '주택담보대출잔액', '주택외담보대출잔액', '기타대출잔액']].sum(axis=1)
    X['총대출건수'] = X[['신용대출건수', '장기카드대출건수', '단기카드대출건수', '주택담보대출건수', '주택외담보대출건수', '기타대출건수']].sum(axis=1)
    X['건당평균대출잔액'] = X['총대출잔액'] / (X['총대출건수'] + eps)
    fin_total = X[['1금융권대출잔액', '2금융권대출잔액', '3금융권대출잔액']].sum(axis=1)
    X['2금융권잔액비중'] = X['2금융권대출잔액'] / (fin_total + eps)
    X['3금융권잔액비중'] = X['3금융권대출잔액'] / (fin_total + eps)
    X['비1금융권잔액비중'] = (X['2금융권대출잔액'] + X['3금융권대출잔액']) / (fin_total + eps)
    X['신용성대출비중'] = (X['신용대출잔액'] + X['장기카드대출잔액'] + X['단기카드대출잔액']) / (X['총대출잔액'] + eps)
    rate_pairs = [('신용대출잔액', '신용대출금리'), ('장기카드대출잔액', '장기카드대출금리'), ('단기카드대출잔액', '단기카드대출금리')]
    X['잔액가중평균금리'] = sum(X[b] * X[r] for b, r in rate_pairs) / (sum(X[b] for b, _ in rate_pairs) + eps)
    X['최고금리'] = X[['신용대출금리', '장기카드대출금리', '단기카드대출금리']].max(axis=1)
    X['카드사용액합계'] = X['월평균신용카드사용액'] + X['월평균체크카드사용액']
    X['신용카드사용비중'] = X['월평균신용카드사용액'] / (X['카드사용액합계'] + eps)
    X['카드사용액대비총부채'] = X['총대출잔액'] / (X['카드사용액합계'] + eps)
    X['기관당평균대출건수'] = X['총대출건수'] / (X['대출기관수'] + eps)
    X['보험료합계'] = X[['월납입보험료', '연금보험료', '운전자보험료', '종신보험료', '질병보험료']].sum(axis=1)
    X['보험료대비부채'] = X['총대출잔액'] / (X['보험료합계'] + eps)
    X['연체이력강도'] = X[DELIN_COLS].sum(axis=1)
    return X


def undersample(X, y, ratio=1, seed=0):
    r = np.random.RandomState(seed)
    pos = np.where(y.values == 1)[0]; neg = np.where(y.values == 0)[0]
    neg_s = r.choice(neg, size=min(len(neg), len(pos) * ratio), replace=False)
    idx = np.concatenate([pos, neg_s]); r.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]


class EasyEnsembleHGB:
    """서로 다른 1:ratio 언더샘플 표본으로 n_bags개 HGB를 학습, 확률 평균."""

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


# ---------- Platt 보정 (원 노트북 셀 16 포팅) ----------
def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt_calibrator(Xf_tr, y_tr, n_splits=5, verbose=True):
    """학습셋 내부 OOF 예측으로 Platt 보정기 학습 (홀드아웃 누수 없음).

    반환: (calibrator, oof_tr) — oof_tr은 보정 전 학습셋 OOF 확률.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_tr = np.zeros(len(y_tr))
    for k, (tr_i, va_i) in enumerate(skf.split(Xf_tr, y_tr)):
        m = EasyEnsembleHGB(n_bags=20, ratio=1).fit(Xf_tr.iloc[tr_i], y_tr.iloc[tr_i])
        oof_tr[va_i] = m.predict_proba1(Xf_tr.iloc[va_i])
        if verbose:
            print(f'  OOF fold {k + 1}/{n_splits} 완료')
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(to_logit(oof_tr).reshape(-1, 1), y_tr)
    return calibrator, oof_tr


def calibrate(calibrator, probs):
    """보정 전 확률 → Platt 보정 확률(PD)."""
    return calibrator.predict_proba(to_logit(np.asarray(probs)).reshape(-1, 1))[:, 1]
