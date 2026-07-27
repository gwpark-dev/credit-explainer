# credit-explainer

「AI와 신용평가모형」 발표과제(30%) — 기존 실습과제 모형(EasyEnsemble 20×HGB) 위에
SHAP 설명·공정성 감사·LLM 번역 계층을 얹는 '신뢰 계층' 데모.

## 기준 문서

- `docs/프로젝트명세서.md` — 첫 번째 기준. 모든 작업은 이 명세서를 따른다.
- `docs/아이디어기획서.md` — 참고용.
- `docs/decisions/` — ADR은 채팅 세션 담당. **이 저장소 작업에서 ADR을 생성하지 않는다.**

## 핵심 제약

- **기존 모형 재사용**: 새 모형 개발 금지. 실습과제 파이프라인(`src/pipeline.py`)의 로직 변경 금지.
  기준 수치: 홀드아웃 AUC 0.9247, KS 0.7355 (seed 42, test_size 0.2 층화분할). ±0.005 밖이면 멈추고 보고.
- **의존성 고정**: pandas, numpy, scikit-learn, shap, matplotlib, jupyter, ipykernel, joblib만.
  추가 라이브러리 필요 시 멈추고 보고. Counterfactual은 수동 구현(DiCE 등 금지).
- **외부 API 사용 없음** (비용 상한 $0). LLM 설명문 생성은 별도 프롬프트에서 진행.
- **데이터**: `../data/AI_Play_DB.csv` (cp949), 타깃 `향후12개월내연체여부`. 데이터는 커밋하지 않음.
- **커밋**: Conventional Commits(한국어), Co-Authored-By: Claude 포함. push 금지.
  outputs/의 png는 커밋 대상, joblib은 제외.

## 로드맵

### Phase 1 — 데모 노트북 (진행 중)

`notebooks/01_demo_core.ipynb` — 모든 셀 실행된 상태로 저장.

- [x] 스캐폴딩: uv(Python 3.12) + git + 의존성
- [x] `src/pipeline.py`: 실습과제 파이프라인 모듈화(split_xy, safe_log1p, make_preprocess,
      add_features, undersample, EasyEnsembleHGB, ks_stat) + Platt 보정(학습셋 OOF 기반)
- [x] 기준 재현: AUC 0.9247 / KS 0.7355 ±0.005
- [x] Platt 보정 + 임계값 0.0443 → 승인/거절 판정
- [x] SHAP Local 사례 3건(고위험 거절·저위험 승인·경계) → `outputs/shap_case_{1,2,3}.png`
- [x] Counterfactual 수동 구현(경계 사례 1건, 행동 가능 변수 격자 탐색)
- [x] 공정성 감사(프록시 그룹: 3금융권대출잔액>0) → `outputs/fairness_audit.png`
- [x] 부록: 신용사면 실험 차트 → `outputs/amnesty_experiment.png`
- [x] LLM 자연어 설명문 생성 (`notebooks/02_demo_explain.ipynb`, ADR-0001):
      claude-opus-4-8, 캐시 모드(`outputs/explanations_cache.json`) → 발표장 API 호출 0회,
      HITL 검토 3건 통과, 규제 매핑 표 포함

**DoD**: `uv run jupyter nbconvert --to notebook --execute`로 전체 무오류 실행,
outputs/ png 5개 이상, 기준 수치 재현, 커밋 완료(푸시 X).

### Phase 2 — 발표자료 (예정)

- 10분 PPT 10~12장 (도입 → 문제 정의 → 4계층 아키텍처 → 데모 → 한계와 통제 → 마무리)
- 발표자 노트, Q&A 카드(사면 실험·차별점·책임 소재)
- Phase 1 산출물(차트·수치) 확정 후 착수. 데모 없이 슬라이드부터 만들지 않는다.

## 구조

```
src/pipeline.py        # 실습과제 파이프라인 모듈 (로직 변경 금지)
notebooks/01_demo_core.ipynb
experiments/           # 사면 시뮬레이션 (완료된 실험)
outputs/               # 차트 png (커밋 대상), joblib (커밋 제외)
docs/                  # 명세서·기획서, decisions/(채팅 세션 담당)
```
