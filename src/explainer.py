# -*- coding: utf-8 -*-
"""③ 생성형 AI 계층 — SHAP 결과를 소비자 눈높이 자연어 설명문으로 "번역".

ADR-0001 준수:
- Anthropic Claude API 호출, 프롬프트 5요소 공식(역할·맥락·목표·형식·제약)
- 발표용 캐시 모드: outputs/explanations_cache.json 우선, 미존재 시에만 API 호출
- API 키는 .env(ANTHROPIC_API_KEY)로만 주입, 코드·로그·커밋에 노출 금지
- 모델 버전은 코드 상수로 고정(재현성)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / '.env')

# 구현 시점(2026-07) Anthropic 권장 기본 모델로 고정 (ADR-0001 4항)
MODEL = 'claude-opus-4-8'
MAX_TOKENS = 800
PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK = 5.00, 25.00  # USD, claude-opus-4-8 기준

# 안전장치
MAX_API_CALLS = 12       # 세션 내 총 호출 한도
COST_LIMIT_USD = 5.00    # 예상 비용 상한 (ADR-0001)

CACHE_PATH = _ROOT / 'outputs' / 'explanations_cache.json'

# 세션 누적 카운터
api_calls = 0
total_cost_usd = 0.0


def display_value(name, value):
    """LLM 입력용 사람이 읽는 표시값 변환.

    금리 컬럼은 % × 1000 스케일로 저장되어 있음(원본 분포 검증: 0 제외 최댓값
    24,000 = 데이터 시기의 법정 최고금리 24%와 일치). 잔액류는 데이터 정의서가
    없어 화폐 단위를 확정할 수 없으므로 단위 없이 수치만 표시한다.
    """
    v = float(value)
    if '금리' in name:
        return f'연 {round(v / 1000, 2):g}%'
    if '비중' in name:
        return f'{v * 100:.1f}%'
    if '여부' in name:
        return '있음' if v >= 0.5 else '없음'
    if name == '연체이력강도':
        return f'연체 관련 지표 4개 중 {int(round(v))}개 해당'
    if '건수' in name:
        return f'{int(round(v)):,}건'
    if name.endswith('수'):
        return f'{int(round(v)):,}개'
    if abs(v) >= 1000:
        return f'{v:,.0f}'
    return f'{v:,.4g}'


SYSTEM_PROMPT = """\
[역할] 당신은 금융회사 소비자보호 부서의 신용평가 결과 설명 담당자입니다.

[맥락] 신용정보법 제36조의2는 자동화된 신용평가에 대해 소비자가 결과와 주요 기준, \
기초정보에 대한 설명을 요구할 권리를 보장합니다. 지금 작성하는 문서는 그 권리에 따라 \
소비자에게 발송되는 공식 설명문입니다. 독자는 금융 비전문가이므로 전문용어는 풀어서 씁니다.

[목표] 소비자가 (1) 자신의 평가 결과(승인/거절)와 (2) 그 결과에 영향을 준 주요 요인을 \
이해하고, (3) 앞으로 참고할 수 있는 사항과 (4) 자신의 권리(설명요구·정보제출·이의제기)를 \
알 수 있게 합니다.

[형식] 아래 네 부분을 순서대로, 전체 400자 내외의 존댓말로 작성합니다. 인사말·서명은 생략합니다.
① 결과 통지: 평가 결과를 한 문장으로 안내
② 주요 요인: 입력으로 제공된 요인 3~5개를, 제공된 수치 근거를 인용하며 쉬운 언어로 설명
③ 개선 참고사항: 입력에 개선 시뮬레이션 결과가 있으면 그 내용을 반영하되, \
"~하시면 재평가 시 결과가 달라질 수 있습니다" 수준의 비확정적 표현만 사용
④ 권리 안내: 설명요구권, 기초정보 제출권(정정 요청 포함), 이의제기권을 한두 문장으로 안내

[제약]
- 입력에 없는 수치·사실을 만들어내지 않습니다.
- 수치는 입력에 적힌 표시값을 그대로 인용합니다(단위 변환·재계산 금지).
- 점수 조작을 유도하는 표현, 승인을 확정적으로 약속하는 표현, 투자 권유, \
특정 금융상품 추천을 하지 않습니다.
- 평가 자체를 다시 수행하거나 결과를 바꾸지 않습니다(당신의 역할은 설명뿐입니다)."""


def build_prompt(case):
    """사례 dict → user 메시지 텍스트.

    case: {case_id, 사례명, 판정, pd, threshold, factors: [{변수, 표시값, 방향}], counterfactual}
    """
    lines = [
        f"[평가 결과] {case['판정']}",
        f"[부실 위험 추정치] {case['pd'] * 100:.2f}% (판정 기준값: {case['threshold'] * 100:.2f}% 이상이면 거절)",
        '[주요 요인 (영향력 순)]',
    ]
    for i, f in enumerate(case['factors'], 1):
        lines.append(f"  {i}. {f['변수']} = {f['표시값']} — 위험 {f['방향']} 요인")
    if case.get('counterfactual'):
        lines.append(f"[개선 시뮬레이션 결과] {case['counterfactual']}")
    lines.append('\n위 정보만 사용하여 형식에 맞는 설명문을 작성해 주세요.')
    return '\n'.join(lines)


def _load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding='utf-8'))
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')


def generate_explanation(case, use_cache=True):
    """캐시 우선 설명문 생성. 반환: {text, model, usage, cost_usd, from_cache, created_at}"""
    global api_calls, total_cost_usd

    cache = _load_cache()
    key = case['case_id']
    if use_cache and key in cache:
        return {**cache[key], 'from_cache': True}

    # 안전장치
    if api_calls >= MAX_API_CALLS:
        raise RuntimeError(f'세션 API 호출 한도({MAX_API_CALLS}회) 초과 — 중단')
    est_next = (2000 * PRICE_IN_PER_MTOK + MAX_TOKENS * PRICE_OUT_PER_MTOK) / 1e6
    if total_cost_usd + est_next > COST_LIMIT_USD:
        raise RuntimeError(f'예상 비용이 상한 ${COST_LIMIT_USD}를 초과 — 멈추고 보고')

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': build_prompt(case)}],
    )
    api_calls += 1

    if response.stop_reason not in ('end_turn', 'stop_sequence'):
        raise RuntimeError(f'비정상 종료(stop_reason={response.stop_reason}) — 멈추고 보고')

    text = ''.join(b.text for b in response.content if b.type == 'text').strip()
    usage = {'input_tokens': response.usage.input_tokens,
             'output_tokens': response.usage.output_tokens}
    cost = (usage['input_tokens'] * PRICE_IN_PER_MTOK
            + usage['output_tokens'] * PRICE_OUT_PER_MTOK) / 1e6
    total_cost_usd += cost

    entry = {'text': text, 'model': MODEL, 'usage': usage,
             'cost_usd': round(cost, 6),
             'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    cache[key] = entry
    _save_cache(cache)
    return {**entry, 'from_cache': False}
