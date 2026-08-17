"""
파이프라인 전체 — LLM을 부르는 두 단계가 여기 다 있다.

    광고 문안
       ↓ [1] extract_claims()  ← LLM 호출 ①  검증 대상 주장으로 쪼갬
       ↓ [2] retriever          ← 임베딩 검색  관련 규정만 추림
       ↓ [3] judge()            ← LLM 호출 ②  규정만 근거로 판정
       ↓ [4] parse_review_payload()  (schema.py) 지어낸 근거 폐기
    ReviewResult

프롬프트가 이 프로젝트 품질의 절반이라, 파이프라인 코드와 같은 파일에 둔다.
"""

from __future__ import annotations

from functools import lru_cache

from src import config, llm
from src.retriever import PolicyRetriever
from src.schema import Decision, ReviewResult, RetrievedPolicy, parse_review_payload

# ---------------------------------------------------------------------------
# [1] claim 추출
# ---------------------------------------------------------------------------
# 광고 한 줄에 성격이 다른 주장이 섞여 있으면("No.1 에어컨, 전기료 50% 절감")
# 통째로 임베딩했을 때 벡터가 두 주장 사이 중간에 찍혀 어느 규정과도 안 가깝다.
# 먼저 쪼개고 각각 검색하는 것이 retrieval 품질에 가장 크게 기여했다.

_CLAIM_SCHEMA = {
    "type": "OBJECT",
    "required": ["claims"],
    "properties": {"claims": {"type": "ARRAY", "items": {"type": "STRING"}}},
}

_CLAIM_PROMPT = """\
아래 광고 문안에서 '검증이 필요한 주장(claim)'을 뽑아 주세요.

검증이 필요한 주장이란:
- 최상급/순위 표현 (예: 최고, 1위, No.1, 세계 최초)
- 수치로 표현된 성능·효능·절감 효과 (예: 50% 절감, 99.9% 살균)
- 가격·할인·무료·최저가 표현
- 경쟁사와의 비교 또는 경쟁사 언급
- 안전성·의학적 효능 표현 (예: 치료, 무해, 부작용 없음)
- 경품·당첨·이벤트 조건
- 고객 후기, 인물 사진 사용, 출시 예정 기능 언급

규칙:
- 광고 문안에 실제로 등장한 표현을 **그대로** 인용하세요. 요약하지 마세요.
- 단순 감성 문구나 제품 소개("여름 준비, 시원하게")는 claim이 아닙니다. 제외하세요.
- 한 문장에 주장이 두 개면 두 개로 나누세요.
- 없으면 빈 배열을 반환하세요.

광고 문안:
\"\"\"{ad_text}\"\"\"
"""


def extract_claims(ad_text: str) -> list[str]:
    """광고 문안 → 검증 대상 claim 리스트. 없으면 빈 리스트."""
    payload = llm.generate_json(
        _CLAIM_PROMPT.format(ad_text=ad_text),
        _CLAIM_SCHEMA,
        system_instruction=(
            "당신은 광고 심의 실무자입니다. 광고 문안에서 사실 여부나 근거를 "
            "따져봐야 하는 주장(claim)만 정확히 골라냅니다."
        ),
    )
    # 공백 제거 + 순서 유지 중복 제거
    return list(dict.fromkeys(c.strip() for c in payload.get("claims", []) if c.strip()))


# ---------------------------------------------------------------------------
# [3] 판정
# ---------------------------------------------------------------------------
_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "required": ["decision", "risk_score", "issues"],
    "properties": {
        "decision": {"type": "STRING", "enum": ["PASS", "REVIEW", "BLOCK"]},
        "risk_score": {"type": "NUMBER"},
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["claim", "rule_id", "reason", "suggestion"],
                "properties": {
                    "claim": {"type": "STRING"},
                    "rule_id": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                    "suggestion": {"type": "STRING"},
                },
            },
        },
    },
}

# 국가별 심의 태도. 규정 텍스트(무엇을 요구하나)와 별개로, '같은 표현을 얼마나 엄격히
# 보느냐'는 관할마다 다르다. 그 맥락을 판정 프롬프트에 주입한다.
# (규정만 국가별로 바꾸고 이 맥락을 안 주면 "미국 규정을 한국식으로 해석"하는 오류가 난다)
COUNTRY_CONTEXT = {
    "KR": "대한민국 표시·광고의 공정화에 관한 법률 기준으로 판단합니다. "
          "최상급·정량 효능·가격 표시에 객관적 실증 근거가 필요합니다.",
    "US": "미국 FTC 가이드라인 기준으로 판단합니다. 모든 객관적 주장은 "
          "'competent and reliable evidence'가 뒷받침되어야 하며, "
          "'eco-friendly' 같은 일반적 친환경 표현은 구체적 입증 없이는 기만으로 봅니다.",
    "EU": "EU 소비자보호 지침(Green Claims, Omnibus, UCPD) 기준으로 판단합니다. "
          "일반적 친환경 표현('eco-friendly', 'climate neutral')과 미검증 비교광고에 "
          "특히 엄격하며, 입증되지 않은 경우 사용 금지(BLOCK) 수준으로 봅니다.",
}


_JUDGE_PROMPT = """\
아래 광고 문안이 {country} 광고 심의 규정에 저촉되는지 판단해 주세요.

## 심의 기준 (관할: {country})
{country_context}

## 광고 문안
\"\"\"{ad_text}\"\"\"

## 검증 대상 주장(claim)
{claims}

## 참고 가능한 규정 (이 목록에 있는 것만 근거로 사용할 수 있습니다)
{rules}

## 판정 기준
- BLOCK  : 규정이 "사용할 수 없다", "금지한다"고 명시한 표현을 실제로 사용한 경우.
           보완 표기로 해소되지 않고 문구 자체를 바꿔야 하는 수준.
- REVIEW : 표현 자체는 허용되지만 규정이 요구하는 근거·조건·기준 표시가 누락된 경우.
- PASS   : 검증할 주장이 없거나, 규정이 요구하는 표시가 모두 갖춰진 경우.

## 작성 규칙
1. issues의 rule_id는 반드시 위 규정 목록에 있는 ID여야 합니다. 없는 ID를 만들지 마세요.
2. 근거 규정 문장이 실제로 그 내용을 담고 있을 때만 issue로 올리세요.
3. claim에는 광고 문안에 등장한 표현을 그대로 인용하세요.
4. suggestion에는 무엇을 추가/변경하면 되는지 구체적으로 쓰세요.
5. risk_score: PASS 0.0~0.2, REVIEW 0.3~0.7, BLOCK 0.7~1.0.
6. 문제가 없으면 decision=PASS, issues=[].
7. 애매하면 PASS가 아니라 REVIEW. 위험한 광고를 통과시키는 것이
   불필요한 검토 요청보다 훨씬 큰 손실입니다.
"""


def judge(ad_text: str, claims: list[str], hits: list[RetrievedPolicy],
          country: str = "KR") -> ReviewResult:
    """claim + 검색된 규정 → 판정. hits가 곧 '허용된 근거'의 전부다."""
    rules = "\n".join(
        f"- {h.policy.rule_id} | {h.policy.category} | {h.policy.title} "
        f"(심각도: {h.policy.severity})\n    {h.policy.text}"
        for h in hits
    )

    payload = llm.generate_json(
        _JUDGE_PROMPT.format(
            country=country,
            country_context=COUNTRY_CONTEXT.get(country, ""),
            ad_text=ad_text,
            claims="\n".join(f"- {c}" for c in claims),
            rules=rules or "(검색된 규정이 없습니다.)",
        ),
        _JUDGE_SCHEMA,
        system_instruction=(
            "당신은 광고 심의 담당 법무 검토자입니다. 주어진 규정만을 근거로 "
            "판단하며, 규정에 없는 내용을 추측하지 않습니다."
        ),
    )

    # 프롬프트로 "지어내지 마"는 확률, 여기서 거르는 게 보장. schema.py 참고.
    return parse_review_payload(payload, ad_text, claims, [h.policy.rule_id for h in hits])


# ---------------------------------------------------------------------------
# 엮기
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_retriever() -> PolicyRetriever:
    """검색기는 한 번만 만들어 재사용한다(Chroma 접속 + 규정 로딩 비용)."""
    r = PolicyRetriever()
    if r.count() == 0:
        raise RuntimeError("인덱스가 비어 있습니다. 먼저 `python cli.py index`를 실행하세요.")
    return r


def review(ad_text: str, top_k: int = config.TOP_K, country: str = "KR") -> ReviewResult:
    """광고 문안 하나를 검토한다. country로 관할을 지정한다(규정 검색 + 판정 기준)."""
    claims = extract_claims(ad_text)

    # 검증할 주장이 없으면 검색도 판정도 불필요 → LLM 호출 1회 절약
    if not claims:
        return ReviewResult(ad_text, Decision.PASS, 0.0)

    # country로 해당 국가 규정만 검색하고, 같은 국가 기준으로 판정한다.
    hits = get_retriever().search_many(claims, k=top_k, country=country)
    return judge(ad_text, claims, hits, country=country)
