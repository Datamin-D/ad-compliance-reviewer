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
from src.prompts import (  # 프롬프트/스키마는 leaf 모듈로 분리(로컬·Databricks 공유). src/prompts.py 참고
    CLAIM_PROMPT as _CLAIM_PROMPT,
    CLAIM_SCHEMA as _CLAIM_SCHEMA,
    COUNTRY_CONTEXT,
    JUDGE_PROMPT as _JUDGE_PROMPT,
    JUDGE_SCHEMA as _JUDGE_SCHEMA,
)
from src.retriever import PolicyRetriever
from src.schema import Decision, ReviewResult, RetrievedPolicy, parse_review_payload

# ---------------------------------------------------------------------------
# [1] claim 추출
# ---------------------------------------------------------------------------
# 광고 한 줄에 성격이 다른 주장이 섞여 있으면("No.1 에어컨, 전기료 50% 절감")
# 통째로 임베딩했을 때 벡터가 두 주장 사이 중간에 찍혀 어느 규정과도 안 가깝다.
# 먼저 쪼개고 각각 검색하는 것이 retrieval 품질에 가장 크게 기여했다.
# (프롬프트·스키마 정의는 src/prompts.py로 이동 — 위 import 참고)


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
# [3] 판정   (JUDGE_SCHEMA / COUNTRY_CONTEXT / JUDGE_PROMPT 정의는 src/prompts.py)
# ---------------------------------------------------------------------------
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
