"""
광고 심의 RAG — 02(평가)와 03(배포)이 함께 쓰는 단 하나의 구현.
"평가한 코드 = 배포한 코드"를 보장하려고 한 파일에 둔다.

- 02 노트북: `from serving_model import review` 로 가져와 25건 평가
- 03 노트북: 이 파일 자체를 MLflow 모델로 기록(models-from-code) → Model Serving 엔드포인트

프롬프트와 grounding 체크(parse_review_payload)는 로컬 src/ 를 그대로 재사용한다.
Gemini 키는 환경변수 GEMINI_API_KEY 로 받는다(노트북=입력창, 서빙=secret scope).
"""

import json
import os
import time
from functools import cache

import mlflow
from src.prompts import CLAIM_PROMPT, CLAIM_SCHEMA, COUNTRY_CONTEXT, JUDGE_PROMPT, JUDGE_SCHEMA
from src.schema import parse_review_payload

# ---- CONFIG (01에서 만든 이름과 같아야 함) ----------------------------------
CATALOG, SCHEMA = "workspace", "ad_compliance"
VS_ENDPOINT = "adcompliance-vs"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.policies_index"
EMBED_ENDPOINT = "databricks-gte-large-en"
GEMINI_MODEL = "gemini-3.6-flash"
TOP_K = 5
# ---------------------------------------------------------------------------


@cache
def _gemini():
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@cache
def _index():
    from databricks.vector_search.client import VectorSearchClient

    return VectorSearchClient(disable_notice=True).get_index(VS_ENDPOINT, INDEX_NAME)


def generate_json(prompt, schema=None, system_instruction=None, temperature=0.0, tries=5):
    """Gemini 호출 + JSON 강제. 503(과부하)은 기다리면 풀리므로 5·10·20·40초 간격으로 재시도."""
    from google.genai import types

    for attempt in range(tries):
        try:
            r = _gemini().models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=temperature,
                    response_mime_type="application/json", response_schema=schema))
            return json.loads((r.text or "").strip() or "{}")
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def search_many(queries, k=TOP_K, country=None):
    """claim 여러 개 검색 → 중복 규정 병합(최고 점수). 반환: (rule_id, category, title, text, severity, score)."""
    best = {}
    for q in queries:
        res = _index().similarity_search(
            query_text=q,
            columns=["rule_id", "category", "title", "text", "severity"],
            num_results=k,
            filters={"country": country} if country else None,
        )
        for row in res["result"].get("data_array", []):
            score = row[-1]  # 마지막 컬럼 = 유사도 점수
            if row[0] not in best or score > best[row[0]][-1]:
                best[row[0]] = (*row[:5], score)
    return sorted(best.values(), key=lambda r: r[-1], reverse=True)


@mlflow.trace(span_type="LLM")
def extract_claims(ad_text):
    payload = generate_json(CLAIM_PROMPT.format(ad_text=ad_text), CLAIM_SCHEMA,
                            system_instruction="당신은 광고 심의 실무자입니다. 검증이 필요한 주장만 정확히 골라냅니다.")
    return list(dict.fromkeys(c.strip() for c in payload.get("claims", []) if c.strip()))


@mlflow.trace(span_type="RETRIEVER")
def retrieve(claims, country, k=TOP_K):
    return search_many(claims, k=k, country=country)


@mlflow.trace(span_type="LLM")
def judge(ad_text, claims, hits, country):
    rules = "\n".join(f"- {h[0]} | {h[1]} | {h[2]} (심각도: {h[4]})\n    {h[3]}" for h in hits)
    payload = generate_json(
        JUDGE_PROMPT.format(country=country, country_context=COUNTRY_CONTEXT.get(country, ""),
                            ad_text=ad_text, claims="\n".join(f"- {c}" for c in claims),
                            rules=rules or "(검색된 규정이 없습니다.)"),
        JUDGE_SCHEMA,
        system_instruction="당신은 광고 심의 법무 검토자입니다. 주어진 규정만 근거로 판단합니다.")
    # ★ 로컬과 동일한 grounding 체크: 검색 안 된 rule_id는 코드에서 폐기
    return parse_review_payload(payload, ad_text, claims, [h[0] for h in hits])


@mlflow.trace(span_type="CHAIN")
def review(ad_text, country="KR", k=TOP_K):
    claims = extract_claims(ad_text)
    if not claims:
        return parse_review_payload({"decision": "PASS", "risk_score": 0.0, "issues": []}, ad_text, [], [])
    return judge(ad_text, claims, retrieve(claims, country, k), country)


class AdReviewer(mlflow.pyfunc.PythonModel):
    """서빙용 껍데기. 입력: ad_text, country 컬럼 / 출력: 심사 결과 JSON 문자열(한 행당 하나)."""

    def predict(self, context, model_input, params=None):
        return [review(r["ad_text"], country=r.get("country") or "KR").to_json(indent=None)
                for r in model_input.to_dict("records")]


mlflow.models.set_model(AdReviewer())
