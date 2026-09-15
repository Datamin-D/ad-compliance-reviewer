# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · RAG 재구성 + MLflow 평가/트레이싱
# MAGIC
# MAGIC `01`의 Vector Search 인덱스 + Gemini로 `review()`를 재구성한다.
# MAGIC **프롬프트·grounding 체크는 `src/`에서 그대로 재사용** — 도메인 로직은 재작성하지 않는다.
# MAGIC
# MAGIC > 초보 경로: Gemini 키는 **맨 위 입력창에 붙여넣기**, `REPO_PATH`는 자동 인식. 자세한 클릭은 README 참고.
# MAGIC > ⚠️ 스캐폴드 — 셀 단위로 실행하며 버전에 맞춰 확인할 것.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch mlflow google-genai
# MAGIC %restart_python

# COMMAND ----------

# ---- CONFIG (01과 같은 카탈로그/스키마로) -----------------------------------
CATALOG, SCHEMA = "workspace", "default"     # ← 01에서 쓴 이름과 똑같이
VS_ENDPOINT = "adcompliance-vs"
INDEX_NAME  = f"{CATALOG}.{SCHEMA}.policies_index"
EVAL_TABLE  = f"{CATALOG}.{SCHEMA}.eval_set"
GEMINI_MODEL = "gemini-3.6-flash"            # 지금 내 키로 쓸 수 있는 모델명
MLFLOW_EXPERIMENT = "/Users/min91155@gmail.com/adcompliance-rag"
TOP_K = 5
# ---------------------------------------------------------------------------

# REPO_PATH 자동 인식: 이 노트북은 <repo>/databricks/ 안에 있으므로 상위 폴더가 repo 루트.
# (Git folder로 가져왔다면 여기서 src/ 를 찾는다. 자동 인식이 안 되면 아래 줄을 직접 채운다.)
import os, sys
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    REPO_PATH = "/Workspace" + os.path.dirname(os.path.dirname(_nb))
except Exception:
    REPO_PATH = "/Workspace/Users/min91155@gmail.com/RAG"   # ← 수동 fallback (폴더 우클릭 → Copy path)
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
print("REPO_PATH =", REPO_PATH)

# ★ 도메인 로직 재사용 (chromadb·genai 의존 없는 leaf 모듈이라 그냥 import된다)
from src.prompts import CLAIM_PROMPT, CLAIM_SCHEMA, JUDGE_PROMPT, JUDGE_SCHEMA, COUNTRY_CONTEXT
from src.schema import parse_review_payload

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Gemini 키 입력
# MAGIC 이 셀을 실행하면 **노트북 맨 위에 입력창(`gemini_api_key`)** 이 생긴다. 거기에 키를 붙여넣고 셀을 다시 실행.
# MAGIC (코드에 키를 안 써서 GitHub에 안 올라간다. 키 발급: https://aistudio.google.com/apikey )

# COMMAND ----------

dbutils.widgets.text("gemini_api_key", "", "Gemini API Key (여기에 붙여넣기)")   # noqa: F821
GEMINI_API_KEY = dbutils.widgets.get("gemini_api_key")                          # noqa: F821
assert GEMINI_API_KEY, "↑ 맨 위 입력창에 Gemini 키를 붙여넣고 이 셀을 다시 실행하세요."

import json, time
from google import genai
from google.genai import types

_gemini = genai.Client(api_key=GEMINI_API_KEY)

def generate_json(prompt, schema=None, system_instruction=None, temperature=0.0):
    """Gemini 직접 호출. response_schema로 JSON을 '강제'(로컬 llm.generate_json과 동일 보장) + 짧은 재시도."""
    for attempt in range(3):
        try:
            r = _gemini.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=temperature,
                    response_mime_type="application/json", response_schema=schema))
            return json.loads((r.text or "").strip() or "{}")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Vector Search 검색기 (로컬 `search_many`와 같은 인터페이스)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
_index = VectorSearchClient(disable_notice=True).get_index(VS_ENDPOINT, INDEX_NAME)

def search_many(queries, k=TOP_K, country=None):
    """claim 여러 개 검색 → 중복 규정 병합(최고 점수). 반환: (rule_id, category, title, text, severity, score) 리스트."""
    best = {}
    for q in queries:
        res = _index.similarity_search(
            query_text=q,
            columns=["rule_id", "category", "title", "text", "severity"],
            num_results=k,
            filters={"country": country} if country else None,
        )
        for row in res["result"]["data_array"]:
            rid, cat, title, text, sev = row[0], row[1], row[2], row[3], row[4]
            score = row[-1]                       # 마지막 컬럼 = 유사도 점수
            if rid not in best or score > best[rid][-1]:
                best[rid] = (rid, cat, title, text, sev, score)
    return sorted(best.values(), key=lambda r: r[-1], reverse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) `review()` 재구성 + MLflow Tracing
# MAGIC 각 단계(@mlflow.trace)가 span으로 남는다 = 모든 판정의 **감사 추적**. 프롬프트·grounding은 `src/` 재사용.

# COMMAND ----------

import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT)

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
    # ★ 로컬과 동일한 grounding 체크 재사용: 검색 안 된 rule_id는 코드에서 폐기
    return parse_review_payload(payload, ad_text, claims, [h[0] for h in hits])

@mlflow.trace(span_type="CHAIN")
def review(ad_text, country="KR", k=TOP_K):
    claims = extract_claims(ad_text)
    if not claims:
        return parse_review_payload({"decision": "PASS", "risk_score": 0.0, "issues": []}, ad_text, [], [])
    return judge(ad_text, claims, retrieve(claims, country, k), country)

# 스모크 — EU 판정이 찍히면 검색+판정이 붙은 것
r = review("100% eco-friendly air conditioner, zero emissions!", country="EU")
print(r.decision.value, r.retrieved_rule_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) MLflow 평가 — risky-recall(자체 지표) + 검색 품질
# MAGIC 이 프로젝트의 1순위 지표는 **위험 광고를 놓치지 않는 Recall**. 추적 run으로 기록하고,
# MAGIC 검색 품질은 Recall@k로 분리 측정한다(로컬 `evaluate.py` 로직 이식).
# MAGIC 실행 후 사이드바 **Experiments → adcompliance-rag** 에서 확인.

# COMMAND ----------

import pandas as pd
eval_rows = spark.table(EVAL_TABLE).toPandas().to_dict("records")
RISKY = {"REVIEW", "BLOCK"}

with mlflow.start_run(run_name=f"rag-eval-topk{TOP_K}"):
    mlflow.log_param("top_k", TOP_K)
    mlflow.log_param("embed", "databricks-gte-large-en")
    mlflow.log_param("judge", GEMINI_MODEL)

    cases = []
    for row in eval_rows:
        res = review(row["ad_text"], country=row.get("country", "KR"), k=TOP_K)
        want = set(row.get("expected_rules") or [])
        got = set(res.retrieved_rule_ids)
        cases.append({
            "id": row["id"], "expected": row["expected"], "predicted": res.decision.value,
            "correct": row["expected"] == res.decision.value,
            "retrieval_hit": bool(want & got) if want else None,
            "cited": ",".join(sorted({i.rule_id for i in res.issues})),
        })

    tp = sum(c["expected"] in RISKY and c["predicted"] in RISKY for c in cases)
    fn = sum(c["expected"] in RISKY and c["predicted"] not in RISKY for c in cases)
    fp = sum(c["expected"] not in RISKY and c["predicted"] in RISKY for c in cases)
    scored = [c for c in cases if c["retrieval_hit"] is not None]

    mlflow.log_metrics({
        "accuracy": sum(c["correct"] for c in cases) / len(cases),
        "risky_recall": tp / (tp + fn) if tp + fn else 0.0,        # ★ 헤드라인
        "risky_precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_negatives": fn,                                      # 놓친 위험 광고 수
        "retrieval_recall_at_k": sum(bool(c["retrieval_hit"]) for c in scored) / len(scored) if scored else 0.0,
    })
    mlflow.log_table(pd.DataFrame(cases), "cases.json")             # per-case 상세(오답 분석)
    print("완료. Experiments → adcompliance-rag 에서 risky_recall / recall@k / Traces 확인.")
    print("실험: CONFIG의 TOP_K를 3으로 바꿔 이 셀만 다시 돌리면 run이 하나 더 → 두 run을 Compare.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 선택 A · External Model 게이트웨이로 Gemini 태우기 (여유 있을 때)
# MAGIC Gemini를 Databricks 서빙 엔드포인트 뒤에 숨긴다 → 키 관리·요청 로깅을 서빙 레이어에서.
# MAGIC **비밀 스코프(CLI)가 필요**해 초보에겐 마지막에 권한다. 성공하면 위 `generate_json`을 게이트웨이 버전으로 교체.
# MAGIC ```
# MAGIC # 터미널(로컬 Databricks CLI)에서 먼저:
# MAGIC #   databricks secrets create-scope adcompliance
# MAGIC #   databricks secrets put-secret  adcompliance gemini_api_key
# MAGIC ```

# COMMAND ----------

# from mlflow.deployments import get_deploy_client
# deploy = get_deploy_client("databricks")
# deploy.create_endpoint(
#     name="gemini-chat",
#     config={"served_entities": [{
#         "name": "gemini",
#         "external_model": {
#             "name": GEMINI_MODEL, "provider": "google", "task": "llm/v1/chat",
#             "google_ai_studio_config": {"google_api_key": "{{secrets/adcompliance/gemini_api_key}}"},
#         }}]})   # provider/필드명은 Databricks 버전 문서로 확인
#
# def generate_json(prompt, schema=None, system_instruction=None, temperature=0.0):
#     import json, re
#     msgs = ([{"role":"system","content":system_instruction}] if system_instruction else []) + \
#            [{"role":"user","content":prompt + "\n\n반드시 유효한 JSON만 출력하세요."}]
#     resp = deploy.predict(endpoint="gemini-chat", inputs={"messages": msgs, "temperature": temperature})
#     text = resp["choices"][0]["message"]["content"]
#     m = re.search(r"\{.*\}", text, re.DOTALL)   # 게이트웨이는 response_schema를 못 넘길 수 있어 방어 파싱
#     return json.loads(m.group(0)) if m else {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 선택 B · LLM judge 평가 (groundedness / correctness)
# MAGIC 손으로 짠 `parse_review_payload` grounding 체크가 MLflow **groundedness judge**로 승격되는 지점.
# MAGIC MLflow 버전에 맞는 셀만 사용.

# COMMAND ----------

# --- (B-1) MLflow 3 / databricks-agents : mlflow.genai.evaluate ---
# import mlflow.genai
# def predict_fn(ad_text, country="KR"):
#     r = review(ad_text, country=country)
#     return {"response": r.decision.value + " | " + "; ".join(i.reason for i in r.issues)}
# data = [{"inputs": {"ad_text": row["ad_text"], "country": row.get("country","KR")},
#          "expectations": {"expected_response": row["expected"]}} for row in eval_rows]
# mlflow.genai.evaluate(data=data, predict_fn=predict_fn,
#                       scorers=[mlflow.genai.scorers.Correctness(), mlflow.genai.scorers.Groundedness()])

# --- (B-2) 구형 mlflow.evaluate ---
# pdf = pd.DataFrame([{"inputs": row["ad_text"], "targets": row["expected"],
#                      "predictions": review(row["ad_text"], country=row.get("country","KR")).decision.value}
#                     for row in eval_rows])
# mlflow.evaluate(data=pdf, targets="targets", predictions="predictions", model_type="text")
