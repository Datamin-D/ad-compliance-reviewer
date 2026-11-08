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

# REPO_PATH 자동 인식: 이 노트북은 <repo>/databricks/ 안에 있으므로 상위 폴더가 repo 루트.
# (Git folder로 가져왔다면 여기서 src/ 를 찾는다. 자동 인식이 안 되면 아래 줄을 직접 채운다.)
import os, sys
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    REPO_PATH = "/Workspace" + os.path.dirname(os.path.dirname(_nb))
except Exception:
    REPO_PATH = "/Workspace/Users/min91155@gmail.com/ad-compliance-reviewer"  # ← 수동 fallback (폴더 우클릭 → Copy path)
for p in (REPO_PATH, f"{REPO_PATH}/databricks"):
    if p not in sys.path:
        sys.path.append(p)
print("REPO_PATH =", REPO_PATH)

MLFLOW_EXPERIMENT = "/Users/min91155@gmail.com/adcompliance-rag"
# 나머지 CONFIG(카탈로그·스키마·인덱스·모델명·TOP_K)는 serving_model.py 맨 위 한 곳에만 있다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-a) 키 입력창 만들기 — **이 셀부터 실행**
# MAGIC 실행하면 **노트북 맨 위(제목 바로 아래)** 에 `Gemini API Key` 입력창이 생긴다.
# MAGIC 거기에 키를 붙여넣은 뒤 **다음 셀(1-b)** 로 넘어갈 것. 이 셀은 에러 없이 끝난다.
# MAGIC (코드에 키를 안 써서 GitHub에 안 올라간다. 키 발급: https://aistudio.google.com/apikey )

# COMMAND ----------

dbutils.widgets.text("gemini_api_key", "", "Gemini API Key (여기에 붙여넣기)")   # noqa: F821
print("✅ 입력창을 만들었습니다.")
print("   → 노트북 맨 위로 스크롤(Ctrl+Home)해서 'Gemini API Key' 칸에 키를 붙여넣으세요.")
print("   → 붙여넣은 뒤 아래 1-b 셀을 실행하면 됩니다. (입력창이 안 보이면 상단 Widgets 아이콘을 눌러 펼치세요)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-b) 키 읽어서 Gemini 연결
# MAGIC 위 입력창에 키를 넣은 **뒤에** 실행. 비어 있으면 여기서 멈춘다(정상 — 키를 넣고 다시 실행).

# COMMAND ----------

os.environ["GEMINI_API_KEY"] = dbutils.widgets.get("gemini_api_key")           # noqa: F821
assert os.environ["GEMINI_API_KEY"], "↑ 노트북 맨 위 'Gemini API Key' 입력창에 키를 붙여넣고 이 셀을 다시 실행하세요."
print("✅ 키 확인")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2~3) `review()` 가져오기 + MLflow Tracing
# MAGIC 심사 로직(Gemini 호출·Vector Search 검색·판정)은 **`serving_model.py` 한 곳**에 있다.
# MAGIC 여기서 평가하는 코드 = `03`에서 Model Serving으로 배포하는 코드.
# MAGIC 각 단계(@mlflow.trace)가 span으로 남는다 = 모든 판정의 **감사 추적**.

# COMMAND ----------

import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT)

from serving_model import review, CATALOG, SCHEMA, TOP_K, GEMINI_MODEL, EMBED_ENDPOINT
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.eval_set"

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
    mlflow.log_param("embed", EMBED_ENDPOINT)
    mlflow.log_param("judge", GEMINI_MODEL)

    cases, errors = [], []
    for row in eval_rows:
        try:
            res = review(row["ad_text"], country=row.get("country", "KR"), k=TOP_K)
        except Exception as e:   # 한 건 실패(예: Gemini 503)로 25건 전체가 멈추지 않게. 실패 건은 따로 기록
            errors.append({"id": row["id"], "error": str(e)[:300]})
            print("❌", row["id"], str(e)[:100])
            continue
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
        "n_scored": len(cases),                                     # 점수에 들어간 건수
        "n_errors": len(errors),                                    # 실패해서 빠진 건수 (0이어야 깔끔)
    })
    mlflow.log_table(pd.DataFrame(cases), "cases.json")             # per-case 상세(오답 분석)
    if errors:
        mlflow.log_table(pd.DataFrame(errors), "errors.json")
    print(f"완료: {len(cases)}건 채점, {len(errors)}건 실패. Experiments → adcompliance-rag → Evaluation runs 에서 확인.")
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
