# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Model Serving 배포 — 심사기를 API로
# MAGIC
# MAGIC `serving_model.py`(02에서 평가한 **바로 그 코드**)를 MLflow 모델로 기록 → Unity Catalog에 등록 → 서빙 엔드포인트.
# MAGIC 끝나면 다른 시스템이 HTTP 요청 한 번으로 광고 문구를 심사받을 수 있다.
# MAGIC
# MAGIC **전제:** 01의 Vector Search 엔드포인트·인덱스가 살아 있을 것(02 평가가 돌았다면 OK).
# MAGIC ⏱ 엔드포인트 준비 15~30분.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch mlflow google-genai databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

import os, sys
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    REPO_PATH = "/Workspace" + os.path.dirname(os.path.dirname(_nb))
except Exception:
    REPO_PATH = "/Workspace/Users/min91155@gmail.com/ad-compliance-reviewer"  # ← 수동 fallback
for p in (REPO_PATH, f"{REPO_PATH}/databricks"):
    if p not in sys.path:
        sys.path.append(p)

from serving_model import CATALOG, SCHEMA, INDEX_NAME, EMBED_ENDPOINT

MODEL_NAME = f"{CATALOG}.{SCHEMA}.ad_reviewer"     # UC에 등록될 모델 이름
SERVING_ENDPOINT = "ad-reviewer"                    # 서빙 엔드포인트 이름(= API 주소의 일부)
SECRET_SCOPE, SECRET_KEY = "adcompliance", "gemini_api_key"
MLFLOW_EXPERIMENT = "/Users/min91155@gmail.com/adcompliance-rag"

dbutils.widgets.text("gemini_api_key", "", "Gemini API Key (여기에 붙여넣기)")   # noqa: F821
print("REPO_PATH =", REPO_PATH, "\n→ 맨 위 'Gemini API Key' 칸에 키를 붙여넣고 다음 셀로")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Gemini 키를 비밀 금고(secret scope)에 저장
# MAGIC 서빙 서버는 노트북 입력창을 볼 수 없다. 그래서 키를 Databricks 금고에 넣고, 서버는 금고에서 꺼내 쓴다(코드·Git엔 안 남음).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

key = dbutils.widgets.get("gemini_api_key")   # noqa: F821
assert key, "↑ 맨 위 입력창에 키를 붙여넣고 이 셀을 다시 실행하세요."
os.environ["GEMINI_API_KEY"] = key            # 2)에서 모델을 기록하기 전 시험 실행에 쓰임

if SECRET_SCOPE not in [s.name for s in w.secrets.list_scopes()]:
    w.secrets.create_scope(scope=SECRET_SCOPE)
w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY, string_value=key)
print("✅ 금고 저장:", f"{SECRET_SCOPE}/{SECRET_KEY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) MLflow 모델로 기록 + Unity Catalog 등록
# MAGIC - `python_model`에 **파일 경로**를 준다(models-from-code) → 그 파일이 곧 모델.
# MAGIC - `code_paths`로 `src/`(프롬프트·grounding 체크)를 같이 싣는다.
# MAGIC - `resources`에 검색 인덱스를 적어두면 서빙 서버가 **검색 권한을 자동으로** 받는다.
# MAGIC - `signature`(입력: ad_text·country / 출력: JSON 문자열)를 직접 적는다 → UC 등록 필수 조건. 코드 검증은 02의 25건 평가가 이미 했다.

# COMMAND ----------

import mlflow
from importlib.metadata import version
from mlflow.models import ModelSignature
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.types import ColSpec, Schema

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# 입출력 형식표(signature) — UC는 이게 없으면 등록을 거부한다.
# 직접 적어두면 MLflow가 '예시로 시험 심사를 돌려 추측'하지 않는다(그 시험이 노트북 환경 탓에 깨졌었음).
signature = ModelSignature(
    inputs=Schema([ColSpec("string", "ad_text"), ColSpec("string", "country")]),
    outputs=Schema([ColSpec("string")]),   # 심사 결과 JSON 문자열
)

with mlflow.start_run(run_name="ad-reviewer-model"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=f"{REPO_PATH}/databricks/serving_model.py",
        code_paths=[f"{REPO_PATH}/src"],
        signature=signature,
        resources=[DatabricksVectorSearchIndex(index_name=INDEX_NAME),
                   DatabricksServingEndpoint(endpoint_name=EMBED_ENDPOINT)],
        pip_requirements=[f"{p}=={version(p)}" for p in ("mlflow", "databricks-vectorsearch", "google-genai")],
        registered_model_name=MODEL_NAME,
    )
VERSION = str(info.registered_model_version)
print(f"✅ 등록: {MODEL_NAME} v{VERSION}  (사이드바 Catalog → {CATALOG} → {SCHEMA} → Models 에서 보임)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) 서빙 엔드포인트 만들기 (또는 새 버전으로 교체)
# MAGIC 실행은 몇 초면 끝나고, 실제 준비는 백그라운드에서 15~30분.
# MAGIC 사이드바 **Serving → `ad-reviewer`** 에서 State가 **Ready**가 되면 4)로.

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

entity = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=VERSION,
    workload_size="Small",
    scale_to_zero_enabled=True,   # 안 쓰면 0대로 줄어 비용 0. 대신 첫 요청은 깨어나느라 느림
    environment_vars={"GEMINI_API_KEY": "{{secrets/adcompliance/gemini_api_key}}"},   # 1)의 금고에서 꺼냄
)

if SERVING_ENDPOINT in [e.name for e in w.serving_endpoints.list()]:
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[entity])
    print("기존 엔드포인트를 새 버전으로 교체 시작:", SERVING_ENDPOINT, "v" + VERSION)
else:
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[entity]))
    print("엔드포인트 생성 시작:", SERVING_ENDPOINT, "→ Serving 화면에서 Ready 기다리기")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) 호출 테스트 — Ready가 된 뒤에 실행
# MAGIC 이제 이 노트북이 아니라 **API**로 심사한다. 다른 팀 시스템도 이 주소로 똑같이 요청하면 된다.

# COMMAND ----------

import json
r = w.serving_endpoints.query(
    name=SERVING_ENDPOINT,
    dataframe_records=[{"ad_text": "전기료 50% 절감! 새로운 인버터 에어컨.", "country": "KR"}],
)
out = json.loads(r.predictions[0])
print("판정:", out["decision"], "| 위험도:", out["risk_score"])
for i in out["issues"]:
    print(f"- [{i['rule_id']}] {i['claim']} → {i['suggestion']}")
