# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · 데이터 랜딩(Unity Catalog) + Vector Search 인덱스
# MAGIC
# MAGIC 로컬 `data/policies/*.jsonl` → **UC Delta 테이블** → **Vector Search Delta-Sync 인덱스**.
# MAGIC
# MAGIC > ⚠️ **스캐폴드.** 셀 단위로 실행하며 워크스페이스 런타임/라이브러리 버전에 맞춰 확인할 것.
# MAGIC > CONFIG 셀의 이름값은 전부 placeholder.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch mlflow
# MAGIC %restart_python

# COMMAND ----------

# ---- CONFIG (본인 워크스페이스 값으로) ------------------------------------
CATALOG   = "workspace"             # Free Edition 기본 카탈로그(내 워크스페이스에 있는 이름으로)
SCHEMA    = "default"
VOLUME    = "raw"                   # jsonl 업로드용 UC Volume
VS_ENDPOINT = "adcompliance-vs"     # Vector Search 엔드포인트 이름
EMBED_ENDPOINT = "databricks-gte-large-en"   # FM API 관리형 임베딩(무설정). Gemini 임베딩 게이트웨이로 바꿔도 됨

POLICIES_TABLE = f"{CATALOG}.{SCHEMA}.policies"
EVAL_TABLE     = f"{CATALOG}.{SCHEMA}.eval_set"
INDEX_NAME     = f"{CATALOG}.{SCHEMA}.policies_index"
VOLUME_PATH    = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"   # 여기에 KR/US/EU.jsonl, eval_set.jsonl 업로드
# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Unity Catalog: catalog / schema / volume
# MAGIC 규정을 **거버넌스되는 테이블**에 둔다 = 접근제어·계보·타임트래블(버전)이 공짜로 붙는다.
# MAGIC 컴플라이언스 도메인엔 이게 곧 감사 추적이다.

# COMMAND ----------

# CATALOG 생성은 Free Edition에서 막힐 수 있다 → 실패하면 이미 있는 카탈로그를 쓰면 되니 무시하고 넘어간다.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    print("CREATE CATALOG 스킵(이미 있거나 권한 없음 — CONFIG의 CATALOG를 실제 존재하는 이름으로):", e)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("업로드 위치:", VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ⛔ 여기서 멈추고 파일 업로드 (다음 셀 실행 전)
# MAGIC 사이드바 **Catalog** → 위 CONFIG의 카탈로그 → 스키마 → **Volumes** → **`raw`** 클릭 →
# MAGIC 오른쪽 위 **Upload to this volume** → 내 PC의 **KR.jsonl · US.jsonl · EU.jsonl · eval_set.jsonl** 4개를 끌어다 놓기.
# MAGIC 업로드가 끝나면 아래 셀부터 계속 실행.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) jsonl → Delta 테이블
# MAGIC **파일명 = 국가코드** 규칙을 그대로 살려 `country` 컬럼을 채운다(로컬 `load_policies`와 동일 로직).
# MAGIC `document` 컬럼 = 임베딩 대상 텍스트 = 로컬 `Policy.to_document()`(`[category] title\ntext`).

# COMMAND ----------

from pyspark.sql import functions as F

# 국가별 파일을 읽어 country 컬럼 부여 (파일명이 곧 국가코드)
policies = None
for cc in ["KR", "US", "EU"]:
    df = (spark.read.json(f"{VOLUME_PATH}/{cc}.jsonl")
                .withColumn("country", F.lit(cc)))
    policies = df if policies is None else policies.unionByName(df)

policies = policies.withColumn(
    "document", F.concat(F.lit("["), F.col("category"), F.lit("] "),
                         F.col("title"), F.lit("\n"), F.col("text"))
)

(policies.write.mode("overwrite").saveAsTable(POLICIES_TABLE))

# Vector Search Delta-Sync는 소스 테이블에 Change Data Feed가 켜져 있어야 한다
spark.sql(f"ALTER TABLE {POLICIES_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

(spark.read.json(f"{VOLUME_PATH}/eval_set.jsonl")
      .write.mode("overwrite").saveAsTable(EVAL_TABLE))

display(spark.table(POLICIES_TABLE).groupBy("country").count())   # KR 28 / US 7 / EU 7 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Vector Search 엔드포인트
# MAGIC ⏱ **생성에 10~20분.** 이 시간에 `02` 노트북 CONFIG를 채워둘 것.
# MAGIC (여기서 막히면 = Free Edition 기능 게이팅. 클라우드 트라이얼로 전환.)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# 이미 있으면 건너뜀
try:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
except Exception as e:
    print("엔드포인트 생성 스킵/에러(이미 존재?):", e)

vsc.wait_for_endpoint(name=VS_ENDPOINT, verbose=True)   # 준비될 때까지 대기

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Delta-Sync 인덱스
# MAGIC `country`를 조회 컬럼으로 포함 → 검색 때 `filters={"country": ...}` 로 관할 분리.
# MAGIC 로컬의 `where={"country": ...}` 와 정확히 같은 역할. **다만 이제는 테이블이 바뀌면 인덱스가 자동 동기화.**

# COMMAND ----------

index = vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=INDEX_NAME,
    source_table_name=POLICIES_TABLE,
    pipeline_type="TRIGGERED",          # 수동 트리거 동기화(비용↓). 실시간이면 "CONTINUOUS"
    primary_key="rule_id",
    embedding_source_column="document", # 이 컬럼을 임베딩
    embedding_model_endpoint_name=EMBED_ENDPOINT,
)
index.wait_until_ready(verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) 스모크 테스트 — 같은 질의, 국가만 바꾸면 결과가 갈린다

# COMMAND ----------

def _search(q, country=None, k=5):
    return index.similarity_search(
        query_text=q,
        columns=["rule_id", "category", "title", "text", "country"],
        num_results=k,
        filters={"country": country} if country else None,
    )

print("US:", [r[0] for r in _search("eco-friendly zero emissions", "US")["result"]["data_array"]])
print("EU:", [r[0] for r in _search("eco-friendly zero emissions", "EU")["result"]["data_array"]])
# 기대: US-*, EU-* 만 각각 나온다 (관할 격리 확인)
