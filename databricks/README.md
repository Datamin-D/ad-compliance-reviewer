# Databricks 포팅 — 초보용 클릭 가이드

로컬(ChromaDB + Gemini) 프로토타입을 Databricks의 두 플래그십 기능으로 업그레이드한다:
**Vector Search**(관리형 벡터 검색) + **MLflow**(추적 평가·트레이싱). 규정 데이터는 **Unity Catalog** 테이블에 둔다.

> Databricks를 처음 켜본다는 가정하에, **어느 버튼을 누르는지**까지 적었다. 화면 문구는 버전에 따라
> 조금씩 다를 수 있으니 "비슷한 이름의 버튼"을 찾으면 된다.

---

## 큰 그림 (3부)

| 부 | 하는 일 | 도구 | 시간 |
|---|---|---|---|
| **A. 준비** | 워크스페이스 만들고, 코드·데이터를 올린다 | GitHub, Catalog | ~30분 |
| **B. 인덱스** | 규정을 Delta 테이블 → Vector Search 인덱스로 | `01` 노트북 | ~1시간(대기 포함) |
| **C. RAG+평가** | 검색+판정 재구성, MLflow로 평가·추적 | `02` 노트북 | ~2시간 |

> **비밀 키/게이트웨이는 기본 경로에서 뺐다.** 초보가 5시간 안에 막히는 지점이라, Gemini는 노트북 상단
> **입력창(widget)** 에 붙여넣는 방식으로 바로 쓰게 했다. Model Serving 게이트웨이는 여유 있을 때
> `02` 맨 아래 "선택" 섹션에서 체험하면 된다.

---

# A. 준비

## A-1. Databricks Free Edition 계정 (~5분)
1. 브라우저에서 **https://www.databricks.com/learn/free-edition** 접속 → **Get started / Sign up**.
2. Google 계정(min91155@gmail.com)으로 로그인하면 됨.
3. 로그인하면 `https://…cloud.databricks.com` 형태의 **워크스페이스**가 열린다. 여기서 다 한다.
   - 화면 **왼쪽 세로 막대(사이드바)** 에 Workspace / Catalog / Compute / Experiments … 메뉴가 있다. 이게 이동 메뉴다.

## A-2. 내 코드를 GitHub에 올리기
이미 계획대로 이 리포를 **GitHub(public)** 로 push해 둔다. (안 했으면 먼저 push — 루트 README의 `git` 명령 참고)
→ Databricks가 이 GitHub 주소로 코드를 통째로 가져온다. 그래야 `src/`(공유 로직)와 `databricks/` 노트북이
한 번에 들어온다.

## A-3. GitHub 코드를 Databricks로 가져오기 (Git folder) (~3분)
1. 사이드바 **Workspace** 클릭.
2. 가운데에 내 폴더가 보인다. 오른쪽 위 파란 **Create** 버튼 클릭 → **Git folder** 선택.
   (예전 이름은 "Repo". 비슷한 걸 고르면 됨)
3. **Git repository URL** 칸에 내 GitHub 리포 주소(`https://github.com/<나>/<리포>.git`) 붙여넣기.
   public이면 로그인 불필요.
4. **Create Git folder** 클릭 → 잠시 후 `RAG` 폴더가 생기고 그 안에 `src/`, `databricks/` 가 다 보인다.

## A-4. 규정 데이터 파일 올리기 (Volume 업로드) (~5분)
Databricks는 파일을 **Volume**(카탈로그 안의 파일 저장소)에 둔다. 내 PC의 4개 파일을 올린다:
`data/policies/KR.jsonl`, `US.jsonl`, `EU.jsonl`, `data/eval_set.jsonl`.

1. 사이드바 **Catalog** 클릭.
2. 왼쪽 목록에서 **쓸 수 있는 카탈로그**를 하나 정한다. Free Edition은 보통 `workspace` 카탈로그가 이미 있다.
   → 이 이름을 기억해 둔다(뒤에서 노트북 CONFIG에 넣는다).
3. 그 카탈로그를 펼쳐 스키마(예: `default`)를 클릭 → 오른쪽 위 **Create** → **Volume** →
   이름 `raw` 입력 → **Create**.
4. 방금 만든 `raw` 볼륨을 클릭 → 오른쪽 위 **Upload to this volume** 클릭 →
   내 PC에서 **4개 jsonl 파일을 끌어다 놓기** → 업로드.

> 파일 경로는 `/Volumes/<카탈로그>/<스키마>/raw/KR.jsonl` 형태가 된다. 이 경로를 노트북이 읽는다.

---

# B. 인덱스 만들기 — `01_setup_and_index` 노트북

## B-1. 노트북 열기 & 컴퓨트 붙이기
1. 사이드바 **Workspace** → `RAG` → `databricks` → **`01_setup_and_index`** 더블클릭. 노트북이 열린다.
2. 오른쪽 위 **Connect**(또는 컴퓨트 드롭다운) 클릭 → **Serverless** 선택.
   (Free Edition은 서버리스라 보통 자동으로 붙어 있다. "Connected"면 OK)

## B-2. CONFIG 고치기
맨 위 **CONFIG 셀**에서 내 값으로 바꾼다:
- `CATALOG` → A-4에서 정한 카탈로그 이름(예: `workspace`)
- `SCHEMA` → 그 스키마(예: `default`)
- `VOLUME` → `raw`
- 나머지(엔드포인트 이름 등)는 그대로 둬도 된다.

## B-3. 위에서부터 셀 실행
- 셀을 클릭하고 **Shift+Enter** (또는 셀 왼쪽의 ▶). 위에서 아래로 순서대로.
- 맨 위 `%pip install …` 셀은 라이브러리를 깐다. 그 다음 `%restart_python` 은 커널을 재시작(정상).
- **catalog/schema/volume 만드는 셀**에서 "권한 없음/이미 존재" 에러가 나면: 이미 만들어 둔 걸 쓰면 되니
  CONFIG의 이름만 실제 존재하는 것으로 맞추고 그 셀은 건너뛴다.

## B-4. ⏱ Vector Search 엔드포인트 대기 (10~20분)
- "3) Vector Search 엔드포인트" 셀을 실행하면 **만드는 데 10~20분** 걸린다. 셀이 대기하며 점점 진행된다.
- **이 시간에 `02` 노트북을 열어 CONFIG를 미리 채워둔다.**
- (진행상황을 눈으로 보고 싶으면: 사이드바 **Compute** → 상단 **Vector Search** 탭 → 내 엔드포인트 상태)
- 여기서 "기능을 못 쓴다" 류 에러가 나면 = Free Edition에서 Vector Search가 막힌 것.
  → 클라우드 **14일 트라이얼**(AWS/Azure/GCP 중 하나)로 갈아타면 된다.

## B-5. 인덱스 + 스모크 테스트
- "4) Delta-Sync 인덱스" 셀 실행 → 준비될 때까지 대기.
- 마지막 "5) 스모크 테스트" 셀 실행 → **US 질의엔 `US-*`, EU 질의엔 `EU-*`** 만 나오면 성공(관할 격리 확인).

---

# C. 검색 + 평가 — `02_rag_and_eval` 노트북

## C-1. 열기 & CONFIG
1. `databricks/02_rag_and_eval` 열고 **Serverless** 연결.
2. CONFIG 셀에서 `CATALOG`/`SCHEMA`를 `01`과 똑같이 맞춘다.
   (`REPO_PATH`는 노트북이 자동으로 알아내니 건드릴 필요 없음)

## C-2. Gemini 키 입력 (초보용: 붙여넣기)
1. `%pip install …` 셀과 그 아래 **위젯 생성 셀**을 실행하면, 노트북 **맨 위에 입력창**이 하나 생긴다
   (`gemini_api_key` 라는 라벨).
2. 그 칸에 **Google AI Studio 키를 붙여넣는다**(https://aistudio.google.com/apikey).
   → 코드에 키를 안 써서 GitHub에 올라갈 위험이 없다.

## C-3. 위에서부터 실행
- Shift+Enter로 순서대로. "스모크" 셀에서 `EU` 판정 결과가 찍히면 검색+판정이 붙은 것이다.
- 그 다음 **"4) MLflow 평가" 셀**이 핵심: eval_set 25건을 돌려 지표를 MLflow에 기록한다(1~2분).

## C-4. 결과 보기 (MLflow) — 여기가 하이라이트
1. 사이드바 **Experiments**(머신러닝 그룹 안) 클릭.
2. `adcompliance-rag` 실험 클릭 → **run 목록**이 보인다. run 하나 클릭.
3. **Metrics**에서 `risky_recall`(★ 위험 광고를 놓치지 않는 비율), `retrieval_recall_at_k`(검색 품질) 확인.
4. **Traces** 탭 → 광고 한 건의 내부 단계(claim추출 → 검색 → 판정)가 span으로 펼쳐진다 = 감사 추적.
5. (실험) `02` CONFIG의 `TOP_K`를 3으로 바꿔 다시 "4) 평가" 셀 실행 → run이 하나 더 생긴다.
   실험 화면에서 **두 run을 체크 → Compare** 로 top_k=3 vs 5를 나란히 비교.

---

# 막히면 / 함정 (초보가 자주 걸리는 것)

- **`import src...` 실패** → `RAG`를 A-3의 **Git folder로** 가져왔는지 확인(그래야 `src/`가 워크스페이스에 있음).
  `02`가 REPO_PATH를 자동으로 못 찾으면, CONFIG의 `REPO_PATH`에 `RAG` 폴더 경로를 직접 넣는다
  (폴더 우클릭 → **Copy path**).
- **파일을 못 읽음** → `/Volumes/<카탈로그>/<스키마>/raw/` 경로와 CONFIG의 카탈로그/스키마 이름이 같은지 확인.
- **Vector Search가 안 만들어짐** → Free Edition 제한. 클라우드 14일 트라이얼로.
- **Gemini 호출 에러** → C-2 입력창에 키를 넣고 그 셀을 실행했는지. 모델명(`GEMINI_MODEL`)이 지금 쓸 수 있는
  이름인지(예: `gemini-2.0-flash`).

# 선택(여유 있을 때) — Model Serving 게이트웨이 체험
`02` 맨 아래 "선택: External Model 게이트웨이" 섹션. Gemini를 Databricks 서빙 엔드포인트 뒤에 숨겨
키 관리·요청 로깅을 서빙 레이어에서 하는 걸 체험한다. **비밀 스코프 설정(CLI)이 필요**해서 초보에겐 마지막에 권한다.

# 우선순위 (5시간 안에 하나만 남긴다면)
**B(Vector Search) + C-4(MLflow 평가)**. 이 둘이 플래그십이자 면접 킬러다. Unity Catalog 거버넌스는
A에서 거의 공짜로 따라온다.
