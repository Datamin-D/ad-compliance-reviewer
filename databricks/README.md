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

# 0. 개념 먼저 — 하나씩 쌓아서 설명

DS/ML 개념(벡터, 임베딩, 코사인 유사도, RAG)은 이미 아는 걸로 치고, **백엔드/인프라 용어만** 기초부터
쌓아 올린다. 순서대로 읽으면 뒤 개념이 앞 개념 위에 얹힌다.

### (1) 워크스페이스 = "내 코드가 실행되는 원격 컴퓨터들의 집합"에 접속하는 계정

로컬에서는 `python cli.py review ...`를 치면 **내 PC**가 그 코드를 실행했다. Databricks는 코드를
**내 PC가 아니라 Databricks가 관리하는 원격 컴퓨터에서** 실행시켜주는 서비스다. 브라우저로 접속하는
저 화면이 곧 그 원격 환경의 콘솔이고, "워크스페이스"는 그 환경 전체를 부르는 이름이다(계정 하나당
워크스페이스 하나가 발급됨). 노트북(코드), 데이터, 컴퓨트(연산 자원), 실험 기록이 전부 이 안에 있다.

### (2) 컴퓨트(compute) = 코드를 실제로 돌리는 원격 서버

노트북에 코드를 적어도, 그걸 실행할 **서버**가 있어야 한다. 이 서버(들)를 Databricks에서는
"컴퓨트"라고 부른다. 전통적으로는 "몇 개짜리 서버를 쓸지, 언제 끌지"를 사람이 직접 설정해야 했다.

**Serverless**는 그 설정을 안 해도 되는 방식이다 — 셀을 실행하는 순간 Databricks가 서버를 알아서
띄우고, 다 쓰면 알아서 끈다. 우리는 Free Edition에서 이것만 쓰면 되므로 "서버를 몇 대 쓸지" 같은
설정 화면은 신경 안 써도 된다. 노트북에서 **Connect → Serverless**를 누르는 게 "이 노트북을 실행할
컴퓨트를 하나 배정해달라"는 요청이다.

### (3) Unity Catalog = 데이터가 어디 있는지 찾는 이름 체계 + 누가 볼 수 있는지 정하는 규칙

로컬에서는 파일 경로(`data/policies/KR.jsonl`)로 데이터를 찾았다. Databricks에서는 데이터를
**3단계 이름**으로 찾는다: `카탈로그.스키마.테이블(또는 볼륨)`. 예를 들어 이 프로젝트에서 만들 이름은:

```
workspace.default.policies     ← 규정 데이터 (테이블)
workspace.default.eval_set     ← 평가셋 (테이블)
workspace.default.raw          ← jsonl 원본 파일들 (볼륨)
```

이게 왜 필요한가: 여러 사람이 같은 워크스페이스를 쓰면 "이 데이터를 누가 읽고 쓸 수 있는지"를 관리해야
한다. Unity Catalog는 이 3단계 이름 각각에 권한(읽기/쓰기 가능한 사람)을 걸 수 있는 시스템이다.
지금은 나 혼자 쓰니 권한 설정을 따로 안 해도 기본값으로 다 된다 — 다만 이름을 이 3단계로 지어야
한다는 규칙만 지키면 된다.

**테이블 vs 볼륨, 왜 둘로 나누나** —
- **볼륨**은 그냥 파일을 두는 곳이다. 우리가 올릴 jsonl 파일은 아직 "행과 열"로 정리되기 전의
  원본이라 여기 둔다. 로컬의 `data/` 폴더와 같은 역할.
- **테이블**은 행(row)·열(column)이 정해진 데이터다. jsonl을 읽어서 `rule_id`, `category`, `text`
  같은 열로 정리한 게 테이블이다. SQL로 `SELECT * FROM policies WHERE country='US'` 같은 조회가
  가능해진다(로컬에서 Chroma에 넣기 전에 `load_policies()`로 파싱하던 것과 같은 단계).

그래서 순서가 "jsonl → 볼륨에 업로드 → 코드로 읽어서 → 테이블로 저장"이다.

### (4) Delta Lake = 테이블을 저장하는 방식 (동시에 여러 명이 써도 안 깨지게)

일반 CSV 파일에 동시에 두 프로그램이 쓰기를 하면 파일이 깨질 수 있다(한쪽이 쓰는 중에 다른 쪽이
읽거나 덮어씀). Delta Lake는 이 문제를 이렇게 푼다: **데이터 파일을 직접 고치지 않고, 매번 새 파일을
추가하면서 "무엇이 바뀌었는지"를 로그에 순서대로 기록**한다. 이 로그는 사실 **git의 커밋 히스토리와
같은 아이디어**다 — git이 파일을 직접 수정하는 대신 커밋을 쌓아서 히스토리를 남기듯, Delta Lake는
쓰기를 할 때마다 로그 엔트리를 쌓는다. 그래서:

- **이전 버전 조회가 가능하다** (`git checkout <커밋>`처럼, 특정 시점의 테이블 상태를 다시 볼 수 있음).
- **"무엇이 바뀌었는지"만 뽑아낼 수 있다** — 이게 **Change Data Feed(CDF)**다. git으로 치면
  `git diff` 같은 것: 전체를 다시 훑지 않고 "이번에 추가/수정/삭제된 행"만 뽑는 기능.
  이 기능이 왜 필요하냐면, 다음 (5)번 Vector Search가 "규정이 하나 수정됐을 때 그 한 줄만 다시
  임베딩하면 되지, 42개를 통째로 다시 계산할 필요는 없다"를 하기 위해 이 CDF를 구독하기 때문이다.
  (지금은 데이터가 42줄이라 체감이 안 되지만, 실제 규정이 수천 줄이면 이 차이가 커진다.)
  노트북에서 `enableChangeDataFeed = true`라는 설정을 켜는 이유가 이것이다.

### (5) Vector Search = 로컬 Chroma와 똑같은 일을 하는, 클라우드에 떠 있는 검색 서비스

로컬에서 만든 `PolicyRetriever`는 이랬다: 내 PC 안의 폴더(`.chroma/`)에 벡터를 저장하고, 내 PC의
파이썬 프로세스가 검색을 계산했다. Databricks의 Vector Search는 **같은 일을 원격 서버에서** 한다.
그러려면 두 가지가 필요하다:

- **Endpoint** — 검색 요청을 받아줄 서버(정확히는 서버 몇 대). "요청을 보내면 응답해주는 프로그램이
  네트워크 너머에 계속 켜져 있다"는 뜻이다. 이걸 처음 만드는 데 10~20분 걸린다(원격 서버를 새로
  준비하는 시간).
- **Index** — 그 서버가 들고 있는 실제 벡터 데이터(로컬의 `.chroma/` 폴더 내용에 해당). 우리가 만드는
  "Delta-Sync 인덱스"는 (4)의 테이블 + CDF를 그대로 연결해서, **테이블이 바뀌면 인덱스도 자동으로
  갱신**되게 만든 것이다. 로컬에서는 규정을 고치면 `python cli.py index`를 손으로 다시 실행해야
  했는데, 여기선 그 재실행이 자동으로 일어난다.

검색할 때 부르는 `similarity_search()` 함수는 로컬의 `retriever.search()`와 하는 일이 완전히
같다: 쿼리 텍스트를 벡터로 바꾸고, 가장 가까운 top-k를 찾고, `country`로 필터링한다(로컬의
`where={"country": ...}`와 동일한 역할).

### (6) MLflow = 실험을 자동으로 기록해주는 도구

로컬의 `evaluate.py`는 결과를 화면에 출력하고 `results.json`에 저장했다. 여러 번 실험하면
(`top_k=3`으로 돌린 결과, `top_k=5`로 돌린 결과…) 그 파일들을 내가 직접 관리해야 한다. MLflow는
이걸 자동으로 정리해주는 시스템이다:

- **Experiment** — 실험들을 모아두는 폴더 하나(이 프로젝트는 `adcompliance-rag` 하나만 쓴다).
- **Run** — "한 번 실행한 기록" 하나. 코드가 `mlflow.log_metric("risky_recall", 0.9)`처럼 값을 남기면
  그게 이번 run에 저장된다. `top_k=5`로 한 번, `top_k=3`으로 한 번 실행하면 run이 2개 쌓이고,
  화면에서 두 run을 선택해 표로 나란히 비교할 수 있다(엑셀에 실험 결과를 표로 정리하던 걸 자동화한 것).
- **Trace(트레이스)** — 함수 하나가 내부적으로 다른 함수를 호출하는 과정을 전부 기록해서 나무 구조로
  보여주는 기능. 예를 들어 `review()` 함수가 `extract_claims()` → `retrieve()` → `judge()`를 차례로
  부르는데, 각 호출이 얼마나 걸렸고 무엇을 입출력했는지가 전부 남는다. `print()`를 여기저기 넣어
  디버깅하던 걸 자동으로, 보기 좋게 해주는 것에 해당한다.

---

# A. 준비

## A-1. Databricks Free Edition 계정 (~5분)
1. 브라우저에서 **https://www.databricks.com/learn/free-edition** 접속 → **Get started / Sign up**.
2. Google 계정(min91155@gmail.com)으로 로그인하면 됨.
3. 로그인하면 `https://…cloud.databricks.com` 형태의 워크스페이스 인스턴스가 열린다. 여기서 다 한다.
   - 화면 **왼쪽 세로 막대(사이드바)** 에 Workspace / Catalog / Compute / Experiments … 메뉴가 있다. 이게 이동 메뉴다.

## A-2. 내 코드를 GitHub에 올리기
이미 계획대로 이 리포를 **GitHub(public)** 로 push해 둔다. (안 했으면 먼저 push — 루트 README의 `git` 명령 참고)
→ Databricks가 이 GitHub 주소로 코드를 통째로 가져온다. 그래야 `src/`(공유 로직)와 `databricks/` 노트북이
한 번에 들어온다.

## A-3. GitHub 코드를 Databricks로 가져오기 (Git folder) (~3분)
1. 사이드바 **Workspace** 클릭 → 왼쪽 트리에서 **🏠 Home** 클릭.
   > ⚠️ 왼쪽에 **Repos**라는 폴더도 보이는데, **거기 아니다.** Repos는 예전 방식이라 비어 있는 게 정상.
   > 요즘 Git folder는 **Home 아래**에 만든다. (화면에도 "Go to home folder and create Git folder" 안내가 뜬다)
2. 오른쪽 위 파란 **Create** 버튼 클릭 → **Git folder** 선택.
3. **Git repository URL** 칸에 내 GitHub 리포 주소(`https://github.com/<나>/<리포>.git`) 붙여넣기.
   public이면 로그인 불필요. **Git folder name**은 리포 이름이 자동으로 채워진다(예: `ad-compliance-reviewer`).
4. **Create Git folder** 클릭 → **Home 아래에** 그 이름의 폴더가 생기고, 안에 `src/`, `databricks/`, `data/` 가 보인다.

> 이후 이 폴더 이름을 **`<리포폴더>`** 라고 부른다. 내 경우엔 `ad-compliance-reviewer`.

## A-4. 규정 데이터 파일 올리기 (Volume 업로드) (~5분)
jsonl은 아직 스키마가 확정 안 된 원본이라 **Volume**에 올린다(0장 참고).

> ⚠️ **올릴 파일은 4개인데 두 폴더에 나뉘어 있다.** `policies` 폴더만 드래그하면 `eval_set.jsonl`이
> 빠져서 나중에 `[PATH_NOT_FOUND] ... eval_set.jsonl` 에러가 난다.
> ```
> data\policies\KR.jsonl      ┐
> data\policies\US.jsonl      ├─ 규정 3개
> data\policies\EU.jsonl      ┘
> data\eval_set.jsonl         ← 한 단계 위 폴더! 잊지 말 것
> ```

1. 사이드바 **Catalog** 클릭.
2. 왼쪽 목록에서 쓸 수 있는 카탈로그를 하나 정한다. Free Edition은 보통 `workspace` 카탈로그가 이미 있다.
   → 이 이름을 기억해 둔다(뒤에서 노트북 CONFIG에 넣는다).
3. 그 카탈로그를 펼쳐 스키마(예: `default`)를 클릭 → 오른쪽 위 **Create** → **Volume** →
   이름 `raw` 입력 → **Create**.
4. 방금 만든 `raw` 볼륨을 클릭 → 오른쪽 위 **Upload to this volume** 클릭 →
   내 PC에서 **4개 jsonl 파일을 끌어다 놓기** → 업로드.

> 파일 경로는 `/Volumes/<카탈로그>/<스키마>/raw/KR.jsonl` 형태가 된다. 이 경로를 노트북이 읽는다.

---

# B. 인덱스 만들기 — `01_setup_and_index` 노트북

## B-1. 노트북 열기 & 컴퓨트 붙이기
1. 사이드바 **Workspace** → **🏠 Home** → **`<리포폴더>`**(예: `ad-compliance-reviewer`) → `databricks`
   → **`01_setup_and_index`** 더블클릭. 노트북이 열린다.
   > 안 보이면 Repos 폴더를 보고 있는 것. **Home**으로 가야 한다(A-3 참고).
2. 오른쪽 위 **Connect**(또는 컴퓨트 드롭다운) 클릭 → **Serverless** 선택.
   (Free Edition은 보통 자동으로 붙어 있다. "Connected"면 OK)

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
- **테이블로 저장하는 셀**(jsonl → `policies` 테이블)에서 `enableChangeDataFeed = true` 를 켜는 줄이 있다
  (0장의 CDF). 이게 있어야 다음 단계의 Delta-Sync 인덱스가 생성된다. 안 켜면 인덱스 생성 단계에서 에러.

## B-4. ⏱ Vector Search 엔드포인트 대기 (10~20분)
- "3) Vector Search 엔드포인트" 셀을 실행하면 endpoint 프로비저닝에 **10~20분** 걸린다. 셀이 대기하며
  진행된다. **이 시간에 `02` 노트북을 열어 CONFIG를 미리 채워둔다.**
- (진행상황을 보고 싶으면: 사이드바 **Compute** → 상단 **Vector Search** 탭 → 내 엔드포인트 상태)
- 여기서 "기능을 못 쓴다" 류 에러가 나면 = Free Edition에서 Vector Search가 막힌 것.
  → 클라우드 **14일 트라이얼**(AWS/Azure/GCP 중 하나)로 갈아타면 된다.

## B-5. 인덱스 + 스모크 테스트
- "4) Delta-Sync 인덱스" 셀 실행 → endpoint 위에 인덱스를 생성. 준비될 때까지 대기.
- 마지막 "5) 스모크 테스트" 셀 실행 → **US 질의엔 `US-*`, EU 질의엔 `EU-*`** 만 나오면 성공(country 필터
  가 정상 작동해 관할이 격리됐다는 뜻).

---

# C. 검색 + 평가 — `02_rag_and_eval` 노트북

## C-1. 열기 & CONFIG
1. `databricks/02_rag_and_eval` 열고 **Serverless** 연결.
2. CONFIG 셀에서 `CATALOG`/`SCHEMA`를 `01`과 똑같이 맞춘다.
   (`REPO_PATH`는 노트북이 자동으로 알아내니 건드릴 필요 없음)

## C-2. Gemini 키 입력 (초보용: 붙여넣기)
1. **`1-a` 셀 실행** → 노트북 **맨 위(제목 바로 아래)** 에 `Gemini API Key` 입력창이 생긴다.
   이 셀은 에러 없이 끝나고 안내 문구만 출력된다.
2. **`Ctrl+Home`으로 맨 위로 스크롤** → 그 칸에 Google AI Studio 키를 붙여넣고, 칸 밖을 한 번 클릭해 값 확정.
   (https://aistudio.google.com/apikey · 입력창이 안 보이면 상단 **Widgets** 아이콘을 눌러 패널을 펼친다)
3. **`1-b` 셀 실행** → 키를 읽어 Gemini에 연결한다.

> 💡 `1-b`에서 `AssertionError`가 나면 = 입력창이 아직 비어 있다는 뜻. 2번을 다시 하고 `1-b`만 재실행하면 된다.
> 코드에 키를 적지 않으므로 GitHub에 키가 올라갈 위험이 없다 — **이 노트북은 Git folder 안에 있으니
> 키를 코드에 직접 적는 건 피할 것.**

## C-3. 위에서부터 실행
- Shift+Enter로 순서대로. "스모크" 셀에서 `EU` 판정 결과가 찍히면 검색+판정이 붙은 것이다.
- 그 다음 **"4) MLflow 평가" 셀**이 핵심: eval_set 25건을 돌려 지표를 MLflow에 기록한다(1~2분).

## C-4. 결과 보기 (MLflow) — 여기가 하이라이트
1. 사이드바 **Experiments**(머신러닝 그룹 안) 클릭.
2. `adcompliance-rag` Experiment 클릭 → Run 목록이 보인다. run 하나 클릭.
3. **Metrics**에서 `risky_recall`(★ 위험 광고를 놓치지 않는 비율), `retrieval_recall_at_k`(검색 품질) 확인.
4. **Traces** 탭 → 광고 한 건의 내부 단계(claim추출 → 검색 → 판정)가 span으로 펼쳐진다 = 감사 추적.
5. (실험) `02` CONFIG의 `TOP_K`를 3으로 바꿔 다시 "4) 평가" 셀 실행 → **Run**이 하나 더 생긴다.
   실험 화면에서 **두 run을 체크 → Compare** 로 top_k=3 vs 5를 나란히 비교.

---

# 막히면 / 함정 (초보가 자주 걸리는 것)

- **Git folder가 안 보임 / Repos 폴더가 비어 있음** → 정상이다. Git folder는 `/Repos`가 아니라
  **Home 아래**에 생긴다. 왼쪽 트리에서 **🏠 Home**을 보라(A-3).
- **`import src...` 실패** → 리포를 A-3의 **Git folder로** 가져왔는지 확인(그래야 `src/`가 워크스페이스에 있음).
  `02`가 REPO_PATH를 자동으로 못 찾으면, CONFIG의 `REPO_PATH`에 리포 폴더 경로를 직접 넣는다
  (폴더 우클릭 → **Copy path**).
- **`[PATH_NOT_FOUND] ... eval_set.jsonl`** → 십중팔구 그 파일만 업로드에서 빠진 것. `eval_set.jsonl`은
  `data\policies\`가 아니라 **`data\` 바로 밑**에 있다(A-4 경고 참고). 볼륨 화면에서 파일이 4개인지 세어볼 것.
- **그 외 파일을 못 읽음** → `/Volumes/<카탈로그>/<스키마>/raw/` 실제 경로와 **`01`·`02` 두 노트북의**
  `CATALOG`/`SCHEMA` 값이 모두 일치하는지 확인. 스키마를 `default` 말고 다른 이름(예: `ad_compliance`)으로
  만들었다면 **두 노트북 다** 고쳐야 한다(`02`의 기본값은 `default`라 자주 놓친다).
- **Vector Search가 안 만들어짐** → Free Edition 제한. 클라우드 14일 트라이얼로.
- **Gemini 호출 에러** → C-2 입력창에 키를 넣고 그 셀을 실행했는지. 모델명(`GEMINI_MODEL`)이 지금 쓸 수 있는
  이름인지(예: `gemini-2.0-flash`).

# 선택(여유 있을 때) — Model Serving 게이트웨이 체험
`02` 맨 아래 "선택: External Model 게이트웨이" 섹션. Gemini를 Databricks 서빙 엔드포인트 뒤에 숨겨
키 관리·요청 로깅을 서빙 레이어에서 하는 걸 체험한다. **비밀 스코프 설정(CLI)이 필요**해서 초보에겐 마지막에 권한다.

# 우선순위 (5시간 안에 하나만 남긴다면)
**B(Vector Search) + C-4(MLflow 평가)**. 이 둘이 플래그십이자 면접 킬러다. Unity Catalog 거버넌스는
A에서 거의 공짜로 따라온다.
