# CONTEXT — 프로젝트 인수인계 / 맥락 정리

> 이 파일 하나만 읽으면 다른 컴퓨터에서도 이 프로젝트가 **무엇이고, 어떻게 돌고, 어떤 파일이
> 무슨 일을 하는지** 파악되도록 정리한 문서. (개인 PC → 회사 PC 이동용)

---

## 0. 한눈에

**무엇** — 광고 문안을 넣으면 사내/해외 광고 규정에 걸리는지 **근거 규정과 함께 판정**(PASS/REVIEW/BLOCK)하고,
규정을 충족하는 **대안 카피까지 만들어 스스로 재검증**하는 RAG + Agent 시스템.

**왜 만들었나 (목적 2개)**
1. **개인 포트폴리오** — Google Cloud FDE(GenAI) 인터뷰용. "business problem → architecture → trade-off → evaluation" 스토리.
2. **사내 해커톤(AX FLEX, 분야 2 현업 주도)** — LangGraph 기반 Agent 제출물.

**한 줄 아키텍처**
```
광고 문안 → [claim 추출(LLM)] → [국가별 규정 검색(RAG)] → [근거 기반 판정(LLM)] → [지어낸 근거 폐기(코드)]
                                                                            └→ [Agent: 구루 3인이 대안 생성 → 각 안 재검증]
```

**중요 원칙**
- 모든 규정·평가 데이터는 **합성(synthetic)**. 실제 사내 규정 미포함.
- 판정 근거는 "검색된 규정 텍스트"뿐. LLM이 없는 규정 ID를 지어내면 **코드에서 폐기**한다(grounding).
- 애매하면 PASS가 아니라 REVIEW(fail-safe). 컴플라이언스에선 놓침(false negative)이 가장 비싸다.

---

## 1. 다른 컴퓨터로 옮기기 (실전 순서)

### 1-1. 무엇을 복사할까
`c:\dev\RAG` 폴더 전체를 옮기면 되지만, **아래는 안 옮겨도 된다**(새 PC에서 재생성됨):

| 폴더/파일 | 옮길 필요 | 이유 |
|---|---|---|
| `src/`, `data/`, `tests/`, `cli.py`, `*.md`, `requirements.txt`, `.env.example` | ✅ 필수 | 소스 |
| `.env` | ⚠️ 결정 필요 | **실제 API 키가 들어있음** — 아래 1-3 참고 |
| `.chroma/` | ❌ 불필요 | 벡터 인덱스. `python cli.py index`로 재생성 |
| `__pycache__/`, `.pytest_cache/`, `*.pyc` | ❌ 불필요 | 파이썬 캐시 |
| `results.json` | ❌ 불필요 | `eval` 실행 산출물 |

> 참고: 지금 이 폴더는 **git 저장소가 아니다**. USB/클라우드/zip으로 폴더째 옮기거나, 회사 PC에서
> `git init` 후 올려도 된다(`.gitignore`는 이미 `.env`·`.chroma/`·캐시를 제외하도록 설정됨).

### 1-2. 새 PC에서 환경 세팅 (4단계)
```powershell
cd <옮긴 경로>\RAG

pip install -r requirements.txt          # 1) 의존성 설치

Copy-Item .env.example .env              # 2) .env 만들고 GEMINI_API_KEY 입력
                                         #    키 발급: https://aistudio.google.com/apikey

python cli.py index                      # 3) 규정 재색인 → "Indexed 42 policies (EU 7, KR 28, US 7)"

python -m pytest tests -q                # 4) 검증 (API 키 없이도 통과, 16종)
python cli.py review "대한민국 No.1 에어컨, 전기료 50% 절감!"   # 실제 동작 확인
```

### 1-3. API 키 (개인 → 회사, 꼭 확인)
- `.env`의 `GEMINI_API_KEY`는 **개인 Google AI Studio 키**다.
- 회사 PC에서는 **회사 정책 확인 후** ▲개인 키를 그대로 쓸지 ▲회사용 키를 새로 발급할지 ▲Vertex AI로 갈지 결정할 것.
- 코드는 `.env`의 `GEMINI_CHAT_MODEL` / `GEMINI_EMBED_MODEL` 환경변수로 **모델을 코드 수정 없이 교체**할 수 있게 되어 있다([src/config.py](src/config.py) 참고). Vertex로 옮길 땐 [src/llm.py](src/llm.py) 한 파일만 고치면 된다.

### 1-4. 파이썬 버전 주의
- 현재 개발은 **Python 3.14**로 했다(`.pyc`가 `cpython-314`). 회사 PC의 파이썬이 다르면 `chromadb` 휠 설치가
  안 될 수 있다. 그 경우 파이썬 3.11~3.12 가상환경(`.venv`)을 따로 만들어 설치하는 게 가장 안전하다.

---

## 2. 파일별 역할

### 진입점
| 파일 | 역할 |
|---|---|
| [cli.py](cli.py) | **유일한 실행 진입점.** 터미널 입력을 해석해 알맞은 함수 호출. 판단 로직은 없음. 서브커맨드 `index / review / agent / eval` |

### `src/` — 실제 로직 (아래층 → 위층 순서)
| 파일 | 파이프라인 단계 | 역할 |
|---|---|---|
| [src/config.py](src/config.py) | 설정(맨 아래층) | 경로·모델명·`TOP_K`·`EMBED_DIM`을 한 곳에. `available_countries()`로 `data/policies/`의 파일에서 국가 목록 자동 발견 |
| [src/schema.py](src/schema.py) | 자료구조 + 검증 | `Policy`/`ReviewResult` 등 dataclass, `Decision` Enum, **`parse_review_payload()`(지어낸 근거 폐기 = 핵심 방어)**, `load_policies()`(파일명→국가코드) |
| [src/llm.py](src/llm.py) | LLM 래퍼 | Gemini 호출 2개만 노출: `embed_texts()`(임베딩), `generate_json()`(JSON 강제 출력 + 재시도). 여기만 고치면 다른 모델/Vertex로 이전 가능 |
| [src/retriever.py](src/retriever.py) | [2] 검색 | `PolicyRetriever` — ChromaDB 색인/검색. `search(query, k, country=...)`로 **국가 필터**(`where={"country":...}`). 의존성 주입(embed_fn/policies/path)이라 테스트가 쉬움 |
| [src/pipeline.py](src/pipeline.py) | [1][3] RAG 본체 | `extract_claims()`(광고→주장 분해), `judge()`(검색된 규정만 근거로 판정), `review()`(전체 엮기). **국가별 심의 태도 `COUNTRY_CONTEXT`** 주입. 프롬프트가 여기 다 있음 |
| [src/agent.py](src/agent.py) | Agent | **LangGraph 구루 패널.** 진단→패널(Iris/Theo/Nora 3인)→검증(재검증+1회 보정)→리포트 |
| [src/evaluate.py](src/evaluate.py) | [4] 평가 | 평가셋 실행 → 정확도·**Recall**·Recall@k·혼동행렬 |

### `data/` — 데이터 (전부 합성)
| 파일 | 역할 |
|---|---|
| [data/policies/KR.jsonl](data/policies/KR.jsonl) | 한국 규정 28개 (R-01…R-28). **파일명 `KR` = 국가코드** |
| [data/policies/US.jsonl](data/policies/US.jsonl) | 미국 규정 7개 (US-01…, FTC 스타일, 영어) |
| [data/policies/EU.jsonl](data/policies/EU.jsonl) | EU 규정 7개 (EU-01…, Green Claims/Omnibus 스타일, 영어) |
| [data/eval_set.jsonl](data/eval_set.jsonl) | 평가용 광고 25건 + 정답(expected, expected_rules) |

> **국가 추가 = 파일 하나 추가.** `data/policies/JP.jsonl`을 넣고 `index`만 다시 돌리면 `--country JP`가 자동 생김. 코드 수정 불필요.

### `tests/` — API 키 없이 도는 테스트
| 파일 | 역할 |
|---|---|
| [tests/test_rag.py](tests/test_rag.py) | 근거 폐기(guardrail)·검색 랭킹·국가 필터·데이터 정합성 |
| [tests/test_agent.py](tests/test_agent.py) | 구루 3인 생성·실패 라벨링·PASS 우선 정렬 |

### 문서 / 설정
| 파일 | 역할 |
|---|---|
| [README.md](README.md) | 영어. 문제정의·아키텍처·설계 결정·평가·실행법 (포트폴리오용) |
| [DEMO.md](DEMO.md) | 데모 대본 (터미널에서 쳐볼 명령어 모음) |
| [CONTEXT.md](CONTEXT.md) | (이 파일) 인수인계·맥락 |
| `requirements.txt` | 의존성: google-genai, langgraph, chromadb, numpy, python-dotenv, pytest |
| `.env.example` | 키 템플릿(빈 값). 복사해서 `.env` 만들 것 |
| `.gitignore` | `.env`·`.chroma/`·캐시 제외 |

---

## 3. Agent(구루 패널) 어떻게 도나 — LangGraph

`src/agent.py`. State(공유 화이트보드)를 노드들이 이어받아 채우는 구조.

```
run() → invoke({ad_text, country, ...})
  START → diagnose  : 원본을 review()로 검토 → 무엇이 왜 걸리나
        → panel     : 구루 3인이 각자 다른 창작 루트로 대안 동시 생성 (LLM 3회)
                       Iris(Human Relief) / Theo(Quiet Intelligence) / Nora(Life's Good Moment)
        → verify    : 각 대안을 같은 review()에 다시 넣어 재검증 → PASS 아니면 1회만 보정
        → report    : PASS를 위로 정렬해 카드 조립 (LLM 없이 코드로)
  END → report(dict) 반환
```

**핵심 아이디어**: 생성한 LLM이 "이거 통과예요"라고 말하는 걸 **안 믿고**, 대안을 원래 RAG 검토기에
다시 넣어 **판정으로 증명**한다. 구루는 사실을 지어낼 수 없다(edit 스키마에 DELETE/SOFTEN/PLACEHOLDER/REFRAME만,
'ADD' 없음). `MAX_REPAIR = 1` — 무한 루프 아님.

> 과거에 "안 될 때까지 계속 고치는 루프 Agent"를 만들었다가 **폐기**하고 이 구루 패널로 바꿨다.
> (문서/코드가 루프를 설명하고 있으면 옛날 버전이다 — 현재는 구루 패널이 정답.)

---

## 4. 산출물 (레포 밖, claude.ai에 있음)

두 개의 발표/제출용 문서를 아티팩트로 만들어 둠(비공개, 링크 아는 사람만):

| 문서 | 용도 | 위치 |
|---|---|---|
| 법무검토 Agent 제출서 | 사내 해커톤 1차 서류 (공식 서식 0·1·2·3 + 심사기준) | claude.ai 계정 내 (비공개) |
| Ad Compliance RAG — 면접 준비 노트 | 인터뷰 대비 (문제-해결-**한계**, 정량 수치) | claude.ai 계정 내 (비공개) |

> 제출서의 화면 캡처 자리는 아직 비어있음 → 새 PC에서 `review`/`agent` 실행 화면을 캡처해 채우면 됨.

---

## 5. 발표·인터뷰용 핵심 talking point

- **먼저 쪼개고 검색한다** — 광고 통째로 임베딩하면 벡터가 여러 주장 사이 중간에 찍혀 어느 규정과도 안 가까움. claim 분해가 검색 품질에 가장 크게 기여.
- **근거를 코드로 강제** — 프롬프트로 "지어내지 마"는 확률, `parse_review_payload()`로 거르는 게 보장. 법무 도메인에선 가짜 인용이 무응답보다 나쁨.
- **Recall 우선** — 위험 광고 놓침(false negative)이 훨씬 비싸서 정확도가 아니라 위험클래스 Recall이 주지표. 검색 품질은 Recall@k로 따로 측정.
- **관할을 데이터로** — 국가별 규정 파일 분리 → 같은 광고가 US=REVIEW, EU=BLOCK. 새 시장 = 파일 하나.
- **한계(인터뷰 차별점)** — 데이터 합성이라 실제 성능 미검증 / 의미 검색이 미묘한 규정 경계를 못 가를 수 있음 / Vision 입력은 로드맵.

---

## 6. 자주 쓰는 명령어

```powershell
$env:PYTHONIOENCODING="utf-8"          # 윈도우에서 한글이 ? 로 깨질 때

python cli.py index                    # 규정 색인 (규정 파일 고쳤으면 다시)
python cli.py review --country EU "100% eco-friendly air conditioner!"
python cli.py agent  --country EU "World's No.1 air conditioner, cuts your energy bill by 50%!"
python cli.py review --json "전 품목 30% 할인!"      # JSON 출력
python cli.py eval                     # 평가셋 전체 + 지표
python -m pytest tests -q              # 테스트
```

전체 데모 시나리오는 [DEMO.md](DEMO.md) 참고.
