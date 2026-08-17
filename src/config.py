"""
설정 — 값을 바꿔가며 실험할 것들을 한 곳에 모아둔다.

이 파일이 따로 있는 이유는 딱 하나다. 모델명이나 TOP_K 같은 값이 코드 여기저기
흩어져 있으면 실험할 때마다 파일 3~4개를 뒤져야 한다. 여기만 보면 되게 한다.

이 파일은 아무도 import 하지 않는다(다들 이걸 import만 한다). 맨 아래층.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# __file__ = 이 파일의 경로.  .resolve()로 절대경로 → .parent.parent로 두 단계 위 = 루트
#   c:\dev\RAG\src\config.py → c:\dev\RAG\src → c:\dev\RAG
# 이렇게 계산하면 어느 폴더에서 실행하든 경로가 안 깨진다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env 파일을 읽어 os.environ에 채워 넣는다. 파일이 없어도 조용히 넘어간다.
# (VS Code가 주입해주든 말든 상관없이 파이썬이 직접 읽는다)
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# 경로  ('/' 연산자로 경로를 잇는 건 pathlib 문법. 윈도우/리눅스 모두 동작)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
# 규정은 '나라별 파일 1개' 구조. data/policies/KR.jsonl, US.jsonl, EU.jsonl ...
# 파일명(스템)이 곧 국가 코드다. 각 나라 법무팀이 자기 파일만 손보면 되도록 분리했다.
# 나라를 추가하는 것 = 파일 하나를 넣는 것 (코드 수정 불필요).
POLICIES_DIR = DATA_DIR / "policies"
EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"   # 평가용 광고 + 정답
CHROMA_PATH = PROJECT_ROOT / ".chroma"        # 벡터 인덱스 저장 폴더 (gitignore됨)
RESULTS_PATH = PROJECT_ROOT / "results.json"  # 평가 산출물

COLLECTION_NAME = "ad_policies"  # Chroma 컬렉션 이름 (SQL의 테이블명에 해당)


def available_countries() -> list[str]:
    """규정 파일이 존재하는 국가 코드 목록. CLI의 --country 선택지가 여기서 나온다."""
    return sorted(p.stem for p in POLICIES_DIR.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
# os.getenv("이름", 기본값) = 환경변수가 있으면 그 값, 없으면 기본값.
# 즉 .env에서 GEMINI_CHAT_MODEL을 지정하면 아래 기본값을 덮어쓴다.
# → 코드를 안 고치고 모델만 바꿔 실험할 수 있다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# gemini-2.5-flash는 신규 사용자에게 더 이상 제공되지 않는다(404 NOT_FOUND).
# 지금 쓸 수 있는 모델 목록은 `client.models.list()`로 확인할 수 있다.
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# 임베딩 벡터의 차원 수. gemini-embedding-001은 기본 3072인데 줄일 수 있고(MRL),
# 규정 30개 규모에선 768로도 충분하고 인덱스가 가볍다.
# ★ 이 값을 바꾸면 반드시 `python cli.py index`로 재색인해야 한다.
#   차원이 다른 벡터끼리는 비교가 안 되기 때문.
EMBED_DIM = 768


# ---------------------------------------------------------------------------
# 검색 파라미터 (튜닝 대상)
# ---------------------------------------------------------------------------
# claim 하나당 가져올 규정 수.
#   크게 잡으면 → 관련 규정을 놓칠 확률↓, 대신 프롬프트에 잡음↑ 토큰비용↑
#   작게 잡으면 → 반대
# 컴플라이언스에선 놓치는 쪽(false negative)이 훨씬 비싸므로 넉넉하게 잡았다.
# `python cli.py eval --top-k 3` 처럼 바꿔가며 실험해 볼 것.
TOP_K = 5
