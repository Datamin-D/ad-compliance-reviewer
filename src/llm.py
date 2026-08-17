"""
Gemini 호출을 감싸는 얇은 층. 외부에 함수 딱 2개만 내보낸다.

    embed_texts()   : 텍스트 → 벡터        (retriever가 씀)
    generate_json() : 프롬프트 → dict      (pipeline이 씀)

왜 SDK를 직접 안 쓰고 한 겹 감싸나
    1. 나중에 Vertex AI나 다른 모델로 옮길 때 이 파일만 고치면 된다.
    2. 테스트에서 이 함수만 가짜(stub)로 바꿔 끼우면 API 없이 전체를 돌릴 수 있다.
    3. "우리가 LLM에게 요구하는 것"이 이 두 줄로 요약된다 — 읽는 사람이 편하다.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

from src import config


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Gemini 클라이언트를 만들어 캐시한다.

    @lru_cache는 "같은 인자로 다시 부르면 계산 안 하고 저장해둔 값을 준다"는
    표준 라이브러리 데코레이터. 인자가 없으니 사실상 "딱 한 번만 실행"이 된다.
    클라이언트를 매 호출마다 새로 만들 이유가 없고, API 키 누락 에러도
    실제로 LLM이 필요한 순간에 한 번만 뜨게 된다.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            ".env.example을 .env로 복사한 뒤 키를 입력해 주세요."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# 임베딩 — 텍스트를 숫자 벡터로
#
# "의미가 비슷한 문장은 벡터도 가깝다"가 전부다. 그 덕에 '단어가 겹치는지'가 아니라
# '뜻이 통하는지'로 검색할 수 있다. RAG의 R이 돌아가는 원리.
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    텍스트 리스트 → 임베딩 벡터 리스트 (입력과 같은 순서, 같은 개수).

    task_type을 주면 모델이 '문서를 색인하는 중'(RETRIEVAL_DOCUMENT)인지
    '질의로 검색하는 중'(RETRIEVAL_QUERY)인지 구분해 서로 다른 벡터를 만든다.
    이걸 비대칭 임베딩이라 하고, 입력 모양이 다를 때 효과가 크다 —
    여기선 짧은 광고 슬로건 vs 긴 규정 문장이라 딱 그 경우다.
    """
    if not texts:
        return []

    client = get_client()
    vectors: list[list[float]] = []

    # 요청 하나에 넣을 수 있는 텍스트 개수에 제한이 있어 32개씩 잘라 보낸다.
    # 규정이 28개니까 실제로는 한 번에 끝난다.
    for i in range(0, len(texts), 32):
        response = client.models.embed_content(
            model=config.EMBED_MODEL,
            contents=texts[i : i + 32],
            config=types.EmbedContentConfig(
                task_type=task_type,
                # 차원 축소(MRL 기법). 3072차원 전부 안 써도 검색 품질 손실이 거의 없고
                # 인덱스 크기와 유사도 계산 비용이 줄어든다.
                output_dimensionality=config.EMBED_DIM,
            ),
        )
        vectors += [list(e.values) for e in response.embeddings]

    # L2 정규화 — 모든 벡터의 길이를 1로 맞춘다.
    #
    # output_dimensionality로 차원을 줄이면 결과 벡터의 길이가 1이 아니게 된다.
    # 지금 설정(Chroma cosine space)에서는 Chroma가 내부에서 알아서 정규화하므로
    # 이 줄이 없어도 검색 순위는 똑같다. 즉 필수는 아니다.
    #
    # 그래도 하는 이유: 길이가 1이면 '코사인 유사도 = 내적'이 되어서
    # 어떤 거리 함수(cosine / 내적 / L2)를 쓰든 순위가 같아진다.
    # 나중에 저장소나 거리 설정을 바꿔도 조용히 틀리는 일이 없다.
    # ponytail: 방어적 3줄. 저장소를 안 바꿀 확신이 있으면 지워도 된다.
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)  # 행마다 벡터 길이
    return (arr / np.where(norms == 0, 1, norms)).tolist()  # 0으로 나누기 방지


# ---------------------------------------------------------------------------
# 구조화된 생성 — LLM이 반드시 JSON을 뱉게 만들기
# ---------------------------------------------------------------------------
def generate_json(
    prompt: str,
    response_schema: dict,
    *,                      # 이 뒤 인자는 반드시 이름을 적어서 넘겨야 함 (실수 방지)
    system_instruction: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    JSON 스키마를 강제해서 모델을 호출하고, 파싱된 dict를 돌려준다.

    response_mime_type + response_schema를 주면 Gemini가 스키마에 맞는 JSON만
    만들도록 '디코딩 단계에서' 제약이 걸린다(constrained decoding).
    프롬프트에 "JSON으로 답해줘"라고 부탁하는 것과는 차원이 다르게 안정적이라,
    파싱 실패 → 재시도 같은 코드가 아예 필요 없어진다.

    system_instruction: 역할 지시. 매 요청의 맨 앞에 붙는 고정 지침.
    temperature=0.0: 같은 입력에 항상 같은 답. 법무 판정은 재현 가능해야 하고,
                     평가 지표도 실행할 때마다 흔들리면 의미가 없다.

    일시적 API 오류(간헐적 5xx 등)에 대비해 짧게 재시도한다. 에이전트 패널은 한 번에
    LLM을 여러 번 부르는데, 그중 하나가 순간적으로 실패했다고 전체가 죽으면 안 된다.
    """
    for attempt in range(3):  # 최대 3회 (2회 재시도)
        try:
            response = get_client().models.generate_content(
                model=config.CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                # 안전 필터 등으로 응답이 빌 수 있다. 예외 대신 빈 dict를 주면
                # 호출부(예: parse_review_payload)가 "근거 0건"으로 안전 처리한다.
                return {}
            return json.loads(text)
        except Exception:  # noqa: BLE001 - 일시 오류면 재시도, 마지막이면 그대로 올림
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))  # 1.5s → 3s 백오프
