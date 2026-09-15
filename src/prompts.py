"""
프롬프트와 출력 스키마 — 의존성 없는 leaf 모듈.

여기 있는 상수들은 "우리가 LLM에게 요구하는 것"의 전부다. pipeline.py가 이걸 쓰고,
Databricks 노트북(클라우드 판)도 이걸 그대로 import해서 쓴다.

★ 왜 pipeline.py에서 떼어냈나
    pipeline.py는 chromadb·google-genai에 묶여 있다. 그래서 `import src.pipeline`만 해도
    그 무거운 라이브러리들이 딸려온다. 반면 이 파일은 순수 문자열·dict뿐이라 아무 데서나
    (예: chromadb가 없는 Databricks 서버리스 노트북) 부담 없이 import된다.
    → 로컬(Chroma)판과 클라우드(Vector Search)판이 '같은 프롬프트·같은 판정 기준'을 공유한다.
      프롬프트는 이 프로젝트 품질의 절반이므로, 두 벌로 복제돼 서로 어긋나면 안 된다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# [1] claim 추출
# ---------------------------------------------------------------------------
CLAIM_SCHEMA = {
    "type": "OBJECT",
    "required": ["claims"],
    "properties": {"claims": {"type": "ARRAY", "items": {"type": "STRING"}}},
}

CLAIM_PROMPT = """\
아래 광고 문안에서 '검증이 필요한 주장(claim)'을 뽑아 주세요.

검증이 필요한 주장이란:
- 최상급/순위 표현 (예: 최고, 1위, No.1, 세계 최초)
- 수치로 표현된 성능·효능·절감 효과 (예: 50% 절감, 99.9% 살균)
- 가격·할인·무료·최저가 표현
- 경쟁사와의 비교 또는 경쟁사 언급
- 안전성·의학적 효능 표현 (예: 치료, 무해, 부작용 없음)
- 경품·당첨·이벤트 조건
- 고객 후기, 인물 사진 사용, 출시 예정 기능 언급

규칙:
- 광고 문안에 실제로 등장한 표현을 **그대로** 인용하세요. 요약하지 마세요.
- 단순 감성 문구나 제품 소개("여름 준비, 시원하게")는 claim이 아닙니다. 제외하세요.
- 한 문장에 주장이 두 개면 두 개로 나누세요.
- 없으면 빈 배열을 반환하세요.

광고 문안:
\"\"\"{ad_text}\"\"\"
"""


# ---------------------------------------------------------------------------
# [3] 판정
# ---------------------------------------------------------------------------
JUDGE_SCHEMA = {
    "type": "OBJECT",
    "required": ["decision", "risk_score", "issues"],
    "properties": {
        "decision": {"type": "STRING", "enum": ["PASS", "REVIEW", "BLOCK"]},
        "risk_score": {"type": "NUMBER"},
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["claim", "rule_id", "reason", "suggestion"],
                "properties": {
                    "claim": {"type": "STRING"},
                    "rule_id": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                    "suggestion": {"type": "STRING"},
                },
            },
        },
    },
}

# 국가별 심의 태도. 규정 텍스트(무엇을 요구하나)와 별개로, '같은 표현을 얼마나 엄격히
# 보느냐'는 관할마다 다르다. 그 맥락을 판정 프롬프트에 주입한다.
# (규정만 국가별로 바꾸고 이 맥락을 안 주면 "미국 규정을 한국식으로 해석"하는 오류가 난다)
COUNTRY_CONTEXT = {
    "KR": "대한민국 표시·광고의 공정화에 관한 법률 기준으로 판단합니다. "
          "최상급·정량 효능·가격 표시에 객관적 실증 근거가 필요합니다.",
    "US": "미국 FTC 가이드라인 기준으로 판단합니다. 모든 객관적 주장은 "
          "'competent and reliable evidence'가 뒷받침되어야 하며, "
          "'eco-friendly' 같은 일반적 친환경 표현은 구체적 입증 없이는 기만으로 봅니다.",
    "EU": "EU 소비자보호 지침(Green Claims, Omnibus, UCPD) 기준으로 판단합니다. "
          "일반적 친환경 표현('eco-friendly', 'climate neutral')과 미검증 비교광고에 "
          "특히 엄격하며, 입증되지 않은 경우 사용 금지(BLOCK) 수준으로 봅니다.",
}


JUDGE_PROMPT = """\
아래 광고 문안이 {country} 광고 심의 규정에 저촉되는지 판단해 주세요.

## 심의 기준 (관할: {country})
{country_context}

## 광고 문안
\"\"\"{ad_text}\"\"\"

## 검증 대상 주장(claim)
{claims}

## 참고 가능한 규정 (이 목록에 있는 것만 근거로 사용할 수 있습니다)
{rules}

## 판정 기준
- BLOCK  : 규정이 "사용할 수 없다", "금지한다"고 명시한 표현을 실제로 사용한 경우.
           보완 표기로 해소되지 않고 문구 자체를 바꿔야 하는 수준.
- REVIEW : 표현 자체는 허용되지만 규정이 요구하는 근거·조건·기준 표시가 누락된 경우.
- PASS   : 검증할 주장이 없거나, 규정이 요구하는 표시가 모두 갖춰진 경우.

## 작성 규칙
1. issues의 rule_id는 반드시 위 규정 목록에 있는 ID여야 합니다. 없는 ID를 만들지 마세요.
2. 근거 규정 문장이 실제로 그 내용을 담고 있을 때만 issue로 올리세요.
3. claim에는 광고 문안에 등장한 표현을 그대로 인용하세요.
4. suggestion에는 무엇을 추가/변경하면 되는지 구체적으로 쓰세요.
5. risk_score: PASS 0.0~0.2, REVIEW 0.3~0.7, BLOCK 0.7~1.0.
6. 문제가 없으면 decision=PASS, issues=[].
7. 애매하면 PASS가 아니라 REVIEW. 위험한 광고를 통과시키는 것이
   불필요한 검토 요청보다 훨씬 큰 손실입니다.
"""
