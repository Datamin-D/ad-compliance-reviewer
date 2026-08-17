"""
자료구조 + LLM 출력 검증.  ★ 이 프로젝트에서 제일 중요한 파일 ★

이 파일이 왜 따로 있나
    파이프라인의 각 단계는 서로 데이터를 주고받는다. 그걸 전부 파이썬 dict로
    들고 다니면 오타(`rule_id`를 `ruleid`로)나 키 누락이 한참 뒤에서 터진다.
    그래서 경계에서 한 번 dataclass로 바꾸며 검증한다.
    소프트웨어 설계에서 이런 층을 anti-corruption layer(부패 방지층)라고 부른다.
    바깥의 못 믿을 데이터가 안쪽 도메인으로 그대로 새어 들어오지 못하게 막는 벽.

    "못 믿을 데이터"의 정체는 LLM 출력이다. 프롬프트에 아무리 "규정 목록에 없는
    ID를 만들지 마라"라고 써도 가끔 만든다. 법무 도메인에서 근거 조작은
    틀린 답보다 나쁘다. 그래서 프롬프트(확률)가 아니라 코드(보장)로 막는다.

이 파일은 아무도 import 하지 않는다(맨 아래층). 그래서 여기부터 읽으면 좋다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class Decision(str, Enum):
    """
    최종 판정 3단계.

    Enum = "정해진 값만 허용되는 타입". 문자열 "REVIEW"를 그냥 쓰면
    "reveiw"라고 오타를 내도 런타임까지 아무도 못 잡는다. Enum이면 즉시 터진다.

    str까지 같이 상속한 이유: 이러면 Decision.PASS는 "PASS"이기도 해서
    json.dumps()나 f-string이 별도 변환 없이 그냥 동작한다.
    """

    PASS = "PASS"      # 문제 없음. 그대로 집행 가능
    REVIEW = "REVIEW"  # 근거·조건 표시를 보완하면 해소 가능. 사람 검토 필요
    BLOCK = "BLOCK"    # 규정이 명시적으로 금지한 표현. 문구를 바꿔야 함

    @classmethod
    def parse(cls, value: str) -> "Decision":
        """
        LLM이 소문자나 공백을 섞어 보내도 받아준다.

        ★ 알 수 없는 값이 오면 PASS가 아니라 REVIEW로 떨어진다.
        판정 불능인데 PASS로 흘려보내면 위험 광고를 놓친다(false negative).
        컴플라이언스에서는 "모르면 사람에게 넘긴다"가 맞다.
        보안 용어로 fail closed — 고장 나면 잠기는 쪽으로.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            return cls.REVIEW


# ---------------------------------------------------------------------------
# 자료구조
#
# @dataclass는 __init__, __repr__, __eq__를 자동으로 만들어주는 표준 라이브러리 기능.
# 아래 Policy는 이것만 쓰면 Policy("R-01", "최상급", ...) 로 바로 생성된다.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)  # frozen=True → 생성 후 값 변경 금지. 규정은 바뀌면 안 되니까
class Policy:
    """규정 1개. data/policies.jsonl의 한 줄에 대응하며, 그대로 RAG의 1 chunk가 된다."""

    rule_id: str    # "R-01" — Chroma의 기본키로도 그대로 쓴다
    category: str   # "최상급 표현"
    title: str      # "객관적 근거 없는 최상급 표현 금지"
    text: str       # 규정 본문
    severity: str   # HIGH / MEDIUM / LOW
    # 관할 국가. 기존 규정 파일에는 이 필드가 없으므로 기본값 KR.
    # (필드에 기본값이 있으면 policies.jsonl의 옛 줄들을 안 고쳐도 된다)
    country: str = "KR"  # KR / US / EU

    def to_document(self) -> str:
        """
        임베딩할 문자열을 만든다.

        본문만 넣는 것보다 카테고리·제목을 앞에 붙이는 편이 검색이 잘 된다.
        광고 문구는 짧고("대한민국 No.1") 규정 본문과 겹치는 단어가 거의 없는데,
        제목("최상급 표현 금지")이 그 어휘 간극을 메워주기 때문이다.
        """
        return f"[{self.category}] {self.title}\n{self.text}"


@dataclass
class RetrievedPolicy:
    """검색 결과 1건 = 규정 + 유사도 점수(1.0에 가까울수록 관련)."""

    policy: Policy
    score: float


@dataclass
class Issue:
    """판정에서 발견된 위반 소지 1건."""

    claim: str        # 문제가 된 광고 문구 조각
    rule_id: str      # 근거 규정 ID — 반드시 '검색된' 규정 중 하나여야 함
    reason: str       # 왜 문제인지
    suggestion: str   # 어떻게 고치면 되는지


@dataclass
class ReviewResult:
    """파이프라인의 최종 산출물. cli.py가 이걸 받아서 출력한다."""

    ad_text: str
    decision: Decision
    risk_score: float

    # field(default_factory=list)는 "기본값은 빈 리스트"라는 뜻.
    # 그냥 `= []`라고 쓰면 모든 인스턴스가 같은 리스트 하나를 공유하는
    # 파이썬의 유명한 함정(mutable default argument)에 빠진다.
    issues: list[Issue] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)           # 추출된 claim들
    retrieved_rule_ids: list[str] = field(default_factory=list)  # 검색된 규정 ID들

    def to_dict(self) -> dict:
        d = asdict(self)               # dataclass → 중첩 dict로 통째 변환
        d["decision"] = self.decision.value  # Enum은 문자열로 풀어준다
        return d

    def to_json(self, indent: int = 2) -> str:
        # ensure_ascii=False: 안 주면 한글이 한국 처럼 이스케이프된다
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# ★ LLM 원시 출력 → 검증된 dataclass
# ---------------------------------------------------------------------------
def parse_review_payload(
    payload: dict,
    ad_text: str,
    claims: list[str],
    allowed_rule_ids: list[str],
) -> ReviewResult:
    """
    judge LLM이 뱉은 JSON dict를 검증하면서 ReviewResult로 바꾼다.

    여기서 하는 방어 3가지:

      (1) grounding check
          issue의 rule_id가 실제로 검색된 규정(allowed_rule_ids)에 없으면 버린다.
          RAG에서 가장 흔한 hallucination이 "존재하지 않는 출처 인용"이다.

      (2) 판정과 근거의 정합성
          근거가 하나도 안 남았는데 BLOCK  → 근거 없는 BLOCK이므로 PASS로 되돌림
          근거가 있는데 PASS               → 모순이므로 REVIEW로 올림

      (3) 타입/범위 정규화
          risk_score가 문자열로 오거나 1.0을 넘는 경우를 잘라낸다.

    Args:
        payload: LLM이 준 JSON (dict)
        ad_text: 원본 광고 문안
        claims: 추출된 claim 목록
        allowed_rule_ids: 검색 단계에서 실제로 가져온 규정 ID들 = 유일하게 허용된 근거
    """
    allowed = set(allowed_rule_ids)  # 매번 리스트를 훑지 않도록 집합으로

    # ── (1) 근거 검증 ──────────────────────────────────────────────
    issues: list[Issue] = []
    for raw in payload.get("issues") or []:   # 키 자체가 없을 수도 있다
        rule_id = str(raw.get("rule_id", "")).strip()
        if rule_id not in allowed:
            continue  # ← 지어낸 조항. issue 통째로 폐기
        issues.append(
            Issue(
                claim=str(raw.get("claim", "")).strip(),
                rule_id=rule_id,
                reason=str(raw.get("reason", "")).strip(),
                suggestion=str(raw.get("suggestion", "")).strip(),
            )
        )

    decision = Decision.parse(payload.get("decision", "REVIEW"))

    # ── (3) 점수 정규화 ────────────────────────────────────────────
    try:
        risk_score = float(payload.get("risk_score", 0.0))
    except (TypeError, ValueError):
        risk_score = 0.0
    risk_score = max(0.0, min(1.0, risk_score))  # 0.0~1.0으로 잘라냄

    # ── (2) 판정 ↔ 근거 정합성 ────────────────────────────────────
    if not issues and decision is not Decision.PASS:
        decision = Decision.PASS
        risk_score = min(risk_score, 0.2)
    elif issues and decision is Decision.PASS:
        decision = Decision.REVIEW
        risk_score = max(risk_score, 0.4)

    return ReviewResult(
        ad_text=ad_text,
        decision=decision,
        risk_score=risk_score,
        issues=issues,
        claims=claims,
        retrieved_rule_ids=list(allowed_rule_ids),
    )


# ---------------------------------------------------------------------------
# 데이터 로딩
#
# .jsonl = JSON Lines. "한 줄에 JSON 하나"인 형식.
# 일반 .json 배열과 달리 한 줄씩 읽을 수 있고, 줄 단위로 추가/수정하기 쉬워서
# 데이터셋 파일에 많이 쓴다.
# ---------------------------------------------------------------------------
def load_policies(path: Path) -> list[Policy]:
    """
    규정 파일을 읽어 Policy 리스트로 만든다.

    path가 디렉터리면 그 안의 모든 *.jsonl을 읽고, **파일명(스템)을 국가 코드로** 삼는다.
    (예: data/policies/US.jsonl → 그 파일의 모든 규정은 country="US")
    이렇게 하면 country를 각 줄에 중복 기입할 필요가 없고, 나라를 추가하는 건
    파일 하나를 넣는 것으로 끝난다.

    path가 단일 파일이면 그 파일만 읽는다(스템을 국가 코드로 사용).
    """
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]

    policies: list[Policy] = []
    for f in files:
        country = f.stem  # 파일명이 국가 코드. country의 단일 출처(source of truth).
        with open(f, encoding="utf-8") as fh:  # encoding 필수. 윈도우 기본은 cp949
            for line in fh:
                if line.strip():               # 빈 줄 건너뜀
                    row = json.loads(line)
                    row["country"] = country   # 파일명이 항상 우선(줄에 country가 있어도 덮어씀)
                    # **는 dict를 키워드 인자로 펼치는 문법.
                    policies.append(Policy(**row))
    return policies


def load_eval_set(path: Path) -> list[dict]:
    """
    eval_set.jsonl → dict 리스트.

    여기만 dataclass로 안 바꾸는 이유: 평가 스크립트 한 곳에서만 쓰이고
    필드가 유동적이라(note 같은 메모 필드) 타입을 고정할 이득이 없다.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
