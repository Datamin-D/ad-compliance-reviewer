"""
구루 패널 에이전트 테스트 — API 키 없이 돈다 (review와 LLM을 stub으로 교체).

검증 대상:
  1. 구루 3인의 추천안이 모두 생성되고, 각각 컴플라이언스 재검증을 거치는가
  2. 재검증에서 통과 못 한 안도 크래시 없이 정직하게 라벨링되는가
  3. 리포트가 바로 쓸 수 있는 안(PASS)을 위로 정렬하는가
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field  # noqa: E402

import src.agent as agent  # noqa: E402
from src.schema import Decision  # noqa: E402


@dataclass
class _FakeIssue:
    claim: str = "대한민국 No.1"
    rule_id: str = "R-01"
    reason: str = "근거 없음"
    suggestion: str = "조사기관 표시"


@dataclass
class _FakeResult:
    decision: Decision
    issues: list = field(default_factory=list)
    risk_score: float = 0.0


def _patch(monkeypatch, decision):
    """review는 항상 같은 판정을 내고, generate_json은 가짜 추천안을 낸다."""
    counter = {"n": 0}

    def fake_review(text, **kw):
        issues = [] if decision is Decision.PASS else [_FakeIssue()]
        return _FakeResult(decision, issues, 0.0 if decision is Decision.PASS else 0.6)

    def fake_generate_json(prompt, schema, **kw):
        counter["n"] += 1
        return {"headline": f"Less fuss {counter['n']}", "support": "One clear benefit.",
                "why_lg": "human-first", "edits": [
                    {"type": "SOFTEN", "target": "No.1", "note": "removed superlative"}]}

    monkeypatch.setattr(agent, "review", fake_review)
    monkeypatch.setattr(agent.llm, "generate_json", fake_generate_json)


def test_panel_produces_three_distinct_personas(monkeypatch):
    _patch(monkeypatch, Decision.PASS)
    r = agent.run("대한민국 No.1 에어컨")

    props = r["proposals"]
    assert len(props) == 3                                  # 구루 3인
    assert {p["id"] for p in props} == {"relief", "quiet", "moment"}
    assert {p["name"] for p in props} == {"Iris", "Theo", "Nora"}
    assert all(p["decision"] == "PASS" for p in props)      # 전부 재검증 통과
    assert all(p["headline"] for p in props)                # 헤드라인이 비어있지 않음


def test_failing_proposals_are_labeled_not_crashed(monkeypatch):
    # review가 계속 REVIEW → 보정 1회 시도해도 안 되면 정직하게 REVIEW로 라벨
    _patch(monkeypatch, Decision.REVIEW)
    r = agent.run("대한민국 No.1 에어컨")

    props = r["proposals"]
    assert len(props) == 3
    assert all(p["decision"] == "REVIEW" for p in props)
    assert all(p["homework"] for p in props), "REVIEW 안에는 남은 숙제가 있어야 함"


def test_report_sorts_pass_first():
    # node_report의 정렬만 단위 검증 (LLM 불필요)
    state = {
        "ad_text": "x", "country": "KR",
        "diagnosis": {"decision": "REVIEW", "issues": []},
        "proposals": [
            {"id": "a", "name": "A", "route": "R", "headline": "h1", "support": "s1", "why_lg": "", "edits": [],
             "compliance": {"decision": "REVIEW", "risk_score": 0.5, "issues": [{"rule_id": "R-01", "suggestion": "s"}]}},
            {"id": "b", "name": "B", "route": "R", "headline": "h2", "support": "s2", "why_lg": "", "edits": [],
             "compliance": {"decision": "PASS", "risk_score": 0.0, "issues": []}},
        ],
    }
    out = agent.node_report(state)["report"]["proposals"]
    assert out[0]["decision"] == "PASS"      # PASS가 위로
    assert out[1]["decision"] == "REVIEW"
