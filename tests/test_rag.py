"""
테스트 — API 키 없이 돈다 (LLM은 전부 stub).

두 가지만 본다:
  1. LLM이 근거를 지어냈을 때 걸러지는가   ← 이 프로젝트의 핵심 방어
  2. 검색 랭킹/중복 병합 로직이 맞는가

모델 품질은 여기서 안 잰다. 그건 `python cli.py eval`의 몫.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src import config  # noqa: E402
from src.retriever import PolicyRetriever  # noqa: E402
from src.schema import (  # noqa: E402
    Decision, Policy, load_eval_set, load_policies, parse_review_payload,
)


def _payload(decision="REVIEW", risk=0.6, issues=()):
    return {"decision": decision, "risk_score": risk, "issues": list(issues)}


def _issue(rule_id):
    return {"claim": "x", "rule_id": rule_id, "reason": "-", "suggestion": "-"}


# ── 1. LLM 출력 검증 (guardrail) ──────────────────────────────────────
def test_decision_parse_fails_safe():
    assert Decision.parse("block") is Decision.BLOCK
    assert Decision.parse("  Review ") is Decision.REVIEW
    # 알 수 없는 값은 PASS가 아니라 REVIEW로. 모르면 사람에게 넘긴다.
    assert Decision.parse("무슨말이지") is Decision.REVIEW


def test_hallucinated_rule_id_is_dropped():
    """검색되지 않은 규정(R-99)을 근거로 든 issue는 폐기된다."""
    p = _payload("BLOCK", 0.9, [_issue("R-01"), _issue("R-99")])
    r = parse_review_payload(p, "광고", ["x"], allowed_rule_ids=["R-01", "R-04"])
    assert [i.rule_id for i in r.issues] == ["R-01"]
    assert r.decision is Decision.BLOCK


def test_all_issues_dropped_falls_back_to_pass():
    """근거가 전부 폐기되면 판정도 근거를 잃는다 → PASS로 되돌림."""
    r = parse_review_payload(_payload("BLOCK", 0.95, [_issue("R-99")]), "광고", ["x"], ["R-01"])
    assert r.decision is Decision.PASS and r.issues == [] and r.risk_score <= 0.2


def test_pass_with_issues_is_escalated():
    """issue를 지적해놓고 PASS라는 모순은 REVIEW로 올린다."""
    r = parse_review_payload(_payload("PASS", 0.1, [_issue("R-08")]), "광고", ["x"], ["R-08"])
    assert r.decision is Decision.REVIEW and r.risk_score >= 0.4


def test_risk_score_is_clamped():
    assert parse_review_payload(_payload("PASS", "이상한값"), "광고", [], []).risk_score == 0.0
    r = parse_review_payload(_payload("BLOCK", 7.5, [_issue("R-01")]), "광고", ["x"], ["R-01"])
    assert r.risk_score == 1.0


# ── 2. 검색 ──────────────────────────────────────────────────────────
_VOCAB = ["최상급", "할인", "살균", "경쟁사"]

_POLICIES = [
    Policy("R-01", "최상급", "최상급 표현 규정", "최상급 표현 근거 필요", "HIGH"),
    Policy("R-08", "할인", "할인 표시 규정", "할인 기준가격 표시", "HIGH"),
    Policy("R-17", "살균", "살균 표현 규정", "살균 시험성적서 필요", "HIGH"),
    Policy("R-14", "경쟁사", "경쟁사 언급 규정", "경쟁사 실명 금지", "HIGH"),
]


def _stub_embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """단어 포함 여부를 그대로 벡터로 만든 뒤 L2 정규화. 결정적이라 테스트에 적합."""
    out = []
    for t in texts:
        v = [1.0 if w in t else 0.0 for w in _VOCAB]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


@pytest.fixture
def retriever(tmp_path):
    # 진짜 Chroma를 임시 폴더에 띄운다. 빠르고, 실제 저장소 동작까지 같이 검증된다.
    r = PolicyRetriever(embed_fn=_stub_embed, policies=_POLICIES, path=tmp_path)
    r.build()
    return r


def test_build_indexes_all(retriever):
    assert retriever.count() == len(_POLICIES)


def test_search_ranks_relevant_first(retriever):
    hits = retriever.search("할인 문구를 검토해 주세요", k=2)
    assert hits[0].policy.rule_id == "R-08"
    assert 0.0 <= hits[0].score <= 1.0


def test_search_respects_k(retriever):
    assert len(retriever.search("할인", k=1)) == 1
    assert len(retriever.search("할인", k=3)) == 3


def test_search_many_merges_and_sorts(retriever):
    hits = retriever.search_many(["할인 표현", "할인 기준", "살균 표현"], k=3)
    ids = [h.policy.rule_id for h in hits]
    assert len(ids) == len(set(ids)), "중복 규정이 병합되지 않았다"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_rebuild_does_not_duplicate(retriever):
    """upsert라 몇 번 돌려도 개수가 안 늘어야 한다."""
    retriever.build()
    assert retriever.count() == len(_POLICIES)


def test_country_filter_isolates(tmp_path):
    """country를 주면 그 국가 규정만, 안 주면 전체가 검색돼야 한다."""
    policies = [
        Policy("R-01", "최상급", "KR 최상급", "최상급 근거 필요", "HIGH", country="KR"),
        Policy("US-01", "최상급", "US 최상급", "최상급 근거 필요", "HIGH", country="US"),
    ]
    r = PolicyRetriever(embed_fn=_stub_embed, policies=policies, path=tmp_path)
    r.build()

    kr = r.search("최상급", k=5, country="KR")
    assert [h.policy.rule_id for h in kr] == ["R-01"]      # KR만

    us = r.search("최상급", k=5, country="US")
    assert [h.policy.rule_id for h in us] == ["US-01"]     # US만

    both = {h.policy.rule_id for h in r.search("최상급", k=5)}  # 필터 없음
    assert both == {"R-01", "US-01"}                        # 전체


# ── 3. 데이터 ────────────────────────────────────────────────────────
def test_data_files_are_wellformed():
    policies = load_policies(config.POLICIES_DIR)
    rule_ids = {p.rule_id for p in policies}
    assert len(rule_ids) == len(policies) >= 20, "rule_id 중복"

    for row in load_eval_set(config.EVAL_SET_PATH):
        assert row["expected"] in {"PASS", "REVIEW", "BLOCK"}
        for rid in row.get("expected_rules", []):
            assert rid in rule_ids, f"{row['id']}가 없는 규정 {rid}를 참조"
