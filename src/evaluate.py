"""
평가 — 파이프라인 [4]단계.

왜 Accuracy만 보지 않는가
    False Positive: 멀쩡한 광고를 REVIEW로 올림 → 검토자가 5분 더 씀
    False Negative: 위험한 광고를 PASS로 통과   → 제재/소송/브랜드 리스크
비용이 다르므로 "위험(REVIEW+BLOCK)"을 positive로 둔 Recall이 1순위 지표다.

retrieval을 따로 재는 이유
    최종 판정만 보면 "검색은 실패했는데 LLM 상식으로 우연히 맞춘" 경우를 못 거른다.
    정답 규정이 top-k에 들어왔는지(Recall@k)를 따로 재야 오답의 원인이
    '검색 실패'인지 '판단 실패'인지 구분된다.
"""

from __future__ import annotations

import time

from src import config
from src.pipeline import review
from src.schema import Decision, load_eval_set

RISKY = {Decision.REVIEW, Decision.BLOCK}  # positive 클래스
ORDER = [Decision.PASS, Decision.REVIEW, Decision.BLOCK]


def run(limit: int | None = None, top_k: int = config.TOP_K) -> dict:
    """평가셋을 돌려 케이스별 결과 + 집계 지표를 반환한다."""
    rows = load_eval_set(config.EVAL_SET_PATH)[:limit]
    cases, started = [], time.time()

    for i, row in enumerate(rows, 1):
        expected = Decision.parse(row["expected"])
        result = review(row["ad_text"], top_k=top_k)

        # 정답 규정이 실제로 검색됐는가 (없는 케이스는 None = 채점 제외)
        want = set(row.get("expected_rules") or [])
        got = set(result.retrieved_rule_ids)
        hit = bool(want & got) if want else None

        cases.append({
            "id": row["id"],
            "ad_text": row["ad_text"],
            "expected": expected.value,
            "predicted": result.decision.value,
            "correct": expected is result.decision,
            "risk_score": result.risk_score,
            "expected_rules": sorted(want),
            "cited_rules": sorted({x.rule_id for x in result.issues}),
            "retrieval_hit": hit,
            "retrieval_coverage": len(want & got) / len(want) if want else None,
        })
        print(f"[{i:>2}/{len(rows)}] {'O' if cases[-1]['correct'] else 'X'} {row['id']} "
              f"expected={expected.value:<6} predicted={result.decision.value}")

    elapsed = time.time() - started
    metrics = _metrics(cases) | {
        "elapsed_sec": round(elapsed, 1),
        "sec_per_case": round(elapsed / max(len(cases), 1), 2),
        "top_k": top_k,
    }
    return {"metrics": metrics, "cases": cases}


def _metrics(cases: list[dict]) -> dict:
    n = len(cases)
    if n == 0:
        return {}

    # 위험(REVIEW+BLOCK)을 positive로 둔 2-class 집계
    tp = fn = fp = tn = 0
    matrix = {e.value: {p.value: 0 for p in ORDER} for e in ORDER}
    for c in cases:
        exp_risky = Decision.parse(c["expected"]) in RISKY
        pred_risky = Decision.parse(c["predicted"]) in RISKY
        if exp_risky and pred_risky:
            tp += 1
        elif exp_risky:
            fn += 1        # ← 컴플라이언스에서 가장 중요한 숫자
        elif pred_risky:
            fp += 1
        else:
            tn += 1
        matrix[c["expected"]][c["predicted"]] += 1

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    scored = [c for c in cases if c["retrieval_hit"] is not None]
    return {
        "n": n,
        "accuracy": round(sum(c["correct"] for c in cases) / n, 3),
        "risky_recall": round(recall, 3),
        "risky_precision": round(precision, 3),
        "risky_f1": round(f1, 3),
        "true_positive": tp, "false_negative": fn,
        "false_positive": fp, "true_negative": tn,
        "retrieval_recall_at_k": round(
            sum(bool(c["retrieval_hit"]) for c in scored) / len(scored), 3) if scored else 0.0,
        "retrieval_coverage": round(
            sum(c["retrieval_coverage"] for c in scored) / len(scored), 3) if scored else 0.0,
        "confusion_matrix": matrix,
    }


def print_report(payload: dict) -> None:
    m, cases = payload["metrics"], payload["cases"]
    print(f"\n{'=' * 72}")
    print(f"EVALUATION  (n={m['n']}, top_k={m['top_k']}, "
          f"{m['sec_per_case']}s/case, total {m['elapsed_sec']}s)")
    print("=" * 72)

    print("\n── Decision quality ──────────────────────────────────")
    print(f"  Accuracy (3-class)  : {m['accuracy']:.3f}")
    print(f"  Risky Recall  ★     : {m['risky_recall']:.3f}   놓친 위험 광고 {m['false_negative']}건")
    print(f"  Risky Precision     : {m['risky_precision']:.3f}   과잉 검토 {m['false_positive']}건")
    print(f"  Risky F1            : {m['risky_f1']:.3f}")

    print("\n── Retrieval quality ─────────────────────────────────")
    print(f"  Recall@k (정답 규정 1개 이상 적중) : {m['retrieval_recall_at_k']:.3f}")
    print(f"  Coverage (정답 규정 적중 비율)     : {m['retrieval_coverage']:.3f}")

    print("\n── Confusion matrix (행=정답, 열=예측) ───────────────")
    print("  " + " " * 10 + "".join(f"{d.value:>9}" for d in ORDER))
    for e in ORDER:
        print(f"  {e.value:<10}" + "".join(f"{m['confusion_matrix'][e.value][p.value]:>9}" for p in ORDER))

    wrong = [c for c in cases if not c["correct"]]
    if wrong:
        print("\n── 오답 케이스 ───────────────────────────────────────")
        for c in wrong:
            miss = "" if c["retrieval_hit"] in (True, None) else "  [검색 실패]"
            print(f"  {c['id']}  {c['expected']} → {c['predicted']}{miss}\n        {c['ad_text']}")
    print()
