"""
진입점 하나. 프로젝트 루트에서 실행한다.

    python cli.py index                          # 규정 28개 임베딩 → Chroma
    python cli.py review "대한민국 No.1 에어컨"     # 단건 검토
    python cli.py review --json "30% 할인!"
    python cli.py eval                           # 평가셋 전체 + 지표
    python cli.py eval --limit 5 --top-k 3

이 파일이 하는 일은 "터미널에 친 글자를 해석해서 알맞은 함수를 부르는 것"뿐이다.
판단 로직은 하나도 없다. 실제 일은 전부 src/ 안에서 일어난다.

왜 루트에 있나
    파이썬은 실행한 .py 파일이 있는 폴더를 sys.path(=import 검색 경로)에 자동으로 넣는다.
    이 파일이 루트에 있으면 sys.path에 프로젝트 루트가 들어가므로 `from src...`가 그냥 된다.
    scripts/ 안에 뒀다면 sys.path에 scripts/만 들어가서 src를 못 찾는다.
    (예전에 그걸 때우려고 scripts/_bootstrap.py가 있었는데, 파일을 루트로 옮겨서 없앴다)
"""

import argparse
import json

from src import config
from src.schema import Decision

# 터미널에서 판정을 한눈에 구분하려고 붙이는 표식. 색은 안 쓴다(윈도우 터미널 호환).
_MARK = {
    Decision.PASS: "[ PASS  ]",
    Decision.REVIEW: "[ REVIEW ]",
    Decision.BLOCK: "[ BLOCK ]",
}


# ---------------------------------------------------------------------------
# 서브커맨드 3개. 각 함수는 argparse가 파싱해준 args 객체 하나만 받는다.
#
# import를 함수 안에서 하는 이유:
#   src.retriever를 불러오면 chromadb가, src.pipeline을 불러오면 google-genai가
#   같이 로딩된다(수 초 걸림). `python cli.py --help`만 치는데 그걸 다 기다릴 이유가 없다.
#   실제로 그 명령을 실행할 때만 로딩되도록 함수 안으로 내렸다.
# ---------------------------------------------------------------------------
def cmd_index(args):
    """규정을 임베딩해서 벡터 인덱스를 만든다. 규정 파일을 고쳤으면 다시 실행."""
    from collections import Counter

    from src.retriever import PolicyRetriever
    from src.schema import load_policies

    # 나라별 규정 수를 먼저 보여준다 (data/policies/*.jsonl 구조가 눈에 보이게)
    by_country = Counter(p.country for p in load_policies(config.POLICIES_DIR))
    breakdown = ", ".join(f"{c} {n}" for c, n in sorted(by_country.items()))

    n = PolicyRetriever().build()
    print(f"Indexed {n} policies ({breakdown}) → {config.CHROMA_PATH}")


def cmd_review(args):
    """광고 문안 1건을 검토해서 결과를 출력한다."""
    from src.pipeline import review

    result = review(args.ad_text, top_k=args.top_k, country=args.country)

    # --json: 다른 프로그램이 받아먹기 좋은 형태. 사람이 볼 땐 아래 리포트.
    if args.json:
        print(result.to_json())
        return

    print(f"\n{'=' * 72}\n[{args.country}] 광고 문안: {result.ad_text}\n{'=' * 72}")
    print(f"{_MARK[result.decision]}  risk_score = {result.risk_score:.2f}\n")

    if result.claims:
        print("추출된 claim:")
        for c in result.claims:
            print(f"  - {c}")
        print()

    if not result.issues:
        print("발견된 위반 소지가 없습니다.")
    else:
        print(f"발견된 위반 소지 {len(result.issues)}건:")
        for i, x in enumerate(result.issues, 1):
            print(f"\n  {i}. \"{x.claim}\"")
            print(f"     근거 규정 : {x.rule_id}")
            print(f"     사유      : {x.reason}")
            print(f"     수정 제안 : {x.suggestion}")

    # 어떤 규정이 후보로 올라갔는지 = 검색이 잘 됐는지 눈으로 확인하는 용도.
    # 판정이 이상할 때 "검색이 문제였나, 판단이 문제였나"를 여기서 먼저 본다.
    print(f"\n(검색된 규정: {', '.join(result.retrieved_rule_ids) or '없음'})\n")


def cmd_eval(args):
    """평가셋 전체를 돌려 지표를 내고 results.json에 저장한다."""
    from src import evaluate

    payload = evaluate.run(limit=args.limit, top_k=args.top_k)
    evaluate.print_report(payload)

    # 케이스별 상세는 화면에 다 못 담으니 파일로. 오답 분석할 때 이걸 연다.
    config.RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"상세 결과 저장: {config.RESULTS_PATH}")


_BADGE = {"PASS": "✅ PASS", "REVIEW": "⚠ REVIEW", "BLOCK": "⛔ BLOCK"}


def cmd_agent(args):
    """광고 문안을 구루 패널에 태운다: 진단 → 3인 추천 → 재검증 → 리포트."""
    from src.agent import run

    r = run(args.ad_text, country=args.country)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    # (1) 원본 진단 — 무엇이 걸렸나
    print(f"\n{'=' * 72}\n[{r['country']}] 원본: {r['ad_text']}\n{'=' * 72}")
    dg = r["diagnosis"]
    if dg["issues"]:
        print(f"진단: {_BADGE.get(dg['decision'], dg['decision'])}")
        for i in dg["issues"]:
            print(f"   └ {i['rule_id']} \"{i['claim']}\" — {i['reason']}")
    else:
        print(f"진단: {_BADGE.get(dg['decision'], dg['decision'])} (원본에 걸린 표현 없음)")

    # (2) 구루 3인의 추천안 (PASS가 위로 정렬됨)
    circ = ["①", "②", "③", "④", "⑤"]
    for n, p in enumerate(r["proposals"]):
        print(f"\n{'─' * 72}")
        print(f"추천안 {circ[n]} {p['name']} · {p['route']}   {_BADGE.get(p['decision'], p['decision'])}")
        print(f'   “{p["headline"]}”')
        print(f"   {p['support']}")
        print(f"   왜 LG다운가: {p['why_lg']}")
        if p["homework"]:
            print(f"   집행 전 확보할 근거:")
            for i in p["homework"]:
                print(f"     · {i['suggestion']}  (규정 {i['rule_id']})")
    print()


# ---------------------------------------------------------------------------
# argparse 조립
#
# argparse는 파이썬 표준 라이브러리다. 하는 일은 한 가지:
#   터미널에 친 문자열 목록  ["review", "--json", "30% 할인!"]
#   → 파이썬 객체            args.cmd="review", args.json=True, args.ad_text="30% 할인!"
# 직접 sys.argv를 잘라서 if/else로 처리해도 되지만, --help 자동 생성,
# 오타 시 에러 메시지, 타입 변환(--top-k 3 → int 3)을 공짜로 해준다.
# ---------------------------------------------------------------------------
def main():
    # (1) 공통 옵션은 '부모 파서'에 담는다.
    #     add_help=False가 필요하다. 안 그러면 자식마다 -h가 중복 정의돼서 에러난다.
    #
    #     왜 이렇게 하나: 공통 옵션을 최상위 파서에 그냥 달면
    #     `python cli.py --top-k 3 eval`처럼 서브커맨드 '앞'에만 쓸 수 있다.
    #     사람은 보통 `python cli.py eval --top-k 3`이라고 친다. 부모 파서를 상속시키면
    #     서브커맨드 '뒤'에 붙일 수 있다.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top-k", type=int, default=config.TOP_K,
                        help="claim 하나당 검색할 규정 수 (기본 %(default)s)")

    # 관할 국가 옵션. review와 agent가 공유한다.
    # 선택지는 data/policies/*.jsonl 파일에서 자동 발견한다 → 나라 추가 = 파일 추가.
    countries = config.available_countries() or ["KR"]
    default_country = "KR" if "KR" in countries else countries[0]
    country_opt = argparse.ArgumentParser(add_help=False)
    country_opt.add_argument("--country", default=default_country, choices=countries,
                             help="심의 관할 국가 (기본 %(default)s, 규정 파일에서 자동 발견)")

    parser = argparse.ArgumentParser(description="광고 컴플라이언스 검토 (Policy RAG)")

    # (2) 서브커맨드를 담을 자리. required=True → 커맨드 없이 실행하면 사용법을 띄운다.
    sub = parser.add_subparsers(dest="cmd", required=True)

    # (3) 커맨드별 파서. set_defaults(fn=...)로 "이 커맨드가 오면 부를 함수"를 심어둔다.
    #     이러면 아래에서 if cmd == "index": ... elif ... 를 쓸 필요가 없다.
    sub.add_parser("index", help="규정을 임베딩해 벡터 인덱스 구축").set_defaults(fn=cmd_index)

    p = sub.add_parser("review", parents=[common, country_opt], help="광고 문안 1건 검토")
    p.add_argument("ad_text", help="검토할 광고 문안 (따옴표로 감쌀 것)")
    p.add_argument("--json", action="store_true", help="사람용 리포트 대신 JSON 출력")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("eval", parents=[common], help="평가셋 전체 실행")
    p.add_argument("--limit", type=int, default=None, help="앞에서 N건만 (빠른 확인용)")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("agent", parents=[country_opt],
                       help="검토→수정→재검토 에이전트 (LangGraph)")
    p.add_argument("ad_text", help="검토할 광고 문안 (따옴표로 감쌀 것)")
    p.add_argument("--json", action="store_true", help="사람용 리포트 대신 JSON 출력")
    p.set_defaults(fn=cmd_agent)

    # (4) 실제 파싱 → (3)에서 심어둔 함수를 호출.
    args = parser.parse_args()
    args.fn(args)


# 이 파일을 직접 실행했을 때만 main()을 부른다.
# 다른 파일이 `import cli`로 가져다 쓸 땐 실행되지 않는다(파이썬 관용구).
if __name__ == "__main__":
    main()
