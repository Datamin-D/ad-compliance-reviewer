"""
LG 스타일 마케팅 구루 패널 에이전트 (LangGraph).

기존 파이프라인(pipeline.review)은 '검토'만 했다 — 문제를 짚고 끝.
이 에이전트는 걸린 광고를 **LG 브랜드 보이스를 공유하되 창작 루트가 다른 카피라이터 3명**이
각자 다시 쓰고, 각 안을 **같은 RAG 검토기로 재검증**한다. "컴플라이언스 충족"을
말이 아니라 판정으로 증명한다.

    원본 광고
       │ [1] diagnose  기존 review로 '무엇이 걸리는지' 진단
       ▼
    ┌──── [2] panel : 구루 3인이 각자 LG 창작 루트로 다시 씀 (영어/UK) ────┐
    │  Iris  — Human Relief      "Less [friction]. More [life]."          │
    │  Theo  — Quiet Intelligence"[it] gets it right, before you need to."│
    │  Nora  — Life's Good Moment"[a real moment], with less in the way." │
    └──────────────────────────┬──────────────────────────────────────────┘
       │ [3] verify : 각 안을 review로 재검증 → 아직 걸리면 1회 보정
       ▼
    [4] report : 3개 추천안 + 각각의 컴플라이언스 상태 + 남은 숙제

설계 메모 (AI 엔지니어 판단)
    브랜드 스킬 원문(LG_Marketing_Style_SKILL.md)은 ~500줄이라 매 LLM 호출에 통째로
    넣으면 토큰·비용·지연이 폭발한다. 그래서 핵심만 압축한 LG_VOICE 상수로 프롬프트에
    박았다(원문은 이 상수를 증류한 레퍼런스). 3인의 창작 루트는 그 스킬 §14를 그대로 따랐고,
    "근거 없는 단정 금지"(스킬 §13)는 우리 컴플라이언스 목표와 방향이 같아 자연히 맞물린다.

핵심 안전 제약
    구루들도 없는 사실(조사기관·시험성적·수상)을 지어내면 안 된다. 규제 표현은
    삭제·완화·리프레이밍으로만 해소하고(스키마에 'ADD' 없음), 최종적으로 verify 단계의
    RAG 검토가 통과를 보장한다.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src import llm
from src.pipeline import review  # 기존 검토 파이프라인을 그대로 재사용

# 검증에서 걸린 추천안을 자동 보정할 최대 횟수. 무한 루프가 아니라 딱 1회.
MAX_REPAIR = 1


# ---------------------------------------------------------------------------
# LG 브랜드 보이스 — 3인이 공통으로 지키는 규칙 (스킬에서 증류)
# ---------------------------------------------------------------------------
LG_VOICE = """\
[공통 브랜드 보이스 — LG 전자 'Life's Good'. 세 카피 모두 아래를 지킨다]
- 사람이 주인공, 기술은 배경. 기능 이름을 헤드라인에 넣지 않는다.
- 짧고 쉽게. 헤드라인은 영어 2~8단어, 한 가지 생각만. 소리 내어 읽어 자연스럽게.
- 조용한 지성의 어휘: understands / senses / anticipates / quietly handles.
  과시·로봇·마법·벤치마크 톤 금지.
- 일상의 낙관: 거창한 꿈이 아니라 '생각할 거리 하나가 줄어드는' 작은 편익.
- 금지어: revolutionary, ultimate, unmatched, next-generation, game-changing,
  그리고 공허한 프리미엄어(elevate your lifestyle, redefine luxury 등).
- 기본적으로 느낌표(!)를 쓰지 않는다. 명령형(Upgrade your life)보다 돕는 어조.
- 근거 없는 단정은 부드럽게. (예: "Knows what you need" → "Adapts to how you use it")
- 마지막 정서가 'Life's Good'과 자연스럽게 이어져야 한다.
- 출력은 영어. 
"""


# ---------------------------------------------------------------------------
# 구루 3인 — 브랜드 보이스는 공유, 창작 루트는 다르게 (스킬 §14)
# ---------------------------------------------------------------------------
PERSONAS = [
    {
        "id": "relief",
        "name": "Iris",
        "route": "Human Relief",
        "system": "You are Iris, an LG copywriter who writes in the 'Human Relief' route: "
                  "start from the burden being removed.",
        "strategy": "제거되는 부담에서 출발한다. 구조: \"Less [friction]. More [life].\" "
                    "대중 캠페인과 Life's Good 정렬에 강하다. 예) Less checking. More chilling.",
    },
    {
        "id": "quiet",
        "name": "Theo",
        "route": "Quiet Intelligence",
        "system": "You are Theo, an LG copywriter who writes in the 'Quiet Intelligence' route: "
                  "start from what the product quietly understands or handles.",
        "strategy": "제품/공간이 조용히 알아서 하는 것에서 출발한다. 구조: "
                    "\"[it] gets it right, before you need to.\" AI·센싱·개인화 제품에 강하다. "
                    "예) The room gets it right before you notice.",
    },
    {
        "id": "moment",
        "name": "Nora",
        "route": "Life's Good Moment",
        "system": "You are Nora, an LG copywriter who writes in the 'Life's Good Moment' route: "
                  "start from the human moment made possible once the technology disappears.",
        "strategy": "기술이 사라진 뒤 가능해지는 인간적 순간에서 출발한다. 구조: "
                    "\"[a real moment], with less getting in the way.\" 라이프스타일·주방·TV·가족 맥락에 강하다. "
                    "예) More time sharing dinner, less time watching it.",
    },
]
_PERSONA_BY_ID = {p["id"]: p for p in PERSONAS}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    ad_text: str       # 원본 광고 문안 (고정)
    country: str       # 관할 국가 (KR/US/EU, 고정)
    diagnosis: dict    # 원본을 검토한 결과 (무엇이 걸리나)
    proposals: list    # 구루별 추천안
    report: dict       # 최종 리포트


# ---------------------------------------------------------------------------
# LLM 출력 스키마 — LG 카피 한 편 (헤드라인 + 서포트 + 왜 LG인지)
# ---------------------------------------------------------------------------
_PROPOSAL_SCHEMA = {
    "type": "OBJECT",
    "required": ["headline", "support", "why_lg", "edits"],
    "properties": {
        "headline": {"type": "STRING"},   # 영어 2~8단어, 느낌표 없이
        "support": {"type": "STRING"},    # 영어 한 문장, 제품 진실 → 인간 편익
        "why_lg": {"type": "STRING"},     # 왜 LG다운지 한 줄
        "edits": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["type", "target", "note"],
                "properties": {
                    # 사실 '추가(ADD)'는 없다 = 지어낼 수 없다.
                    "type": {"type": "STRING",
                             "enum": ["DELETE", "SOFTEN", "PLACEHOLDER", "REFRAME"]},
                    "target": {"type": "STRING"},
                    "note": {"type": "STRING"},
                },
            },
        },
    },
}


def _issue_dict(i) -> dict:
    return {"claim": i.claim, "rule_id": i.rule_id,
            "reason": i.reason, "suggestion": i.suggestion}


def _copy_of(prop: dict) -> str:
    """헤드라인 + 서포트를 합쳐 '검토 대상 광고 문안'을 만든다."""
    return f"{prop['headline']}. {prop['support']}".strip()


def _check(copy: str, country: str) -> dict:
    """문안 하나를 검토기로 재검증한다. (verify의 핵심 — '충족'을 증명하는 단계)"""
    r = review(copy, country=country)
    return {
        "decision": r.decision.value,
        "risk_score": r.risk_score,
        "issues": [_issue_dict(i) for i in r.issues],
    }


def _fmt_issues(issues: list) -> str:
    if not issues:
        return "(걸린 사유 없음)"
    return "\n".join(
        f"- \"{i['claim']}\" (규정 {i['rule_id']}): {i['reason']} → {i['suggestion']}"
        for i in issues
    )


# ---------------------------------------------------------------------------
# 노드 1 — DIAGNOSE
# ---------------------------------------------------------------------------
def node_diagnose(state: State) -> dict:
    r = review(state["ad_text"], country=state["country"])
    return {"diagnosis": {
        "decision": r.decision.value,
        "issues": [_issue_dict(i) for i in r.issues],
    }}


# ---------------------------------------------------------------------------
# 노드 2 — PANEL
# ---------------------------------------------------------------------------
_GEN_PROMPT = """\
당신은 LG 전자 카피라이터 '{name}'입니다. 창작 루트: {route}.
{strategy}

{voice}

아래 광고를 위 브랜드 보이스로, {country} 광고 규정을 충족하도록 영어(UK)로 다시 써 주세요.

## 원본 문안
"{ad_text}"

## 규정 위반 지적 (이 문제들을 해소해야 합니다)
{issues}

## 규칙
- 없는 사실·수치·기관명·수상·조사결과를 **절대 만들어내지 마세요.**
- 규제되는 표현(근거 없는 최상급/수치, 안전성 단정, 경쟁사 비방)은 삭제·완화·리프레이밍하세요.
- 세 카피라이터가 확연히 달라야 합니다. 당신은 '{route}' 루트를 끝까지 지킵니다.
- headline: 영어 2~8단어, 느낌표 없이. support: 영어 한 문장. why_lg: 왜 LG다운지 한 줄(한국어 가능).
- edits: 원본의 규정 위반 표현을 무엇으로 어떻게 바꿨는지.
"""


def _generate(persona: dict, ad_text: str, country: str, issues: list) -> dict:
    payload = llm.generate_json(
        _GEN_PROMPT.format(
            name=persona["name"], route=persona["route"], strategy=persona["strategy"],
            voice=LG_VOICE, country=country, ad_text=ad_text, issues=_fmt_issues(issues)),
        _PROPOSAL_SCHEMA,
        system_instruction=persona["system"],
    )
    return {
        "id": persona["id"], "name": persona["name"], "route": persona["route"],
        "headline": (payload.get("headline") or "").strip(),
        "support": (payload.get("support") or "").strip(),
        "why_lg": (payload.get("why_lg") or "").strip(),
        "edits": payload.get("edits") or [],
    }


def node_panel(state: State) -> dict:
    issues = state["diagnosis"]["issues"]
    proposals = [
        _generate(p, state["ad_text"], state["country"], issues)
        for p in PERSONAS
    ]
    return {"proposals": proposals}


# ---------------------------------------------------------------------------
# 노드 3 — VERIFY (재검증 + 1회 보정)
# ---------------------------------------------------------------------------
_REPAIR_PROMPT = """\
당신(LG 카피라이터 '{name}', 루트: {route})이 쓴 아래 광고가 {country} 규정에 아직 걸립니다.
LG 브랜드 보이스와 당신의 루트를 유지하되 영어(UK)로 고쳐 주세요.

## 현재 문안
"{copy}"

## 아직 걸린 사유
{issues}

## 규칙
- 없는 사실을 만들지 마세요. 삭제·완화·리프레이밍으로만 해소하세요.
- headline / support / why_lg / edits 형식으로 답하세요.
"""


def _repair(persona: dict, copy: str, issues: list, country: str) -> dict:
    payload = llm.generate_json(
        _REPAIR_PROMPT.format(
            name=persona["name"], route=persona["route"],
            country=country, copy=copy, issues=_fmt_issues(issues)),
        _PROPOSAL_SCHEMA,
        system_instruction=persona["system"],
    )
    return {
        "headline": (payload.get("headline") or "").strip(),
        "support": (payload.get("support") or "").strip(),
        "why_lg": (payload.get("why_lg") or "").strip(),
        "edits": payload.get("edits") or [],
    }


def node_verify(state: State) -> dict:
    verified = []
    for prop in state["proposals"]:
        prop = dict(prop)
        comp = _check(_copy_of(prop), state["country"])

        # 아직 통과 못 했으면 딱 1회 보정 (무한 루프 아님)
        if comp["decision"] != "PASS" and MAX_REPAIR:
            persona = _PERSONA_BY_ID[prop["id"]]
            fix = _repair(persona, _copy_of(prop), comp["issues"], state["country"])
            if fix["headline"]:  # 보정본이 비지 않았을 때만
                fixed = {**prop, "headline": fix["headline"], "support": fix["support"],
                         "why_lg": fix["why_lg"] or prop["why_lg"]}
                comp2 = _check(_copy_of(fixed), state["country"])
                if comp2["decision"] == "PASS" or len(comp2["issues"]) < len(comp["issues"]):
                    prop = {**fixed, "edits": prop["edits"] + fix["edits"]}
                    comp = comp2

        prop["compliance"] = comp
        verified.append(prop)
    return {"proposals": verified}


# ---------------------------------------------------------------------------
# 노드 4 — REPORT (LLM 없이 코드로만 조립 — 결정적·저비용)
# ---------------------------------------------------------------------------
def node_report(state: State) -> dict:
    cards = []
    for p in state["proposals"]:
        comp = p["compliance"]
        cards.append({
            "id": p["id"], "name": p["name"], "route": p["route"],
            "headline": p["headline"], "support": p["support"], "why_lg": p["why_lg"],
            "edits": p["edits"],
            "decision": comp["decision"], "risk_score": comp["risk_score"],
            "homework": comp["issues"],   # 재검증에도 남은 지적(주로 근거자료 필요)
        })
    # 바로 쓸 수 있는 안(PASS)을 위로 정렬
    order = {"PASS": 0, "REVIEW": 1, "BLOCK": 2}
    cards.sort(key=lambda c: order.get(c["decision"], 3))

    return {"report": {
        "ad_text": state["ad_text"],
        "country": state["country"],
        "diagnosis": state["diagnosis"],
        "proposals": cards,
    }}


# ---------------------------------------------------------------------------
# 그래프 조립 — 진단 → 패널 → 검증 → 리포트 (직선)
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("diagnose", node_diagnose)
    g.add_node("panel", node_panel)
    g.add_node("verify", node_verify)
    g.add_node("report", node_report)

    g.add_edge(START, "diagnose")
    g.add_edge("diagnose", "panel")
    g.add_edge("panel", "verify")
    g.add_edge("verify", "report")
    g.add_edge("report", END)
    return g.compile()


def run(ad_text: str, country: str = "KR") -> dict:
    """광고 문안 하나를 패널에 태워 최종 report(dict)를 돌려준다."""
    app = build_graph()
    final = app.invoke({
        "ad_text": ad_text,
        "country": country,
        "diagnosis": {},
        "proposals": [],
        "report": {},
    })
    return final["report"]