# 데모 대본 — 국가별 광고 컴플라이언스

같은 광고라도 관할 국가에 따라 판정이 달라지는 것을 보여주는 데모.
규정은 **나라별 파일**로 나뉜다: `data/policies/KR.jsonl`(28) · `US.jsonl`(7) · `EU.jsonl`(7).
파일명이 곧 국가 코드다. 각 나라 법무팀이 자기 파일만 손보면 되고, 나라 추가 = 파일 추가.
전부 synthetic.

Windows 터미널에서 한글이 `?`로 깨지면 먼저:
```powershell
$env:PYTHONIOENCODING="utf-8"
```

## 0. 사전 준비 (최초 1회)

```powershell
cd c:\dev\RAG
python cli.py index      # → "Indexed 42 policies (EU 7, KR 28, US 7)"
```

규정 파일을 고쳤으면 다시 `index`. `--country` 선택지는 `data/policies/`의 파일에서
자동으로 잡힌다(지금은 `KR`(기본)/`US`/`EU`). 새 나라는 파일만 넣고 `index`만 다시 돌리면 된다.

---

## 데모 A — 같은 광고, 다른 국가, 다른 판정  ★ 핵심

친환경 표현. **미국은 조건부 허용(REVIEW), EU는 금지(BLOCK).**

```powershell
python cli.py review --country US "100% eco-friendly air conditioner, zero emissions!"
python cli.py review --country EU "100% eco-friendly air conditioner, zero emissions!"
```

기대:
| 국가 | 판정 | 근거 | 이유 |
|---|---|---|---|
| US | **REVIEW** | US-05 | FTC: 구체적 입증 있으면 쓸 수 있음 → 보완 요구 |
| EU | **BLOCK** | EU-01 | Green Claims 지침: 미입증 포괄적 친환경 표현은 사용 금지 |

> 발표 포인트: "규정 텍스트를 국가별로 분리(Chroma metadata 필터)했고, 판정 기준(심각도·태도)도
> 프롬프트에 관할 맥락으로 주입했습니다. 그래서 **같은 문구가 미국에선 보완, EU에선 차단**입니다."

---

## 데모 B — LG 구루 패널: 한 광고, 3인의 추천안 (영어/UK)  ★ 핵심

에이전트는 걸린 광고를 **하나의 LG 'Life's Good' 보이스를 공유하되 창작 루트가 다른
카피라이터 3명**(Iris / Theo / Nora)이 각자 다시 쓰고, 각 안을 **같은 검토기로 재검증**한다.

> UK 대상이므로 `--country EU`를 쓴다 (UK ASA/CAP 기준은 우리 규정셋의 EU에 가장 가깝다).

```powershell
python cli.py agent --country EU "World's No.1 air conditioner, cuts your energy bill by 50%!"
python cli.py agent --country EU "100% eco-friendly! World's No.1 air conditioner."
```

기대 출력:
```
진단: REVIEW  ← EU-02 최상급(No.1) / EU-05 정량 수치(50%)

추천안 ① Iris · Human Relief          ✅ PASS
   “Less energy anxiety. More everyday comfort.”
추천안 ② Theo · Quiet Intelligence    ✅ PASS
   “It senses the room before you do.”
추천안 ③ Nora · Life's Good Moment     ✅ PASS
   “Cool summer nights, with less getting in the way.”
```

> 발표 포인트 3가지:
> 1. **다중 관점 · LG 톤** — 한 규정 문제에 LG 브랜드 3루트(Human Relief / Quiet Intelligence /
>    Life's Good Moment)로 동시에 답한다. 짧고, 사람 중심이고, Life's Good으로 이어진다.
> 2. **충족을 증명** — 각 안을 RAG 검토기로 재검증한 결과가 ✅. 말로만 "됐다"가 아니다.
> 3. **브랜드 = 컴플라이언스** — LG 보이스가 이미 과장·근거 없는 단정을 금지한다.
>    'LG답게'와 '규정 통과'가 같은 방향으로 움직인다(사실 창작 없이 최상급·수치를 걷어냄).

> 실행에 약 30~60초 걸린다(구루 3인 생성 + 재검증). `--json`으로 데이터만 뽑을 수도 있다.
> 한국어 광고를 넣어도 되고(그러면 KR 규정 기준), 카피는 항상 영어(UK)로 나온다.

---

## 데모 C — 가격 할인 (국가별 강도 차이)

```powershell
python cli.py review --country US "30% off this week!"
python cli.py review --country EU "30% off this week!"
```

기대: 둘 다 REVIEW지만 **EU가 더 엄격**(risk_score 더 높음).
EU Omnibus 지침은 "직전 30일 최저가 기준"을 요구(EU-04)해서 조건이 더 많다.

---

## 데모 D — 한국어 광고 (기본 KR)

`--country`를 안 주면 KR. 기존 한국어 파이프라인 그대로.

```powershell
python cli.py review "대한민국 No.1 에어컨, 전기료 50% 절감!"
python cli.py agent  "대한민국 No.1 에어컨, 전기료 50% 절감!"
```

기대: R-01(최상급)·R-04(정량 효능)로 REVIEW. 에이전트는 최상급을 완화하고 수치는 빈칸으로 남김.

---

## 그 밖의 테스트용 문구

붙여넣어 국가를 바꿔가며 실험해 보세요.

| 문구 | 노리는 규정 |
|---|---|
| `"World's No.1 air purifier"` | US-01 / EU-02 (최상급) |
| `"Kills 99.9% of germs"` | US-07 (살균 근거) |
| `"Cures allergies and asthma"` | EU-06 (미승인 건강 표현) |
| `"Twice as fast as Brand X"` | EU-03 (비교광고) |
| `"Free gift for every sign-up!"` | US-04 ('무료' 조건 고지) |

## 참고

- `--json`을 붙이면 사람용 리포트 대신 JSON. 화면 캡처 대신 데이터로 보여줄 때.
- 판정 문구(사유/제안)는 한국어로 나온다. "한국 마케터가 해외 캠페인을 사전 점검"하는 도구라는 설정.
- LLM 특성상 문장 표현은 실행마다 조금씩 다를 수 있으나 **판정(PASS/REVIEW/BLOCK)과 근거 규정은 안정적**이다.
