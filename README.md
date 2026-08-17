# Ad Compliance Reviewer — a Policy RAG prototype

A small RAG system that reviews Korean advertising copy against a set of internal
ad-review policies and returns a **PASS / REVIEW / BLOCK** decision with the specific
clauses it relied on.

> All policies and evaluation cases in this repository are **synthetic**. They were
> written for this prototype and contain no material from any real company.

---

## 1. Why this exists

The original problem came from a real workflow: every piece of advertising creative had
to be cleared by legal before it could run, and the bottleneck was that reviewers were
re-reading the same rulebook for every asset.

An earlier PoC embedded ad images and looked for visually similar past assets that had
been rejected. It found *something*, but the logic was weak — **visual similarity is not
legal similarity**. Two ads can look nearly identical and differ only in a footnote that
decides whether the claim is compliant.

This version reframes the problem. The evidence for a legal decision should be the
**policy text itself**, retrieved by meaning, not a neighbouring image. Image similarity
is still useful, but as *precedent lookup* — a second, supporting retriever — not as the
basis for the decision. That part is on the roadmap, not in this repo.

## 2. Architecture

```
ad copy (text)
     │
     ▼
[1] Claim Extractor        LLM splits the copy into individually verifiable claims
     │                     "대한민국 No.1 에어컨, 전기료 50% 절감!"
     │                       → ["대한민국 No.1", "전기료 50% 절감"]
     ▼
[2] Policy Retriever       per-claim top-k vector search over 42 policy clauses (KR·US·EU),
     │                     results merged and de-duplicated
     │                       → [R-01 superlatives, R-04 quantified performance, …]
     ▼
[3] Compliance Judge       LLM decides using ONLY the retrieved clauses,
     │                     constrained to a JSON schema
     ▼
{ "decision": "REVIEW", "risk_score": 0.6,
  "issues": [ { "claim": …, "rule_id": "R-01", "reason": …, "suggestion": … } ] }
     │
     ▼
[4] Evaluator              25 labelled cases → accuracy, recall, Recall@k
```

| Component | Choice |
|---|---|
| LLM | `gemini-3.6-flash` (temperature 0, structured output) |
| Embeddings | `gemini-embedding-001`, 768 dims, asymmetric task types |
| Vector store | ChromaDB (persistent, cosine) with a NumPy brute-force fallback |
| Language | Policies and ad copy in Korean; code comments in Korean; docs in English |

## 3. Design decisions worth defending

**Claim extraction before retrieval.**
Embedding the whole ad as one query put the vector somewhere between two unrelated
claims and matched neither policy strongly. Splitting first and searching per claim
was the single largest improvement to retrieval quality.
→ `extract_claims()` in [`src/pipeline.py`](src/pipeline.py)

**Per-country jurisdiction, owned by data not code.**
Each market's rules live in **their own file** — `data/policies/KR.jsonl`, `US.jsonl`,
`EU.jsonl` — so a region's legal team maintains only their file. The filename *is* the
country code. The user selects a country; retrieval filters to it (`where={"country": …}`)
so the search never crosses jurisdictions, and the judge prompt is given that market's
stance. So `"100% eco-friendly"` returns REVIEW under US FTC rules but BLOCK under the EU
Green Claims directive — same ad, different verdict. **Adding a market is dropping in a file:**
the `--country` choices are discovered from the directory (`config.available_countries()`),
no code change. See [`DEMO.md`](DEMO.md).

**One clause = one chunk, no splitting.**
Each policy clause is 2–3 self-contained sentences. Mechanical chunking would strip
"…may not be used" away from the expression it governs, producing chunks that retrieve
well but read as nonsense in the prompt.
→ [`data/policies/`](data/policies/)

**Asymmetric embeddings.**
Documents are embedded with `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY`. The
inputs are shaped very differently — a four-word ad slogan against a formal regulatory
sentence — and the asymmetric task types close part of that gap.
→ [`src/llm.py`](src/llm.py)

**Grounding is enforced in code, not only in the prompt.**
The judge is told to cite only the retrieved clauses. It will still occasionally invent
a `R-99`. So every issue is checked against the set of rule IDs that were actually
retrieved, and unmatched ones are dropped. If *all* issues are dropped, the decision is
reset to PASS rather than left as an unsupported BLOCK. In a legal domain, a fabricated
citation is worse than no answer.
→ `parse_review_payload()` in [`src/schema.py`](src/schema.py)

**Ambiguity resolves upward, never downward.**
An unparseable decision becomes REVIEW, not PASS. The prompt says the same. A missed
violation costs far more than an unnecessary review.
→ `Decision.parse()` in [`src/schema.py`](src/schema.py)

**Recall is the primary metric, not accuracy.**
See below.

## 4. Evaluation

25 labelled ad copies in [`data/eval_set.jsonl`](data/eval_set.jsonl), covering
superlatives, quantified performance claims, pricing and discounts, competitor
comparisons, safety and medical claims, prize events, testimonials, and disclaimers.
Roughly a third are compliant, so a "flag everything" model does not score well.

Two error types have very different costs:

| Error | Consequence |
|---|---|
| False positive — compliant ad flagged | a reviewer spends five extra minutes |
| **False negative — non-compliant ad passed** | **regulatory action, brand damage** |

So the headline metric is **recall on the risky class** (REVIEW ∪ BLOCK), with precision
reported alongside as a proxy for reviewer workload.

Retrieval is scored separately as **Recall@k** — did the labelled ground-truth clause
appear in the top-k at all? Without it you cannot tell a correct decision that was
properly grounded from one the model guessed from general knowledge, and you cannot tell
whether a wrong decision was a retrieval failure or a judgement failure.

```
python cli.py eval
```

Prints a decision-quality block, a retrieval-quality block, a 3×3 confusion matrix, and
a list of the incorrect cases (marking which of them were retrieval failures). Full
per-case output is written to `results.json`.

Target thresholds for this prototype: risky recall ≥ 0.85, Recall@5 ≥ 0.80.

## 5. The agent — a panel of three LG-voiced copywriters

The reviewer above only diagnoses. The agent (`src/agent.py`, built on **LangGraph**) turns
that diagnosis into choices: three copywriter personas each rewrite the flagged ad — sharing one
**LG "Life's Good" brand voice** but taking a different creative route — and every proposal is
**re-checked by the same RAG reviewer**. "Compliant" is not claimed by the generator; it is
*proven* by verification.

```
ad copy
   │ [1] diagnose   review() → what's non-compliant
   ▼
   ├─[2] panel   three gurus, one LG voice, three routes (English / UK)
   │      Iris · Human Relief        "Less energy anxiety. More everyday comfort."
   │      Theo · Quiet Intelligence  "It senses the room before you do."
   │      Nora · Life's Good Moment  "Cool summer nights, with less getting in the way."
   ▼
   │ [3] verify   re-review each proposal; one bounded repair if it still fails
   ▼
[4] report   three recommendations, each with its compliance badge + homework
```

**Why a panel of routes, not one rewrite:** a marketer wants options, not "the fix". The three
routes come straight from the LG brand playbook (Human Relief / Quiet Intelligence / Life's Good
Moment), so each option is a genuinely different creative angle that still feels like LG.

**Brand voice and compliance push the same way.** The LG voice (`LG_VOICE` in `src/agent.py`,
distilled from an LG marketing style guide) already forbids hype and unsupported claims — the
same thing the compliance reviewer enforces. So making copy *more LG* and making it *compliant*
are aligned, not in tension.

**The safety rule still holds: gurus may not invent facts.** The edit schema allows only
`DELETE / SOFTEN / PLACEHOLDER / REFRAME` — no "add a fact" option — and each proposal is verified
by the real reviewer, so the compliance guarantee comes from the RAG judge, not from trusting the
rewrite.

```powershell
# English / UK copy — use EU as the closest proxy in the rule set (UK ASA/CAP ≈ EU rules)
python cli.py agent --country EU "World's No.1 air conditioner, cuts your energy bill by 50%!"
python cli.py agent "대한민국 No.1 에어컨, 전기료 50% 절감!"
#   → diagnosis, then 3 distinct LG-voiced compliant rewrites, each re-verified (see DEMO.md)
```

`tests/test_agent.py` stubs the reviewer and the LLM so the panel, the verify-and-repair step,
and the PASS-first sorting all run with no API key.

## 6. Running it

```powershell
pip install -r requirements.txt

Copy-Item .env.example .env      # then put your key in GEMINI_API_KEY
                                 # https://aistudio.google.com/apikey

python cli.py index              # embeds all policies → "Indexed 42 policies (EU 7, KR 28, US 7)"

python cli.py review "대한민국 No.1 에어컨, 전기료 50% 절감!"
python cli.py review --json "전 품목 30% 할인!"

python cli.py agent "업계 최고의 공기청정기, 깨끗한 공기를 만나보세요!"   # review→rewrite loop

# multi-country — same ad, different verdict (see DEMO.md)
python cli.py review --country US "100% eco-friendly air conditioner!"   # → REVIEW
python cli.py review --country EU "100% eco-friendly air conditioner!"   # → BLOCK

python cli.py eval               # full evaluation run
python cli.py eval --limit 5     # quick check
python cli.py eval --top-k 3     # retrieval-depth experiment

python -m pytest tests -q        # 16 tests, no API key needed
```

On Windows, if Korean output renders as `?`, set `$env:PYTHONIOENCODING="utf-8"` first.

Re-run `python cli.py index` whenever any file in `data/policies/` changes. It uses `upsert`
keyed by rule ID, so re-running never duplicates rows.

## 7. Layout

```
cli.py                  single entry point: index | review | agent | eval
data/policies/          one file per market (filename = country): KR/US/EU .jsonl
data/eval_set.jsonl     25 labelled ad copies with ground-truth decisions
src/config.py           tunable parameters in one place
src/schema.py           dataclasses + LLM-output validation and grounding check
src/llm.py              thin Gemini wrapper: embed_texts / generate_json
src/retriever.py        [2] ChromaDB indexing and search
src/pipeline.py         [1] claim extraction + [3] judgement — both prompts live here
src/agent.py            LangGraph panel: diagnose → 3 gurus → verify → report
src/evaluate.py         [4] metrics
tests/test_rag.py       guardrail and retrieval tests (stubbed LLM, no API key)
tests/test_agent.py     agent loop-termination tests (stubbed LLM, no API key)
```

`PolicyRetriever` takes its embedding function, its policy list, and its storage path by
injection. That is what lets the tests run a real Chroma collection in a temp directory
against a deterministic bag-of-words stub, with no network calls.

## 8. Roadmap

- **Vision input** — accept the image itself; a vision model extracts copy, price,
  disclaimer presence, logos, and on-image text, replacing the text-only entry point.
- **Precedent retrieval** — the original idea, in its proper place: a multimodal image
  index of previously reviewed creatives, retrieved as supporting precedent shown next to
  the decision, never as the evidence for it.
- **Reviewer feedback loop** — capture reviewer overrides as new labelled cases and grow
  the evaluation set from real disagreements.
- **Cost and latency tracking** — tokens and seconds per review, per stage.
