# Golden-set evaluation — consolidated findings

Consolidated from the golden-set evaluation project (`.superpowers/sdd/EVAL_PLAN/`,
gitignored — this file is the durable, committed record). Every claim below
traces to a measurement reproducible with:

```bash
cd chat
python scripts/run_eval.py --verbose --json
pytest tests/ -q --durations=10
```

**Interpreter:** `~/miniconda3/python.exe`. **Run from `chat/`** —
`config.py`'s `env_file=".env"` resolves relative to the working directory,
so a run from the repo root silently drops `RAG_MODEL_PRESET=e5` and every
number becomes meaningless.

**This harness is deterministic.** The same `data/index.json` +
`data/gate_vectors.json` produce bit-identical numbers on repeat runs —
verified three consecutive runs during Task 14. An earlier progress note
speculated that some metric movement was "embedding/gate-value noise"; that
was wrong. The actual cause was Finding J (below): a test fixture was
overwriting the real gate calibration between runs. There is no noise in
this harness — any apparent drift is a corrupted artifact, not measurement
variance.

Measured against: index built `2026-08-01T00:48:45Z`, e5 preset, 230 chunks
(146 en + 84 zh), `golden.jsonl` at 120 cases (96 positives + 24 shared
negatives), `data/eval_baseline.json` generated `2026-08-01T02:58:03Z`.

---

## Task 24 update (2026-08-01): full hit@4 rose, page-only hit@4 did not

**Read this before trusting the `66/96` hit@4 number in the section below —
a newer build supersedes it, and the honest story has two numbers, not one.**

Task 24 pointed the en off-topic gate at the curated
`chat/knowledge/about_en.md` corpus (55 sections) instead of all indexed
English chunks, and rebuilt. That rebuild was the first since Tasks 22/23
grew `about_en.md`/`about_zh.md` from 5/22 to 55/53 sections each — so it
also, incidentally, folded 108 new curated chunks (55 en + 53 zh) into the
retrieval index for the first time. **The reported hit@4 gain (66/96 →
90/96) is mostly that fold-in, not anything task 24's own code change
touched** (that change only reshapes the gate-corpus branch of
`index_builder.py`; retrieval/chunk construction is untouched).

Because several curated sections carry an explicit `link:` line pointing at
a real page (e.g. a Gyrotris-focused `about_en.md` section links
`pages/gyrotris.html`), a curated chunk can satisfy `expected_urls` for a
golden case **without the real page's own text ever entering the top-4** —
the corpus, authored in part by reading those same pages, answers on the
page's behalf. This is scoring an answer key partly against itself, not a
genuine retrieval improvement on the site's own content.

**Both numbers, measured on the same build, reconcile plainly:**

```
hit@4 FULL   (as shipped, all 308 chunks):        90/96
hit@4 PAGE-ONLY (chat/knowledge/*.md excluded):    59/96
pre-task-24 baseline (66/96, itself already
  assisted by the smaller 5+19-section corpus):    66/96
```

`combat_design_recruiter/zh` — the one cell flagged in Finding F below as
never moving across multiple builds — is the clearest symptom: full hit@4
went 4/12 → 12/12, but page-only hit@4 for that same cell is **5/12**,
essentially the same as the pre-rebuild 4/12. 45 of its 48 full-mode top-4
slots trace to knowledge chunks, not page chunks.

**A second, independent check clarifies what actually changed and what
didn't:** re-deriving the OLD build's OWN page-only number (excluding its
own, smaller 26-chunk knowledge assist) gives **59/96 — identical, per cell,
to the new build's page-only number.** The real site pages' own
retrievability did not change at all between builds (same HTML, same
chunking, same embedding model, deterministic per the note above) — what
changed is only how large a lift the knowledge-corpus assist provides
(26 chunks → 66/96 full; 108 chunks → 90/96 full). The true, zero-assist
page-retrieval baseline has been sitting at 59/96 all along; it was simply
never reported as a separate number before this build.

**Fix, not just a caveat:** `scripts/run_eval.py`'s positive-cells table (and
`--json` output, and `data/eval_baseline.json`) now carries a `hit@4(pg)` /
`hit_at_4_page_only` column alongside `hit@4`, computed by excluding
`Runtime.knowledge_chunk_ids` from the retrieval candidate pool before
top-k is taken (`Runtime.retrieve(..., exclude_ids=...)`), so this gap is
visible on every future run without needing a manual re-derivation. It is a
reported diagnostic, not a pass/fail metric — deliberately not compared by
`tests/test_golden.py::test_no_metric_regressed`, the same treatment already
given to `gate_margin` (see `build_baseline`'s docstring).

**What this does and does not mean:** the *product* genuinely improved — a
visitor asking these 96 questions now gets a better answer more often,
because the curated sections are real, grounded, and freely blend with page
content in front of the LLM. But the specific claim "site-page retrieval
improved" is false; site-page retrieval is unchanged, and the growing
knowledge corpus increasingly does the retrieving on the pages' behalf. Full
detail, the code, and the independent re-verification:
`.superpowers/sdd/EVAL_PLAN/task-24-report.md`.

---

## Current measured state (pre-task-24; numbers below are superseded, see above)

```
positive cells                gate   hit@4  keywords  retrieved
ai_agent_recruiter/en        12/12    7/12     12/24  en:48
ai_agent_recruiter/zh          n/a   10/12     16/24  zh:48
client_dev_recruiter/en      11/12    9/12     15/24  en:47 zh:1
client_dev_recruiter/zh        n/a   10/12     16/25  zh:48
combat_design_recruiter/en   12/12    8/12     10/20  en:47 zh:1
combat_design_recruiter/zh     n/a    4/12      5/15  zh:48
visitor/en                   12/12    8/12     11/24  en:46 zh:2
visitor/zh                     n/a   10/12     11/23  zh:48
TOTAL                       47/48*   66/96    96/179

shared negatives   off_topic/easy   off_topic/adjacent   injection
en                          4/4                0/4            1/4
zh                          n/a                n/a            n/a
```

`*` = gate total excludes the 4 `zh` cells with no gate available.

**Read the `n/a` rows as "unmeasured," never as "0%" or "the gate failed."**
There is no Chinese gate on this machine because the 2026-08-01 build's zh
calibration did not separate (off-topic max 0.517 vs on-topic min 0.515,
margin -0.4%), so `build_index` correctly declined to ship one. All 48 zh
cases (12 positive per cell x 4 + 12 negatives) take the name-blind
`cjk_bypass` path and reach the LLM behind only the system prompt's
instruction to refuse off-topic requests — not measured by this harness at
all. This is the degradation contract working as designed, not a defect.

Test suite: **75 passed, 3 failed, 1 skipped** in 6.54s
(`pytest tests/ -q --durations=10`, verified 2026-07-31). The 3 failures are
pre-existing MiniLM assumptions that `RAG_MODEL_PRESET=e5` — now the
documented required configuration — invalidates; they are owned by Task 17,
not this task:

- `test_embedder.py::test_semantic_neighbors_beat_strangers` — asserts a
  cosine gap `> 0.15`; e5 gives 0.134 (0.8821 vs 0.7481). This is the
  documented e5 signature (compressed cosines), not a regression.
- `test_embedder.py::test_documents_and_query_share_one_code_path` — asserts
  `embed_documents == embed_query` exactly; only true for a preset with empty
  prefixes. e5 uses `"query: "`/`"passage: "`.
- `test_index_builder.py::test_builds_schema_with_deterministic_ids_and_vectors`
  — asserts `index["model_preset"] == "minilm"`; settings now say `e5`.

No test-artifact corruption occurred during this measurement:
`git status --short chat/data/` is clean after the full suite run, and
`tests/conftest.py`'s content-hash guard (Task 18) did not fire.

---

## What to do first

Ordered by leverage, cheapest/highest-value first:

1. **Rule on the en gate threshold (Finding M).** Lowering it from 0.2536 to
   ~0.235 recovers the one live false refusal at zero cost — adjacent
   leakage is already 0/4 and cannot get worse. This is a one-line config
   change with measured, bounded impact. **Not implemented here — this
   report recommends it; the user rules on it.**
2. **Finish Task 17** (`RAG_MODEL_PRESET` build guard + pin the 3 MiniLM-
   assumption tests). Turns "works if you remember to `cd chat`" into "fails
   loudly if misconfigured." Also clears the only 3 failing tests.
3. **Task 19** (strip `<textarea>`, rebuild, remeasure). Finding C is a
   one-line fix (`loader._BOILERPLATE_TAGS`) with a concrete, already-sighted
   payoff: at least 2 of 96 positives currently lose their top rank to junk
   content.
4. **Task 21** (curated question-shaped gate corpus, both languages). This is
   the real fix for Findings D/E/F — a bigger effort, correctly sequenced
   last so it isn't built on top of a corpus still being polluted by Finding
   C.
5. **Widen `about_zh.md` toward combat-design phrasings** (Finding F) as part
   of the Task 21 rewrite — `combat_design_recruiter/zh` is the one cell that
   has not moved across two index builds.

---

## Fixed

### A — Index staleness
`data/index.json` predated the `10be374` content rewrite of `chat-agent.html`
and `chat/knowledge/`. Proven by wording drift between the live page and the
indexed chunk text ("ingestion **and indexing** pipeline" / "**prebuilt**
vector index" live vs "ingestion pipeline" / "**static** vector index"
indexed). **Fixed** by the 2026-08-01 rebuild (commit `92b23f7`).

### J — A test fixture was destroying the real gate artifacts
`tests/test_index_builder.py`'s `tiny_site` fixture monkeypatched
`settings.index_path`/`settings.roles_path` to a tmp dir but not
`settings.gate_vectors_path`/`settings.fallback_vectors_path`. Running the
full suite silently overwrote the real `data/gate_vectors.json` (146
vectors, threshold 0.2536) with a 5-chunk toy calibration
(threshold 0.0608). `gate_vectors.json` is gitignored, so this was
unrecoverable by git — every subsequent `run_eval.py` invocation in that
session would have measured against a meaningless gate. Latent for the
whole project; only armed once `RAG_MODEL_PRESET=e5` activated the e5
gate-writing branch. **Fixed** in Task 18 (commit `7973227`, merged
`1cd52e6`): the two missing paths are now monkeypatched, and
`tests/conftest.py` adds a session-scoped guard that content-hashes
`index.json`/`gate_vectors.json`/`fallback_vectors.json`/`eval_baseline.json`
before and after the run and fails loudly naming any file a test mutated.
The guard was verified firing by deliberately reverting the fix.

### L — A one-sided `gate_available` check would have produced false-positive regressions
The regression test originally skipped gate-derived metric comparison based
on the *baseline's* `gate_available` flag only. A bypassed gate always
reports `refusal_easy`/`refusal_adjacent`/`refusal_injection` as 0 (not
`n/a`) at the raw-count level — the `n/a` presentation is a display-layer
decision, not a stored value — so a baseline generated *with* a zh gate,
compared against a later run *without* one, would report a regression that
never happened. **Fixed** in Task 14 (commit `9e73a03`, merged `ac09b80`):
gate-derived metrics are now skipped when *either* side lacks a gate;
`hit_at_4`/`keyword_coverage` are unaffected (retrieval is gate-free) and
stay strictly compared in all four availability combinations.

---

## Outstanding

### N — Curated knowledge sections can satisfy `expected_urls` without the real page ever retrieving (leads this section; see the Task 24 update above)

`chat/knowledge/about_en.md`/`about_zh.md` sections may carry a `link:` line
pointing at a real page (`loader.load_knowledge`); when one does, a chunk
built entirely from curated prose can match a golden case's `expected_urls`
even though the actual page's own text never lands in the top-4. As the two
corpora grew (5→55 en, 22→53 zh sections, Tasks 22/23) this stopped being a
rounding error: 108 of 308 indexed chunks are now knowledge-derived, and the
task-24 rebuild's full hit@4 (90/96) is inflated by this relative to the
page-only number (59/96) — see the Task 24 update above for the full
measurement and reconciliation.

**Not classified as a defect to "fix" by removing `link:` or shrinking the
corpus** — the curated sections are grounded, real, and genuinely help the
LLM answer well; the issue is purely that `hit@4` alone can no longer be
read as "the site's own pages are retrievable," and a reader must check
`hit@4(pg)` (below) for that claim specifically. **Mitigated**, not fixed:
`scripts/run_eval.py`, `--json`, and `data/eval_baseline.json` now carry a
`hit_at_4_page_only` diagnostic (`Runtime.knowledge_chunk_ids` /
`Runtime.retrieve(..., exclude_ids=...)`) reporting hit@4 with knowledge
chunks excluded from the candidate pool, printed as `hit@4(pg)` next to the
normal column. It is not wired into `test_no_metric_regressed` (a reported
diagnostic, not a pass/fail gate, mirroring how `gate_margin` is already
treated) — a future task should judge whether page-only retrieval quality
itself needs work, now that it is visible rather than blended away.

### M — The adjacency axis quantifies the gate's headline failure mode

Measured en gate values against the live threshold 0.2536 (reviewer-verified
against real score buckets that sit well clear of the threshold on both
sides — not a score-fitted split):

```
easy off_topic      0.115 - 0.199   -> 4/4 refused   correct
on-topic MINIMUM    0.240           -> BELOW threshold -> false refusal
adjacent off_topic  0.313 - 0.402   -> 0/4 refused    leak
injection                           -> 1/4 refused (one leaked at 0.6058)
```

Confirmed against this session's own run: `neg-en-01..04` (easy) refuse at
0.1147-0.1991; `neg-en-05..08` (adjacent) all pass at 0.3130-0.4020;
`neg-en-09..11` (injection) pass at 0.3453/0.6058/0.3589, `neg-en-12`
refuses at 0.2366. `client-en-12` — a genuine positive
("Has he ever shipped something completely on his own...") — is refused at
0.2323, the concrete case behind `client_dev_recruiter/en` reading 11/12.

**The ordering is easy < on-topic < adjacent.** Adjacent probes score
*higher* than genuine on-topic questions, so no single threshold value can
admit on-topic questions while rejecting adjacent ones — this is a
separation problem, not a threshold-placement problem (this generalizes
Finding E, which established the same shape from ad-hoc probing before the
`adjacency` field existed to measure it directly).

**Consequence:** lowering the en threshold from 0.2536 to ~0.235 recovers
`client-en-12`'s false refusal at **zero cost** to adjacent leakage, which is
already 0/4 and cannot get worse. 0.2536 is strictly dominated by ~0.235 on
the evidence in this dataset. This is a recommendation for the user to rule
on — not implemented here, and no threshold was changed in this task.

### E — Neither gate separates "about the domain" from "about YC" (subsumed/quantified by M)
Originally measured via ad-hoc probing (Task 9, prior index): 16 easy
off-domain probes refused 13/16 (81%), but 4 generic domain-adjacent probes
("what engine should I use for my indie game," "tips for designing a boss
fight") refused 0/4, and a nonsense probe (`zzzz qqqq xxxx nonsense`) passed
at 0.3411 against a 0.2312 threshold. Finding M is this same shape, now
measured systematically through the golden set's `adjacency` field instead
of one-off probes.

### B / Task 17 — `RAG_MODEL_PRESET` footgun (partly fixed)
`chat/.env` sets `RAG_MODEL_PRESET="e5"` and `.env.example` now carries the
same line (verified present, this session) — a fresh clone will get the
correct default. **Still outstanding, owned by Task 17:** no build-time
guard exists yet that raises when `settings.model_preset` disagrees with an
existing `index.json`'s recorded `model_preset`. Without it, running
`python scripts/build_index.py` from the wrong working directory (silently
falling back to the `minilm` default) would rebuild the whole index in a
different embedding space and desync `gate_vectors.json`, `fallback_vectors.json`,
and the deployed Tencent package — with no error, just wrong numbers
downstream. Also owned by Task 17 (Finding K): pin the 3 MiniLM-assumption
test failures listed above to an explicit preset instead of the ambient
environment.

Also worth noting, not a defect: `RAG_MODEL_PRESET` selects only the
*retrieval* model. The gate models (`gate_model: "minilm"`,
`gate_model_zh: "bge_zh"`) are declared inside the `e5` preset itself — so
setting `e5` is what *enables* the MiniLM English gate and the bge-zh
Chinese gate; setting `minilm` disables both and writes no
`gate_vectors.json` at all.

### C — `<textarea>` demo text is indexed as real content (Task 19 fixes)
`loader._BOILERPLATE_TAGS` strips `script, style, nav, header, footer,
noscript, canvas` but not `textarea`. The Word Cloud Generator page's
`<textarea>` placeholder text ("Nothing Can Go Wrong is a...", train tracks,
whirlpools, spring pads) is indexed as page content. It pollutes 4 chunks in
both languages; `pages/toolbox.html#sec0:zh:0` opens
"词云 二维码生成器 … Nothing Can Go Wrong is a…".

Verified live, this session, against the current e5 index: `toolbox.html`
takes **rank 1**, displacing the expected page, in at least 2 of the 96
positive cases:

| case | wants | toolbox.html score |
|---|---|---|
| `combat-en-06` ("How does he validate that a system actually plays well?") | `pages/game-design-workshop.html` | 0.8424 |
| `client-en-05` ("Has he shipped anything that actually runs live in an engine...") | `pages/3d-rendering.html` | 0.8455 |

Both are also `genuine_miss` cases in the right-page/wrong-chunk catalog
below, i.e. Finding C is a direct, demonstrated contributor to two of this
run's 28 pure retrieval misses. The ledger records earlier sightings on a
prior index build (4 combat-en cases in Task 9, `client-en-05` again in
Task 11) — the count fluctuates across rebuilds because embeddings change,
but the mechanism (junk chunk with a high-magnitude, topic-incoherent
vector) has been live across at least three separate index builds and is
not fixed. Fix: add `"textarea"` to `_BOILERPLATE_TAGS`, rebuild (Task 19).

### D — The 40-character floor silently discards Chinese gate corpus
`loader.py:131` (`load_knowledge`) drops any knowledge section whose body is
under 40 characters. The floor is language-blind, but Chinese encodes the
same content in far fewer characters, so it disproportionately discards
short Chinese sections. Verified directly against the current, already-
rewritten `chat/knowledge/about_zh.md` this session (reproducing
`load_knowledge`'s exact split/strip logic): **22 authored `##` sections,
19 survive the floor, 3 are silently dropped:**

- `王元辰是谁` ("who is YC") — 26 characters — an identity-question shape
- `自动微分工具项目` — 38 characters
- `科研背景` — 32 characters

These question shapes never enter the zh gate's decision boundary purely
because they were phrased concisely. Directly implicated in the Chinese
gate's calibration failures across multiple corpus versions (Finding M's
mirror on the zh side). Not fixed — recommended fix is a language-aware
floor (e.g. count CJK characters differently) or lengthening these three
sections, folded into the Task 21 corpus rewrite.

### F — Per-role retrieval gaps are real and diagnosable
`combat_design_recruiter/zh` is the weakest cell at hit@4 **4/12**, and it
has not moved across two index builds (7/12 pre-rebuild with a live zh gate,
4/12 post-rebuild with retrieval-only scoring — the underlying retrieval
number is flat). Comparing cases that hit the *same page*: `client-zh-06/07`
retrieve `cemented-dreams.html` at rank 1, while `combat-zh-02/03/04` never
retrieve it at all despite comparably good paraphrasing. `about_zh.md`
covers engine-programming question shapes well and combat-design ones
poorly — this is a corpus-coverage gap, not a global gate or retrieval
defect. Actionable: widen `about_zh.md` toward combat-design phrasings
(folded into Task 21).

### G — `visitor` is the hardest cell for retrieval, and it is the default role
hit@4 8/12 (en) and 10/12 (zh) on broad-bio and point-me-somewhere questions
— lower than the specialist roles' typical range. `visitor` is
`default_role`, so this is the retrieval quality most actual site visitors
experience. The `<meta name="description">` summary chunks and
`knowledge/about_*.md` exist precisely to serve this traffic, which is the
strongest argument for prioritizing the Task 21 corpus widening over
role-specific tuning.

### H — Fit-on-data proximity inflates scores, measured
Task 13 review caught `visitor-en-01`/`visitor-zh-01` paraphrasing
`gate_calibration.ON_TOPIC`/`ON_TOPIC_ZH` strings. Rewording
`visitor-en-01` to stop reusing `ON_TOPIC`'s "who is Yuanchen Wang?" shape
dropped its measured gate value from **0.7508 to 0.5547** — a drop of
**~0.20**, entirely attributable to no longer resembling fit-on data. This
is the concrete, measured cost of the disjointness authoring rule
(`eval/README.md` rule 1) existing at all: without it, this class of case
would silently overstate the gate's real permissiveness.

### Right-page/wrong-chunk: full catalog, this run

Every positive case, classified by the `hit@4` x `keywords` quadrant defined
in `eval/README.md`, verified this session by parsing `run_eval.py
--verbose` output directly (not estimated):

| outcome | count | reading |
|---|---:|---|
| healthy (hit + all keywords) | 42 | working as intended |
| **right page, wrong chunk** (hit, keyword(s) missing) | **24** | chunking problem |
| genuine miss (no hit, keyword(s) missing) | 28 | retrieval problem |
| gate false refusal (positive refused before retrieval ran) | 1 | `client-en-12`, see Finding M |
| fact-lives-elsewhere (no hit, all keywords found anyway) | 1 | `aiagent-en-12`, see below |

`42 + 24 = 66` reconciles exactly with the reported `hit@4 66/96`.

The 24 right-page/wrong-chunk cases (`hit@4` passes, keyword coverage does
not — the fact was on the retrieved page but not in the specific chunk that
made the cut):

```
combat-en-01   combat-en-05   combat-en-11   combat-zh-05
client-en-04   client-en-07   client-zh-03   client-zh-05
client-zh-07   client-zh-08   client-zh-09   aiagent-en-03
aiagent-en-04  aiagent-en-05  aiagent-zh-09  aiagent-zh-10
visitor-en-02  visitor-en-07  visitor-en-12  visitor-zh-01
visitor-zh-03  visitor-zh-09  visitor-zh-11  visitor-zh-12
```

25% of the entire positive set (24/96) lands on the right page but the wrong
chunk. This is chunking granularity/boundary placement, not a retrieval
failure — `expected_urls` is satisfied but `expected_keywords` is not. No
single case in this list was edited to improve its score (authoring rule 5,
`eval/README.md`); each is a live work-queue item.

`aiagent-en-12` ("Which of his degrees actually included statistics-heavy
modeling coursework...") is the one case where the *opposite* asymmetry
shows up: `education.html` (the expected page) never lands in the top 4, yet
both expected keywords ("Applied Bayesian Analysis," "Machine Learning")
were found anyway — because the same facts also appear in a page that did
retrieve. Per `eval/README.md`'s reading table, this means "the fact lives
somewhere unexpected; check `expected_urls`" rather than a retrieval defect.

---

## Deferred by user decision

### I — `test_index_carries_a_calibrated_gate` asserts on dead fields
`tests/test_gate.py::test_index_carries_a_calibrated_gate` checks
`index.json`'s `gate_stat`/`gate_threshold` fields, which nothing reads for
an e5 index: `runtime.py` builds gates from
`gate_vectors.json`/`fallback_vectors.json`, and the deployed Tencent
function (`functions/tencent/index.py`) reads its own bundled copy. The
fields in `index.json` are an advisory duplicate with no live consumer.
Confirmed still present in the current test suite (line 47 of
`tests/test_gate.py`). Explicitly deferred — the user declined to fix it
inside this plan; it should be pointed at by the final whole-branch review.

---

## Traceability note

Findings A, J, L above are stated as fixed based on the commit history and
the specific test/guard behavior described; this session independently
re-ran the full suite and confirmed `chat/data/` is untouched afterward,
which is consistent with (though does not by itself re-derive) J and L
being fixed. Findings D, C (the two toolbox rank-1 sightings), M's exact
score values, the 66/96 quadrant breakdown, and the 24-case catalog above
were independently recomputed against the live `data/` artifacts in this
session, not copied from the ledger. Everything else (A's wording-drift
proof, H's before/after values, F's cross-cell page comparison, the
historical toolbox sighting count from Tasks 9/11) is carried forward from
`.superpowers/sdd/EVAL_PLAN/progress.md` and was not independently
re-derived here — flagged so a reader knows which numbers this session
verified firsthand versus which it is relaying.
