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

Measured against: index built `2026-08-01T20:45:15Z`, e5 preset, 308 chunks
(192 en + 116 zh), `golden.jsonl` at 120 cases (96 positives + 24 shared
negatives), `data/eval_baseline.json` generated `2026-08-01T20:46:05Z`. This
is the Task 24 build described in the section immediately below — the
230-chunk/146+84 figures this line used to carry were the PRE-task-24 build
and are superseded throughout this document exactly as flagged inline below.

**Task 29 Part 2 note (filenames only, added without touching history below):**
`data/index.json` → `data/chunks_e5.json` (or `chunks_{model_preset}.json`
generally), `data/gate_vectors.json` → `data/gate_en_minilm.json` (committed)
+ `data/gate_zh_bge.json` (gitignored), `data/fallback_vectors.json` →
`data/gate_en_minilm.json` + a new `data/chunks_en_minilm.json` (degraded-mode
retrieval corpus — see `../README.md`'s file-layout table). Every mention of
the three retired names in the findings below is a HISTORICAL fact about the
build/artifact state at the time that finding was written and is deliberately
left as-is, not retroactively renamed — rewriting history here would make the
"measured against" timestamps and git-commit references above no longer
correspond to what the surrounding prose describes. Read old filenames in
this document as their Task-29-Part-2 successors; the numbers/margins/counts
next to them are unaffected (this was a pure file split + a bundled-package
naming/shape change, not a re-embedding — see
`.superpowers/sdd/EVAL_PLAN/task-29b-report.md` for confirmation the eval
numbers below did not move).

---

## CRITICAL — everything this file measures is the LOCAL artifact, not what's deployed

**Status: open. Documented here and in `../README.md`; not fixed by this
branch, because fixing it means redeploying, which this branch deliberately
does not do.**

Every gate number in this document — English threshold `0.2006`, margin
`+2.5%`, `48/48*` gate-pass, `easy 4/4`/`adjacent 1/4`/`injection 1/4`, "no
Chinese gate on this machine" — describes `chat/data/gate_vectors.json` on
disk in this repo. It does **not** describe the backend visitors actually
hit.

**Superseded (final whole-branch review) — the local-state figures in the
paragraph above, not the local-vs-deployed point it makes.** The
local-vs-deployed gap is still real and still open. But the enumerated
figures predate the Task 29 Part 2 file split and the zh-gate-deletion fix,
and this was the one stale block in this file carrying no such note. Current
local artifacts: the gate lives in `data/gate_en_minilm.json` (English) and
`data/gate_zh_bge.json` (Chinese), not `gate_vectors.json`; the English
threshold/margin are unchanged at `0.2006` / `+2.5%`; **there IS a separating
Chinese gate on this machine** (`top`, threshold `0.3979`, margin `+5.9%`, 53
sections), so `run_eval.py` reports **gate 96/96**, not `48/48*` with four
`n/a` cells. Shared negatives now read en `4/4 · 1/4 · 1/4`, zh
`3/4 · 0/4 · 0/4`. See `eval/README.md`'s Known Limits, which was updated for
this and is the authority. `chat/functions/tencent/index.py` never reads
`chat/data/gate_vectors.json`; it reads its own copy, bundled at
`functions/tencent/tencent-function-e5.zip` by
`functions/tencent/build_package.py`, and that zip is untouched by this
branch (confirmed: `git diff --stat main...feat/golden-eval -- chat/functions/`
is empty).

Opened programmatically (read-only, zip left untouched) as of this writing,
the deployed zip carries:

```
build_info.json:  build_id e5-20260721-3de2a80, built_at 2026-07-21T22:35:16Z
gate_vectors.json:
  en: 149 vectors (raw page chunks, NOT about_en.md's curated corpus),
      threshold 0.2312, model_preset minilm
  zh: 19 vectors, threshold 0.4919, model_preset bge_zh   <- LIVE IN PRODUCTION
```

Consequences:
- Production's English gate still scores against the pre-Task-24 corpus
  (149 raw page chunks) at threshold `0.2312`, not the curated 55-section
  `about_en.md` corpus at `0.2006`/+2.5% this document reports.
- **Production runs a live Chinese gate at threshold `0.4919`.** This is not
  a theoretical risk: `chat-widget.js` sets `state.index.gate_remote = true`
  for an e5 index and always defers to the server's `/embed` gate decision
  when reachable, so this threshold is what Chinese-speaking visitors hit
  today. A prior measurement of essentially this same threshold (on-topic
  floor 0.492, threshold 0.4919 — a near-zero margin) found held-out Chinese
  positives failing it at a real rate (see `eval/README.md`'s Known Limits).
  This document's own `n/a` for every Chinese gate metric reflects **this
  machine having no `"zh"` key in `data/gate_vectors.json`** — it says
  nothing about production, which has never been gate-free for Chinese.
- Retrieval itself is fine: the deployed zip's embedding model (e5) matches
  the current index's embedding space, so answers are grounded correctly;
  only the gate is stale.

**Compatibility check (so whoever redeploys knows it's safe): verified this
session, read-only.** `functions/tencent/index.py`'s `_load_embedder()`
(lines ~132-156) reads exactly `spec["vectors"]`, `spec["gate_stat"]`,
`spec["gate_threshold"]`, `spec.get("query_prefix", "")`,
`spec.get("pooling", "mean")` from each language's entry in
`gate_vectors.json`; it never reads `chunk_ids` (present in the OLD deployed
`en` entry, absent from the new one — irrelevant either way) and silently
ignores the new `gate_margin` key (present in the new entry, absent from the
old one — also harmless, nothing reads it). `gate_decision()` (lines
~181-210) does `gate = _gates["zh"]; if gate is None: return
{"pass": True, ..., "reason": "cjk_bypass"}` — and the current
`chat/data/gate_vectors.json` has no `"zh"` key at all (confirmed
programmatically: `json.load(...).keys() == ['en']`), so `payload.get("zh")`
in `_load_embedder()` returns `None` and `_gates["zh"]` stays `None`,
correctly falling through to `cjk_bypass` rather than erroring. **A repackage
(`python functions/tencent/build_package.py`) and redeploy is therefore a
pure win with no code change required** — it would bring production's
English gate to `0.2006`/+2.5% and remove the live, false-refusing Chinese
gate, replacing it with the same `cjk_bypass` this document already reports
as the local state.

**Not done here, on purpose:** this task's instructions were to document,
not to repackage or redeploy. The action item — rebuild the SCF zip, verify
`build_info.json`'s new `build_id`, redeploy, confirm `/` health check
reports it — is the actual next step and belongs to whoever owns production
access; see `../README.md`'s "Deploying a backend" section.

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

**Follow-up (final whole-branch review):** `knowledge_chunk_ids` only stays
correct against the exact build its chunks came from — the chunks are frozen
at build time, but the heading set they're matched against is re-read from
`about_<lang>.md` on every call. Renaming a heading with no rebuild desyncs
the two and silently moves `hit_at_4_page_only` (demonstrated: renaming 5
`about_en.md` headings dropped it from 59/96 to 56/96 with no rebuild).
`Runtime.stale_knowledge_headings` now catches this — it reports current
`about_<lang>.md` headings with no matching chunk in the index — and
`scripts/run_eval.py` prints a note when it's non-empty. It is, like
`hit_at_4_page_only` itself, a reported diagnostic, not wired into
`test_no_metric_regressed`.

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

**Superseded (final whole-branch review, second pass):** the 75/3/1 count
above and the 3 named failures are historical, from 2026-07-31, before Task
17 landed the `RAG_MODEL_PRESET` build guard and pinned the 3
MiniLM-assumption tests to an explicit preset. Current test suite, run the
same way, from `chat/`: **194 passed, 0 skipped**. (An earlier revision of
this note said "105 passed, 1 skipped" and was never refreshed through Tasks
26-34 or the final review's fix wave. The former single skip no longer fires
now that a Chinese gate is locally available.) Verified this session;
`git status --short chat/data/` stayed clean and the conftest guard did not
fire.

---

## What to do first

Everything this section originally recommended is done — see "Fixed" below
for what actually happened to each (in several cases, not the specific fix
recommended here, but a broader change that achieved the same goal or more).
What's genuinely still open, ordered by leverage:

1. **Redeploy the production backend (see the CRITICAL section above).** The
   single highest-leverage action item left: none of this branch's gate work
   (English threshold `0.2006`, the curated-corpus gate, the CJK-weighted
   floor) reaches visitors until `functions/tencent/build_package.py` is
   re-run and the result redeployed. Verified safe (no code change needed) —
   see the compatibility check above. **Not implemented here** — this task's
   scope is documentation, not deployment; ruling on / performing the
   redeploy belongs to whoever owns production access.
2. **Judge whether page-only retrieval quality itself needs work (Finding
   N / the Task 24 update above).** `hit@4` is now 90/96, but `hit@4(pg)` —
   what the site's own pages retrieve with zero curated-corpus assist — is
   59/96 and did not move. `combat_design_recruiter/zh` and `visitor/zh` are
   the weakest page-only cells (5/12 each on the current build) despite
   scoring well (12/12) on the full, assisted metric. This is explicitly
   *not* classified as a defect to fix by shrinking the corpus — the curated
   sections are real and grounded — but a future task should decide whether
   the site's own page content/chunking needs improving for these cells, now
   that the gap is visible instead of blended away.
3. Everything else flagged in the final whole-branch review's Minor section
   and "follow-up" triage rows (conftest guard hardening, `retrieveFallback`'s
   positional mapping, `get_embedder()`'s cache key) is real but not urgent;
   see that review for the full list rather than duplicating it here.

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

### B / Task 17 — `RAG_MODEL_PRESET` footgun — **FIXED**
`chat/.env` sets `RAG_MODEL_PRESET="e5"` and `.env.example` now carries the
same line (verified present, this session) — a fresh clone will get the
correct default. **The build-time guard now exists** (`index_builder.py`,
~lines 119-131): if an existing `data/index.json` records a
`model_preset` that disagrees with `settings.model_preset`, `build_index()`
raises before writing anything, naming both presets and pointing at
`RAG_ALLOW_PRESET_CHANGE=1` as the deliberate opt-out. Verified present and
reads correctly this session (not verified by triggering an actual raise,
which would require running `build_index.py` — prohibited by this task's
constraints — but the guard clause was read directly and its condition and
message confirmed). Finding K (pin the 3 MiniLM-assumption test failures to
an explicit preset) is also done — see the test-suite note above: 105
passed, 1 skipped, none of the 3 named failures remain.

Also worth noting, not a defect: `RAG_MODEL_PRESET` selects only the
*retrieval* model. The gate models (`gate_model: "minilm"`,
`gate_model_zh: "bge_zh"`) are declared inside the `e5` preset itself — so
setting `e5` is what *enables* the MiniLM English gate and the bge-zh
Chinese gate; setting `minilm` disables both and writes no
`gate_vectors.json` at all.

### C — `<textarea>` demo text is indexed as real content — **FIXED (Task 19)**
`"textarea"` is now in `loader._BOILERPLATE_TAGS` (verified this session:
the tuple reads `("script", "style", "nav", "header", "footer", "noscript",
"canvas", "textarea")`), so the Word Cloud Generator page's placeholder text
is stripped before indexing rather than becoming a junk chunk. The
description below is kept for its diagnostic value (how the bug manifested,
which cases it broke) but no longer describes the current index. Original
report:

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

### D — The 40-character floor silently discards Chinese gate corpus — **FIXED (Task 21)**
`loader.py` now weights CJK characters instead of counting raw `len()`
(`_effective_length`, `_CJK_WEIGHT = 2.5`) — verified this session: all 53
currently-authored `about_zh.md` sections clear the floor, including the
three below. `_CJK_WEIGHT` is a chosen heuristic with headroom, not a cited
information-density ratio — see `loader.py`'s comment for the empirical
number (the minimum weight that would admit these three sections is ~1.61).
Original report, kept for the evidence:

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
mirror on the zh side).

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

**Update (Task 33):** `aiagent-en-03`/`aiagent-zh-03` (the degraded-mode
fallback question — "What happens to the chat feature if the backend
embedding service is unreachable?") are a fresh instance of this exact
pattern, found and deliberately left as-is rather than "fixed." Task 33
restored `pages/chat-agent.html`'s exact page-side phrasing ("retrieval
locally" / "本地完成检索") after an earlier rewrite in the same task had
accidentally paraphrased it away — so the *page* is factually corroborated
again, and reads correctly. But in practice both cases still score their
keywords **exclusively** from the curated `about_en.md`/`about_zh.md`
fallback section, because the page's own matching paragraph does not reach
the real top-4 for these specific queries. The textual goal (the page states
the true fact, in the words the golden case looks for) is met; the
retrieval-corroboration goal this finding is about is only partly realized.
Read "restored page-side corroboration" in the Task 33 report as meaning
exactly that and no more — these two cases did not stop scoring off the
corpus, and nobody should expect `hit@4(pg)` to reflect the restoration
without the page's own chunk actually reaching top-4.

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

**Status update (final whole-branch review): the specific false refusal is
fixed; the underlying separation problem this finding diagnoses is still
real, just less severe.** Task 24 did not implement the manual threshold
edit recommended above — it rewired the gate to a curated corpus instead
(`about_en.md`'s 55 sections replacing all raw English chunks), which moved
the threshold to `0.2006` (positive margin, `+2.5%`) as a side effect. All 4
English positive cells now gate `12/12` — every genuine positive passes, so
`client-en-12`'s specific false refusal (or its equivalent — golden.jsonl's
negative cases were also reauthored since, so the exact case wording has
changed) no longer exists. Re-verified this session, read-only, against the
current gate corpus: fit-on-data bounds are off-topic max `0.1934`
("best restaurants nearby") and on-topic min `0.2078` ("resume highlights").
Against golden.jsonl's current English negatives: `easy` still refuses
cleanly (4/4, scores `0.0752`-`0.1517`, well under the on-topic floor —
unchanged in kind), but `adjacent` still mostly leaks (1/4 refused, the other
3 score `0.3393`-`0.3853` — comfortably *above* on-topic min `0.2078`) and
`injection` likewise (1/4 refused, 3 leak at `0.3163`-`0.4457`). **The
ordering easy < on-topic < adjacent still holds** — this finding's core claim
(no single threshold can admit on-topic while rejecting adjacent probes,
because adjacent probes score higher than genuine on-topic ones) is
unresolved and is not something a threshold or corpus change can fix; it
would need the gate to consider more than top-1 similarity against a fixed
corpus. Not a defect introduced by this branch — a pre-existing, now more
precisely measured limitation.

### E — Neither gate separates "about the domain" from "about YC" (subsumed/quantified by M)
Originally measured via ad-hoc probing (Task 9, prior index): 16 easy
off-domain probes refused 13/16 (81%), but 4 generic domain-adjacent probes
("what engine should I use for my indie game," "tips for designing a boss
fight") refused 0/4, and a nonsense probe (`zzzz qqqq xxxx nonsense`) passed
at 0.3411 against a 0.2312 threshold. Finding M is this same shape, now
measured systematically through the golden set's `adjacency` field instead
of one-off probes.

### F — Per-role retrieval gaps are real and diagnosable — **improved at the full metric, gap persists at page-only**
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

**Update (final whole-branch review):** the widening happened (Task 22/23
grew `about_zh.md` to 53 sections) and the full `hit@4` for this cell is now
**12/12** — a complete turnaround from 4/12. But `hit@4(pg)` (page-only, the
Task 24 diagnostic — see above) for the same cell is **5/12**, essentially
unchanged from the pre-widening 4/12. Read together with Finding N: the
widened `about_zh.md` corpus itself now answers most of these questions
directly (via its curated text, which is grounded and real), but the site's
own combat-design pages still aren't what's being retrieved for them. The
underlying retrieval gap this finding describes is not fixed, only masked by
a (legitimate, but distinct) corpus improvement.

### G — `visitor` is the hardest cell for retrieval, and it is the default role — **same pattern as F**
hit@4 8/12 (en) and 10/12 (zh) on broad-bio and point-me-somewhere questions
— lower than the specialist roles' typical range. `visitor` is
`default_role`, so this is the retrieval quality most actual site visitors
experience. The `<meta name="description">` summary chunks and
`knowledge/about_*.md` exist precisely to serve this traffic, which is the
strongest argument for prioritizing the Task 21 corpus widening over
role-specific tuning.

**Update (final whole-branch review):** full hit@4 for `visitor` is now
10/12 (en) and 12/12 (zh) — improved, as intended. `hit@4(pg)`, however, is
7/12 (en) and 5/12 (zh) — still the weakest page-only range across all
8 cells. Same reading as Finding F: the knowledge corpus is doing real work
for the `visitor` role specifically (matching its own framing above, that
`about_*.md` "exist[s] precisely to serve this traffic"), but that is a
description of the corpus compensating for weaker page-level retrieval, not
evidence the page-level gap closed.

### H — Fit-on-data proximity inflates scores, measured
Task 13 review caught `visitor-en-01`/`visitor-zh-01` paraphrasing
`gate_calibration.ON_TOPIC`/`ON_TOPIC_ZH` strings. Rewording
`visitor-en-01` to stop reusing `ON_TOPIC`'s "who is Yuanchen Wang?" shape
dropped its measured gate value from **0.7508 to 0.5547** — a drop of
**~0.20**, entirely attributable to no longer resembling fit-on data. This
is the concrete, measured cost of the disjointness authoring rule
(`eval/README.md` rule 1) existing at all: without it, this class of case
would silently overstate the gate's real permissiveness.

### Q — The zh gate does not reliably refuse generic "do this task for me" requests, and barely refuses well-formed anaphoric follow-ups at all (Task 9)

Found while authoring Task 9's post-context negative golden cases (the ones
that pair an off-topic question with prior on-topic conversation, to check
that a rewrite cannot smuggle topicality into a question that should still be
refused — see `eval/README.md`'s Task 9 section). One candidate,
`帮我翻译一下这段话` ("help me translate this passage") — a plain
general-assistant task request, off-topic by any reading — measured
**0.4209** against the live zh gate threshold **0.3979**: it passes on its
own, with no conversation needed at all. The case was retuned to a phrasing
that does fail (`帮我写封邮件`, "help me write an email," measured 0.3825) so
`neg-post-zh-02` itself is not compromised, but the underlying gap is not
fixed. It is not an isolated surprise: swept the **full** 16-entry
`gate_calibration.OFF_TOPIC_ZH` set against the live zh gate (not just the
two task-request examples first noticed) and found **8 of 16 — 1 in 2 —
pass (leak) this same gate on their own**, not the "roughly 1 in 8" an
earlier draft of this entry estimated from those two alone (correction:
task 9 review caught this undercount by a factor of 4 and it was
re-verified directly):

```
给我讲个笑话 0.4433 · 帮我写作业 0.4338 · 帮我写一段Python代码 0.5169
把这句话翻译成英文 0.4901 · 谁赢了世界杯 0.4297 · 人民币兑美元今天的汇率是多少 0.4181
推荐一部好看的科幻电影 0.4289 · 新手健身应该怎么练 0.4542
```

(the other 8 — weather, poem, restaurants, tallest mountain, cooking,
Spanish "thank you," time zone, boiling an egg — do refuse, 0.3262-0.3853,
all comfortably under the 0.3979 threshold; the zero-false-refusal policy,
see `gate_calibration.py`'s module docstring, only ever promises to catch a
*subset*, and here it catches almost exactly half of its own fit-on
off-topic set).

**Contrast, for scale: the equivalent sweep of the full 16-entry
`gate_calibration.OFF_TOPIC` (English) set against the live en gate (also
threshold-relative, 0.2029) catches all 16 — 0 leak.** This is direct
evidence for this finding's "strictly weaker separation than the English
gate" claim below, not just an assertion: the same zero-false-refusal
calibration policy, run over comparably-sized fit-on sets in each language,
leaves the zh gate's own calibration corpus half-uncaught while the en
gate's is fully caught.

**A second, broader symptom surfaced empirically while retuning Task 9's zh
follow-up positives**, worth recording alongside this one because it's the
same mechanism at larger scale: of roughly 90 candidate short/elliptical
Chinese phrasings tried (elided arguments, dangling pronouns, "那...呢"
topic-shift continuations, generic temporal fillers, task-agnostic
"他/这个 + interrogative" questions), essentially none scored below the
0.3979 threshold on their own — most land 0.42-0.56 regardless of topic,
including phrasings with **no specific referent to any prior turn at all**
(e.g. `那这个是谁做的`, `那这个用了什么工具` — both score >0.5 with zero
context). The only candidates found under threshold were content-free
continuation fillers (`那后来呢` 0.3957, `再后来` 0.3768) with 0.002-0.021
margins — too thin to be robust given this gate's documented history of
threshold instability across rebuilds (`eval/README.md`'s Known Limits:
0.3979 today, 0.4919 in an earlier corpus version, non-separating
in between). **Reading: the current zh gate's "top" statistic, calibrated
against `about_zh.md`'s broad Q&A-shaped corpus, gives almost any
grammatically well-formed Chinese question about "他" a high floor
regardless of whether it is actually on-topic or resolvable from context** —
a strictly weaker separation than the English gate exhibits for the
equivalent shapes (see the EN candidates tried for the same task, which
found off-threshold phrasings easily, several with >0.05 margin). This
generalizes Finding M (adjacency) to a new axis (task-request phrasing and
anaphoric-completeness).

**Conclusion (reframed after review): this is not "Task 9 couldn't author a
zh case," it is a measured property of the system — follow-up refusal is an
EN phenomenon at current calibration, and a zh visitor does not hit the bug
this feature fixes.** The rewrite-and-rescue path this task adds only ever
fires when the RAW question, judged alone, fails the gate (see
`eval/README.md`'s Task 9 section on the escalated path). If essentially
every natural zh question about 他 clears 0.3979 regardless of context or
actual topicality, then no natural zh follow-up ever reaches that path — zh
has no rescue phenomenon to measure, not a gap in coverage. This is
corroborated, not contradicted, by the FIRST half of this finding: Task 9's
zh post-context *negatives* (`neg-post-zh-01`/`02`) refuse correctly on their
own — `帮我写封邮件` at 0.3825, comfortably under threshold. So the zh gate
does separate clearly-off-topic requests from on-topic ones; it just also
admits essentially any domain-adjacent-sounding question standalone,
including ones a rewrite would need to resolve to make truly on-topic. Both
things are true about the same gate at once, and the follow-up-authoring
sweep is what surfaced the second one. `FOLLOWUP_POSITIVES_PER_LANG` in
`evaluation.py` therefore declares `{"en": 2, "zh": 0}` — a checked,
enforced zero (`test_followup_positive_pool_has_the_right_composition`), not
an absent key indistinguishable from an oversight.

**Trigger condition (superseded below — kept for the historical record).**
This is not hypothetical: zh calibration moves with `about_zh.md`'s content
(every edit to that file moves the threshold, per this document's own
"Coupled cost, same root cause" note on the EN side), and the zh gate's
threshold has already ranged from 0.3979 (current local build) to 0.4919
(an earlier corpus version, and — per the CRITICAL section at the top of
this document — what the currently DEPLOYED backend still runs). A tighter
zh gate would make natural zh follow-ups start failing standalone, at which
point zh follow-up positives become both authorable and necessary, and
`FOLLOWUP_POSITIVES_PER_LANG["zh"]` should move off 0. Not classified as
fixed by this task, since fixing the underlying separation (if it is even
a defect to fix, rather than the zero-false-refusal policy working as
designed) means recalibrating or reconsidering the gate statistic, out of
this task's scope.

**Update (a later task): live-API confirmation that the only zh shape under
threshold is categorically unresolvable, not just unfound — supersedes the
"too thin to be robust" / no-candidate-found framing above with a stronger,
structural claim.**

A subsequent re-sweep first confirmed the paragraph above still holds for
the shape that actually matters most: ~50 fresh structural analogues of the
EN winners' shape (`followup-en-01`/`02` — an elided argument plus a
dangling pronoun; e.g. 那优化呢, 那这部分呢, 那性能呢, 那怎么调) were tried
against the live zh gate and **every one leaked**, scoring 0.42–0.53
regardless of topic — the same finding as above, reproduced exactly, not a
fluke of the first sweep.

But widening to plain **narrative-continuation fillers** ("and then?") found
several that genuinely clear the gate with real margin — not just the two
this document already knew about (`那后来呢` 0.3957, `再后来` 0.3768), but a
better one still: `那再后来` (0.3645, margin 0.0334), plus `那后来` (0.3857,
margin 0.0122) and `再往下` (0.3877, margin 0.0102). Two of these
(`再后来`, `那后来`) were built into real golden cases and run through the
**live rewrite API** (not a hand-written stand-in) to test whether a
reference-resolution rewriter could actually turn them into something that
clears the gate:

```
followup-zh-01  '再后来'  -> '再后来'    [echoed, unchanged]        FAILED
followup-zh-02  '那后来'  -> '那后来？'  [only a '？' added]         FAILED
```

Both failed to resolve into anything gate-passing or page-retrieving. **This
is not a rewriter bug — it is the correct behavior of a reference-resolution
rule applied to text that has no reference to resolve.** `再后来`/`那后来`
mean "and then?": a **narrative continuation, not a referring expression**.
There is no dangling pronoun and no elided argument with a recoverable
filler; resolving one would require *generating the next event in a
sequence* from outside knowledge, not substituting a referent already
implied by the conversation. `followup-zh-02` gaining only a `？` makes the
point precisely — the model found nothing to substitute and, correctly
under its own instructions, declined to invent content. This is the exact
same "echo when nothing needs resolving" behavior the post-context negatives
(`neg-post-zh-01`/`02`) are designed to rely on; here it is simply pointed at
a positive case where that behavior means the case can never pass.

**The stronger conclusion this supports:** the zh gate's compressed dynamic
range (own off-topic calibration bottoms out around 0.3262 against a 0.3979
threshold, versus English reaching 0.09 — see `evaluation.py`'s
`FOLLOWUP_POSITIVES_PER_LANG` comment) leaves a narrow band of candidates
under threshold at all, and — measured directly, not inferred — that band is
occupied *entirely* by a shape a reference-resolution rewriter cannot act on
by construction. This is not "we haven't searched hard enough yet," it is
"the zh gate's under-threshold candidates and what reference resolution can
fix do not overlap" at this calibration — a structural mismatch between the
gate's separation behavior and the rewrite mechanism, not a search-coverage
gap one more sweep would close.

**Trigger condition, sharpened (superseded below — kept for the historical
record; the referring-expression framing turns out to be the wrong axis, not
wrong in kind).** zh follow-up coverage becomes possible
only if the zh gate tightens enough that *referring-expression* questions —
genuine dangling pronouns or elided arguments, the shape `followup-en-01`/
`02` actually exercise, not narrative-continuation fillers — fall below
threshold. Widening the gap by tightening the threshold further, on its
own, does not help: the filler shape already clears the gate with room to
spare (down to 0.3645 measured), so a lower threshold mostly encloses more
of the same unresolvable shape before it reaches far enough to admit a
genuine referring expression. Recalibration must be evaluated by re-running
this same live-rewrite-API check against whatever newly-under-threshold
candidates a tighter gate admits, not by threshold movement alone.

**Update (a later session): the subject-elided hypothesis — tested, and its
failure is what upgrades this finding from a sweep result to a structural
claim. Supersedes the "referring-expression" framing above with the
mechanism underneath it.**

The sweeps above cover two shapes: 他/这个-style interrogatives (~90
candidates) and content-free narrative-continuation fillers (~50 structural
analogues of the filler shape). Both came back the same way — everything
clears the gate except a handful of fillers that a rewriter correctly
declines to touch — which supports "zh has no rescue phenomenon" but is still
a search-coverage claim: it says these two shapes don't work, not that no
shape can. The repo owner named the specific gap directly: a third shape,
**contentful but subject-elided** questions (a dropped referent — "what's
optimized inside it," not "and then?"), carries a genuine dangling reference,
unlike the fillers, so it was the strongest available candidate for a
counterexample to "zh has none." It was measured with `rt.gate()` against the
live zh gate (threshold `0.3979`):

```
有什么特别的        0.4271
那里面呢            0.4250
里面有什么优化      0.4731
有什么巧思          0.4532
有什么难点          0.4563
性能上做了什么      0.4772
怎么优化的          0.5048
优化上有什么设计    0.5118
里面用了什么技术    0.5377
里面有什么设计      0.5420
```

All ten pass (leak) the gate on their own, several by a wide margin. For
direct contrast, the two under-threshold fillers already on record in this
finding: `再后来` `0.3768`, `那后来` `0.3857` — both comfortably below
`0.3979`, and both already shown above to resolve into nothing a rewriter can
use.

**This is the actual reason for `zh: 0`, stated as a tension rather than a
coincidence about which shapes happened to get tried.** The zh gate scores
similarity against `chat/knowledge/about_zh.md`'s curated corpus (the `top`
statistic in `gate_calibration.py`); 优化/设计/性能/技术/难点 — the words that
make `里面有什么优化` etc. genuinely resolvable and genuinely on-topic — are
exactly that corpus's vocabulary (check any of the ten against `about_zh.md`
directly to verify). A question has to reuse domain vocabulary to have a
recoverable referent to resolve in the first place, and reusing that
vocabulary is precisely what the gate's similarity score rewards. So: below
threshold implies content-free, hence nothing for a rewriter to resolve (the
filler case, confirmed live above, `followup-zh-01`/`02`); enough content to
be resolvable implies domain vocabulary, which the gate reads as on-topic and
passes standalone before any rewrite runs. There is no region of "resolvable
but under threshold" between them for a sweep to have missed — the two
required properties are in tension by construction.

**Not a gate defect.** `里面有什么优化`, asked mid-conversation about Prime
Engine, genuinely is an on-topic question, and the gate admitting it
standalone at `0.4731` is correct — nothing about that decision is wrong.
What this shows is that follow-up-rescue is a no-op for Chinese at current
calibration, because the zh gate does not refuse the class of question the
mechanism exists to rescue. English draws its line on the content side of
that boundary: `followup-en-01`'s raw question, "what about tuning it,"
refuses at `0.1466`; its Chinese equivalent, "里面有什么优化" ("what's
optimized inside it" — same referential shape, same missing subject), is
admitted at `0.4731`. One pair of numbers is the whole asymmetry: English
refuses before the content boundary that would need rescuing; Chinese admits
past it.

**Trigger condition, sharpened again.** zh follow-up coverage becomes
possible only if the zh gate tightens enough that questions carrying
**domain vocabulary** — not referring expressions in the abstract, but
specifically the 优化/设计/性能/技术/难点 class that is what makes them
resolvable — fall below threshold. Given the corpus-overlap mechanism just
measured, threshold movement alone is unlikely to reach that: the gate would
need to also refuse a large share of genuine on-topic vocabulary use, not
just these ten follow-ups. The more plausible lever is changing what the gate
corpus scores against — narrowing or restructuring `about_zh.md`'s coverage,
or scoring by something other than raw top-1 similarity to it — not the
threshold number in isolation. Either way, **any such change would need
re-checking against the existing zh negatives** (`neg-post-zh-01`/`02`, and
the `OFFTOPIC_ZH` sweep earlier in this finding): raising strictness enough
to push domain-vocabulary follow-ups below threshold raises it for everything
else scored against the same corpus too, and some of what currently passes
correctly could stop doing so.

**Why this sweep is worth more than the two before it.** The 他/这个 and
filler sweeps each show "we tried a shape and it didn't work" — a
search-coverage claim, always open to "did you try X." Subject-elided-but-
contentful questions were exactly that X: the one shape where both required
properties (resolvable, under-threshold) could plausibly coexist, proposed
specifically because it was the strongest case *against* `zh: 0`, not a
straw shape easy to fail. A clean 10/10 fail — using the gate's own contrast,
fillers at `0.3768`/`0.3857` versus content at `0.4271`-`0.5420`, a gap of
over `0.04` at the closest pair — is what turns "no counterexample found" into
"no counterexample can exist here, and here is the mechanism." `zh: 0` rests
on that structural claim now, not on the size of a sweep.

**Measurement caveat (final whole-branch review, fix wave): only one of the
two EN follow-up positives is a strong test of retrieval, not just the gate.**
Recorded here, alongside the rest of this finding's evidence about what the
follow-up-resolution feature does and does not measure, so it is on the
record before `followup_rescued` is ever quoted as a result on its own.

Verified directly with `rt.retrieve()` against the current index, raw
question only (no rewrite, no history):

```
followup-en-01 "what about tuning it"
  -> pages/automatic-differentiation.html, pages/chat-agent.html,
     pages/aegis-sword.html, pages/game-design-workshop.html
  wants pages/prime-engine.html -- ABSENT from the top-4 entirely.

followup-en-02 "was it tested"
  -> pages/chat-agent.html, pages/chat-agent.html, pages/gyrotris.html,
     pages/nothing-can-go-wrong.html
  wants pages/chat-agent.html -- ALREADY present, ranks 1 and 2.
```

`followup-en-01` is the genuinely strong case: the raw question's retrieval
misses the expected page completely, so a measured rescue there is evidence
the rewrite both cleared the gate AND fixed retrieval. `followup-en-02`'s
raw retrieval already lands the expected page in the top-4 -- its `rescued`
outcome rides almost entirely on the gate flip (refused standalone at 0.1498
vs. threshold 0.2029, per its golden.jsonl note), not on the rewrite
repairing a retrieval miss, because there is no retrieval miss to repair.
Both cases are legitimate follow-up-resolution positives (the raw question
genuinely fails the gate standalone, which is the feature's actual
precondition), but a reader treating `followup_rescued` as "N cases where
the rewrite fixed a broken retrieval" would overcount by this one case.

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

## Accepted by design

### Y — Two `zh`-labelled chunks contain English text, and that is correct

`pages/publications.html` has no Chinese translation of its citation list, so
the Chinese view falls back to the English text and the builder labels the
resulting chunks `lang="zh"`:

```
pages/publications.html#publications:zh:0   cjk=0.0%   "Publications Huang, J., He, Y., ..."
pages/publications.html#publications:zh:1   cjk=0.0%   ", Wang, Z., Yu, L., ... (2024). Enhancing semantic segme..."
```

**Confirmed by the site owner as intended: there is no Chinese version of the
publications page.** Citations are not translated in any case, so a Chinese
question about papers retrieving an English author list is the right answer,
not a leak — the LLM answers in Chinese and cites the titles as published.

Do **not** "fix" this by suppressing `zh` chunks for untranslated sections.
That would make the publications content unreachable to Chinese visitors
entirely, which is strictly worse than serving it in English.

**The mechanism is worth watching, though**, because it is the same one behind
Finding C: any page lacking a Chinese translation yields `zh`-labelled chunks
carrying English text. Two exist today and both are benign. If that count
grows, check whether the new cases are also legitimately untranslated content
(fine) or a loader defect leaking boilerplate across the language views (not).

Related: this is why `retrieved_langs` reports language by **chunk label**, not
by detected script. Across all 8 cells the current build shows a clean
`en:48` / `zh:48` — 384 retrieved chunks, zero cross-language drift — which is
the evidence that the single mixed-language e5 index does not need splitting
by language. e5 separates them without help.

### O — A 0.0001-margin near-tie flipped one Chinese case after Task 33's page corrections — verified, not a retrieval degradation

**Status: verified near-tie, accepted, not fixed.** Task 33 corrected
`pages/chat-agent.html` and `chat/knowledge/about_{en,zh}.md`, which still
claimed retrieval happens in the visitor's browser after Task 29 had moved
it server-side. That correction legitimately moved two golden-set numbers
(`hit@4` 90/96 → 89/96, `keywords` 132/179 → 131/179), traced to exactly one
case: `combat-zh-03` ("策划能不能自己调参数，不用每次找程序？", wants
`pages/cemented-dreams.html`, keyword "Blueprint").

Computed twice, independently — once while fixing it, once by a re-reviewer
who reconstructed the pre-edit index from
`git show 0422d0b:chat/data/chunks_e5.json` and recomputed from scratch
rather than trusting the first pass — with matching results both times:

```
pages/chat-agent.html#sec1:zh:1        0.8577 -> 0.8586   (rose after the page rewrite)
pages/cemented-dreams.html#sec7:zh:0   0.8585 -> 0.8585   (correct chunk, curated; held still)
gap: 0.0001
```

`chat-agent.html#sec1:zh:1` is the tail of the corrected "Thin client"
paragraph running into the head of the flow-diagram caption under the same
`<h4>` (no intervening heading), `chunk_size=800`/`overlap=100`. Both
paragraphs are accurate as rewritten — the collision with this specific,
unrelated Chinese combat-design question is coincidental, not a symptom of
bad content. The correct chunk is still retrieved, at rank 5 instead of
rank 4, one place outside `top_k=4`.

**Why not fixed:** the only two available levers are a `chat/src/` chunking
change (out of scope for a documentation-driven page correction) or
rewording the caption specifically to win a 0.0001 similarity contest
against an unrelated query — which is exactly the metric-gaming pattern
`eval/README.md` rule 1's disjointness tests exist to catch elsewhere on
this project (it has fired twice before; see that rule). Doing it by hand
here would be the same defect, not a fix for it.

**Risk, named explicitly:** a 0.0001 gap is coin-flip noise. It happened to
land in the regression direction this time and was caught, explained twice
independently, and only baked into the baseline after explicit user
ratification with the evidence in hand. The same coincidence landing the
other way on some future rebuild would silently *improve* a number for no
real reason — and small single-case swings like this, left unexamined, are
exactly how a baseline ratchets downward over time without anyone deciding
it should. Treat any future single-case swing at this margin with the same
scrutiny this one got, not as routine noise to re-baseline on sight.

**Coupled cost, same root cause:** `chat/knowledge/about_en.md` is
simultaneously retrieval corpus and the English gate's calibration corpus
(Finding N above; `eval/README.md`'s Known Limits). Correcting its stale
"browser-side retrieval" section moved the EN gate: threshold `0.2006` →
`0.2029`, margin `+2.5%` → `+1.7%`. Still comfortably positive — verified
against all 48 English golden positives, not sampled: 0 refused at the new
threshold — but this is the concrete, measured cost of editing that file:
any wording change to `about_en.md` moves the gate boundary as directly as
an explicit threshold edit would, not just retrieval. Worth knowing before
the next person edits that corpus for an unrelated reason.

---

### P — Adding the 17 Chinese page summaries cost 2 `hit@4` and bought 6 `hit@4(pg)`

**Status: measured near-ties, accepted, baseline updated.** The page-aware
feature gave every page a `data-zh` meta description, so `load_page` now emits
a summary chunk (`anchor="top"`) for the Chinese half of the index, which had
never had one. That is 17 new legitimate zh chunks — 319 → 336 — and they
compete for `top_k=4` like any other chunk.

Full movement, English **entirely flat** (its descriptions did not change):

```
                             hit@4      hit@4(pg)    keywords
ai_agent_recruiter/zh        11 -> 11    9 -> 10 up  18 -> 17 down
client_dev_recruiter/zh      11 -> 10 DN 9 ->  7 DN  18 -> 17 down
combat_design_recruiter/zh   11 -> 10 DN 5 ->  7 up  12 -> 12
visitor/zh                   12 -> 12    5 -> 10 up  19 -> 19
TOTAL (all 8 cells)          90 -> 88   59 -> 65    135 -> 133
```

The two `hit@4` losses are both sub-0.0025 rank-4/rank-5 near-ties, in the
same regime as Finding O above and computed the same way:

```
combat-zh-08  "他的设计作品有没有被公开展示或选中过？"  wants game-design-workshop
  4. 0.851400  pages/cemented-dreams.html#top:zh:0          <- new summary chunk
  5. 0.850700  pages/game-design-workshop.html#sec7:zh:0    <- expected, margin 0.0007

client-zh-08  "他是只会一个引擎，还是真的在好几个引擎里都做过东西？"  wants skills
  2. 0.878800  pages/nothing-can-go-wrong.html#top:zh:0     <- new summary chunk
  4. 0.876900  pages/prime-engine.html#sec3:zh:0
  5. 0.874700  pages/skills.html#sec2:zh:0                  <- expected, margin 0.0022
```

Note the second one's mechanism: the new chunk landed at **rank 2**, so every
chunk below it shifted down one slot and `skills.html` fell out of the window
at a 0.0022 boundary it was already sitting on. Nothing was out-matched; the
window moved.

**Why accepted rather than tuned away.** `hit@4(pg)` — the same metric with
`chat/knowledge/*.md`'s curated corpus excluded — rose by 6, and `visitor/zh`
alone went 5 → 10. Finding N and the Task 24 update call the gap between
`hit@4` and `hit@4(pg)` this project's Critical 1: a gap means the curated
corpus, not the site's own pages, is answering the question. That gap narrowed
31 → 23. Trading 2 corpus-answered cases for 6 page-answered ones is the
direction this file has been asking for.

The only lever that would recover the two is rewording `cemented-dreams`'s or
`nothing-can-go-wrong`'s Chinese description to lose a similarity contest
against an unrelated query — the exact metric-gaming pattern `eval/README.md`
rule 1 exists to catch, and the same reasoning that left Finding O unfixed.
Description length was checked first and ruled out as a cause: all 17 are
65–108 CJK characters, inside the 60–100 target, so this is not overlong text
matching too broadly.

**Second movement, same feature, separate cause.** Documenting page-awareness
on `pages/chat-agent.html` (a new `<h4>` with two paragraphs, plus an expanded
"Observable by turn") added 9 more chunks, 336 → 345. Both primary metrics
held exactly flat — `hit@4` 88, `hit@4(pg)` 65 — and `keywords` went
133 → 131: `aiagent-zh-04` **gained** 夹带指令, `ai_agent_recruiter/en` and
`visitor-zh-07` each lost. The `visitor-zh-07` loss is the same near-tie
shape:

```
visitor-zh-07  "如果只能看一个项目，你会让我先点开哪一个？"  wants projects.html
  1. 0.883200  pages/chat-agent.html#sec2:zh:2   <- new page-awareness prose
  2. 0.882800  pages/chat-agent.html#sec4:zh:1   <- new observability prose
  3. 0.882400  pages/projects.html#sec3:zh:0     <- EXPECTED PAGE, still retrieved
  4. 0.881200  pages/nothing-can-go-wrong.html#sec1:zh:0
  --- top_k=4 cutoff ---
  7. 0.877700  pages/projects.html#sec4:zh:0     <- carries the two keywords, 0.0035 out
```

Ranks 1–4 span 0.0020 end to end. The expected page is still retrieved at
rank 3, so `hit@4` never moved; only the specific keyword-bearing chunk fell
outside the window. The lever to recover it is trimming accurate portfolio
prose about a real feature to move a 0.0035 similarity on an unrelated
question — the same metric-gaming refusal as Finding O.

**Risk, named explicitly — same warning as Finding O, which applies to this
entry too.** Three single-case swings at this margin were re-baselined across
this feature. That is the second and third time; the ratchet Finding O warns
about is cumulative, and this entry is part of what it warns about. The
defence is that the *aggregate* moved decisively the right way on the
page-only metric, not that any individual case was inspected and forgiven. If
a future change costs `hit@4` again **without** a corresponding `hit@4(pg)`
gain, that is not this pattern and must not be re-baselined by pointing at
this entry.

**One thing worth watching.** `pages/chat-agent.html` now has the most zh
prose of any project page and took ranks 1 and 2 on a generic
"which project should I look at" question. That is a content-balance
observation, not a near-tie: if it starts winning questions that are plainly
about other projects, the fix is on that page, not in the retrieval code.

---

## Deferred by user decision

### I — `test_index_carries_a_calibrated_gate` asserts on dead fields — RESOLVED
**Status: resolved by Task 29 Part 2. The "needs a user ruling before merge
to main" flag is retired** — three of the six final whole-branch reviewers
reached that conclusion independently.

The original finding: `tests/test_gate.py::test_index_carries_a_calibrated_gate`
checked `index.json`'s `gate_stat`/`gate_threshold` fields, which nothing read
for an e5 index — `runtime.py` built gates from
`gate_vectors.json`/`fallback_vectors.json`, and the deployed Tencent function
reads its own bundled copy — so they were an advisory duplicate with no live
consumer, and the test was inert in the sense that matters.

What changed: Task 29 Part 1/2 moved those fields to `data/meta.json` and
retargeted the test, now
`tests/test_gate.py::test_meta_json_carries_a_calibrated_gate`. `meta.json`'s
`gate_threshold`/`gate_stat` have a **live** consumer:
`scripts/chat-widget.js`'s `gateThreshold()` and `gateValue()` read them on
every page load, in both light and degraded mode. The final review proved the
test now bites by zeroing `meta.json`'s `gate_threshold` and watching it go
red. Nothing to defer.

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
