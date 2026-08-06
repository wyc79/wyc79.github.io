# Golden set — held-out evaluation for the chat agent

`golden.jsonl` is **measurement**, not corpus. It never enters `data/chunks_e5.json`.
Editing it changes only what the score says.

For consolidated findings from running this harness against the current
index — what's fixed, what's outstanding, and the full right-page/wrong-
chunk catalog — see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

Contrast `../knowledge/*.md`, which IS corpus: `build_index` embeds those
sections into the index, so editing them changes what the agent knows and how it
answers. Both `about_en.md` and `about_zh.md` additionally *are* their
language's gate matrix — `about_en.md`'s 55 sections are the English gate's
on-topic corpus (wired since Task 24), and `about_zh.md`'s sections are
calibrated the same way on every build; whether a Chinese gate ships from this
repo's own artifacts depends on whether that calibration currently separates
(see Known Limits below — it has flipped between builds before, so check
`data/gate_zh_bge.json`'s own `gate_margin` rather than trusting a claim here).

## Format

One JSON object per line. Adding a case is a one-line diff.

| field | required | meaning |
|---|---|---|
| `id` | yes | unique, `{role-short}-{lang}-{nn}`; role-short ∈ `client`, `aiagent`, `combat`, `visitor` |
| `role` | positives only | a role id present in `../data/roles.json`. Negatives belong to no role — see below |
| `lang` | yes | `en` or `zh` |
| `type` | yes | `positive`, `off_topic`, or `injection` |
| `q` | yes | the question, as a visitor would type it |
| `expected_urls` | positives only | site-relative page paths; a hit is a non-empty intersection with the top-4 |
| `expected_keywords` | positives only | 1–4 terms that must appear in the retrieved text; scored as coverage |
| `adjacency` | `off_topic` only | `"easy"` or `"adjacent"` — how close the probe sits to the site's domain. Empty on `positive` and `injection` (an injection is always adjacent by design: it is worded to look on-topic) |
| `history` | no | prior conversation turns, `[{"role": "user"/"assistant", "content": "..."}, ...]`. Empty (or omitted) for the single-turn cases that make up most of the set. See "Multi-turn cases" below. |
| `note` | no | why this case exists |

**Positives are per (role, lang).** Every `(role, lang)` cell holds exactly
12 `positive` cases with no `history` (`POSITIVES_PER_CELL` in
`evaluation.py`) — one cell per role in `roles.json` crossed with `en`/`zh`.
Multi-turn positives (see below) are **not** part of this count — they are
excluded from it deliberately, both in the dataset invariant
(`test_positive_cells_have_the_right_composition`) and in
`evaluation.aggregate()`'s runtime counting, and are held to their own
separate, per-language quota instead (`FOLLOWUP_POSITIVES_PER_LANG` in
`evaluation.py`, a mapping — currently `{"en": 2, "zh": 0}`, enforced by
`test_followup_positive_pool_has_the_right_composition`). **zh is a declared
zero, not a missing case, and not a search-coverage gap:** at this project's
current zh gate calibration, a zh question below the gate's threshold and a
zh question a rewriter can actually resolve are mutually exclusive — the gate
scores against `about_zh.md`'s own vocabulary, which is exactly the
vocabulary that makes a question resolvable in the first place, so a zh
follow-up is never refused and the rewrite-and-rescue path it would test
never fires. See `KNOWN_ISSUES.md` Finding Q for the full measurements
(including the subject-elided-question test this claim rests on) and its
trigger condition: tightening the threshold alone will not do it; the more
likely lever is changing the gate corpus itself, re-checked against the
existing zh negatives.

**Negatives are one shared pool per language**, not per role: the gate either
refuses off-domain and injected probes or it doesn't, and that has nothing to
do with which role's system prompt is loaded. Each language's pool holds
exactly 4 `off_topic` tagged `"easy"`, 4 `off_topic` tagged `"adjacent"`, 4
`injection`, and 2 post-context negatives (`off_topic` with `history` — see
below) (`NEGATIVES_PER_LANG` in `evaluation.py`). A negative with `history` is
bucketed as `post_context` regardless of its `adjacency` tag — see "Multi-turn
cases" below for why.

`adjacency` splits `off_topic` into two questions that demand opposite fixes:

- **`easy`** — obviously unrelated to the site (weather, recipes, tax law).
  A refusal failure here means the gate itself is broken.
- **`adjacent`** — plausibly related but still off-domain (game industry
  news, a *different* studio's postmortem, general career advice that
  happens to mention games). A refusal failure here means the gate can't
  separate *this* domain from things that merely resemble it.

Reporting them separately is the point: 90% refusal on `easy` with 90% pass
on `adjacent` is a precise, actionable gate. A single blended refusal number
would report the same 90% either way.

`expected_urls` names **pages, never chunk ids**. Ids like
`pages/prime-engine.html#sec2:en:3` move whenever the page is edited or
`chunk_size` changes; a page URL survives re-chunking and still catches real
retrieval failures. Page URLs are also language-agnostic, which is what lets one
field serve both languages.

`expected_keywords` adds the precision `expected_urls` gives up. A page has up
to 31 chunks and landing on any of them scores a hit, even when the chunk
carrying the answer never surfaced. Keywords are matched against the
concatenated text of the post-floor top-4 — exactly the chunks the function puts
in the prompt and returns as `sources` — so they answer "was the fact available
to the model?".

Read the two together:

| `hit@4` | keywords | reading |
|---|---|---|
| pass | pass | healthy |
| pass | fail | right page, wrong chunk — a **chunking** problem, not retrieval |
| fail | pass | the fact lives somewhere unexpected; check `expected_urls` |
| fail | fail | genuine retrieval miss |

Matching is case-insensitive. Word boundaries apply only where the keyword's own
edge is ASCII alphanumeric, so `AI` will not match inside "available" while
`C++`, `3C` and `UE5` still match. CJK keywords match as plain substrings.

## Authoring rules

1. **Disjoint from fit-on data.** Never reuse a string from
   `gate_calibration.ON_TOPIC/OFF_TOPIC/ON_TOPIC_ZH/OFF_TOPIC_ZH`,
   `tests/test_gate.py::OFF_TOPIC`, or any `roles.json` starter. Those set the
   gate threshold or are already asserted — scoring against them measures
   nothing. `test_golden.py` enforces this.
2. **Chinese cases are authored natively**, never translated, and never lifted
   from an `about_zh.md` heading. `about_zh.md` currently holds 53 sections
   and is the zh gate's calibration corpus regardless of whether a zh gate is
   actually shipping on a given machine (see Known Limits) — a question
   copied from one of its headings scores ~1.0 by construction. The count
   changes as the corpus is authored further; read it from the file, not
   from a number quoted here.
3. **Visitor vocabulary, not site vocabulary.** "What's his shipped title count"
   beats "tell me about Cemented Dreams" — the latter is retrieval on easy mode.
4. **Keywords are proper nouns and technical terms, not prose.** `Blueprint`,
   `UE5`, `frustum culling`, `Hive` — never `mobility and responsiveness`. This
   is the one field coupled to the site's wording: rewrite a page and prose
   keywords break for reasons that are not regressions, while the name of a
   technology survives. Avoid Latin keywords shorter than three characters.
5. **Never edit a case to make a regression pass.** A failing case is a work
   queue item. Record reality in the baseline.
6. **A negative's `adjacency` must be `"easy"` or `"adjacent"`, never guessed
   under time pressure.** It is the label a later reader uses to tell "the
   gate is broken" apart from "the gate can't see this specific edge" —
   picking the easy bucket for a case that's actually adjacent (or vice
   versa) quietly corrupts that signal instead of raising a visible error.

## Multi-turn cases (Task 9)

A case with `history` exercises the *escalated follow-up path*: the widget
condenses the conversation plus the new question into a standalone rewrite,
but only after the raw question, judged alone, fails the off-topic gate — a
follow-up that clears the gate on its own never reaches the rewriter, in
production or in `score_case`'s replay of it. `history` is a list of
`{"role": "user"/"assistant", "content": "..."}` turns and must **alternate,
start on `user`, and end on `assistant`** — the widget's own conversation
state is only ever appended to in user/assistant pairs, so a case whose
history could not have come from a real session is rejected by `load_cases`
rather than silently accepted.

**Rewrites are recorded, not called live.** `chat/eval/rewrites.json` (case
id → the recorded standalone rewrite) is written *only* by
`scripts/refresh_rewrites.py`, the one thing in this repo that calls a real
LLM for a rewrite. `run_eval.py` and `pytest` replay that recorded fixture
instead — never the live model — which is what keeps the harness
deterministic and network-free. **Re-run `refresh_rewrites.py` whenever
`REWRITE_SYSTEM_PROMPT` (in `functions/tencent/index.py`) changes, or
whenever a multi-turn case is added or edited**, and read the diff before
committing: a rewrite that starts smuggling a project name into an
off-topic follow-up is exactly the regression the post-context negatives
below exist to catch, and it shows up in that diff first, in plain text. A
case added without a matching `rewrites.json` entry does not error — it
silently scores as still-refused (an absent measurement, never a fabricated
pass; see `load_rewrites`'s docstring) — `run_eval.py` renders that cell's
`post_context`/`rescued` column as `n/a` instead of a count whenever any of
its multi-turn cases lacks a recorded entry (see `_fixture_coverage`), so a
gap is visible in the table itself rather than requiring a separate note.

`chat/eval/rewrites.json` is recorded, against the live DeepSeek API — both
multi-turn positives resolved and all four post-context negatives echoed
verbatim. `followup_rescued` and `refusal_post_context` are consequently real
measurements, not placeholders, and are wired into
`tests/test_golden.py::_REGRESSION_METRICS` (`_GATE_METRICS` too — see that
constant's declaration for why each metric needed it). This makes the re-run
rule above matter more than it used to: a `REWRITE_SYSTEM_PROMPT` edit with
no matching `refresh_rewrites.py` re-run no longer just prints a stale
number, it silently invalidates the regression gate — `data/eval_baseline.json`
would keep comparing against rewrites the deployed prompt no longer produces.
Read the current fixture's effect on the numbers from `data/eval_baseline.json`
or `run_eval.py`'s own output rather than trusting any count quoted in this
file, for the same reason every other measured number in this doc carries
that instruction: both move whenever the fixture, the golden set, or the
index does.

**A good multi-turn positive is a question that genuinely cannot stand
alone.** The raw `q`, judged with no conversation behind it, must fail the
gate — if it passes on its own, `history` is decorative and the case is
measuring nothing about the rewrite path (this is checkable offline with
`rt.gate(case.q)`, no fixture needed, and was the concrete failure mode Task
9 found empirically: several natural-sounding follow-up phrasings — elided
arguments like "is there any optimization work," dangling pronouns like "how
was it tested" — turned out to pass the gate standing alone against this
project's live, calibrated gate, even though an equivalent-sounding
production symptom had failed against an older deployed gate at a different
threshold over a different corpus; see `KNOWN_ISSUES.md` Finding A). A
hand-written *resolved* form — what a correct rewriter should produce — must
both pass the gate and retrieve `expected_urls`/`expected_keywords`; verify
both before recording the real fixture, so a broken case is caught before it
ever reaches `refresh_rewrites.py`.

**A good post-context negative is a question with nothing to resolve, paired
with on-topic history.** The point is *echo-and-refuse*: a correct rewriter
adds nothing (there is no dangling reference to resolve — "make up a poem for
me" after a Prime Engine question is just as off-topic as it would be with no
history at all), so the resolved form is the same text and must still fail
the gate, exactly like the raw form. **The regression this case exists to
catch is a rewrite that resolves it anyway** — if a rewriter turns "make up a
poem for me" into "write a poem about Prime Engine" because Prime Engine was
mentioned one turn earlier, the gate now has real, on-topic-sounding text to
pass, and the off-topic gate has effectively gone permanently open for the
rest of that conversation. Like multi-turn positives, the raw `q` must fail
the gate on its own — otherwise the escalated path never fires and the case
never exercises the rewriter at all — so verify with `rt.gate(case.q)` before
committing, not after.

`evaluation.aggregate()` reports multi-turn cases in dedicated cells, never
blended with the single-turn numbers: positive cells gain `n_followup` /
`followup_rescued` (a rescue requires a REPLAYED rewrite — one the raw
question actually needed — to both clear the gate and land the expected page;
see `CaseResult.rescued`'s docstring), and shared negative cells gain a fourth
bucket, `post_context`, alongside `easy`/`adjacent`/`injection` — bucketed by
whether the case carries `history`, ahead of `adjacency`, because a
post-context regression is a different failure (the rewriter leaking context)
than an adjacency one (the raw gate itself failing to separate the domain)
and averaging them into the `easy`/`adjacent` numbers would hide exactly the
regression this bucket exists to catch. Multi-turn positives are *excluded*
from `n_positive`/`gate_pass`/`hit_at_4`/`hit_at_4_page_only`/
`keywords_found`/`keywords_total`/`retrieved_langs` entirely — folding them in
would grow those cells' denominators every time a follow-up case is added,
moving numbers for cells the case was never meant to touch. A follow-up's
retrieval quality is already captured by `rescued`, which requires landing
the expected page.

## Running

```bash
python scripts/run_eval.py                      # positive table + shared negatives block
python scripts/run_eval.py --role visitor --lang zh --verbose
python scripts/run_eval.py --update-baseline    # after a deliberate change
```

`--update-baseline` refuses a filtered run (`--role`/`--lang`) and refuses a
run whose index is known stale (`Runtime.stale_knowledge_headings` non-empty —
rebuild first, or pass `--allow-stale-index` to freeze that state on purpose).
Both refusals exist because the baseline is what `test_no_metric_regressed`
compares against: a baseline captured from a state that does not represent the
system silently re-anchors the regression gate.

The table prints blocks that are never blended into each other: a
per-`(role, lang)` positive-cell table (gate-pass, `hit@4`, keyword coverage,
retrieved languages — single-turn cases only, see "Multi-turn cases" above),
a shared-negatives block with one row per language showing `off_topic/easy`,
`off_topic/adjacent`, `injection`, and `post_context` refusal counts side by
side, and — only when the golden set holds multi-turn positives — a
follow-up-resolution block (`rescued N/M` per cell). Either block renders
`n/a` instead of a count for any cell whose multi-turn case(s) lack a
recorded `rewrites.json` entry, rather than a separate note naming the case
(see "Multi-turn cases" above).

`--role` filters the positive table only — negatives belong to no role, so
the shared-negatives block always covers the full pool for whichever
language(s) are selected, regardless of `--role`. Filtering it too would
make the block look like the shared pool while actually showing an
arbitrary, incomplete slice of it. `run_eval.py` prints a note to stderr
whenever `--role` is set, so this is never silent.

`tests/test_golden.py` fails only when a metric drops below
`../data/eval_baseline.json` — an absolute threshold would be meaningless,
because `hit@4` measures this file's difficulty as much as the system's quality.
Keyword coverage is compared as a ratio, not a raw count, so editing a case's
keyword list does not read as a regression. A metric or cell that exists on
only one side of the comparison (because the schema changed since the
baseline was generated) is treated as new, not a regression, and skipped;
anything present on both sides is still compared strictly.

Gate-derived metrics (`gate_pass`, `refusal_easy`, `refusal_adjacent`,
`refusal_injection`, `refusal_post_context`, `followup_rescued`) are
additionally skipped whenever **either** the baseline or the current run
reports `gate_available: false` for that cell — comparing a real refusal
count against the 0 a bypassed gate always produces would report a
regression that never happened. `refusal_post_context` is gate-derived the
same way `refusal_easy`/`adjacent`/`injection` are; `followup_rescued` is a
compound metric (a rescue needs both a gate pass and a retrieval hit — see
`CaseResult.rescued`) that carries the same exposure through its gate half,
currently a no-op in this repo (the en gate is always committed and no zh
follow-up positive exists — see `tests/test_golden.py`'s
`_GATE_METRICS`/`_FOLLOWUP_METRICS` comments for the full reasoning) but
declared for when that changes. `hit_at_4` and `keyword_coverage` have no
such exception; retrieval is gate-free, so they are always compared
strictly.

## Known limits

- The **Chinese gate** is built from `../knowledge/about_zh.md`.
  `loader.py`'s section floor used to be a raw character count, which is
  language-blind — Chinese has no word-boundary spaces and packs more content
  per character than Latin text — and silently dropped some authored `##`
  sections (three of 22 at the time, including an identity-question shape,
  "who is YC") before they ever reached the gate corpus. The floor is now
  script-aware (a CJK character counts for more than a Latin one instead of
  1-for-1; the weight is a chosen heuristic with headroom above the ~1.6
  minimum that would admit those three sections, not a cited ratio — see
  `loader.py`'s `_CJK_WEIGHT` comment), so every currently-authored section
  clears it; this is a property of the floor, not a fixed count, since Tasks
  22-23 grew the corpus (to 53 sections as of this writing — read the current
  count from the file, it keeps growing). **Two rebuilds have run since the
  floor fix** (Tasks 19 and 24), so the margin has been re-measured, not left
  at its pre-fix value: recomputed against the current 53-section corpus
  (read-only, matching `_build_zh_gate`'s own calibration call), stat `top`,
  threshold `0.5273`, off-topic max `0.5169` vs on-topic min `0.5153` — margin
  **-0.49%**, still not separating at THAT point, so `build_index` correctly
  did **not** ship a zh gate at that measurement — every CJK question took the
  name-blind `cjk_bypass` path instead. **Superseded (Task 29 Part 2 rebuild):**
  the current `data/gate_zh_bge.json` DOES separate (stat `top`, threshold
  `0.3979`, margin **+5.9%**, 53 sections) — read the live number from that
  file's own `gate_margin` rather than trusting either figure quoted here,
  since this has flipped between separating and not across multiple rebuilds
  as `about_zh.md` grew and the calibration/floor logic changed. The
  pre-floor-fix figure this section used to report
  (22-section corpus, off-topic max 0.517 vs on-topic min 0.515, margin
  -0.4%) is superseded by the above, not the current state. An even earlier
  corpus version did calibrate, at threshold 0.4919 against an on-topic floor
  of 0.492 — essentially zero margin — and held-out Chinese positives still
  failed it at a real rate; that same 0.4919 threshold is, as of this
  writing, what the currently *deployed* backend still runs live (it bundles
  its own, older gate vectors from a zip built before Task 29 Part 2's file
  split — see `../README.md`'s "Deploying a
  backend" section). Neither local state (no gate, or now a marginal one) nor
  the deployed state (a live, differently-marginal gate) is a defect in the
  golden cases; both are the finding.
- `data/gate_zh_bge.json` is gitignored, and may simply be absent on a fresh
  clone that hasn't rebuilt — either way a machine without a working zh gate
  reports the Chinese gate columns as `n/a`, never as `0%`. Chinese `hit@4`
  still runs — retrieval does not depend on the gate.
- Only the gate and retrieval are scored. Answer quality and role emphasis need
  the LLM and are deliberately out of scope; the `role` field is carried on
  every case so that tier can be added without re-authoring.
- **This harness is deterministic.** Repeated runs against the same
  `data/chunks_e5.json` + `data/gate_en_minilm.json`/`data/gate_zh_bge.json`
  produce bit-identical results —
  dot products of fixed vectors don't drift between runs. If numbers ever
  appear to "shift" between two runs, suspect a mutated artifact (see
  `tests/conftest.py`'s guard), not measurement noise.
