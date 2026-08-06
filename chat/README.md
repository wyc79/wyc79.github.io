# Portfolio Chat Agent — client-side RAG on GitHub Pages

A role-aware AI chat agent for [wyc79.github.io](https://wyc79.github.io), built as a
demonstration of a full RAG pipeline that fits inside a static site:

- **Write path (this package, Python):** site HTML → section loader → sliding-window
  chunker → ONNX embeddings → `data/chunks_{model_preset}.json` (`chunks_e5.json` in
  production), a static vector index served by GitHub Pages.
- **Read path (browser, `../scripts/chat-widget.js`):** the visitor picks a role
  (recruiter for game client dev / game AI & agents / combat design, or visitor —
  each mirrors a real campus-hiring JD); normal mode's retrieval happens server-side
  (see below), while light mode and degraded mode embed the question **in the
  browser** with transformers.js and score a dot product against a committed chunks
  file, with the top chunks rendered as linked source cards.
- **Generation (optional, `functions/tencent/`):** a backend holds the LLM
  API key and turns retrieved chunks into grounded answers. Without one the widget
  runs in retrieval-only demo mode — still useful, still fully client-side.

```
build time (python)                     visit time (browser)
site *.html                             question + role
  └─ loader.py      sections              └─ transformers.js (WASM, self-hosted)
  └─ chunker.py     800/100 overlap       └─ dot product vs a chunks file → source cards
  └─ embedder.py    e5 / MiniLM ONNX      └─ POST /chat → Tencent SCF → DeepSeek
  └─ index_builder  data/chunks_e5.json         (holds API key; logs every turn)
```

## Why it's built this way

**Retrieval is server-side in normal mode, client-side in light/degraded mode
(Task 29).** GitHub Pages can only serve static files, so with no backend configured
(light mode: `--model minilm`, no backend) or the backend unreachable (degraded mode)
the widget does retrieval itself: a committed chunks file (`data/chunks_{model_preset}.json`
in light mode, `data/chunks_en_minilm.json` in degraded mode — see the file-layout
table below) is small enough that a brute-force dot product over it runs in well under
a millisecond in the browser. In normal mode (a backend configured and reachable), the
Tencent function retrieves server-side from its own bundled copy of the production
chunks file instead: the browser fetches only `data/meta.json` (~6 KB combined with
`data/roles.json`) on load, not the multi-MB chunks file, and gets chunk text/URLs back
in `/chat`'s response. Either way the retrieval layer stays inspectable (open
DevTools, watch the scores, or read the function's logs). Exact chunk/page/section
counts and file size move on every rebuild as site content and the curated corpus
change — read them straight from `data/meta.json` (`chunk_count`, `built_at`,
`chunks_file`) or `build_index.py`'s own summary line rather than trusting a number
quoted here; `eval/KNOWN_ISSUES.md` records the counts as of its most recent
measurement.

**The chat knows which page you are on.** The widget sends the current page's
site-relative url (`pages/skills.html`) with every `/chat` request, and the
function appends up to `PAGE_CONTEXT_MAX` of that page's own chunks to the
context, tagged `current_page="true"`, with the page's `<meta
name="description">` summary chunk first. There is no intent detection: the
model decides from the tag whether a question is about the page in front of the
visitor ("summarize this page") or about the site in general. Retrieval
returning nothing no longer refuses on its own — a deictic question has no
useful query vector, which is exactly why it used to hit the canned refusal.
Only the url crosses the wire; the page title in the system prompt is read off
the function's own chunk records, so no client-supplied string reaches the
prompt. Page-injected chunks are context only and never render as source cards
— the visitor is already on that page.

That summary chunk exists in both languages because each page's `<meta
name="description">` carries a `data-zh` attribute alongside its English
`content=`, following the site's own i18n convention (`scripts/i18n.js`).
**Never put `data-en` on a `<meta>` tag** — i18n.js selects `[data-en][data-zh]`
and assigns `textContent`, which is meaningless on a void element;
`tests/test_loader.py::test_no_meta_description_carries_data_en` enforces this.
A page with no `data-zh` yields no Chinese summary chunk rather than falling
back to the English string, which would put English text in the Chinese half of
the index where the zh gate cannot judge it.

### File layout (`chat/data/`, Task 29 Part 2)

One file, one job — this split is what makes degraded mode's source links reliable
(see "Off-topic use is refused three times over" and the degraded-mode paragraph
below for the bug this fixes):

| file | model | contents | git | fetched by browser |
|---|---|---|---|---|
| `chunks_{model_preset}.json` (`chunks_e5.json` in production) | e5 | all indexed chunks, pages + `knowledge/*.md`, both languages | committed | **never** in production — `localModelMatchesIndex()` fetches this file only when `meta.model` is MiniLM, i.e. a light-mode `chunks_minilm.json` build |
| `gate_en_minilm.json` | MiniLM | English off-topic gate vectors (`knowledge/about_en.md`'s curated sections) | committed | degraded mode only |
| `gate_zh_bge.json` | bge-zh | Chinese off-topic gate vectors (`knowledge/about_zh.md`) | **gitignored** | **never** (MiniLM, the only in-browser model, cannot embed Chinese) |
| `chunks_en_minilm.json` | MiniLM | the retrieval corpus's English chunks, re-embedded with MiniLM | committed | degraded mode only |
| `meta.json` | — | build metadata + `chunks_file` naming the retrieval corpus above | committed | every load |
| `roles.json` | — | persona prompts/starters | committed | every load |

`chunks_en_minilm.json` is **not** a language-filtered copy of `chunks_e5.json` for
convenience — it exists because degraded mode needs its own real, id-addressable
chunk records distinct from the gate's curated corpus (see below).

**One retrieval model, one gate model, both self-hosted quantized ONNX.** Document
vectors are precomputed at build time by `multilingual-e5-small` (e5, 384-dim, bilingual
en+zh — `RAG_MODEL_PRESET=e5`, see `.env.example`); the deployed Tencent function embeds
each query with the *same* model at `/embed`, so both sides of the retrieval dot product
stay in one embedding space. e5 is too large to self-host in the browser (~130 MB on
disk), so a second, much smaller model — `all-MiniLM-L6-v2` (~23 MB, self-hosted at
`models/Xenova/all-MiniLM-L6-v2/`, cached by the browser after first load) — runs
entirely client-side via transformers.js/WASM and does two jobs, each against its own
single-purpose file (Task 29 Part 2): the local off-topic gate (`data/gate_en_minilm.json`,
below) and the degraded-mode retrieval fallback (`data/chunks_en_minilm.json`) used when
the function is unreachable. Both models run the same mean-pooling +
L2-normalize recipe on the Python (onnxruntime) and JS (transformers.js) sides. Measured
cross-runtime parity for the MiniLM path: cosine(browser vector, python vector) ≈ 0.99 —
native vs WASM int8 kernels round differently — with top-4 retrieval overlap of 3–4/4 on
test queries. Documents are embedded one at a time, unpadded, because padded batches
shift the dynamic-quantization scales and would break this parity.

**The only secret lives in the backend function.** Anything shipped to a static site is
public, so the LLM API key (DeepSeek by default) sits in the Tencent SCF function's
environment config, never in the repo. The client sends a role *id*, never prompt text
— the function reads `data/roles.json` from the site itself, so prompts can't be
injected through the API surface. The function validates and size-caps every field,
checks the `Origin` header, caps `max_tokens`, and rate-limits per IP.

**Off-topic use is refused three times over.** The chat is not a general assistant:
(1) the widget gates on retrieval score — if the best chunk scores below the
threshold, it refuses locally and re-suggests role-specific questions without any
LLM call. The threshold is not hardcoded: build_index calibrates it per embedding
model from canonical on-/off-topic query sets (src/portfolio_rag/gate_calibration.py).
Because e5 compresses cosines and can't separate on-/off-topic itself, gating is
delegated to a MiniLM copy of **`knowledge/about_en.md`'s curated on-topic sections**
(`data/gate_en_minilm.json`, whose `gate_threshold`/`gate_margin` are also copied into
`data/meta.json` for reference) — not a copy of the full page-chunk index, which is
what earlier builds used. `_check_en_gate_margin`
(`index_builder.py`) **aborts the build** (raises, writes nothing) if calibration doesn't
clear a configurable margin floor (default: any non-negative margin; see
`RAG_MIN_GATE_MARGIN`/`RAG_ALLOW_NEGATIVE_MARGIN`) — a negative-margin English gate can
no longer ship silently the way it once did. Read the calibrated stat/threshold/margin
straight from `data/gate_en_minilm.json` (or the build log's `gate calibration`
line) rather than trusting a number quoted here, since recalibration moves it on every
rebuild of `about_en.md`. A second gate, `bge-small-zh-v1.5` over the hand-written
`knowledge/about_zh.md` corpus, is calibrated on every build but only shipped (as
`data/gate_zh_bge.json`, gitignored) if it actually separates the distributions —
read whether it currently does from that file's own `gate_margin` (or the build
log's `gate calibration` line), not from a number quoted here, since calibration
quality moves with `about_zh.md`'s content. When it doesn't ship, Chinese questions
bypass the local gate entirely (name-blind `cjk_bypass`), relying on the system
prompt's off-topic instruction alone. **This local-repo state does not describe
the currently deployed backend** — see "Deploying a backend" below.
The gate is name-blind: mentioning "Yuanchen Wang" inflates similarity (a
name-dropped joke request scores 0.61), so name-bearing questions are gated on the
question with the name stripped out, unless the remainder is a bio-intent stub
("who is", "tell me about", empty) — those are genuinely about YC and pass; (2) the function independently refuses empty-context requests, so bypassing the
widget doesn't buy anything; (3) the system prompt instructs the model to decline
general-purpose requests and ignore instruction-injection in questions. Pages'
`<meta name="description">` tags are indexed as summary chunks so broad-but-legitimate
questions ("who is YC") clear the gate comfortably.

**A failed gate is not terminal when there's a conversation behind it (follow-up
resolution).** Gate (1) judges one string in isolation, so a genuine follow-up —
"is there any optimization work" right after a Prime Engine turn — can refuse on an
artifact of that isolation rather than on being off-topic. Only the local, normal-mode
gate path (`emb.gate` in `chat-widget.js`, i.e. the server-computed `/embed` verdict) can
escalate, and only when `state.history.length` is non-empty; a refused first turn, a
refused self-contained question with no history, and a declared intent (the action chips
bypass the gate entirely) all take the old, terminal refusal. When it does escalate, the
widget sets `gate_failed: true` and resends the question to `/chat` with its capped
history (`state.history.slice(-6)`). There, `rewrite_question` condenses history + question
into one standalone string with a small LLM call pinned to temperature 0 and thinking
disabled — sampling would let a visitor resample past an authoritative gate, and an
unpinned thinking budget has already once eaten the whole completion and left nothing to
gate (see `REWRITE_SYSTEM_PROMPT`'s neighboring comments in `functions/tencent/index.py`) —
then runs the *same* name-blind `gate_form()` and `gate_decision()` the `/embed` path uses
on the result. That re-gate is server-side and authoritative: a client cannot skip it, and
`gate_failed` only ever adds this check, never substitutes for the original one. The
standalone string is used only to gate and retrieve; the visitor's own wording, unrewritten,
is still what reaches the model (`body["question"]`, not the rewrite). A rewrite that fails
the re-gate refuses exactly as before, distinguished from a retrieval-empty refusal by the
server's own `reason` field (`regate_failed` vs the default `retrieval_empty`) so the two
causes don't get conflated after the fact. The client's `refused_by` is not the same
vocabulary: it is `resp.reason` when the (redeployed) server sent one, but falls back to a
distinct literal, `server_retrieval_empty`, only when it's missing — i.e. only against an
old, un-redeployed function that predates this field entirely (see the deployment callout
below for exactly that scenario). Light mode and degraded mode have no backend LLM to
condense with, so a refusal there is still the old, terminal one.

Known limit, accepted rather than fixed: the gate is a **subject-matter** filter, not an
intent filter, so `"write a poem about it"` right after a Prime Engine turn resolves to
something genuinely about Prime Engine, clears the gate legitimately, and is left entirely
to the system prompt's decline-general-requests instruction — the same defense that already
has to catch that exact string typed as a first turn. This isn't a new hole the feature
opens; it just routes more traffic toward an existing one.

**Everything is logged.** Each turn (input, retrieved chunk ids + scores, output) goes
to `console.debug`, to Google Analytics as a `chat_turn` event when available, and
server-side via the Tencent SCF function (JSON lines in the SCF console's 日志查询;
see `.claude/DEPLOY.md`).

**Curated chunks bridge vocabulary gaps — and are now the English gate's corpus
too.** Visitors ask in hiring vocabulary the pages never use ("resume highlights",
"CV", "qualifications") — without help those queries score below the off-topic gate.
`knowledge/*.md` holds short authored summaries (each `## Heading` block is one chunk,
`link:` sets its source card) indexed alongside the scraped pages. `about_en.md` is no
longer retrieval-only: it is also the corpus the English gate calibrates against (see
above), so a wording change there can move the live gate's margin as directly as a
threshold edit would — read `about_en.md`'s own header before editing it. Keep facts
consistent with the site and **rebuild after editing** — an edit that isn't rebuilt
desyncs `Runtime.knowledge_chunk_ids` from the index and can silently corrupt the
`hit@4(pg)` diagnostic (`src/portfolio_rag/runtime.py`'s `stale_knowledge_headings`
reports this when it happens; `scripts/run_eval.py` prints a note if so).

## Layout

```
chat/
├── build.py               # CLI: THE master build -- one call for site artifacts, `--function`
│                          #   also for the Tencent SCF zip (see "Rebuilding the index" below)
├── src/portfolio_rag/     # the pipeline: config, loader, chunker, embedder, roles, index_builder
├── knowledge/             # curated .md chunks (resume/CV vocabulary the pages lack)
├── scripts/build_index.py # CLI: rebuild data/ after editing site content (what build.py calls)
├── scripts/run_eval.py    # CLI: score the golden set against data/eval_baseline.json
├── eval/                  # golden.jsonl (held-out cases) + README.md (authoring guide)
├── tests/                 # pytest suite
├── data/                  # generated: chunks_e5.json (vectors), gate_en_minilm.json +
│                          #   gate_zh_bge.json (off-topic gate vectors), chunks_en_minilm.json
│                          #   (degraded-mode retrieval), meta.json (small sidecar the widget
│                          #   fetches instead of the full chunks file), roles.json (personas)
├── models/                # self-hosted MiniLM ONNX (weights + tokenizer)
└── functions/tencent/     # Tencent SCF backend + build_package.py (what build.py --function calls)
../scripts/chat-widget.js  # the site-side widget (self-contained, no framework)
../scripts/vendor/         # self-hosted transformers.min.js + ONNX Runtime WASM
```

## Rebuilding the index

Run after editing any site page. `build.py` is the one-call master build — it wraps
`scripts/build_index.py` (and, with `--function`, `functions/tencent/build_package.py` too) so
you never have to remember which script does what:

```bash
cd chat
pip install -e ".[dev]"
python build.py                # rebuild data/ (chunks/gate/meta/roles) -- fast, no network needed
python build.py --function     # ALSO rebuild the Tencent SCF zip -- downloads any missing
                                #   models/wheels first; see "Deploying a backend" below for
                                #   what to do with the zip afterward
pytest -q   # chunker contract, loader, embedder parity, index schema, gate, eval harness
```

Without `--function`, `build.py` just runs `scripts/build_index.py --model <preset>` (default
preset `e5`, matching production). With `--function` it instead runs
`functions/tencent/build_package.py`, which downloads any missing models/wheels, calls
`scripts/build_index.py` itself as one of its steps, and writes
`functions/tencent/tencent-function-<preset>.zip`. Call `scripts/build_index.py` directly only
if you specifically want the site artifacts without `build.py`'s summary/next-steps output —
everything below about what gets printed and what to commit applies either way, since
`build.py` doesn't change what `build_index.py` writes.

`build_index.py` prints a one-line summary (page/section/chunk counts, chunks-file
size, gate threshold, elapsed seconds) — the numbers depend on current site
content and the active `model_preset`, so a captured transcript would go
stale the next time either changes and isn't reproduced here. The committed
`data/chunks_e5.json` records its own `built_at`/`model_preset`/chunk count for
whatever was last built (also mirrored, without the chunks array, in `data/meta.json`);
`eval/KNOWN_ISSUES.md` has the current build's actual stats and a case study in what
happens when this file is left stale.

Commit the regenerated `data/chunks_e5.json`, `data/gate_en_minilm.json`,
`data/chunks_en_minilm.json` and `data/meta.json` (the small metadata sidecar —
`gate_threshold`/`gate_stat`/`gate_remote`/`model`/`query_prefix`/`chunks_file`/etc. —
the widget fetches on every load instead of the full chunks file; see "Why it's built
this way" above). `data/gate_zh_bge.json` is gitignored and never committed. Chunk ids
are deterministic (`{url}#{anchor}:{i}`), so diffs stay readable.

If you ran `python build.py --function`, upload and redeploy the zip **before** committing
and publishing the site — see "Deploying a backend" below for why the order matters and the
exact steps; it is the reverse of the order you'd guess.

## Evaluating retrieval and the gate (golden set)

`eval/golden.jsonl` is a held-out measurement set, separate from both the
site content (`../pages/*.html`) and the gate's own fit-on calibration data
(`gate_calibration.py`) — see `eval/README.md` for the full authoring guide.
It holds 126 cases:

- **96 positives** — 12 per `(role, lang)` cell, one cell for each of the 4
  roles in `data/roles.json` crossed with `en`/`zh`. Scored for gate-pass,
  `hit@4` (did the expected page land in the top 4?) and keyword coverage
  (did the answer-bearing text actually make it into the retrieved chunks?).
- **28 shared negatives** — one pool per language (not per role: the gate
  never sees the role), 14 cases each: 4 `off_topic`/`"easy"`,
  4 `off_topic`/`"adjacent"`, 4 `injection`, and 2 post-context (`off_topic`
  with `history` attached — see below).
  - **`easy`** off-topic probes are obviously unrelated to the site (weather,
    recipes, tax law) — a refusal failure here means the gate itself is
    broken.
  - **`adjacent`** off-topic probes are plausibly related but still
    off-domain (game-industry news, a different studio's postmortem) — a
    refusal failure here means the gate can't separate *this* domain from
    things that merely resemble it. Reporting the two separately is the
    point: a single blended refusal number can't tell these apart.
- **2 multi-turn positives (`en` only) plus the 4 post-context negatives
  above** exercise the follow-up-resolution path described under "Off-topic
  use is refused three times over" — a case with `history` that the raw
  question fails the gate on its own, so the escalated rewrite-and-re-gate
  path actually fires. These are counted separately from the 96/28 above
  (`n_followup`/`followup_rescued` and the `post_context` negative bucket,
  not `n_positive`/`gate_pass`), so they never move those cells' denominators
  just by existing. `en`-only is a measured property of the current zh gate
  calibration, not a coverage gap — see `eval/README.md`'s "Multi-turn cases"
  section for the full authoring rules and `KNOWN_ISSUES.md` Finding Q for
  why zh has none: virtually any well-formed Chinese question about 他 clears
  the zh gate standalone regardless of whether it depends on prior turns
  (measured 0.42–0.56 against the zh threshold across ~90 candidate
  phrasings), so a zh follow-up never reaches the rewriter to be measured,
  and `FOLLOWUP_POSITIVES_PER_LANG = {"en": 2, "zh": 0}` records that as a
  checked zero rather than an oversight. Revisit if the zh gate is ever
  recalibrated tighter — Finding Q names the exact trigger.

```bash
python scripts/run_eval.py                          # positive table + shared negatives block
python scripts/run_eval.py --role visitor --lang zh --verbose
python scripts/run_eval.py --update-baseline         # after a deliberate, reviewed change
pytest tests/test_golden.py                          # fails on any regression vs data/eval_baseline.json
```

**The rewrite fixture (`eval/rewrites.json`) keeps multi-turn scoring offline.**
`scripts/refresh_rewrites.py` is the only thing in this repo that calls a real LLM for a
follow-up rewrite — `run_eval.py` and `pytest` never do; they replay whatever
`eval/rewrites.json` has recorded, which is what keeps the harness deterministic and
network-free. Re-run it, and read the diff before committing, whenever
`REWRITE_SYSTEM_PROMPT` (`functions/tencent/index.py`) changes or a multi-turn case is
added or edited — a stale fixture reports numbers that describe neither the old prompt nor
the new one, and now that these metrics are gated (below), a stale fixture can silently
invalidate the regression gate itself rather than just mis-describing it. A case with no
recorded rewrite doesn't error; it silently scores as still-refused (an absent measurement,
never a fabricated pass — see `load_rewrites`'s docstring), which is why `run_eval.py`
renders that cell's `rescued`/`post_context` column as `n/a` rather than a count (see
`_fixture_coverage`).

`eval/rewrites.json` is recorded, against the live DeepSeek API — both multi-turn positives
resolved and all four post-context negatives echoed verbatim (see the file itself for the
exact rewrites). `followup_rescued` and `refusal_post_context` are consequently **wired
into** `tests/test_golden.py::_REGRESSION_METRICS` (and, for the reasons detailed at that
constant's declaration, `_GATE_METRICS` too), and `data/eval_baseline.json` carries their
real measured values. Do not trust the numbers quoted in any prose here, in this file or
`eval/README.md` — read them straight from `data/eval_baseline.json` (`n_followup`/
`followup_rescued` per positive cell, `n_post_context`/`refusal_post_context` per shared
cell) or from `run_eval.py`'s own `follow-up resolution` block and `post_context` column,
since both move the moment the fixture or the golden set changes and a stale quote here
would not.

```bash
python scripts/refresh_rewrites.py --dry-run   # review before writing
python scripts/refresh_rewrites.py             # re-record eval/rewrites.json
```

`--update-baseline` refuses to write from a run that cannot represent the
system: a filtered run (`--role`/`--lang`), or one where the committed chunks
file's knowledge headings no longer match `chat/knowledge/about_*.md` on disk
(rebuild the index first — `--allow-stale-index` is the deliberate override).

Run everything from `chat/`, with the interpreter that has `portfolio_rag`
installed editable — `env_file=".env"` in `config.py` resolves relative to
the working directory, so running from the repo root silently drops
`RAG_MODEL_PRESET` back to its `minilm` default and every score becomes
meaningless. This harness is deterministic: the same index and gate vectors
produce bit-identical numbers on every run.

A gate that isn't available on the machine (`data/gate_zh_bge.json` absent, e.g. no
Chinese gate on a fresh clone that hasn't rebuilt — see Known Limits in
`eval/README.md`) reports that cell's gate metrics as `n/a`, never as `0%`.
`hit@4` and keyword coverage are gate-free and always report a real number.

For the consolidated findings from the most recent run — what's fixed, what's
outstanding, and every right-page/wrong-chunk case — see
[`eval/KNOWN_ISSUES.md`](eval/KNOWN_ISSUES.md).

## Previewing locally

Browsers block module imports and `fetch()` on `file://` pages, so opening
`index.html` directly disables the chat (the widget explains this instead of
erroring). Preview over HTTP from the repo root:

```bash
python -m http.server 8000   # then open http://localhost:8000
```

## Deploying a backend (enables LLM answers)

**Redeploying the existing Tencent SCF backend** (console credentials in hand — the upload
itself is a manual console step, not automated here, since it needs your Tencent login
rather than a token this repo could hold):

```bash
cd chat && python build.py --function   # rebuilds data/ AND functions/tencent/tencent-function-e5.zip
```

Then, in order: **(1) upload the new zip in the SCF console and redeploy the function
FIRST**, confirm it's live, **(2) only then** `git add/commit/push chat/data/` and publish the
site. Reversed, every visitor loses the LLM answer and falls to the widget's offline search
path (a ~23 MB in-browser model) until the function catches up — see the load-bearing-order
callout below for the exact mechanism, and `.claude/DEPLOY.md` (gitignored, local-only) for
the step-by-step console walkthrough.

The site works without this step — the widget stays in retrieval-only mode until
`WORKER_URL` is set. The only backend is **Tencent SCF (chosen for China
reachability + DeepSeek):** sources in `functions/tencent/`; the step-by-step
console guide is kept locally in `.claude/DEPLOY.md` (gitignored, not
published). Speaks the current `/chat` contract (retrieves server-side, no
`contexts` in the request).

> **Deployment order is load-bearing (Task 29).** The widget's `/chat` request no
> longer sends `contexts`; the OLD deployed Tencent function's `validate_chat_body`
> requires it and 400s without it. **Redeploy `functions/tencent/` (rebuild
> `tencent-function-e5.zip` and upload it) before publishing a site build that
> includes this widget change.** The reverse direction is safe: the NEW function
> still accepts an old cached widget's `contexts` field, ignoring it and logging
> `client_contexts_ignored` — so redeploying the function first, ahead of the site,
> never breaks an already-loaded page.

> **Follow-up resolution (this branch) is additive in both directions — hygiene, not
> a hard requirement like the ordering above.** A NEW widget's `gate_failed: true`
> reaching an OLD, un-redeployed function is silently ignored: the old function's
> request handling has never heard of that field, so the turn proceeds as an
> ordinary, context-free `/chat` call always has, and the widget's `refused_by`
> falls back to its pre-this-branch literal (`server_retrieval_empty`) since the old
> function's refusal response carries no `reason`. An OLD widget, symmetrically,
> never sends `gate_failed` at all, so a NEW function never sees it and never takes
> the escalated path. Redeploy the function first regardless — the ordering rule
> above governs `contexts`, not this field, and still applies for its own reasons.

> **The deployed Tencent SCF package must be rebuilt and redeployed before any
> gate change in this repo reaches production.** `functions/tencent/index.py`
> never reads `chat/data/gate_en_minilm.json`/`chat/data/gate_zh_bge.json` — it
> reads its own bundled copies (packaged as `gate_en_minilm.json`/`gate_zh_bge.json`
> inside the zip too, as of Task 29 Part 2 — see `build_package.py`),
> zipped into `functions/tencent/tencent-function-e5.zip` by
> `functions/tencent/build_package.py`. That zip is untouched by this branch
> (or any branch that only edits `chat/knowledge/`, `chat/src/`, `chat/data/`)
> and still contains whatever gate was packaged at its last build — including,
> for a zip built before Task 29 Part 2, an internal `gate_vectors.json` under
> the OLD combined shape (that older package still runs the SAME gate
> semantics; only the bundled file's own name/shape changed on the packaging
> side, not the gate logic itself). Concretely,
> as of this writing the deployed zip carries `build_id e5-20260721-3de2a80`
> (2026-07-21): an English gate of 149 raw page-chunk vectors at threshold
> `0.2312` (the pre-Task-24 corpus this repo's own `data/gate_en_minilm.json` no
> longer matches), **and a live Chinese gate at threshold `0.4919`** —
> `chat-widget.js` trusts `state.meta.gate_remote === true` for an e5 build
> and always defers to this server-side decision, so that Chinese gate is what
> actually runs for visitors today, independent of whatever this repo's own
> `eval/` harness currently reports for `chat/data/gate_zh_bge.json`. Until
> someone re-runs `build_package.py` and
> redeploys, **production runs the old gate, on both languages** — this
> repo's gate numbers describe
> `chat/data/`, not what visitors hit. See `eval/KNOWN_ISSUES.md`'s "Critical"
> section for the full writeup, including a compatibility check confirming
> `index.py` needs no code change to consume the new gate file shape
> — a repackage-and-redeploy is a pure win, not a risky one.

This gap is the concrete origin of the bug follow-up resolution fixes. The question that
motivated the feature, "is there any optimization work," measures **0.2133** — below the
deployed gate's threshold quoted above (`0.2312`, so it refuses in production) and above
this repo's own recalibrated one (`0.2029`, so it passes here, standalone, with no history
needed at all). Once the function above is repackaged and redeployed, that specific
symptom stops reproducing on its own; follow-up resolution still
earns its place for whatever now sits closer to the (new, lower) live threshold — a rewrite
that pulls a question further below the gate than the raw text sat can still get refused
without it, deployed gate or not.

Deploying the function for the first time (new SCF function, not a redeploy) is a
console-driven process — see `.claude/DEPLOY.md` (gitignored, local-only) for the
full walkthrough, including secrets and rate limits. Once it's live, point the
widget at it: in `scripts/chat-widget.js` set

```js
var WORKER_URL = 'https://<函数URL>';
```

using the URL shown on the function's 函数URL page in the SCF console.

## Model provenance

Three self-hosted, quantized ONNX models, each with a distinct job (see "Why it's built
this way" above for how they fit together):

- **Retrieval** — `models/Xenova/multilingual-e5-small/`, the standard transformers.js
  export of [`Xenova/multilingual-e5-small`](https://huggingface.co/Xenova/multilingual-e5-small)
  (an ONNX/quantized port of `intfloat/multilingual-e5-small`), 384-dim, mean pooling,
  asymmetric `query: `/`passage: ` prefixes, ~130 MB on disk. Document vectors are
  embedded once at build time (`data/chunks_e5.json`); query vectors are embedded
  server-side by the deployed Tencent function's `/embed` endpoint, since e5 is too
  large to self-host in the browser.
- **English gate (`data/gate_en_minilm.json`) + degraded-mode retrieval
  (`data/chunks_en_minilm.json`)** — `models/Xenova/all-MiniLM-L6-v2/`, the
  standard transformers.js export of
  [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
  (Apache-2.0), 384-dim, dynamically quantized, ~23 MB — small enough to self-host and
  run client-side via transformers.js/WASM. `scripts/vendor/` holds transformers.js
  2.17.2 and its ONNX Runtime WASM, so this path has zero runtime dependencies outside
  this repository and the (optional) backend function.
- **Chinese gate (`data/gate_zh_bge.json`, gitignored)** — `models/Xenova/bge-small-zh-v1.5/`,
  the standard transformers.js
  export of [`Xenova/bge-small-zh-v1.5`](https://huggingface.co/Xenova/bge-small-zh-v1.5)
  (an ONNX/quantized port of `BAAI/bge-small-zh-v1.5`), 512-dim, CLS pooling, ~24 MB.
  Calibrated against `knowledge/about_zh.md` on every build but only shipped when
  calibration actually separates on-/off-topic scores (as of the most recent local
  rebuild, it does — read the real margin from `data/gate_zh_bge.json`'s own
  `gate_margin`, or `run_eval.py`'s printed gate-calibration table, rather than
  trusting a number quoted here; calibration quality moves with `about_zh.md`'s
  content). This model is server-side only — see "gate_zh_bge.json is never served
  to a browser" — and whether it's actually live for visitors depends on when the
  deployed Tencent SCF package was last rebuilt (see "Deploying a backend" below).
