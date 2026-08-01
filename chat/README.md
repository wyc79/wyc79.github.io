# Portfolio Chat Agent — client-side RAG on GitHub Pages

A role-aware AI chat agent for [wyc79.github.io](https://wyc79.github.io), built as a
demonstration of a full RAG pipeline that fits inside a static site:

- **Write path (this package, Python):** site HTML → section loader → sliding-window
  chunker → ONNX embeddings → `data/index.json`, a static vector index served by
  GitHub Pages.
- **Read path (browser, `../scripts/chat-widget.js`):** the visitor picks a role
  (recruiter for game client dev / game AI & agents / combat design, or visitor —
  each mirrors a real campus-hiring JD), their question is
  embedded **in the browser** with transformers.js, retrieval is a dot product over the
  static index, and the top chunks are rendered as linked source cards.
- **Generation (optional, `worker/`):** a Cloudflare Worker holds the Anthropic API key
  and turns retrieved chunks into grounded answers. Without it the widget runs in
  retrieval-only demo mode — still useful, still fully client-side.

```
build time (python)                     visit time (browser)
site *.html                             question + role
  └─ loader.py      sections              └─ transformers.js (WASM, self-hosted)
  └─ chunker.py     800/100 overlap       └─ dot product vs index.json  → source cards
  └─ embedder.py    MiniLM ONNX           └─ POST /chat → Cloudflare Worker → Claude
  └─ index_builder  data/index.json            (holds API key; logs every turn)
```

## Why it's built this way

**Retrieval is client-side.** GitHub Pages can only serve static files, but retrieval
doesn't need a server: the committed index (`data/index.json`, currently 226 chunks ×
384 dims across 16 pages / 132 sections) is a ~1.05 MB JSON file and a brute-force dot
product over it runs in well under a millisecond in the browser. No vector DB to host,
nothing to pay for, and the retrieval layer stays inspectable (open DevTools, watch the
scores).

**One retrieval model, one gate model, both self-hosted quantized ONNX.** Document
vectors are precomputed at build time by `multilingual-e5-small` (e5, 384-dim, bilingual
en+zh — `RAG_MODEL_PRESET=e5`, see `.env.example`); the deployed Tencent function embeds
each query with the *same* model at `/embed`, so both sides of the retrieval dot product
stay in one embedding space. e5 is too large to self-host in the browser (~130 MB on
disk), so a second, much smaller model — `all-MiniLM-L6-v2` (~23 MB, self-hosted at
`models/Xenova/all-MiniLM-L6-v2/`, cached by the browser after first load) — runs
entirely client-side via transformers.js/WASM and does two jobs: the local off-topic
gate (below) and the degraded-mode retrieval fallback (`data/fallback_vectors.json`)
used when the Worker/function is unreachable. Both models run the same mean-pooling +
L2-normalize recipe on the Python (onnxruntime) and JS (transformers.js) sides. Measured
cross-runtime parity for the MiniLM path: cosine(browser vector, python vector) ≈ 0.99 —
native vs WASM int8 kernels round differently — with top-4 retrieval overlap of 3–4/4 on
test queries. Documents are embedded one at a time, unpadded, because padded batches
shift the dynamic-quantization scales and would break this parity.

**The only secret lives in the Worker.** Anything shipped to a static site is public,
so the Anthropic key sits in a Cloudflare Worker secret. The client sends a role *id*,
never prompt text — the Worker reads `data/roles.json` from the site itself, so prompts
can't be injected through the API surface. The Worker validates and size-caps every
field, checks the `Origin` header, caps `max_tokens`, and (with the optional KV
binding) rate-limits per IP.

**Off-topic use is refused three times over.** The chat is not a general assistant:
(1) the widget gates on retrieval score — if the best chunk scores below the
threshold, it refuses locally and re-suggests role-specific questions without any
LLM call. The threshold is not hardcoded: build_index calibrates it per embedding
model from canonical on-/off-topic query sets (src/portfolio_rag/gate_calibration.py).
Because e5 compresses cosines and can't separate on-/off-topic itself, gating is
delegated to a MiniLM copy of the chunk vectors (`data/gate_vectors.json`, mirrored
into `index.json`'s `gate_threshold` for reference); the current build's English gate
calibrates to `0.2536` with a small **negative** margin (off-topic max sits *above*
on-topic min by 1.3%), so build_index ships it anyway with a warning rather than
silently disabling it — expect an occasional on-topic question to score just below
threshold and get refused. A second gate, `bge-small-zh-v1.5` over the hand-written
`knowledge/about_zh.md` corpus, is calibrated on every build but only shipped if it
actually separates the distributions; on the current build it doesn't (margin -0.4%),
so no Chinese gate ships and Chinese questions bypass the local gate entirely
(name-blind `cjk_bypass`), relying on the system prompt's off-topic instruction alone.
The gate is name-blind: mentioning "Yuanchen Wang" inflates similarity (a
name-dropped joke request scores 0.61), so name-bearing questions are gated on the
question with the name stripped out, unless the remainder is a bio-intent stub
("who is", "tell me about", empty) — those are genuinely about YC and pass; (2) the Worker independently refuses empty-context requests, so bypassing the
widget doesn't buy anything; (3) the system prompt instructs the model to decline
general-purpose requests and ignore instruction-injection in questions. Pages'
`<meta name="description">` tags are indexed as summary chunks so broad-but-legitimate
questions ("who is YC") clear the gate comfortably.

**Everything is logged.** Each turn (input, retrieved chunk ids + scores, output) goes
to `console.debug`, to Google Analytics as a `chat_turn` event when available, and
server-side via the Worker (`wrangler tail` live; 30-day KV persistence when bound).

**Curated chunks bridge vocabulary gaps.** Visitors ask in hiring vocabulary the
pages never use ("resume highlights", "CV", "qualifications") — without help those
queries score below the off-topic gate. `knowledge/*.md` holds short authored
summaries (each `## Heading` block is one chunk, `link:` sets its source card)
indexed alongside the scraped pages. Keep facts consistent with the site and
rebuild after editing.

## Layout

```
chat/
├── src/portfolio_rag/     # the pipeline: config, loader, chunker, embedder, roles, index_builder
├── knowledge/             # curated .md chunks (resume/CV vocabulary the pages lack)
├── scripts/build_index.py # CLI: rebuild data/ after editing site content
├── scripts/run_eval.py    # CLI: score the golden set against data/eval_baseline.json
├── eval/                  # golden.jsonl (held-out cases) + README.md (authoring guide)
├── tests/                 # pytest suite
├── data/                  # generated: index.json (vectors), roles.json (personas)
├── models/                # self-hosted MiniLM ONNX (weights + tokenizer)
└── worker/                # Cloudflare Worker (LLM proxy + logging) + wrangler.toml
../scripts/chat-widget.js  # the site-side widget (self-contained, no framework)
../scripts/vendor/         # self-hosted transformers.min.js + ONNX Runtime WASM
```

## Rebuilding the index

Run after editing any site page:

```bash
cd chat
pip install -e ".[dev]"
python scripts/build_index.py
pytest -q   # chunker contract, loader, embedder parity, index schema, gate, eval harness
```

`build_index.py` prints a one-line summary (page/section/chunk counts, index
size, gate threshold, elapsed seconds) — the numbers depend on current site
content and the active `model_preset`, so a captured transcript would go
stale the next time either changes and isn't reproduced here. The committed
`data/index.json` records its own `built_at`/`model_preset`/chunk count for
whatever was last built; `eval/KNOWN_ISSUES.md` has the current build's
actual stats and a case study in what happens when this file is left stale.

Commit the regenerated `data/index.json`. Chunk ids are deterministic
(`{url}#{anchor}:{i}`), so diffs stay readable.

## Evaluating retrieval and the gate (golden set)

`eval/golden.jsonl` is a held-out measurement set, separate from both the
site content (`../pages/*.html`) and the gate's own fit-on calibration data
(`gate_calibration.py`) — see `eval/README.md` for the full authoring guide.
It holds 120 cases:

- **96 positives** — 12 per `(role, lang)` cell, one cell for each of the 4
  roles in `data/roles.json` crossed with `en`/`zh`. Scored for gate-pass,
  `hit@4` (did the expected page land in the top 4?) and keyword coverage
  (did the answer-bearing text actually make it into the retrieved chunks?).
- **24 shared negatives** — one pool per language (not per role: the gate
  never sees the role), 12 cases each split 4 `off_topic`/`"easy"`,
  4 `off_topic`/`"adjacent"`, 4 `injection`.
  - **`easy`** off-topic probes are obviously unrelated to the site (weather,
    recipes, tax law) — a refusal failure here means the gate itself is
    broken.
  - **`adjacent`** off-topic probes are plausibly related but still
    off-domain (game-industry news, a different studio's postmortem) — a
    refusal failure here means the gate can't separate *this* domain from
    things that merely resemble it. Reporting the two separately is the
    point: a single blended refusal number can't tell these apart.

```bash
python scripts/run_eval.py                          # positive table + shared negatives block
python scripts/run_eval.py --role visitor --lang zh --verbose
python scripts/run_eval.py --update-baseline         # after a deliberate, reviewed change
pytest tests/test_golden.py                          # fails on any regression vs data/eval_baseline.json
```

Run everything from `chat/`, with the interpreter that has `portfolio_rag`
installed editable — `env_file=".env"` in `config.py` resolves relative to
the working directory, so running from the repo root silently drops
`RAG_MODEL_PRESET` back to its `minilm` default and every score becomes
meaningless. This harness is deterministic: the same index and gate vectors
produce bit-identical numbers on every run.

A gate that isn't available on the machine (`data/gate_vectors.json` missing
a language, e.g. no Chinese gate today — see Known Limits in
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

The site works without this step — the widget stays in retrieval-only mode until
`WORKER_URL` is set. Two interchangeable backends exist:

- **Tencent SCF (chosen for China reachability + DeepSeek):** sources in
  `functions/tencent/`; the step-by-step console guide is kept locally in
  `.claude/DEPLOY.md` (gitignored, not published).
- **Cloudflare Worker (Anthropic API):** below.

```bash
cd chat/worker
npx wrangler login
npx wrangler secret put ANTHROPIC_API_KEY     # paste your key; never committed
npx wrangler deploy                            # prints https://portfolio-chat.<acct>.workers.dev
```

Optional but recommended — logging persistence + per-IP rate limiting:

```bash
npx wrangler kv namespace create CHAT_KV       # then uncomment the binding in wrangler.toml
npx wrangler deploy
```

Finally, point the widget at the Worker: in `scripts/chat-widget.js` set

```js
var WORKER_URL = 'https://portfolio-chat.<acct>.workers.dev';
```

Watch live logs with `npx wrangler tail`. Model and origin allowlist are plain vars in
`wrangler.toml`.

## Model provenance

Three self-hosted, quantized ONNX models, each with a distinct job (see "Why it's built
this way" above for how they fit together):

- **Retrieval** — `models/Xenova/multilingual-e5-small/`, the standard transformers.js
  export of [`Xenova/multilingual-e5-small`](https://huggingface.co/Xenova/multilingual-e5-small)
  (an ONNX/quantized port of `intfloat/multilingual-e5-small`), 384-dim, mean pooling,
  asymmetric `query: `/`passage: ` prefixes, ~130 MB on disk. Document vectors are
  embedded once at build time (`data/index.json`); query vectors are embedded
  server-side by the deployed Tencent function's `/embed` endpoint, since e5 is too
  large to self-host in the browser.
- **English gate + degraded-mode fallback** — `models/Xenova/all-MiniLM-L6-v2/`, the
  standard transformers.js export of
  [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
  (Apache-2.0), 384-dim, dynamically quantized, ~23 MB — small enough to self-host and
  run client-side via transformers.js/WASM. `scripts/vendor/` holds transformers.js
  2.17.2 and its ONNX Runtime WASM, so this path has zero runtime dependencies outside
  this repository and the (optional) Worker.
- **Chinese gate** — `models/Xenova/bge-small-zh-v1.5/`, the standard transformers.js
  export of [`Xenova/bge-small-zh-v1.5`](https://huggingface.co/Xenova/bge-small-zh-v1.5)
  (an ONNX/quantized port of `BAAI/bge-small-zh-v1.5`), 512-dim, CLS pooling, ~24 MB.
  Calibrated against `knowledge/about_zh.md` on every build but only shipped when
  calibration actually separates on-/off-topic scores — not currently the case (see
  "Off-topic use is refused three times over" above), so this model isn't in the
  current deployment's request path.
