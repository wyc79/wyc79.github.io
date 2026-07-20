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
doesn't need a server: 123 chunks × 384 dims is a ~570 KB JSON file and a brute-force
dot product runs in well under a millisecond. No vector DB to host, nothing to pay for,
and the retrieval layer stays inspectable (open DevTools, watch the scores).

**One embedding space, one model file.** The build pipeline (Python/onnxruntime) and
the widget (transformers.js/WASM) run the *same* quantized ONNX file, self-hosted at
`models/Xenova/all-MiniLM-L6-v2/` — mean pooling + L2 normalize on both sides.
Self-hosting (~23 MB, cached by the browser after first load) means no third-party
model CDN at runtime and identical behavior everywhere.
Measured cross-runtime parity: cosine(browser vector, python vector) ≈ 0.99 — native
vs WASM int8 kernels round differently — with top-4 retrieval overlap of 3–4/4 on test
queries. Documents are embedded one at a time, unpadded, because padded batches shift
the dynamic-quantization scales and would break this parity.

**The only secret lives in the Worker.** Anything shipped to a static site is public,
so the Anthropic key sits in a Cloudflare Worker secret. The client sends a role *id*,
never prompt text — the Worker reads `data/roles.json` from the site itself, so prompts
can't be injected through the API surface. The Worker validates and size-caps every
field, checks the `Origin` header, caps `max_tokens`, and (with the optional KV
binding) rate-limits per IP.

**Off-topic use is refused three times over.** The chat is not a general assistant:
(1) the widget gates on retrieval score — if the best chunk scores below 0.22
(calibrated: on-topic questions measure ≥ ~0.30, jokes/homework/translation requests
≤ ~0.25), it refuses locally and re-suggests role-specific questions without any LLM
call. The gate is name-blind: mentioning "Yuanchen Wang" inflates similarity (a
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
├── tests/                 # pytest suite (20 tests)
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
#  pages  sections  chunks  index_kb  seconds
#     15        56     123     569.1     1.60
pytest -q   # 20 tests: chunker contract, loader, embedder parity, index schema
```

Commit the regenerated `data/index.json`. Chunk ids are deterministic
(`{url}#{anchor}:{i}`), so diffs stay readable.

## Previewing locally

Browsers block module imports and `fetch()` on `file://` pages, so opening
`index.html` directly disables the chat (the widget explains this instead of
erroring). Preview over HTTP from the repo root:

```bash
python -m http.server 8000   # then open http://localhost:8000
```

## Deploying the Worker (enables LLM answers)

The site works without this step — the widget stays in retrieval-only mode until
`WORKER_URL` is set.

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

`models/Xenova/all-MiniLM-L6-v2/` is the standard transformers.js export of
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
(Apache-2.0), 384-dim, dynamically quantized. `scripts/vendor/` holds transformers.js
2.17.2 and its ONNX Runtime WASM, so the demo has zero runtime dependencies outside
this repository and the (optional) Worker.
