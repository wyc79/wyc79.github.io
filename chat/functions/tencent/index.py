"""Tencent SCF web function — the LLM backend for the portfolio chat agent.

Zero-dependency (stdlib only) port of chat/worker/worker.js, speaking the
OpenAI-compatible chat-completions protocol (DeepSeek by default, or any
provider that exposes /v1/chat/completions). Listens on :9000 as SCF web
functions require; scf_bootstrap starts it.

Same contract and guarantees as the Cloudflare worker:
  POST /chat  {session, role, lang, question, history?} -> {answer, model, rid, sources[]}
  POST /embed {session?, text, gate_text?, gate_only?} -> {vector?, gate?, rid}
  POST /log   client-side event record -> 204
  GET  /      health
- Origin allowlist, size caps on every field.
- Retrieval (Task 29): /chat retrieves from its own bundled chunks.json —
  the client no longer sends contexts (a `contexts` field on an old cached
  widget is ignored outright, logged as client_contexts_ignored, never used
  to build the prompt). Retrieval returning nothing -> canned refusal, no LLM
  call (server-side off-topic guard, now independent of what the client
  sends).
- Role prompts come from the site's roles.json (client sends a role id only);
  a bundled roles.json copy is the fallback if github.io is unreachable from
  the function's region.
- Bundled data (Task 29 Part 2 file split — see chat/README.md): chunks.json
  (the retrieval corpus, packaged from chat/data/chunks_{preset}.json),
  gate_en_minilm.json and gate_zh_bge.json (the per-language off-topic gate
  vectors, packaged from the identically-named chat/data/ files). Each is a
  single-purpose file — a gate file carries no chunk records, chunks.json
  carries no gate fields — mirroring the split on the chat/data/ side.
- Every turn logged as JSON (SCF 日志查询), keyed by a short human-scannable
  request id (rid) + hashed session (sid). /chat logs the exact LLM input
  (system_head + clipped chunks with ids/scores + messages) and output
  (answer, finish_reason, usage) — the chunks and their scores now come from
  this function's own retrieval, not client-sent contexts; /embed logs the
  gate decision only, never the query vector. The widget can still POST its
  own client-side top-k {id, score} to /log (degraded/light-mode audit) — all
  correlate by rid/sid.
Config via environment variables: LLM_API_KEY (required), LLM_BASE_URL,
LLM_MODEL, SITE_BASE, ALLOWED_ORIGINS, RATE_LIMIT_PER_HOUR.
"""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# Build identity: build_package.py stamps build_info.json into the package at
# build time; index.py reads it here so every log line carries build_id and the
# health check reports the full stamp. Answers "which build is running?" (the
# stale-zip ambiguity that made the zh-answer bug hard to see) at a glance.
def _load_build_info() -> dict:
    try:
        return json.loads(Path(__file__).with_name("build_info.json").read_text(encoding="utf-8"))
    except Exception:
        return {"build_id": "unknown", "built_at": None}


BUILD_INFO = _load_build_info()
BUILD_ID = BUILD_INFO.get("build_id", "unknown")


LIMITS = {
    "question": 1000,
    "history_turns": 8,
    "history_text": 1200,
    "log_bytes": 4096,
    # Log-only clip lengths. CLS (日志服务) truncates very long lines, so logs
    # keep the exact LLM input/output but clip free text — ids/scores/usage
    # stay full so retrieval quality and token accounting are never lost.
    "log_ctx_text": 200,
    "log_msg_text": 240,
    "log_answer": 800,
}

# Human-scannable request id: MMDD-HHMMSS-NNNN. The 4-digit per-instance counter
# makes it collision-free within a process (SCF may run several instances, so
# ids are unique per instance — pair with sid to disambiguate across them).
_req_counter = [0]
_req_lock = threading.Lock()

_roles_cache: dict = {"data": None, "ts": 0.0}
_rate: dict = {"hour": None, "counts": {}}
_rate_lock = threading.Lock()

# ── optional server-side embedding + gating (stage 2) ─────────────────────
# If MODEL_DIR exists in the package (an ONNX sentence-embedding model, e.g.
# Xenova/multilingual-e5-small), POST /embed serves query vectors. Without it
# /embed returns 503 and the widget falls back to its in-browser model.
# If GATE_MODEL_DIR + gate_en_minilm.json are also packaged, /embed
# additionally answers the off-topic gate with the GATE model (MiniLM: e5
# can't separate on/off-topic — its cosines compress into one band).
_embed = {"lock": threading.Lock(), "session": None, "tokenizer": None, "error": None,
          "prefix": "", "pooling": "mean"}
# Two gate bundles: "en" (MiniLM vs knowledge/about_en.md, packaged from
# gate_en_minilm.json) and optionally "zh" (bge-zh vs knowledge/about_zh.md,
# packaged from gate_zh_bge.json when it's present -- see build_package.py).
# CJK queries use zh when available, otherwise bypass to the LLM-prompt guard.
_gates: dict = {"en": None, "zh": None}

# ── retrieval (Task 29) ────────────────────────────────────────────────────
# chunks.json, bundled from chat/data/chunks_{preset}.json next to
# gate_en_minilm.json/gate_zh_bge.json (see build_package.py). /chat
# retrieves from this directly instead of trusting client-sent contexts --
# see the module docstring. matrix is None for EVERY state in which retrieval
# cannot work (not packaged, failed to load, preset mismatch, zero chunks), so
# the single check `_index["matrix"] is None` is a complete availability test
# for /chat's guard and GET /'s health flag. WHICH failure it was lives in
# _index["error"], which the health payload also reports -- an earlier version
# distinguished "packaged but genuinely empty" by loading it as an empty
# matrix, which is not None and therefore reported itself as working.
_index: dict = {"matrix": None, "chunks": None, "error": None}

# Mirrors chat-widget.js's TOP_K/MIN_SCORE and portfolio_rag.runtime's same
# constants -- see rank_chunks below and chat/tests/test_retrieval_sync.py,
# which executes all three implementations against one shared fixture.
TOP_K = 4
MIN_SCORE = 0.18

CJK_RE = None  # compiled lazily


def _load_model(dir_name: str) -> tuple:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(__file__).parent / dir_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{dir_name} not packaged")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=int(env("EMBED_MAX_TOKENS", "256")))
    # Deterministic session. Dynamically-quantized MatMul is otherwise
    # nondeterministic run-to-run: with several quantized models loaded (embed +
    # en/zh gates), CPU memory-arena reuse can corrupt a few activations and push
    # a gate score across the threshold (a rare false refusal). One short query
    # per request, so single-threaded + no arena costs nothing.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    session = ort.InferenceSession(
        str(model_dir / "onnx" / "model_quantized.onnx"), so, providers=["CPUExecutionProvider"]
    )
    return tok, session


def _load_embedder() -> None:
    try:
        tok, session = _load_model(env("MODEL_DIR", "model"))
        with _embed["lock"]:
            _embed["tokenizer"], _embed["session"] = tok, session
            _embed["prefix"] = env("QUERY_PREFIX", "")
        log({"type": "embedder_loaded"})
    except Exception as err:
        _embed["error"] = repr(err)
        log({"type": "embedder_load_failed", "error": repr(err)})
        return

    # Task 29 Part 2: one gate file per language (gate_en_minilm.json,
    # gate_zh_bge.json), each already a complete gate spec on its own -- no
    # more combined gate_vectors.json with per-language keys to unpack.
    import numpy as np

    dirs = {"en": env("GATE_MODEL_DIR", "gate_model"), "zh": env("GATE_MODEL_ZH_DIR", "gate_model_zh")}
    files = {"en": "gate_en_minilm.json", "zh": "gate_zh_bge.json"}
    for lang in ("en", "zh"):
        gate_file = Path(__file__).with_name(files[lang])
        if not gate_file.exists():
            continue
        try:
            # The JSON read belongs INSIDE this try. It used to sit above it,
            # so a truncated or malformed gate file raised JSONDecodeError out
            # of _load_embedder() -- which main() calls BEFORE binding :9000 --
            # and killed the whole process: /chat, /embed, /log and health all
            # down, for a corrupt file whose only honest blast radius is "no
            # gate for this language". A missing KEY in the same file was
            # already caught cleanly here (gate_load_failed, service stays up),
            # and _load_index() below has always handled JSONDecodeError
            # correctly -- this was the one path that did not degrade.
            # Degrading here means: en -> gate_decision() returns None (no
            # gate); zh -> cjk_bypass, the same fallback a missing zh file
            # already gets.
            spec = json.loads(gate_file.read_text(encoding="utf-8"))
            tok, session = _load_model(dirs[lang])
            _gates[lang] = {
                "tokenizer": tok,
                "session": session,
                "matrix": np.array(spec["vectors"], dtype=np.float32),
                "stat": spec["gate_stat"],
                "threshold": spec["gate_threshold"],
                "prefix": spec.get("query_prefix", ""),
                "pooling": spec.get("pooling", "mean"),
            }
            log({"type": "gate_loaded", "lang": lang, "stat": spec["gate_stat"],
                 "threshold": spec["gate_threshold"]})
        except Exception as err:
            log({"type": "gate_load_failed", "lang": lang, "error": repr(err)})


def _load_index() -> None:
    """Load the bundled retrieval corpus (Task 29). Same "before binding the
    port" window as _load_embedder() -- see main(). Leaves _index["matrix"]
    None (not an empty matrix) on any failure, which /chat's availability
    check treats as "retrieval not available" (503), never as a silent
    fall-through to client-sent contexts (see the module docstring).

    chunks.json is the zip-internal name for whatever chat/data/
    chunks_{preset}.json build_package.py bundled for this package (Task 29
    Part 2 renamed the chat/data/ source from index.json; the zip's own
    internal filename is a packaging detail decoupled from that, chosen to
    match -- so a `grep -rn "index.json"` sweep for the retired name finds
    nothing here or anywhere else in the repo)."""
    index_file = Path(__file__).with_name("chunks.json")
    if not index_file.exists():
        _index["error"] = "chunks.json not packaged"
        log({"type": "index_load_failed", "error": _index["error"]})
        return
    try:
        import numpy as np

        payload = json.loads(index_file.read_text(encoding="utf-8"))
        # Defense in depth (Task 29 fix round 1 review): build_package.py
        # only bundles chunks.json when its own model_preset matches the
        # preset it is packaging (chunks_preset_status()), but a zip can be
        # assembled by hand (or by a future refactor of that script)
        # without going through it. Compare against BUILD_INFO["preset"] --
        # the SAME string build_package.py's make_build_info() stamped into
        # build_info.json when it packaged THIS zip (see the module-level
        # BUILD_INFO load above). An exact, in-package identity check, no
        # environment variable involved and nothing an operator can forget
        # to set.
        #
        # Fix round 1 originally compared against the QUERY_PREFIX env var
        # instead (fix round 2 review, Important): that conflated two very
        # different failure classes under one all-or-nothing refusal --
        # "wrong preset entirely" (a real bug, and already fully caught at
        # the bundling side regardless of this check) versus "correct
        # preset, forgotten QUERY_PREFIX console step" (an operator
        # omission that used to be a silent retrieval-quality degradation
        # and would, under that check, become a permanent /chat 503 for an
        # otherwise-correctly-built deployment -- worse than what it
        # replaced). BUILD_INFO["preset"] has no such false-positive mode:
        # it says nothing about what env vars are configured.
        build_preset = BUILD_INFO.get("preset")
        index_preset = payload.get("model_preset")
        if build_preset is None:
            # An old or hand-assembled zip with no usable preset stamp in
            # build_info.json. An UNKNOWN signal is not evidence of a
            # mismatch -- the bundling side already guarantees the match
            # for anything build_package.py itself produced, and refusing
            # here on absence alone would recreate the exact "correct
            # deployment, missing config -> total outage" trap this fix
            # exists to avoid. Log loudly and load anyway.
            log({
                "type": "index_preset_unverifiable",
                "index_model_preset": index_preset,
                "warning": "build_info.json has no usable 'preset' -- cannot verify "
                           "chunks.json matches the packaged retrieval model; loading anyway",
            })
        elif index_preset != build_preset:
            _index["error"] = (
                f"chunks.json model_preset {index_preset!r} does not match this "
                f"package's build_info.json preset {build_preset!r} -- refusing "
                "to load a mismatched-embedding-space index"
            )
            log({"type": "index_load_failed", "error": _index["error"]})
            return
        chunks = payload.get("chunks") or []
        if not chunks:
            # "packaged but genuinely empty" has no legitimate production
            # meaning, and it used to load as np.empty((0, 0)) -- which is not
            # None, so it passed /chat's `_index["matrix"] is None` guard and
            # GET /'s health flag while rank_chunks returned [] for every
            # question. Result: HTTP 200 + the canned refusal forever, the
            # widget never falling to degraded mode (200 is success), and
            # health cheerfully reporting "retrieval": true. Nothing anywhere
            # said "broken".
            #
            # Reachable, not hypothetical: build_package.py's
            # chunks_preset_status() validates only model_preset, so
            # chat/data/meta.json (which sits beside chunks_e5.json and also
            # carries "model_preset": "e5") bundled as chunks.json used to
            # sail through the guard that exists specifically to catch
            # mis-bundling. That guard now also requires a non-empty chunks
            # list; this is the load-side half of the same fix.
            _index["error"] = "chunks.json has no chunks"
            log({"type": "index_load_failed", "error": _index["error"],
                 "model_preset": payload.get("model_preset")})
            return
        _index["matrix"] = np.array([c["vector"] for c in chunks], dtype=np.float32)
        _index["chunks"] = chunks
        log({"type": "index_loaded", "chunks": len(chunks),
             "model_preset": payload.get("model_preset")})
    except Exception as err:
        _index["error"] = repr(err)
        log({"type": "index_load_failed", "error": repr(err)})


def rank_chunks(matrix, chunks: list, query_vec) -> list:
    """Pure ranking function: score every chunk row by dot product against
    query_vec (all vectors are unit-normalized, so this is cosine), sort
    descending, take the top TOP_K, THEN drop any below MIN_SCORE. Order
    matters -- floor after top-k, not before.

    Deliberately separate from the embedding call (retrieve(), below) so it
    can be executed directly, without an ONNX model. Mirrored, with the same
    two constants and the same floor-after-top-k order, by chat-widget.js's
    scoreChunks and portfolio_rag.runtime's rank_hits --
    chat/tests/test_retrieval_sync.py executes all three against one shared
    fixture and asserts they agree. Returns a list of {"chunk": <chunk
    dict>, "score": float}, highest score first.

    Two corrections from a cross-implementation sync-test review (Task 29
    fix round 1), both required to actually match chat-widget.js:
    - `kind="stable"`: np.argsort's default is not stable, so ties broke
      differently than JS's Array.prototype.sort (stable since ES2019),
      which keeps original chunk order on equal scores.
    - MIN_SCORE is compared against the RAW (unrounded) score, not the
      4-decimal-rounded one -- rounding first let scores like 0.179960
      round up to 0.18 and clear a floor they shouldn't. scoreChunks never
      rounds internally, so it always compared raw. round() is applied only
      to the returned "score", for output, never for the floor test.
    """
    import numpy as np

    if matrix.shape[0] == 0:
        return []
    scores = matrix @ np.asarray(query_vec, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    top = [(chunks[i], float(scores[i])) for i in order]
    return [{"chunk": c, "score": round(s, 4)} for c, s in top if s >= MIN_SCORE]


def retrieve(question: str) -> list:
    """Embed the query with the packaged retrieval model (reusing
    _run_embedding, which already applies QUERY_PREFIX -- no second embedding
    path) and rank the bundled index. Raises if the index or embedder isn't
    loaded; callers must check availability first (see _chat)."""
    if _index["matrix"] is None:
        raise RuntimeError(_index["error"] or "index not loaded")
    query_vec = _run_embedding(_embed, question)
    return rank_chunks(_index["matrix"], _index["chunks"], query_vec)


def sources_from_hits(hits: list) -> list:
    """The /chat response's `sources` array: seven fields per source --
    id, url, anchor, page_title, section_title, text, score -- exactly what
    chat-widget.js's addSources, dedupeForDisplay and sourcesForLog read
    (see resultsFromSources there, the mirror-image function). Do not rename
    these keys. Pure and separate from the HTTP handler so
    chat/tests/test_chat_contract_sync.py can call it directly without
    spinning up a server."""
    return [
        {
            "id": h["chunk"].get("id"),
            "url": h["chunk"].get("url"),
            "anchor": h["chunk"].get("anchor"),
            "page_title": h["chunk"].get("page_title"),
            "section_title": h["chunk"].get("section_title"),
            "text": h["chunk"].get("text"),
            "score": h["score"],
        }
        for h in hits
    ]


def _run_embedding(bundle: dict, text: str) -> "object":
    import numpy as np

    enc = bundle["tokenizer"].encode(bundle["prefix"] + text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in {i.name for i in bundle["session"].get_inputs()}:
        feed["token_type_ids"] = np.zeros_like(input_ids)
    (hidden,) = bundle["session"].run(["last_hidden_state"], feed)
    if bundle.get("pooling") == "cls":
        pooled = hidden[:, 0]
    else:
        mask = attention_mask[:, :, None].astype("float32")
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(1e-9)
    return pooled[0] / max(float((pooled[0] ** 2).sum() ** 0.5), 1e-12)


def embed_text(text: str) -> list[float]:
    return [round(float(v), 6) for v in _run_embedding(_embed, text)]


def gate_decision(gate_text: str) -> dict | None:
    """Language-routed off-topic gate; None when no gate is packaged."""
    global CJK_RE
    if _gates["en"] is None:
        return None
    if CJK_RE is None:
        import re

        CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
    if CJK_RE.search(gate_text):
        gate = _gates["zh"]
        if gate is None:
            # No zh gate packaged/enabled - let CJK questions through to the
            # LLM-prompt guard rather than refusing all Chinese visitors.
            return {"pass": True, "value": None, "reason": "cjk_bypass"}
        lang = "zh"
    else:
        gate, lang = _gates["en"], "en"

    import numpy as np

    scores = gate["matrix"] @ np.asarray(_run_embedding(gate, gate_text), dtype=np.float32)
    top, mean = float(np.max(scores)), float(np.mean(scores))
    if gate["stat"] == "contrast":
        value = top - mean
    elif gate["stat"] == "zscore":
        value = (top - mean) / (float(np.std(scores)) + 1e-6)
    else:
        value = top
    return {"pass": value >= gate["threshold"], "value": round(value, 4), "lang": lang}


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def log(record: dict) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record.setdefault("build", BUILD_ID)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def new_request_id() -> str:
    """Short, time-ordered, human-scannable id for one request (vs the opaque
    uuid session). Scannable in 日志查询; sortable by time within an instance."""
    with _req_lock:
        _req_counter[0] = (_req_counter[0] + 1) % 10000
        n = _req_counter[0]
    return time.strftime("%m%d-%H%M%S", time.gmtime()) + f"-{n:04d}"


def session_hash(session) -> str:
    """8-char stable hash of the session uuid: same session → same sid across
    turns (correlates a conversation in the logs) without dumping the raw id."""
    if not isinstance(session, str) or not session:
        return "anon"
    return hashlib.sha1(session.encode("utf-8")).hexdigest()[:8]


def clip(text, limit: int) -> str:
    """Trim free text for logs, marking how much was dropped so truncation is
    never mistaken for the whole value."""
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit})"


def log_contexts(hits: list) -> list:
    """Compact per-chunk log rows: ids/urls/scores kept FULL (that's the
    retrieval-quality signal), chunk text clipped to fit CLS line limits.
    `hits` is rank_chunks()'s own output ({"chunk": ..., "score": ...}) --
    Task 29 moved retrieval server-side, so this reads this function's own
    ranking, not a client-supplied contexts list."""
    rows = []
    for i, h in enumerate(hits):
        c, score = h["chunk"], h["score"]
        rows.append({
            "i": i + 1,
            "id": c.get("id"),
            "title": c.get("page_title"),
            "url": c.get("url"),
            "text": clip(c.get("text", ""), LIMITS["log_ctx_text"]),
            "score": round(float(score), 4),
        })
    return rows


def allowed_origin(origin: str) -> bool:
    allowed = env("ALLOWED_ORIGINS", "https://wyc79.github.io")
    return origin in [s.strip() for s in allowed.split(",") if s.strip()]


def load_roles() -> dict:
    if _roles_cache["data"] and time.time() - _roles_cache["ts"] < 3600:
        return _roles_cache["data"]
    url = env("SITE_BASE", "https://wyc79.github.io").rstrip("/") + "/chat/data/roles.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as res:
            _roles_cache.update(data=json.load(res), ts=time.time())
    except Exception as err:  # github.io can be unreachable from some regions
        log({"type": "roles_fetch_failed", "error": str(err), "fallback": "bundled roles.json"})
        bundled = Path(__file__).with_name("roles.json")
        _roles_cache.update(data=json.loads(bundled.read_text(encoding="utf-8")), ts=time.time())
    return _roles_cache["data"]


def rate_limited(ip: str, bucket: str = "chat", per_hour_env: str = "RATE_LIMIT_PER_HOUR", default: str = "30") -> bool:
    # Per-instance, in-memory: a real cap for casual abuse, not a hard
    # guarantee (SCF may run several instances). Pair with a low concurrency
    # quota in the console — see DEPLOY.md.
    hour = int(time.time() // 3600)
    key = f"{bucket}:{ip}"
    with _rate_lock:
        if _rate["hour"] != hour:
            _rate["hour"], _rate["counts"] = hour, {}
        _rate["counts"][key] = _rate["counts"].get(key, 0) + 1
        return _rate["counts"][key] > int(env(per_hour_env, default))


def validate_chat_body(body) -> str | None:
    """Task 29: `contexts` is no longer part of the request contract, so it is
    no longer validated here either. If an old cached widget still sends one,
    the caller ignores it outright (see _chat) rather than rejecting the
    request -- an old client must keep working while the site rolls out."""
    if not isinstance(body, dict) or not isinstance(body.get("question"), str) or not body["question"].strip():
        return "question required"
    if len(body["question"]) > LIMITS["question"]:
        return "question too long"
    history = body.get("history")
    if history is not None:
        if not isinstance(history, list) or len(history) > LIMITS["history_turns"]:
            return "bad history"
        for h in history:
            if not isinstance(h, dict) or h.get("role") not in ("user", "assistant"):
                return "bad history role"
            if not isinstance(h.get("content"), str) or len(h["content"]) > LIMITS["history_text"]:
                return "bad history item"
    return None


def call_llm(system: str, messages: list) -> tuple[str, str, dict | None, str | None]:
    url = env("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": env("LLM_MODEL", "deepseek-chat"),
        "max_tokens": 512,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + env("LLM_API_KEY"),
        },
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.load(res)
    choice = data["choices"][0]
    return choice["message"]["content"], data.get("model", ""), data.get("usage"), choice.get("finish_reason")


REFUSAL = (
    "I can only answer questions about YC and his work — his projects, "
    "skills, education, and publications. Nothing on the site matches that "
    "question, so try asking about one of those instead."
)


def refusal_response(rid: str) -> dict:
    """/chat's 200 body for "retrieval returned nothing": the canned REFUSAL,
    `refused: True`, and an empty sources list.

    `refused` is load-bearing on the client. chat-widget.js's send() keys its
    own LOCALIZED refusal off this field and only renders REFUSAL's English
    text if the field goes missing -- which is exactly what happened while the
    widget declared the field and never read it (a zh visitor got English, with
    no starters and no way forward). Pure and separate from the handler, like
    sources_from_hits, so chat/tests/test_chat_contract_sync.py can assert on
    the real shape rather than a retyped copy of it."""
    return {"answer": REFUSAL, "refused": True, "rid": rid, "sources": []}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ── plumbing ────────────────────────────────────────────────────────
    def _origin(self) -> str:
        return self.headers.get("Origin", "")

    def _ip(self) -> str:
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def _cors(self) -> dict:
        origin = self._origin()
        return {
            "Access-Control-Allow-Origin": origin if allowed_origin(origin) else "null",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json") -> None:
        self.send_response(status)
        for k, v in self._cors().items():
            self.send_header(k, v)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _read_body(self, cap: int = 64 * 1024) -> bytes:
        length = min(int(self.headers.get("Content-Length", 0) or 0), cap)
        return self.rfile.read(length) if length else b""

    def log_message(self, fmt, *args):  # silence default per-request stderr lines
        pass

    # ── routes ──────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        if not allowed_origin(self._origin()):
            return self._json(403, {"error": "origin not allowed"})
        if self.path.split("?")[0] == "/":
            # chunk_count + retrieval_error make the two ways retrieval can be
            # unavailable distinguishable from outside: "not packaged" (error
            # names the missing file), "packaged but empty" (error says so),
            # "preset mismatch" (error names both presets). Before this, an
            # unusable index could report "retrieval": true, so the health
            # endpoint could not be used to check a deployment at all.
            return self._json(200, {
                "service": "portfolio-chat",
                "ok": True,
                "embed": _embed["session"] is not None,
                "retrieval": _index["matrix"] is not None,
                "chunk_count": len(_index["chunks"] or []),
                "retrieval_error": _index["error"],
                "build": BUILD_INFO,
            })
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not allowed_origin(self._origin()):
            return self._json(403, {"error": "origin not allowed"})
        path = self.path.split("?")[0]
        try:
            if path == "/chat":
                return self._chat()
            if path == "/embed":
                return self._embed_route()
            if path == "/log":
                return self._log()
            self._json(404, {"error": "not found"})
        except Exception as err:
            log({"type": "error", "message": repr(err)})
            self._json(500, {"error": "internal error"})

    def _embed_route(self):
        if _embed["session"] is None:
            return self._json(503, {"error": "embedding not available", "detail": _embed["error"] or "still loading"})
        if rate_limited(self._ip(), bucket="embed", per_hour_env="RATE_LIMIT_EMBED_PER_HOUR", default="120"):
            return self._json(429, {"error": "rate limited, try later"})
        try:
            body = json.loads(self._read_body() or b"null")
        except json.JSONDecodeError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("text"), str) or not body["text"].strip():
            return self._json(400, {"error": "text required"})
        if len(body["text"]) > LIMITS["question"]:
            return self._json(400, {"error": "text too long"})
        gate_text = body.get("gate_text") if isinstance(body.get("gate_text"), str) else body["text"]
        gate = gate_decision(gate_text[: LIMITS["question"]])
        rid, sid = new_request_id(), session_hash(body.get("session"))
        response = {"model": env("MODEL_NAME", "server-embedder"), "rid": rid}
        # gate_only (Task 29): the widget calls /embed first just to get the
        # gate decision (and to detect the backend is up) -- retrieval moved
        # server-side into /chat, so it no longer needs a query vector at
        # all. Skip the ONNX inference that would only be thrown away.
        # Everything else (gate decision, rid, logging, rate limiting, 503
        # semantics) is unchanged; old clients that omit the field still get
        # a vector.
        if not body.get("gate_only"):
            response["vector"] = embed_text(body["text"])
        if gate is not None:
            response["gate"] = gate
        # Log the gate decision only — NEVER the query vector. gate carries
        # pass/value/lang(/reason); the clipped gate_text shows what was judged.
        log({"type": "embed", "rid": rid, "sid": sid, "gate": gate, "q": clip(gate_text, 120)})
        self._json(200, response)

    def _chat(self):
        if not env("LLM_API_KEY"):
            return self._json(503, {"error": "function not configured: missing LLM_API_KEY env var"})
        # Retrieval (Task 29): /chat does its own retrieval now, so it needs
        # both the query embedder AND the bundled index. Neither loading is
        # an old-zip-vs-new-zip partial state worth distinguishing in the
        # response -- either way /chat cannot honor the new contract, and
        # must NOT silently fall back to trusting client-sent contexts (that
        # would quietly reopen the prompt-injection surface this change
        # closes). The widget treats a failing /chat as backend-down and
        # goes to degraded mode.
        if _embed["session"] is None or _index["matrix"] is None:
            return self._json(503, {"error": "retrieval not available"})
        if rate_limited(self._ip()):
            return self._json(429, {"error": "rate limited, try later"})

        try:
            body = json.loads(self._read_body() or b"null")
        except json.JSONDecodeError:
            body = None
        invalid = validate_chat_body(body)
        if invalid:
            return self._json(400, {"error": invalid})

        rid, sid = new_request_id(), session_hash(body.get("session"))

        # An old cached widget may still send `contexts` -- ignore it
        # completely (never let client-supplied text reach the prompt) but
        # log how often it happens so the old-client tail is visible.
        client_contexts = body.get("contexts")
        client_contexts_ignored = len(client_contexts) if isinstance(client_contexts, list) else 0

        hits = retrieve(body["question"])

        if not hits:
            log({
                "type": "chat_refused",
                "rid": rid,
                "sid": sid,
                "role": body.get("role"),
                "question": clip(body["question"], LIMITS["log_msg_text"]),
                "client_contexts_ignored": client_contexts_ignored,
            })
            return self._json(200, refusal_response(rid))

        roles_data = load_roles()
        role = roles_data["roles"].get(body.get("role")) or roles_data["roles"][roles_data["default_role"]]

        context_block = "\n".join(
            f'<chunk index="{i + 1}" page="{h["chunk"].get("page_title", "")}" '
            f'url="{h["chunk"].get("url", "")}">\n{h["chunk"]["text"]}\n</chunk>'
            for i, h in enumerate(hits)
        )
        # system_head = the role/persona prompt WITHOUT the injected chunks; the
        # chunks are logged separately (clipped) via log_contexts, so the whole
        # LLM input is reconstructable without a giant log line.
        system_head = (
            f"{roles_data['base_system_prompt']}\n\n"
            f"Visitor role: {role['label']}. {role['system_prompt']}"
        )
        # Answer in the site's active language (the widget sends it from the
        # 中/EN toggle), regardless of the question's own language.
        answer_lang = body.get("lang") if body.get("lang") in ("en", "zh") else None
        if answer_lang == "zh":
            system_head += "\n\nAlways answer in Chinese (简体中文), regardless of the language the question is written in."
        elif answer_lang == "en":
            system_head += "\n\nAlways answer in English, regardless of the language the question is written in."
        system = f"{system_head}\n\nContext retrieved from the site for this question:\n{context_block}"
        messages = list(body.get("history") or []) + [{"role": "user", "content": body["question"]}]

        try:
            answer, model, usage, finish = call_llm(system, messages)
        except urllib.error.HTTPError as err:
            log({"type": "llm_error", "rid": rid, "sid": sid, "status": err.code,
                 "detail": err.read().decode("utf-8", "replace")[:500]})
            return self._json(502, {"error": "llm call failed"})

        # Full LLM I/O for every turn: system_head + clipped chunks (ids &
        # this function's own retrieval scores kept full) + clipped messages
        # in; answer + finish_reason + usage out. Enough to reproduce and
        # audit any answer.
        log({
            "type": "chat",
            "rid": rid,
            "sid": sid,
            "role": body.get("role"),
            "client_contexts_ignored": client_contexts_ignored,
            "in": {
                "system_head": system_head,
                "contexts": log_contexts(hits),
                "messages": [
                    {"role": m.get("role"), "content": clip(m.get("content"), LIMITS["log_msg_text"])}
                    for m in messages
                ],
            },
            "out": {
                "answer": clip(answer, LIMITS["log_answer"]),
                "finish_reason": finish,
                "model": model,
                "usage": usage,
            },
        })
        self._json(200, {"answer": answer, "model": model, "rid": rid, "sources": sources_from_hits(hits)})

    def _log(self):
        raw = self._read_body(LIMITS["log_bytes"] + 1)
        if len(raw) > LIMITS["log_bytes"]:
            return self._json(413, {"error": "log too large"})
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        if not isinstance(record, dict):
            record = {"data": record}
        # Correlate with server turns: hash the session to sid (drop the raw
        # uuid), and keep any rid the widget echoes back from /chat or /embed.
        # This is where the widget's client-side top-k {id, score} land.
        if "session" in record:
            record["sid"] = session_hash(record.pop("session"))
        log({"type": "client_log", **record})
        self._send(204)


def main() -> None:
    port = int(env("PORT", "9000"))  # SCF web functions must listen on 9000
    log({"type": "startup", "port": port, "llm_base": env("LLM_BASE_URL", "https://api.deepseek.com"),
         "build_info": BUILD_INFO})
    # Load models BEFORE binding the port: SCF considers a web function ready
    # only once it listens, so cold-start requests queue until the models are
    # in memory instead of racing a background load and seeing 503s (the init
    # timeout covers this — see DEPLOY.md). /embed's 503 now only ever means
    # "not packaged". _load_index() (Task 29) loads the bundled retrieval
    # index in the same window -- /chat's 503 means either piece is missing.
    _load_embedder()
    _load_index()
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
