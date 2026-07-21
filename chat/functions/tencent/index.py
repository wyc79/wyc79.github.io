"""Tencent SCF web function — the LLM backend for the portfolio chat agent.

Zero-dependency (stdlib only) port of chat/worker/worker.js, speaking the
OpenAI-compatible chat-completions protocol (DeepSeek by default, or any
provider that exposes /v1/chat/completions). Listens on :9000 as SCF web
functions require; scf_bootstrap starts it.

Same contract and guarantees as the Cloudflare worker:
  POST /chat  {session, role, question, history?, contexts[]} -> {answer, model}
  POST /log   client-side event record -> 204
  GET  /      health
- Origin allowlist, size caps on every field.
- Empty contexts -> canned refusal, no LLM call (server-side off-topic guard).
- Role prompts come from the site's roles.json (client sends a role id only);
  a bundled roles.json copy is the fallback if github.io is unreachable from
  the function's region.
- Every request/response printed as JSON -> visible in SCF 日志查询.
Config via environment variables: LLM_API_KEY (required), LLM_BASE_URL,
LLM_MODEL, SITE_BASE, ALLOWED_ORIGINS, RATE_LIMIT_PER_HOUR.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LIMITS = {
    "question": 1000,
    "contexts": 6,
    "context_text": 1600,
    "history_turns": 8,
    "history_text": 1200,
    "log_bytes": 4096,
}

_roles_cache: dict = {"data": None, "ts": 0.0}
_rate: dict = {"hour": None, "counts": {}}
_rate_lock = threading.Lock()

# ── optional server-side embedding + gating (stage 2) ─────────────────────
# If MODEL_DIR exists in the package (an ONNX sentence-embedding model, e.g.
# Xenova/multilingual-e5-small), POST /embed serves query vectors. Without it
# /embed returns 503 and the widget falls back to its in-browser model.
# If GATE_MODEL_DIR + gate_vectors.json are also packaged, /embed additionally
# answers the off-topic gate with the GATE model (MiniLM: e5 can't separate
# on/off-topic — its cosines compress into one band).
_embed = {"lock": threading.Lock(), "session": None, "tokenizer": None, "error": None,
          "prefix": "", "pooling": "mean"}
# Two gate bundles: "en" (MiniLM vs the chunk vectors) and optionally "zh"
# (bge-zh vs the hand-written knowledge/about_zh.md corpus). CJK queries use zh when
# available, otherwise bypass to the LLM-prompt guard.
_gates: dict = {"en": None, "zh": None}

CJK_RE = None  # compiled lazily


def _load_model(dir_name: str) -> tuple:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(__file__).parent / dir_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{dir_name} not packaged")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=int(env("EMBED_MAX_TOKENS", "256")))
    session = ort.InferenceSession(
        str(model_dir / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"]
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

    gate_file = Path(__file__).parent / "gate_vectors.json"
    if gate_file.exists():
        import numpy as np

        payload = json.loads(gate_file.read_text(encoding="utf-8"))
        dirs = {"en": env("GATE_MODEL_DIR", "gate_model"), "zh": env("GATE_MODEL_ZH_DIR", "gate_model_zh")}
        for lang in ("en", "zh"):
            spec = payload.get(lang)
            if not spec:
                continue
            try:
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
    print(json.dumps(record, ensure_ascii=False), flush=True)


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
    if not isinstance(body, dict) or not isinstance(body.get("question"), str) or not body["question"].strip():
        return "question required"
    if len(body["question"]) > LIMITS["question"]:
        return "question too long"
    contexts = body.get("contexts")
    if not isinstance(contexts, list) or len(contexts) > LIMITS["contexts"]:
        return "bad contexts"
    for c in contexts:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str) or len(c["text"]) > LIMITS["context_text"]:
            return "bad context item"
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


def call_llm(system: str, messages: list) -> tuple[str, str, dict | None]:
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
    answer = data["choices"][0]["message"]["content"]
    return answer, data.get("model", ""), data.get("usage")


REFUSAL = (
    "I can only answer questions about YC and his work — his projects, "
    "skills, education, and publications. Nothing on the site matches that "
    "question, so try asking about one of those instead."
)


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
            return self._json(200, {
                "service": "portfolio-chat",
                "ok": True,
                "embed": _embed["session"] is not None,
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
        response = {"vector": embed_text(body["text"]), "model": env("MODEL_NAME", "server-embedder")}
        if gate is not None:
            response["gate"] = gate
        self._json(200, response)

    def _chat(self):
        if not env("LLM_API_KEY"):
            return self._json(503, {"error": "function not configured: missing LLM_API_KEY env var"})
        if rate_limited(self._ip()):
            return self._json(429, {"error": "rate limited, try later"})

        try:
            body = json.loads(self._read_body() or b"null")
        except json.JSONDecodeError:
            body = None
        invalid = validate_chat_body(body)
        if invalid:
            return self._json(400, {"error": invalid})

        if len(body["contexts"]) == 0:
            log({
                "type": "chat_refused",
                "session": str(body.get("session", ""))[:64],
                "role": body.get("role"),
                "question": body["question"],
            })
            return self._json(200, {"answer": REFUSAL, "refused": True})

        roles_data = load_roles()
        role = roles_data["roles"].get(body.get("role")) or roles_data["roles"][roles_data["default_role"]]

        context_block = "\n".join(
            f'<chunk index="{i + 1}" page="{c.get("title", "")}" url="{c.get("url", "")}">\n{c["text"]}\n</chunk>'
            for i, c in enumerate(body["contexts"])
        )
        system = (
            f"{roles_data['base_system_prompt']}\n\n"
            f"Visitor role: {role['label']}. {role['system_prompt']}\n\n"
            f"Context retrieved from the site for this question:\n{context_block}"
        )
        messages = list(body.get("history") or []) + [{"role": "user", "content": body["question"]}]

        try:
            answer, model, usage = call_llm(system, messages)
        except urllib.error.HTTPError as err:
            log({"type": "llm_error", "status": err.code, "detail": err.read().decode("utf-8", "replace")[:500]})
            return self._json(502, {"error": "llm call failed"})

        log({
            "type": "chat",
            "session": str(body.get("session", ""))[:64],
            "role": body.get("role"),
            "question": body["question"],
            "retrieved": [{"id": c.get("id"), "title": c.get("title"), "url": c.get("url")} for c in body["contexts"]],
            "answer": answer,
            "model": model,
            "usage": usage,
        })
        self._json(200, {"answer": answer, "model": model})

    def _log(self):
        raw = self._read_body(LIMITS["log_bytes"] + 1)
        if len(raw) > LIMITS["log_bytes"]:
            return self._json(413, {"error": "log too large"})
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        log({"type": "client_log", **(record if isinstance(record, dict) else {"data": record})})
        self._send(204)


def main() -> None:
    port = int(env("PORT", "9000"))  # SCF web functions must listen on 9000
    log({"type": "startup", "port": port, "llm_base": env("LLM_BASE_URL", "https://api.deepseek.com")})
    # Load models BEFORE binding the port: SCF considers a web function ready
    # only once it listens, so cold-start requests queue until the models are
    # in memory instead of racing a background load and seeing 503s (the init
    # timeout covers this — see DEPLOY.md). /embed's 503 now only ever means
    # "not packaged".
    _load_embedder()
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
