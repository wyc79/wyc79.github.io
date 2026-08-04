"""Client-side resilience guarantees, executed against the REAL widget source.

Every function under test is extracted verbatim from scripts/chat-widget.js and
run in a node subprocess (see tests/_node_harness.py) rather than retyped here
-- a Python reimplementation would drift from the file it is meant to mirror,
which is the exact failure mode the other *_sync.py tests exist to prevent.
"""

import pytest

from portfolio_rag.config import settings
from tests._node_harness import extract_js_function, extract_js_var, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"


def _widget_src() -> str:
    return WIDGET_PATH.read_text(encoding="utf-8")


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_fetch_with_timeout_rejects_when_the_request_never_settles() -> None:
    """A backend that accepts the connection and then goes silent is the case
    that produced a forever-spinning bubble: without a deadline the promise
    never settles, so none of the widget's fallback paths ever run."""
    script = (
        extract_js_function(_widget_src(), "function fetchWithTimeout(") + "\n"
        # A fetch that honours the abort signal but never resolves on its own.
        "global.fetch = function (url, opts) {\n"
        "  return new Promise(function (resolve, reject) {\n"
        "    opts.signal.addEventListener('abort', function () {\n"
        "      var e = new Error('aborted'); e.name = 'AbortError'; reject(e);\n"
        "    });\n"
        "  });\n"
        "};\n"
        "var started = Date.now();\n"
        "fetchWithTimeout('http://example.invalid', { method: 'POST' }, 50)\n"
        "  .then(function () {\n"
        "    process.stdout.write(JSON.stringify({ settled: 'resolved' }));\n"
        "  })\n"
        "  .catch(function (e) {\n"
        "    process.stdout.write(JSON.stringify({\n"
        "      settled: 'rejected', name: e.name, elapsed: Date.now() - started,\n"
        "    }));\n"
        "  });\n"
    )
    got = run_node_json(script)

    assert got["settled"] == "rejected", "a stalled request must not hang forever"
    assert got["name"] == "AbortError"
    assert got["elapsed"] < 2000, "the deadline must fire promptly, not at some default"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_fetch_with_timeout_passes_through_a_normal_response() -> None:
    """Complement: the deadline must not interfere with a request that
    completes in time (a test that only checked the abort path would pass on an
    implementation that aborts immediately)."""
    script = (
        extract_js_function(_widget_src(), "function fetchWithTimeout(") + "\n"
        "global.fetch = function (url, opts) { return Promise.resolve({ ok: true, url: url }); };\n"
        "fetchWithTimeout('http://example.invalid/chat', { method: 'POST' }, 5000)\n"
        "  .then(function (r) { process.stdout.write(JSON.stringify(r)); });\n"
    )
    got = run_node_json(script)

    assert got["ok"] is True
    assert got["url"] == "http://example.invalid/chat"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_chat_deadline_outlives_the_functions_own_llm_timeout() -> None:
    """index.py's call_llm uses urlopen(timeout=60). A client deadline shorter
    than that would abandon requests the server is still going to answer."""
    src = _widget_src()
    chat_ms = int(extract_js_var(src, "CHAT_TIMEOUT_MS"))
    embed_ms = int(extract_js_var(src, "EMBED_TIMEOUT_MS"))

    assert chat_ms > 60000, "the /chat deadline must sit past call_llm's own 60s timeout"
    assert 0 < embed_ms < chat_ms, "/embed is a short gate call and must give up sooner"
