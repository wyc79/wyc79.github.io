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


def _normalize_answer(raw) -> str:
    import json as _json

    script = (
        extract_js_function(_widget_src(), "function normalizeAnswer(") + "\n"
        "process.stdout.write(JSON.stringify(normalizeAnswer(" + _json.dumps(raw) + ")));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_normalize_answer_strips_markdown_the_plain_text_ui_cannot_render() -> None:
    assert _normalize_answer("**Prime Engine** is his engine work.") == "Prime Engine is his engine work."
    assert _normalize_answer("## Heading\nbody") == "Heading\nbody"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_normalize_answer_reports_nothing_to_render_instead_of_throwing() -> None:
    """The blank-bubble bug: an empty or missing answer must come back as ''
    so send() can raise, not as a value that silently paints an empty bubble --
    and a missing field must not throw a TypeError deep inside rendering."""
    assert _normalize_answer("") == ""
    assert _normalize_answer("   ") == ""
    assert _normalize_answer(None) == ""


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_backend_down_expires_instead_of_lasting_the_whole_session() -> None:
    """One SCF cold start used to demote every later question in the tab to
    the offline-model prompt permanently. The backend must get another chance
    once the TTL has passed."""
    src = _widget_src()
    ttl = int(extract_js_var(src, "REMOTE_DOWN_TTL_MS"))
    script = (
        "var state = { remoteEmbedDownAt: null };\n"
        "var REMOTE_DOWN_TTL_MS = " + str(ttl) + ";\n"
        "var now = 1000000;\n"
        "Date.now = function () { return now; };\n"
        + extract_js_function(src, "function remoteEmbedDown(") + "\n"
        "var fresh = remoteEmbedDown();\n"
        "state.remoteEmbedDownAt = now;\n"
        "var justFailed = remoteEmbedDown();\n"
        "now += REMOTE_DOWN_TTL_MS - 1;\n"
        "var stillDown = remoteEmbedDown();\n"
        "now += 2;\n"
        "var recovered = remoteEmbedDown();\n"
        "process.stdout.write(JSON.stringify({\n"
        "  fresh: fresh, justFailed: justFailed, stillDown: stillDown, recovered: recovered,\n"
        "}));\n"
    )
    got = run_node_json(script)

    assert got["fresh"] is False, "a session that has not failed yet must try the backend"
    assert got["justFailed"] is True, "a just-failed backend must not be retried immediately"
    assert got["stillDown"] is True, "the TTL must actually hold for its full duration"
    assert got["recovered"] is False, "after the TTL the backend must get another chance"
    assert 0 < ttl <= 300000, "a TTL longer than a few minutes is the bug this fixes"
