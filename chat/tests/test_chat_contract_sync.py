"""Cross-implementation sync test for the /chat request/response contract
(Task 29 fix round 1, Minor 5 -- "the single highest-value item... it is the
failure that would break live chat").

Nothing in chat/tests/ previously touched functions/tencent/index.py's HTTP
layer at all: the field-name agreement the whole Task 29 contract flip rests
on (the widget stops sending `contexts`; the function returns `sources` with
seven specific field names) was guarded only by manual runs. This test
executes the REAL request-building code from chat-widget.js's askWorker (via
a real `node` subprocess, extracted verbatim -- not retyped, for the same
drift reason as test_retrieval_sync.py) against the REAL
functions/tencent/index.py's validate_chat_body and sources_from_hits
(loaded via importlib), in both directions:

1. The exact object askWorker sends as its /chat request body must pass
   validate_chat_body (no `contexts` field, all required fields present).
2. The exact seven-field shape sources_from_hits produces must round-trip
   losslessly through the widget's own resultsFromSources -- if either side
   ever renamed or dropped a field, this would show up as a missing/None
   value on the other side instead of silently building a broken UI.
"""

import importlib.util
import json

import pytest

from portfolio_rag.config import settings
from tests._node_harness import extract_js_function, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"


def _load_backend():
    spec = importlib.util.spec_from_file_location("_scf_index_contract_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_ask_worker_request_body_literal(src: str) -> str:
    """The exact object literal askWorker passes to JSON.stringify(), pulled
    from WITHIN askWorker's own body specifically (the file has other,
    unrelated `body: JSON.stringify(...)` calls for /embed and /log)."""
    fn_src = extract_js_function(src, "async function askWorker(")
    marker = "body: JSON.stringify("
    start = fn_src.index(marker) + len(marker)
    depth, i = 1, start  # already past the opening '('
    while depth > 0:
        if fn_src[i] == "(":
            depth += 1
        elif fn_src[i] == ")":
            depth -= 1
        i += 1
    return fn_src[start : i - 1]


def _widget_chat_request_body(question: str) -> dict:
    """Build the exact object askWorker would send for `question`, by
    running its own object-literal source in a real node process with a
    stand-in `state`/`lang()` (askWorker itself does the actual fetch(),
    which needs a live server -- this test is about the request SHAPE, the
    thing validate_chat_body cares about, not the transport)."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    body_literal = _extract_ask_worker_request_body_literal(src)
    script = (
        "const state = { session: 'sess-smoke-test', role: 'visitor', "
        "history: [{role: 'user', content: 'hi'}, {role: 'assistant', content: 'yo'}] };\n"
        "function lang() { return 'en'; }\n"
        f"const question = {json.dumps(question)};\n"
        f"const body = {body_literal};\n"
        "process.stdout.write(JSON.stringify(body));\n"
    )
    return run_node_json(script)


def _widget_results_from_sources(sources: list) -> list:
    """Run the widget's own resultsFromSources (extracted verbatim) in node
    against `sources`, returning what it built."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    fn_src = extract_js_function(src, "function resultsFromSources(")
    script = (
        fn_src + "\n"
        "const sources = " + json.dumps(sources) + ";\n"
        "process.stdout.write(JSON.stringify(resultsFromSources(sources)));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_widget_chat_request_body_passes_validate_chat_body() -> None:
    body = _widget_chat_request_body("What engine programming has he done?")

    assert "contexts" not in body, (
        "askWorker must not send `contexts` -- the whole point of the Task "
        "29 contract flip is that the client stops sending it"
    )
    for field in ("session", "role", "lang", "question", "history"):
        assert field in body, f"askWorker's request body is missing {field!r}"

    mod = _load_backend()
    assert mod.validate_chat_body(body) is None, (
        "the widget's real /chat request body was rejected by index.py's "
        "real validate_chat_body -- the request contract has drifted"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_widget_chat_request_body_with_no_history_still_validates() -> None:
    """A brand-new session sends history: [] (state.history starts empty) --
    must not be mistaken for a malformed/missing field."""
    body = _widget_chat_request_body("who is YC")
    body["history"] = []
    mod = _load_backend()
    assert mod.validate_chat_body(body) is None


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_sources_from_hits_fields_round_trip_through_resultsfromsources() -> None:
    """index.py's sources_from_hits() and chat-widget.js's
    resultsFromSources() are mirror-image functions across the wire (one
    builds `sources`, the other reads it back into {chunk, score}). Round
    -trip a hit through BOTH real functions and confirm every one of the
    seven fields survives -- catches either side silently renaming or
    dropping one."""
    mod = _load_backend()
    hit = {
        "chunk": {
            "id": "pages/x.html#sec1:en:0",
            "url": "pages/x.html",
            "anchor": "sec1",
            "page_title": "Some Page",
            "section_title": "Some Section",
            "text": "hello world",
        },
        "score": 0.8123,
    }
    sources = mod.sources_from_hits([hit])
    assert set(sources[0]) == {"id", "url", "anchor", "page_title", "section_title", "text", "score"}

    results = _widget_results_from_sources(sources)
    assert len(results) == 1
    got = results[0]
    assert got["score"] == 0.8123
    assert got["chunk"] == {
        "id": "pages/x.html#sec1:en:0",
        "url": "pages/x.html",
        "anchor": "sec1",
        "page_title": "Some Page",
        "section_title": "Some Section",
        "text": "hello world",
    }
