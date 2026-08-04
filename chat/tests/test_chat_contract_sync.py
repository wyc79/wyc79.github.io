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
3. /chat's OTHER 200 shape -- the refusal (`{answer, refused, rid,
   sources: []}`) -- must be handled by the widget as a refusal, in the
   visitor's own language. This was the gap the final whole-branch review
   found: askWorker's return comment declared `refused?`, send() never read
   it, so a server-side refusal rendered index.py's hardcoded ENGLISH
   REFUSAL even at lang() === 'zh', with no starters and no page links, and
   record.mode stayed 'llm' so GA and the transcript could not tell a
   refusal from an answer. The test below executes the widget's REAL
   refusal branch (extracted verbatim, run in node) against the REAL
   payload refusal_response() builds.
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


def _run_widget_refusal_branch(resp: dict, page_lang: str) -> dict:
    """Execute send()'s REAL server-refusal branch -- the whole
    `if (resp.refused) { ... }` statement, extracted verbatim including its
    condition (reading that field is the whole point) -- in a node process,
    against `resp`, with the page language set to `page_lang`.

    The widget's own STR table, lang() and t() are extracted verbatim too, so
    the rendered text is the real localized string, not a fixture. Everything
    the branch touches that belongs to send()'s closure or the DOM (record,
    thinking, state, addStarters, pushLog, logTurn) is stubbed and recorded.
    The wrapper returns a sentinel that only survives if the branch did NOT
    return early, which is how "the refusal was handled" is distinguished
    from "control fell through to the normal-answer path"."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    script = (
        "var window = { YCI18N: { current: function () { return " + json.dumps(page_lang) + "; } } };\n"
        + extract_js_function(src, "var STR = ") + ";\n"
        + extract_js_function(src, "function lang(") + "\n"
        + extract_js_function(src, "function t(") + "\n"
        "var calls = [];\n"
        "var record = {};\n"
        "var thinking = { textContent: null, classList: { remove: function (c) { calls.push('remove:' + c); } } };\n"
        "var state = { role: 'visitor', roles: { roles: { visitor: { label: 'Visitor' } } } };\n"
        "function addStarters(role) { calls.push('addStarters'); }\n"
        "function pushLog(e) { calls.push('pushLog:' + e.type); }\n"
        "function logTurn(r) { calls.push('logTurn'); }\n"
        "function handleServerResponse(resp) {\n"
        # Marker deliberately stops before the closing paren: a mutation that
        # weakens the CONDITION (`if (resp.refused && false)`) then still
        # extracts, and shows up as a behavioural failure below rather than as
        # an extraction error that says nothing about what broke.
        + extract_js_function(src, "if (resp.refused") + "\n"
        "  return 'FELL_THROUGH_TO_NORMAL_ANSWER';\n"
        "}\n"
        "var out = handleServerResponse(" + json.dumps(resp) + ");\n"
        "process.stdout.write(JSON.stringify({\n"
        "  fellThrough: out === 'FELL_THROUGH_TO_NORMAL_ANSWER',\n"
        "  record: record, text: thinking.textContent, calls: calls,\n"
        "  localizedRefusal: t('refused'),\n"
        "}));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_a_server_refusal_renders_localized_with_a_way_forward() -> None:
    """The refusal index.py really returns, handled by the widget's real
    branch, on a Chinese page. Must render the widget's own zh refusal (never
    the function's hardcoded English REFUSAL), offer starters, and classify
    the turn as a refusal rather than an LLM answer."""
    mod = _load_backend()
    payload = mod.refusal_response("0801-120000-0001")
    assert payload["refused"] is True and payload["sources"] == []

    got = _run_widget_refusal_branch(payload, page_lang="zh")

    assert not got["fellThrough"], (
        "send() ignored /chat's `refused` field and treated the refusal as an "
        "ordinary LLM answer -- the visitor would see index.py's English REFUSAL"
    )
    assert got["text"] == got["localizedRefusal"], "the refusal must be the widget's localized string"
    assert got["text"] != mod.REFUSAL, (
        "a zh visitor was shown the function's hardcoded English refusal"
    )
    assert "addStarters" in got["calls"], (
        "a server refusal must offer the same way forward the client-side gate "
        "refusal does -- otherwise it is a dead end (no starters, no page links)"
    )
    assert got["record"]["mode"] == "off_topic_refused", (
        "record.mode must not stay 'llm' -- GA and the transcript cannot "
        "distinguish a refusal from an answer if it does"
    )
    assert got["calls"].count("logTurn") == 1, "the refused turn must still be logged exactly once"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_a_normal_answer_is_not_treated_as_a_refusal() -> None:
    """Complement: the branch must not swallow ordinary answers (a test that
    only checked the refusal path would pass on `if (true)`)."""
    got = _run_widget_refusal_branch(
        {"answer": "He built Prime Engine.", "rid": "r1", "sources": []}, page_lang="en"
    )

    assert got["fellThrough"], "a normal /chat answer must fall through to the answer path"
    assert got["record"] == {}
    assert got["calls"] == []


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_refusal_payloads_sources_survive_resultsfromsources() -> None:
    """The refusal carries `sources: []`; resultsFromSources must yield [] and
    not throw, so addSources' early return is reached rather than a TypeError
    deep in rendering."""
    mod = _load_backend()
    assert _widget_results_from_sources(mod.refusal_response("r1")["sources"]) == []


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


def test_usable_answer_rejects_what_would_paint_an_empty_bubble() -> None:
    """A reasoner model returns its text in reasoning_content and leaves
    content empty; a content-filter stop returns "". Both used to reach the
    widget as a 200 and render as a blank bot bubble with no way forward."""
    mod = _load_backend()

    assert mod.usable_answer("He built Prime Engine.") is True
    assert mod.usable_answer("") is False
    assert mod.usable_answer("   \n  ") is False
    assert mod.usable_answer(None) is False
    assert mod.usable_answer(123) is False


# ── llm_payload: max_tokens must not starve a reasoning model's answer ──────
#
# A live /chat turn against LLM_MODEL=deepseek-v4-flash (a reasoning model)
# came back {"answer": "", "finish_reason": "length", ...,
# "usage": {"completion_tokens": 512, "completion_tokens_details":
# {"reasoning_tokens": 512}}} -- the hardcoded max_tokens: 512 ceiling let the
# model spend its ENTIRE completion budget on reasoning_content and leave
# content empty. The visitor saw a blank bot bubble with source cards under
# it. llm_payload is the pure request-building half of call_llm, extracted so
# these tests don't need a live LLM call.


def test_llm_payload_honors_llm_max_tokens_env_override(monkeypatch) -> None:
    mod = _load_backend()
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")

    payload = mod.llm_payload("system prompt", [{"role": "user", "content": "hi"}])

    assert payload["max_tokens"] == 4096


def test_llm_payload_default_max_tokens_is_comfortably_above_the_512_that_caused_the_incident(monkeypatch) -> None:
    """Pins the default well above the 512 ceiling that starved
    deepseek-v4-flash -- a future edit that lowers it back down fails here."""
    mod = _load_backend()
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    payload = mod.llm_payload("system prompt", [])

    assert payload["max_tokens"] >= 2048


def test_llm_payload_puts_system_first_then_messages_and_carries_the_model(monkeypatch) -> None:
    mod = _load_backend()
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    payload = mod.llm_payload("system prompt text", messages)

    assert payload["messages"][0] == {"role": "system", "content": "system prompt text"}
    assert payload["messages"][1:] == messages
    assert payload["model"] == "deepseek-v4-flash"


def test_llm_payload_falls_back_to_the_default_on_a_malformed_max_tokens(monkeypatch) -> None:
    """A typo'd console value (e.g. "abc") must not raise out of a request --
    that would take /chat down entirely instead of just serving one turn with
    the default ceiling."""
    mod = _load_backend()
    monkeypatch.setenv("LLM_MAX_TOKENS", "abc")

    payload = mod.llm_payload("system prompt", [])

    assert payload["max_tokens"] == int(mod.LLM_MAX_TOKENS_DEFAULT)


# ── llm_payload: thinking mode is off by default ────────────────────────────
#
# DeepSeek's reasoning models (deepseek-v4-flash/-pro) enable "thinking" by
# default, effort "high" -- see the incident above, where a live turn spent
# every one of its 512 completion tokens on reasoning_content and returned
# content: "". Raising max_tokens (previous commit) stops the starvation but
# still pays for a reasoning trace this 2-5 sentence, context-already-found
# agent gets no value from. Sending {"thinking": {"type": "disabled"}} at the
# TOP level of the request body (per DeepSeek's docs -- the OpenAI SDK's
# extra_body merges there, and call_llm builds the raw JSON itself) removes
# that cost entirely.


def test_llm_payload_disables_thinking_by_default(monkeypatch) -> None:
    mod = _load_backend()
    monkeypatch.delenv("LLM_THINKING", raising=False)

    payload = mod.llm_payload("system prompt", [])

    assert payload["thinking"] == {"type": "disabled"}


def test_llm_payload_honors_llm_thinking_enabled(monkeypatch) -> None:
    mod = _load_backend()
    monkeypatch.setenv("LLM_THINKING", "enabled")

    payload = mod.llm_payload("system prompt", [])

    assert payload["thinking"] == {"type": "enabled"}


def test_llm_payload_empty_llm_thinking_omits_the_field_entirely(monkeypatch) -> None:
    """The escape hatch: LLM_THINKING="" must drop the key outright, not send
    an empty/None value -- this field is DeepSeek-specific, and LLM_BASE_URL
    can point call_llm at an OpenAI-compatible provider that rejects an
    unknown `thinking` key. Reachable without a code change (just unset/blank
    the env var), for a provider swap this branch's author cannot predict."""
    mod = _load_backend()
    monkeypatch.setenv("LLM_THINKING", "")

    payload = mod.llm_payload("system prompt", [])

    assert "thinking" not in payload


def test_llm_payload_unrecognised_llm_thinking_falls_back_to_disabled(monkeypatch) -> None:
    """A typo'd value (e.g. "maybe") must not be passed through verbatim --
    that risks the API rejecting the whole request over an unknown `type`,
    same failure shape as the malformed-max_tokens case above."""
    mod = _load_backend()
    monkeypatch.setenv("LLM_THINKING", "maybe")

    payload = mod.llm_payload("system prompt", [])

    assert payload["thinking"] == {"type": "disabled"}
    assert "maybe" not in json.dumps(payload)
