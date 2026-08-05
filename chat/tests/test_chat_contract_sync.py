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
import re

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


def _widget_chat_request_body(
    question: str, embed_rid: str = "0804-051310-0001", pathname: str = "/pages/skills.html"
) -> dict:
    """Build the exact object askWorker sends, by running the widget's own
    chatRequestBody() in a real node process with a stand-in state/lang()/
    location.

    Task 29's version brace-matched the object literal out of askWorker's
    body. The literal now calls currentPageUrl(), so extracting the whole
    named function is both simpler and more faithful -- same verbatim-source
    guarantee, one fewer parser to keep in sync with the file it reads."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    script = (
        "const state = { session: 'sess-smoke-test', role: 'visitor', "
        "history: [{role: 'user', content: 'hi'}, {role: 'assistant', content: 'yo'}] };\n"
        "function lang() { return 'en'; }\n"
        f"const window = {{ location: {{ pathname: {json.dumps(pathname)} }} }};\n"
        + extract_js_function(src, "function currentPageUrl(") + "\n"
        + extract_js_function(src, "function chatRequestBody(") + "\n"
        f"process.stdout.write(JSON.stringify(chatRequestBody({json.dumps(question)}, "
        f"{json.dumps(embed_rid)})));\n"
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
    for field in ("session", "role", "lang", "question", "history", "page", "embed_rid"):
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


def _index_empty_answer_error() -> str:
    """The 502 body's `error` string from _chat's usable_answer guard,
    pulled from source -- _chat has a SECOND, unrelated 502 (the call_llm
    HTTPError branch just above it) with its own literal, so the match is
    anchored on the usable_answer guard specifically."""
    src = BACKEND_PATH.read_text(encoding="utf-8")
    guard = re.search(
        r'if not usable_answer\(answer\):.*?_json\(502, \{"error": "([^"]+)"\}\)', src, re.S
    )
    assert guard, "expected the usable_answer guard's 502 body in _chat"
    return guard.group(1)


def _widget_empty_answer_error_check() -> str:
    """The string askWorker compares a 502 body's `error` field against, to
    tell index.py's empty-answer 502 apart from every other 502."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    fn_src = extract_js_function(src, "async function askWorker(")
    m = re.search(r"body\.error === '([^']+)'", fn_src)
    assert m, "expected askWorker's empty-answer body.error comparison"
    return m.group(1)


def test_the_empty_answer_502_error_string_matches_between_index_py_and_the_widget() -> None:
    """index.py's usable_answer guard and chat-widget.js's askWorker each
    hand-type the literal 'llm returned an empty answer' with nothing
    previously asserting the two agree -- a rename on either side would
    silently stop matching, falling through to the generic 'worker 502' ->
    degraded mode instead of send()'s outer catch (the pass-through path
    this module's docstring and test_askworker_flags_the_empty_answer_502_for_pass_through
    in test_widget_resilience.py both cover)."""
    assert _widget_empty_answer_error_check() == _index_empty_answer_error()


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


def _fixture_index() -> list:
    """A miniature index in the real chunks.json shape: two pages, both
    languages, each with a summary chunk (anchor "top") and sections."""
    out = []
    for url, title in (("pages/prime-engine.html", "Prime Engine"), ("pages/skills.html", "Skills")):
        for lang in ("en", "zh"):
            out.append({"id": f"{url}#top:{lang}:0", "url": url, "anchor": "top",
                        "page_title": title, "section_title": title,
                        "text": f"{title} summary ({lang})", "lang": lang})
            # A curated knowledge section that `link:`s to this page. It carries
            # the PAGE's url, which is exactly why url alone cannot separate the
            # two -- the earlier fixture had no such record, so the leak it
            # models was unrepresentable and went unnoticed until a review
            # replayed page_chunks against the real index.
            out.append({"id": f"{url}#curated{lang}:0", "url": url, "anchor": "",
                        "page_title": f"A curated heading about {title}",
                        "section_title": f"A curated heading about {title}",
                        "text": f"curated prose linking to {title} ({lang})",
                        "lang": lang, "curated": True})
            for i in range(8):
                out.append({"id": f"{url}#sec{i}:{lang}:0", "url": url, "anchor": f"sec{i}",
                            "page_title": title, "section_title": f"Section {i}",
                            "text": f"{title} section {i} ({lang})", "lang": lang})
    return out


def test_page_chunks_puts_the_summary_first_and_caps_the_rest() -> None:
    """A deictic question has no useful query vector, so the page's chunks are
    selected by url instead of ranked. The summary chunk leads so the model
    gets the page's thesis before its raw sections."""
    mod = _load_backend()
    got = mod.page_chunks(_fixture_index(), "pages/prime-engine.html", "en")

    assert len(got) == mod.PAGE_CONTEXT_MAX
    assert got[0]["anchor"] == "top", "the summary chunk must lead"
    assert all(c["url"] == "pages/prime-engine.html" for c in got)
    assert all(c["lang"] == "en" for c in got), "an en answer must not be fed zh chunks"


def test_page_chunks_filters_by_answer_language() -> None:
    mod = _load_backend()
    got = mod.page_chunks(_fixture_index(), "pages/skills.html", "zh")

    assert got, "a zh visitor on a page with zh chunks must get them"
    assert all(c["lang"] == "zh" for c in got)


def test_page_chunks_treats_an_unknown_url_as_selecting_nothing() -> None:
    """page_url is client-supplied and used ONLY as a lookup key into our own
    index. Anything unrecognised selects nothing; nothing is trusted."""
    mod = _load_backend()
    index = _fixture_index()

    assert mod.page_chunks(index, "pages/does-not-exist.html", "en") == []
    assert mod.page_chunks(index, "", "en") == []
    assert mod.page_chunks(index, None, "en") == []
    assert mod.page_chunks(index, "../../etc/passwd", "en") == []
    assert mod.page_chunks(index, {"not": "a string"}, "en") == []


def test_build_context_block_tags_only_the_current_page_chunks() -> None:
    """The tag is the entire mechanism: it lets the model tell 'matched the
    question' from 'the page the visitor is looking at' with no classifier
    call, no keyword list and nothing to tune."""
    mod = _load_backend()
    hits = [{"chunk": {"page_title": "Skills", "url": "pages/skills.html", "text": "retrieved"},
             "score": 0.42}]
    page_ctx = [{"page_title": "Prime Engine", "url": "pages/prime-engine.html", "text": "on-page"}]

    block = mod.build_context_block(hits, page_ctx)

    assert block.count('current_page="true"') == 1
    assert block.index("retrieved") < block.index("on-page"), "retrieved chunks come first"
    assert 'index="1"' in block and 'index="2"' in block
    assert 'current_page="true"' not in block.split("</chunk>")[0], "a retrieved chunk must not be tagged"


def test_build_context_block_with_no_page_context_is_unchanged_in_shape() -> None:
    mod = _load_backend()
    block = mod.build_context_block(
        [{"chunk": {"page_title": "Skills", "url": "pages/skills.html", "text": "x"}, "score": 0.4}], []
    )

    assert "current_page" not in block
    assert block.startswith('<chunk index="1" page="Skills" url="pages/skills.html">')


def test_validate_chat_body_accepts_a_page_field_and_rejects_a_malformed_one() -> None:
    mod = _load_backend()
    base = {"question": "summarize this page", "history": []}

    assert mod.validate_chat_body({**base, "page": {"url": "pages/skills.html"}}) is None
    assert mod.validate_chat_body(base) is None, "page is optional -- an old cached widget omits it"
    assert mod.validate_chat_body({**base, "page": "pages/skills.html"}) is not None
    assert mod.validate_chat_body({**base, "page": {"url": 42}}) is not None
    assert mod.validate_chat_body({**base, "page": {"url": "x" * 500}}) is not None


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_request_carries_the_current_page_in_the_index_url_form() -> None:
    """The server matches this against chunk urls verbatim, so it must be
    site-relative ("pages/skills.html"), not a browser pathname."""
    assert _widget_chat_request_body("summarize this page")["page"] == {"url": "pages/skills.html"}


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_landing_page_normalizes_to_index_html() -> None:
    """The index's url for the landing page is "index.html"; the browser's
    pathname is "/". A directory url must normalize the same way."""
    assert _widget_chat_request_body("hi", pathname="/")["page"] == {"url": "index.html"}
    assert _widget_chat_request_body("hi", pathname="/pages/")["page"] == {"url": "pages/index.html"}


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_widget_sends_no_page_title() -> None:
    """Only the url crosses the wire. The page title in the system prompt is
    read off the server's own chunk records, which is what keeps this feature
    out of the prompt-injection surface."""
    body = _widget_chat_request_body("summarize this page")

    assert set(body["page"]) == {"url"}


def test_gate_verdict_is_one_flat_greppable_token() -> None:
    """`gate.pass` nested in a JSON object cannot be filtered on in 日志查询:
    a passed gate and a blocked one look alike at a glance. This is the string
    that distinguishes them."""
    mod = _load_backend()

    assert mod.gate_verdict({"pass": True, "value": 0.4559, "lang": "en"}) == "pass"
    assert mod.gate_verdict({"pass": False, "value": 0.1626, "lang": "en"}) == "block"
    assert mod.gate_verdict({"pass": True, "value": None, "reason": "cjk_bypass"}) == "cjk_bypass"
    assert mod.gate_verdict(None) == "none"


class _StubHandler:
    """The minimal `self` _chat needs, so the REAL handler body runs with no
    socket and no server -- the same shape test_scf_index_loading.py uses.
    `_json` is the sink every reply goes through, so recording it captures the
    real status and the real payload the handler built."""

    def __init__(self, body: bytes, origin: str = "https://wyc79.github.io"):
        self.path = "/chat"
        self._body = body
        self.responses: list = []

    def _origin(self):
        return "https://wyc79.github.io"

    def _ip(self) -> str:
        return "203.0.113.7"

    def _read_body(self, cap: int = 64 * 1024) -> bytes:
        return self._body

    def _json(self, status: int, obj: dict) -> None:
        self.responses.append((status, obj))


def _drive_chat(mod, monkeypatch, body: dict, hits: list, answer: str = "an answer"):
    """Run the real Handler._chat against `body`, with retrieval and the LLM
    stubbed, and return (captured log records, handler responses).

    This replaces two earlier tests that asserted on `inspect.getsource(...)`
    substrings. Those passed whenever an identifier merely APPEARED in the
    method's source -- proven by mutation: replacing the whole refusal
    condition with `if False:` left the entire suite green. Driving the handler
    and reading the emitted record is what actually pins the behaviour."""
    records: list = []
    monkeypatch.setattr(mod, "log", lambda rec: records.append(rec))
    monkeypatch.setattr(mod, "retrieve", lambda q: list(hits))
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setattr(mod, "call_llm", lambda system, messages: (answer, "test-model", {}, "stop"))
    monkeypatch.setattr(mod, "load_roles", lambda: {
        "default_role": "visitor",
        "base_system_prompt": "base",
        "roles": {"visitor": {"label": "Visitor", "system_prompt": "be helpful"}},
    })
    monkeypatch.setitem(mod._embed, "session", object())
    monkeypatch.setitem(mod._index, "matrix", object())
    monkeypatch.setitem(mod._index, "chunks", [])
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    handler = _StubHandler(json.dumps(body).encode("utf-8"))
    mod.Handler._chat(handler)
    return records, handler.responses


def test_the_refused_log_line_carries_embed_rid_and_outcome(monkeypatch) -> None:
    """/embed and /chat write two lines with two different rids. Without
    embed_rid echoed here, a gate decision and the answer it produced can only
    be correlated by sid and wall-clock time."""
    mod = _load_backend()
    records, responses = _drive_chat(
        mod, monkeypatch,
        {"question": "who is YC", "history": [], "embed_rid": "0804-051310-0001"},
        hits=[],
    )

    assert responses and responses[0][0] == 200
    assert responses[0][1].get("refused") is True, (
        "with no retrieval hits and no page context, /chat must refuse -- this "
        "is the assertion that a mutation to the refusal condition has to break"
    )
    refused = [r for r in records if r.get("type") == "chat_refused"]
    assert len(refused) == 1, f"expected one chat_refused record, got {records}"
    assert refused[0]["outcome"] == "refused_no_hits"
    assert refused[0]["embed_rid"] == "0804-051310-0001"


def test_the_answered_log_line_carries_embed_rid_and_outcome(monkeypatch) -> None:
    mod = _load_backend()
    hit = {"chunk": {"id": "pages/skills.html#skills:en:0", "url": "pages/skills.html",
                     "anchor": "skills", "page_title": "Skills", "section_title": "Skills",
                     "text": "UE5, Unity, C++ and C#."}, "score": 0.42}
    records, responses = _drive_chat(
        mod, monkeypatch,
        {"question": "what engines", "history": [], "embed_rid": "0804-051310-0002"},
        hits=[hit],
    )

    assert responses and responses[0][0] == 200
    assert responses[0][1]["answer"] == "an answer"
    chat = [r for r in records if r.get("type") == "chat"]
    assert len(chat) == 1, f"expected one chat record, got {records}"
    assert chat[0]["outcome"] == "answered"
    assert chat[0]["embed_rid"] == "0804-051310-0002"


def test_an_absent_embed_rid_logs_as_none_rather_than_being_omitted(monkeypatch) -> None:
    """An old cached widget sends no embed_rid. The field must still be present
    in the record -- a missing key and a null one look different to a log query,
    and 'this turn had no gate call' is a fact worth being able to search for."""
    mod = _load_backend()
    records, _ = _drive_chat(mod, monkeypatch, {"question": "who is YC", "history": []}, hits=[])

    refused = [r for r in records if r.get("type") == "chat_refused"]
    assert len(refused) == 1, f"expected one chat_refused record, got {records}"
    assert "embed_rid" in refused[0] and refused[0]["embed_rid"] is None


def test_the_embed_log_line_carries_the_flat_verdict(monkeypatch) -> None:
    """`gate.pass` nested in a JSON object cannot be filtered on in the log
    console: a passed gate and a blocked one look alike at a glance. This
    asserts on the emitted record, not on the method's source text."""
    mod = _load_backend()
    records: list = []
    monkeypatch.setattr(mod, "log", lambda rec: records.append(rec))
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setattr(mod, "gate_decision",
                        lambda t: {"pass": False, "value": 0.1626, "lang": "en"})
    monkeypatch.setattr(mod, "embed_text", lambda t: [0.0])
    monkeypatch.setitem(mod._embed, "session", object())

    handler = _StubHandler(json.dumps({"text": "poem about cats", "gate_only": True}).encode("utf-8"))
    handler.path = "/embed"
    mod.Handler._embed_route(handler)

    embed = [r for r in records if r.get("type") == "embed"]
    assert len(embed) == 1, f"expected one embed record, got {records}"
    assert embed[0]["gate_verdict"] == "block", (
        "the /embed log line must carry the flat token, not only the nested gate object"
    )
    assert embed[0]["gate"]["value"] == 0.1626, "the nested object stays, for the score"


def test_validate_chat_body_rejects_an_oversized_embed_rid() -> None:
    mod = _load_backend()
    base = {"question": "who is YC", "history": []}

    assert mod.validate_chat_body({**base, "embed_rid": "0804-051310-0001"}) is None
    assert mod.validate_chat_body(base) is None, "embed_rid is optional"
    assert mod.validate_chat_body({**base, "embed_rid": "x" * 200}) is not None
    assert mod.validate_chat_body({**base, "embed_rid": 42}) is not None


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_ask_worker_builds_its_body_with_chat_request_body() -> None:
    """Task 29's version brace-matched the object literal out of askWorker, so
    the request-shape tests were pinned to what askWorker actually sends. The
    replacement calls chatRequestBody() standalone, which leaves askWorker free
    to stop using it while every shape test stays green. This is the assertion
    that closes that gap: it reads askWorker's own source for the call, which is
    a structural claim about wiring rather than about behaviour, and behaviour is
    covered by test_the_wire_body_is_what_chat_request_body_built in
    test_widget_resilience.py, which asserts on the bytes handed to fetch()."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    fn = extract_js_function(src, "async function askWorker(")

    assert "chatRequestBody(" in fn, (
        "askWorker no longer builds its body through chatRequestBody -- the "
        "request-shape tests in this file are now testing a function the real "
        "request path does not use"
    )
    assert "JSON.stringify(chatRequestBody(" in " ".join(fn.split()), (
        "askWorker must serialise chatRequestBody's output directly"
    )


def test_page_chunks_never_returns_curated_knowledge_sections() -> None:
    """chat/knowledge/about_*.md sections carry the url of their `link:` target,
    so selecting a page's chunks by url alone picks them up too -- and every
    chunk this returns is labelled current_page="true", which the system prompt
    states IS that page's own content. Measured on the real index before the
    build-time `curated` flag existed: four of the six chunks selected for
    pages/skills.html were about_en.md prose, and once dedup against the
    retrieval hits removed the page's real chunks the prompt's page title came
    off a curated section and read 'the page "Languages he speaks"'."""
    mod = _load_backend()
    index = _fixture_index()
    assert any(c.get("curated") for c in index), "fixture must contain a curated chunk"

    got = mod.page_chunks(index, "pages/skills.html", "en")

    assert got, "the page's own chunks must still be selected"
    assert not any(c.get("curated") for c in got), (
        "curated knowledge prose was selected as the page's own content"
    )
    assert all(c["url"] == "pages/skills.html" for c in got)
    assert got[0]["anchor"] == "top", "the page's summary chunk must still lead"


def test_page_chunks_treats_a_missing_curated_flag_as_page_content() -> None:
    """An index built before the flag existed carries no `curated` key at all.
    Absence must read as "page content" rather than excluding everything, or a
    stale bundled index would silently turn page-awareness off."""
    mod = _load_backend()
    index = [dict(c) for c in _fixture_index()]
    for c in index:
        c.pop("curated", None)

    got = mod.page_chunks(index, "pages/skills.html", "en")

    assert len(got) == mod.PAGE_CONTEXT_MAX, "an unflagged index must still yield page context"


def test_validate_chat_body_accepts_a_known_intent_and_rejects_anything_else() -> None:
    """`intent` is a closed set checked by membership, never used to build a
    prompt, an id or a path -- so an unrecognised value is a bad request rather
    than something to sanitise and carry on with."""
    mod = _load_backend()
    base = {"question": "summarize this page", "history": []}

    assert mod.validate_chat_body({**base, "intent": "summarize_page"}) is None
    assert mod.validate_chat_body(base) is None, "intent is optional -- an old cached widget omits it"
    assert mod.validate_chat_body({**base, "intent": "summarise_page"}) is not None, "near-miss spelling"
    assert mod.validate_chat_body({**base, "intent": "../../etc/passwd"}) is not None
    assert mod.validate_chat_body({**base, "intent": 42}) is not None
    assert mod.validate_chat_body({**base, "intent": ""}) is not None


def test_summarize_page_does_not_retrieve(monkeypatch) -> None:
    """The whole point of the declared intent. Ranking a deictic question
    against the index returns chunks about other pages -- measured on the live
    index, "summarize this page" asked from pages/gyrotris.html retrieved two
    index.html chunks and two chat-agent.html ones, none about Gyrotris. They
    would be injected as retrieved context AND rendered as source cards, since
    `sources` is built from hits. On this path retrieve() must not be called at
    all."""
    mod = _load_backend()
    called: list = []

    def _boom(q):
        called.append(q)
        return [{"chunk": {"id": "x", "url": "pages/other.html", "anchor": "",
                           "page_title": "Other", "section_title": "Other", "text": "noise"},
                 "score": 0.85}]

    monkeypatch.setattr(mod, "retrieve", _boom)
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setattr(mod, "call_llm", lambda s, m: ("a summary", "test-model", {}, "stop"))
    monkeypatch.setattr(mod, "load_roles", lambda: {
        "default_role": "visitor", "base_system_prompt": "base",
        "roles": {"visitor": {"label": "Visitor", "system_prompt": "be helpful"}}})
    monkeypatch.setitem(mod._embed, "session", object())
    monkeypatch.setitem(mod._index, "matrix", object())
    monkeypatch.setitem(mod._index, "chunks", [
        {"id": "pages/skills.html#top:en:0", "url": "pages/skills.html", "anchor": "top",
         "page_title": "Skills", "section_title": "Skills", "text": "the page's own summary",
         "lang": "en"}])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    records: list = []
    monkeypatch.setattr(mod, "log", lambda r: records.append(r))

    handler = _StubHandler(json.dumps({
        "question": "Summarize this page", "history": [], "lang": "en",
        "page": {"url": "pages/skills.html"}, "intent": "summarize_page"}).encode("utf-8"))
    mod.Handler._chat(handler)

    assert called == [], "retrieve() must not run on the summarize_page path"
    status, payload = handler.responses[0]
    assert status == 200 and payload["answer"] == "a summary"
    assert payload["sources"] == [], (
        "sources is built from hits, which is empty here -- so no card may point "
        "at a page the answer did not come from"
    )
    chat = [r for r in records if r.get("type") == "chat"][0]
    assert chat["intent"] == "summarize_page"
    assert chat["in"]["contexts"] == [], "no retrieved contexts were used"
    assert chat["in"]["page_context"] == ["pages/skills.html#top:en:0"]
    assert "asked for a summary of this page" in chat["in"]["system_head"]


def test_an_ordinary_question_still_retrieves(monkeypatch) -> None:
    """Complement: the skip is scoped to the declared intent. A test that only
    checked the skip would pass on an implementation that never retrieves."""
    mod = _load_backend()
    called: list = []
    monkeypatch.setattr(mod, "retrieve", lambda q: called.append(q) or [])
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setitem(mod._embed, "session", object())
    monkeypatch.setitem(mod._index, "matrix", object())
    monkeypatch.setitem(mod._index, "chunks", [])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(mod, "log", lambda r: None)

    handler = _StubHandler(json.dumps({"question": "what engines", "history": []}).encode("utf-8"))
    mod.Handler._chat(handler)

    assert called == ["what engines"], "an ordinary question must still retrieve"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_intent_reaches_the_request_body_and_is_absent_otherwise() -> None:
    """The chip's whole contribution is this field. Absent on an ordinary
    question so the server's `intent is not None` check means what it says."""
    src = WIDGET_PATH.read_text(encoding="utf-8")

    def body(intent_js: str) -> dict:
        script = (
            "const state = { session: 's', role: 'visitor', history: [] };\n"
            "function lang() { return 'en'; }\n"
            "const window = { location: { pathname: '/pages/skills.html' } };\n"
            + extract_js_function(src, "function currentPageUrl(") + "\n"
            + extract_js_function(src, "function chatRequestBody(") + "\n"
            "process.stdout.write(JSON.stringify(chatRequestBody('q', 'rid', " + intent_js + ")));\n"
        )
        return run_node_json(script)

    assert body("'summarize_page'")["intent"] == "summarize_page"
    assert "intent" not in body("undefined"), (
        "an ordinary question must omit intent entirely, not send null or ''"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_widget_only_ever_declares_intents_the_backend_accepts() -> None:
    """Both sides hold the same closed set and there is no build step to keep
    them in sync, so this is the only thing that would catch a rename on one
    side -- the same cross-implementation drift test_retrieval_sync.py exists
    for."""
    import re

    mod = _load_backend()
    src = WIDGET_PATH.read_text(encoding="utf-8")
    # pageAction() is the single place the widget names an intent, so read the
    # literals out of that function rather than pattern-matching call sites --
    # the call site passes a variable now, which a call-site pattern silently
    # finds nothing in.
    fn = extract_js_function(src, "function pageAction(")
    declared = set(re.findall(r"intent:\s*'([a-z_]+)'", fn))

    assert declared, "expected pageAction() to name at least one intent"
    assert declared <= set(mod.INTENTS), (
        f"widget sends {sorted(declared - set(mod.INTENTS))}, which index.py's "
        f"INTENTS {mod.INTENTS} would reject as a bad request"
    )


def test_summary_chunks_is_one_per_page_in_the_answer_language() -> None:
    """The whole context for top_projects: every page's own summary, once."""
    mod = _load_backend()
    index = _fixture_index()

    got = mod.summary_chunks(index, "en")

    assert {c["url"] for c in got} == {"pages/prime-engine.html", "pages/skills.html"}
    assert len(got) == 2, "one summary per page, not one per language or per chunk"
    assert all(c["anchor"] == "top" for c in got)
    assert all(c["lang"] == "en" for c in got)
    assert not any(c.get("curated") for c in got), (
        "a curated section carries its link: target's url, so it would appear as "
        "that page's summary"
    )


def test_sources_for_named_returns_only_pages_the_answer_mentions() -> None:
    """Turns prose into clickable cards without asking the model for structured
    output. Unmatched phrasing costs a card, never a parse error."""
    mod = _load_backend()
    cat = [
        {"id": "a", "url": "pages/cemented-dreams.html", "anchor": "top",
         "page_title": "Cemented Dreams", "section_title": "Cemented Dreams", "text": "x"},
        {"id": "b", "url": "pages/gyrotris.html", "anchor": "top",
         "page_title": "Gyrotris", "section_title": "Gyrotris", "text": "y"},
        {"id": "c", "url": "pages/skills.html", "anchor": "top",
         "page_title": "Skills", "section_title": "Skills", "text": "z"},
    ]

    got = mod.sources_for_named(cat, "Start with Cemented Dreams, then Gyrotris.")

    assert [c["url"] for c in got] == ["pages/cemented-dreams.html", "pages/gyrotris.html"]
    assert mod.sources_for_named(cat, "") == []
    assert mod.sources_for_named(cat, None) == []
    assert mod.sources_for_named(cat, "nothing here matches any page") == []


def test_top_projects_uses_the_catalogue_and_does_not_retrieve(monkeypatch) -> None:
    """The hub-page path. index.html has two chunks and projects.html three, so
    summarizing them is close to reading them -- this answers the question the
    visitor actually has there, from every page's summary plus their role."""
    mod = _load_backend()
    called: list = []
    monkeypatch.setattr(mod, "retrieve", lambda q: called.append(q) or [])
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setattr(mod, "call_llm",
                        lambda s, m: ("Look at Cemented Dreams first, then Prime Engine.",
                                      "test-model", {}, "stop"))
    monkeypatch.setattr(mod, "load_roles", lambda: {
        "default_role": "visitor", "base_system_prompt": "base",
        "roles": {"visitor": {"label": "Visitor", "system_prompt": "be helpful"}}})
    monkeypatch.setitem(mod._embed, "session", object())
    monkeypatch.setitem(mod._index, "matrix", object())
    monkeypatch.setitem(mod._index, "chunks", [
        {"id": "1", "url": "pages/cemented-dreams.html", "anchor": "top", "lang": "en",
         "page_title": "Cemented Dreams", "section_title": "Cemented Dreams", "text": "combat"},
        {"id": "2", "url": "pages/prime-engine.html", "anchor": "top", "lang": "en",
         "page_title": "Prime Engine", "section_title": "Prime Engine", "text": "engine"},
        {"id": "3", "url": "pages/toolbox.html", "anchor": "top", "lang": "en",
         "page_title": "Toolbox", "section_title": "Toolbox", "text": "tools"},
    ])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    records: list = []
    monkeypatch.setattr(mod, "log", lambda r: records.append(r))

    handler = _StubHandler(json.dumps({
        "question": "What should I look at?", "history": [], "lang": "en",
        "page": {"url": "index.html"}, "intent": "top_projects"}).encode("utf-8"))
    mod.Handler._chat(handler)

    assert called == [], "top_projects must not retrieve"
    status, payload = handler.responses[0]
    assert status == 200
    assert [s["url"] for s in payload["sources"]] == [
        "pages/cemented-dreams.html", "pages/prime-engine.html"], (
        "cards must be exactly the pages the answer named -- Toolbox was not mentioned")
    assert all("score" not in s for s in payload["sources"]), (
        "nothing was ranked, so no similarity may be reported")
    chat = [r for r in records if r.get("type") == "chat"][0]
    head = chat["in"]["system_head"]
    assert "currently viewing the page" not in head, (
        "the page-awareness wording would name whichever page sorts first, which "
        "is not where the visitor is")
    assert "most worth their time" in head


def test_top_projects_recommends_nothing_it_cannot_link(monkeypatch) -> None:
    """Complement: if the answer names no page, the card strip is empty rather
    than falling back to the whole catalogue."""
    mod = _load_backend()
    monkeypatch.setattr(mod, "retrieve", lambda q: [])
    monkeypatch.setattr(mod, "rate_limited", lambda *a, **k: False)
    monkeypatch.setattr(mod, "call_llm", lambda s, m: ("I have nothing to suggest.", "m", {}, "stop"))
    monkeypatch.setattr(mod, "load_roles", lambda: {
        "default_role": "visitor", "base_system_prompt": "base",
        "roles": {"visitor": {"label": "Visitor", "system_prompt": "x"}}})
    monkeypatch.setitem(mod._embed, "session", object())
    monkeypatch.setitem(mod._index, "matrix", object())
    monkeypatch.setitem(mod._index, "chunks", [
        {"id": "1", "url": "pages/gyrotris.html", "anchor": "top", "lang": "en",
         "page_title": "Gyrotris", "section_title": "Gyrotris", "text": "g"}])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(mod, "log", lambda r: None)

    handler = _StubHandler(json.dumps({
        "question": "What should I look at?", "history": [], "lang": "en",
        "intent": "top_projects"}).encode("utf-8"))
    mod.Handler._chat(handler)

    assert handler.responses[0][1]["sources"] == []
