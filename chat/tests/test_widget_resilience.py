"""Client-side resilience guarantees, executed against the REAL widget source.

Every function under test is extracted verbatim from scripts/chat-widget.js and
run in a node subprocess (see tests/_node_harness.py) rather than retyped here
-- a Python reimplementation would drift from the file it is meant to mirror,
which is the exact failure mode the other *_sync.py tests exist to prevent.
"""

import re

import pytest

from portfolio_rag.config import settings
from tests._node_harness import extract_js_function, extract_js_var, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"


def _widget_src() -> str:
    return WIDGET_PATH.read_text(encoding="utf-8")


def _index_call_llm_timeout() -> int:
    """The literal seconds value from call_llm's own urlopen(...) --
    load_roles has a second, unrelated urlopen with its own (shorter)
    timeout, so the match is scoped to call_llm's definition specifically."""
    src = BACKEND_PATH.read_text(encoding="utf-8")
    fn_src = re.search(r"def call_llm\(.*?\n(?=def )", src, re.S)
    assert fn_src, "expected to find call_llm's definition in index.py"
    t = re.search(r"urlopen\([^)]*timeout=(\d+)", fn_src.group(0))
    assert t, "expected call_llm's urlopen(...) to declare an explicit timeout="
    return int(t.group(1))


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
    implementation that aborts immediately). fetchWithTimeout now resolves
    {res, done} instead of the bare Response (Item 1) -- the deadline is held
    open until the caller's body read finishes, so callers read r.res and
    call r.done() themselves once that read is over."""
    script = (
        extract_js_function(_widget_src(), "function fetchWithTimeout(") + "\n"
        "global.fetch = function (url, opts) { return Promise.resolve({ ok: true, url: url }); };\n"
        "fetchWithTimeout('http://example.invalid/chat', { method: 'POST' }, 5000)\n"
        "  .then(function (r) {\n"
        "    process.stdout.write(JSON.stringify({\n"
        "      ok: r.res.ok, url: r.res.url, hasDone: typeof r.done === 'function',\n"
        "    }));\n"
        "  });\n"
    )
    got = run_node_json(script)

    assert got["ok"] is True
    assert got["url"] == "http://example.invalid/chat"
    assert got["hasDone"] is True, "the caller needs done() to release the deadline once its body read finishes"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_fetch_with_timeout_aborts_a_response_whose_body_never_settles() -> None:
    """The gap Item 1 closes: fetch() itself settles as soon as headers
    arrive, so a deadline that stopped there (the old `.finally` on the fetch
    promise) let a backend that sends headers and then stalls mid-body hang
    forever -- the exact forever-spinner bug one step later. The timer must
    stay armed past the headers and abort the still-open body read too."""
    script = (
        extract_js_function(_widget_src(), "function fetchWithTimeout(") + "\n"
        # Headers arrive immediately; the body (`.json()`) never settles on
        # its own -- only the shared abort signal can end it, same as a real
        # Response whose body stream is tied to the request's AbortController.
        "global.fetch = function (url, opts) {\n"
        "  return Promise.resolve({\n"
        "    ok: true,\n"
        "    json: function () {\n"
        "      return new Promise(function (resolve, reject) {\n"
        "        opts.signal.addEventListener('abort', function () {\n"
        "          var e = new Error('aborted'); e.name = 'AbortError'; reject(e);\n"
        "        });\n"
        "      });\n"
        "    },\n"
        "  });\n"
        "};\n"
        "var started = Date.now();\n"
        "fetchWithTimeout('http://example.invalid', { method: 'POST' }, 50)\n"
        "  .then(function (r) { return r.res.json(); })\n"
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

    assert got["settled"] == "rejected", "a body that stalls after headers arrive must not hang forever"
    assert got["name"] == "AbortError"
    assert got["elapsed"] < 2000, "the deadline must fire promptly, not at some default"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_chat_deadline_outlives_the_functions_own_llm_timeout() -> None:
    """index.py's call_llm uses urlopen(timeout=N). A client deadline shorter
    than that would abandon requests the server is still going to answer --
    N is parsed from index.py's own source, not hardcoded, so raising the
    server timeout can't leave this guard silently green."""
    src = _widget_src()
    chat_ms = int(extract_js_var(src, "CHAT_TIMEOUT_MS"))
    embed_ms = int(extract_js_var(src, "EMBED_TIMEOUT_MS"))
    llm_timeout_s = _index_call_llm_timeout()

    assert chat_ms > llm_timeout_s * 1000, "the /chat deadline must sit past call_llm's own urlopen timeout"
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


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_an_unanswered_question_is_detected_on_replay() -> None:
    """pushLog records the bot reply only when a turn completes, so navigating
    away mid-flight (clicking a source card does exactly this) leaves a user
    entry with nothing after it. Replaying that silently looks like the chat
    ignored the question."""
    script = (
        extract_js_function(_widget_src(), "function lastTurnInterrupted(") + "\n"
        "process.stdout.write(JSON.stringify({\n"
        "  interrupted: lastTurnInterrupted([{type:'note'},{type:'user'}]),\n"
        "  answered: lastTurnInterrupted([{type:'user'},{type:'bot'},{type:'sources'}]),\n"
        "  empty: lastTurnInterrupted([]),\n"
        "  starters: lastTurnInterrupted([{type:'note'},{type:'starters'}]),\n"
        "}));\n"
    )
    got = run_node_json(script)

    assert got["interrupted"] is True
    assert got["answered"] is False
    assert got["empty"] is False, "a fresh transcript is not an interrupted turn"
    assert got["starters"] is False


def test_the_interrupted_notice_exists_in_both_languages() -> None:
    """Every visitor-facing string is bilingual; a missing zh key silently
    falls back to English mid-conversation."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    assert src.count("interrupted:") == 2, "expected one `interrupted` string in STR.en and STR.zh"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_hub_pages_are_dropped_from_the_displayed_cards_only() -> None:
    """index.html has no section anchors and pages/projects.html is a listing,
    so a card pointing at either drops the visitor somewhere that answers
    nothing. Curated about_en.md chunks deliberately link there, so the chunks
    must still reach the LLM -- this filter is display-only."""
    src = _widget_src()
    script = (
        "var HUB_URLS = " + extract_js_var(src, "HUB_URLS") + ";\n"
        + extract_js_function(src, "function displayableSources(") + "\n"
        "var results = [\n"
        "  { chunk: { url: 'pages/prime-engine.html' }, score: 0.5 },\n"
        "  { chunk: { url: 'index.html' }, score: 0.4 },\n"
        "  { chunk: { url: 'pages/projects.html' }, score: 0.3 },\n"
        "  { chunk: { url: 'pages/skills.html' }, score: 0.2 },\n"
        "];\n"
        "process.stdout.write(JSON.stringify({\n"
        "  shown: displayableSources(results).map(function (r) { return r.chunk.url; }),\n"
        "  originalLength: results.length,\n"
        "}));\n"
    )
    got = run_node_json(script)

    assert got["shown"] == ["pages/prime-engine.html", "pages/skills.html"]
    assert got["originalLength"] == 4, "displayableSources must not mutate its input"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_an_all_hub_result_set_renders_no_empty_wrapper() -> None:
    src = _widget_src()
    script = (
        "var HUB_URLS = " + extract_js_var(src, "HUB_URLS") + ";\n"
        + extract_js_function(src, "function displayableSources(") + "\n"
        "process.stdout.write(JSON.stringify(displayableSources([\n"
        "  { chunk: { url: 'index.html' } }, { chunk: { url: 'pages/projects.html' } },\n"
        "])));\n"
    )
    assert run_node_json(script) == []


def test_source_logging_keeps_the_full_ranked_list() -> None:
    """The filter must sit in addSources, not in dedupeForDisplay -- otherwise
    sourcesForLog and record.retrieved lose the audit trail too."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    dedupe = extract_js_function(src, "function dedupeForDisplay(")
    sources_for_log = extract_js_function(src, "function sourcesForLog(")

    assert "displayableSources" not in dedupe, "dedupeForDisplay must stay display-neutral"
    assert "displayableSources" not in sources_for_log, "the log must keep every retrieved source"
    assert "displayableSources" in extract_js_function(src, "function addSources(")


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_degraded_sources_message_is_gated_on_the_displayable_count() -> None:
    """Task 6 made addSources drop hub-only cards (index.html,
    pages/projects.html) at render time. runOfflineSearch's `degradedSources`
    branch decides WHICH sentence to show before that filter ever runs, so if
    it still gates on the raw retrieved.results.length, an all-hub result set
    -- exactly what a broad identity question retrieves, since 11 of 55
    curated about_en.md sections link to a hub -- says "these pages look most
    relevant to your question:" and then renders no cards at all: a visitor
    who just consented to a ~23MB model download hits a dead end. The branch
    must gate on the hub-filtered (displayable) count instead, so an all-hub
    result set falls through to the degradedNoSources + addPageLinks branch
    right below it, which already handles "nothing to point to" correctly."""
    fn_src = extract_js_function(_widget_src(), "async function runOfflineSearch(")

    branch = re.search(r"\}\s*else if\s*\(([^)]*)\)\s*\{[^}]*?degradedSources", fn_src, re.S)
    assert branch, "expected a `} else if (...) { ... degradedSources ... }` branch"
    condition = branch.group(1).strip()

    assert "retrieved.results.length" not in condition, (
        f"the degradedSources branch condition ({condition!r}) must not test the raw "
        "retrieved count -- an all-hub result set has retrieved.results.length > 0 "
        "but addSources renders zero cards for it, so the visitor sees the "
        "'these pages look most relevant' sentence followed by nothing"
    )

    assign = re.search(r"var\s+(\w+)\s*=\s*displayableSources\(retrieved\.results\)", fn_src)
    assert assign, (
        "expected `var <name> = displayableSources(retrieved.results);` computed "
        "before the branch, so the condition can test what will actually render"
    )
    shown_var = assign.group(1)
    assert shown_var in condition, (
        f"the degradedSources branch condition ({condition!r}) must test "
        f"{shown_var}.length (the hub-filtered count), not something else"
    )

    # The filter must only steer WHICH message is shown -- addSources and the
    # audit trail still need the full ranked list (Task 6's own design: the
    # curated hub-linking chunks are good grounding even though the card for
    # them is not worth showing).
    assert "addSources(retrieved.results)" in fn_src, (
        "addSources must still receive the FULL retrieved list, not the filtered one"
    )
    assert "sourcesForLog(retrieved.results)" in fn_src, (
        "the log must still keep the full ranked list, not the filtered one"
    )


# ── askWorker: a 502 is not always "the backend is down" ────────────────────
#
# index.py returns 502 for two different reasons: a genuine LLM call failure
# (degraded mode is the honest answer) and an LLM that answered but with
# EMPTY content (usable_answer()'s guard, Task 3's server half) -- the second
# means the backend is reachable and responding, so telling the visitor to
# download a 23MB offline model would be false. The fix teaches askWorker to
# read the 502 body and mark that one case `passThrough`; send()'s chatErr
# handler (tested separately below) reads that flag instead of the old
# fragile `indexOf('rate limited')` string check.


def _run_ask_worker(status: int, body_js: str) -> dict:
    """Execute the REAL askWorker, together with its REAL fetchWithTimeout,
    in node against a mocked global.fetch -- so the 502-body-sniffing logic
    runs for real rather than being redescribed in Python. `body_js` is a JS
    statement body for the mocked response's `.json()` (e.g.
    "return Promise.resolve({...});", or a rejection for a non-JSON gateway
    502 body)."""
    src = _widget_src()
    script = (
        "var WORKER_URL = 'http://example.invalid';\n"
        "var CHAT_TIMEOUT_MS = 5000;\n"
        "var state = { session: 's1', role: 'visitor', history: [] };\n"
        "function lang() { return 'en'; }\n"
        # askWorker builds its body through chatRequestBody, which reads
        # window.location -- extract both for real rather than stubbing the
        # body, so a change to the request shape cannot pass unnoticed here.
        "var window = { location: { pathname: '/pages/skills.html' } };\n"
        + extract_js_function(src, "function fetchWithTimeout(") + "\n"
        + extract_js_function(src, "function currentPageUrl(") + "\n"
        + extract_js_function(src, "function chatRequestBody(") + "\n"
        + extract_js_function(src, "async function askWorker(") + "\n"
        # Record what askWorker actually handed to fetch(). The stub used to
        # accept opts and discard it, so nothing in this repo ever read the wire
        # body -- every other request-shape test calls chatRequestBody() standalone,
        # which cannot see askWorker dropping it.
        "var sentBody = null;\n"
        "global.fetch = function (url, opts) {\n"
        "  sentBody = opts && opts.body;\n"
        "  return Promise.resolve({\n"
        "    ok: " + ("true" if status < 300 else "false") + ",\n"
        "    status: " + str(status) + ",\n"
        "    json: function () { " + body_js + " },\n"
        "  });\n"
        "};\n"
        "askWorker('who is YC').then(function (r) {\n"
        "  process.stdout.write(JSON.stringify({ threw: false, result: r, sentBody: sentBody }));\n"
        "}).catch(function (e) {\n"
        "  process.stdout.write(JSON.stringify({\n"
        "    threw: true, message: String(e && e.message), passThrough: !!(e && e.passThrough),\n"
        "    sentBody: sentBody,\n"
        "  }));\n"
        "});\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_askworker_flags_the_empty_answer_502_for_pass_through() -> None:
    got = _run_ask_worker(502, "return Promise.resolve({ error: 'llm returned an empty answer' });")

    assert got["threw"] is True
    assert got["passThrough"] is True, "the empty-answer 502 must be marked passThrough"
    assert got["message"] == "the model returned an empty answer", (
        "must match send()'s own client-side empty-answer guard message verbatim, "
        "so the two Task 3 halves are indistinguishable to a visitor"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_askworker_a_different_502_still_falls_to_degraded_mode() -> None:
    """Complement: only the empty-answer 502 bypasses degraded mode. index.py's
    OTHER 502 ("llm call failed", a genuine LLM-call failure) is a real
    backend problem and must keep today's behavior."""
    got = _run_ask_worker(502, "return Promise.resolve({ error: 'llm call failed' });")

    assert got["threw"] is True
    assert got["passThrough"] is False, "a genuine LLM-call-failure 502 must not pass through"
    assert got["message"] == "worker 502"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_askworker_a_gateway_502_with_no_json_body_does_not_throw_while_parsing() -> None:
    """A raw gateway 502 (e.g. Tencent's own error page, not index.py's JSON)
    has no parseable body at all. Reading it for the empty-answer check must
    not itself throw -- that would replace an honest 'worker 502' with a
    confusing JSON-parse error, and skip degraded mode entirely."""
    got = _run_ask_worker(502, "return Promise.reject(new Error('Unexpected token < in JSON'));")

    assert got["threw"] is True
    assert got["passThrough"] is False
    assert got["message"] == "worker 502", "a gateway 502 must keep today's behavior, not a parse error"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_askworker_429_still_passes_through_with_an_unchanged_message() -> None:
    """The rate-limit throw is being converted to the same `passThrough`
    mechanism as the new empty-answer case. The message text -- what a
    visitor ultimately sees via t('somethingWrong', err.message) -- must not
    change in the conversion."""
    got = _run_ask_worker(429, "return Promise.resolve({});")

    assert got["threw"] is True
    assert got["passThrough"] is True
    assert got["message"] == "rate limited — please wait a minute and try again"


# ── send()'s chatErr handler: passThrough decides degraded mode vs. not ─────


def _run_chat_err_branch(err_js: str) -> dict:
    """Execute send()'s REAL `catch (chatErr) { ... }` clause -- extracted
    verbatim, including its passThrough condition -- against a synthetic
    thrown error built by `err_js` (a JS expression), in node. Everything the
    clause touches from send()'s closure (state, question, stripped, record,
    thinking, logTurn, degradedTurn) is stubbed and recorded; `run()`'s own
    outer try/catch is what distinguishes "chatErr rethrew" (send()'s outer
    catch would fire) from "degradedTurn ran and the turn returned"."""
    src = _widget_src()
    catch_src = extract_js_function(src, "catch (chatErr) {")
    script = (
        "var calls = [];\n"
        "var state = { role: 'visitor' };\n"
        "var question = 'who is YC';\n"
        "var stripped = '';\n"
        "var record = {};\n"
        "var thinking = {};\n"
        "function logTurn(e) { calls.push('logTurn:' + e.event); }\n"
        "async function degradedTurn(q, s, t, r) { calls.push('degradedTurn'); }\n"
        "async function run() {\n"
        "  try { throw (" + err_js + "); }\n"
        + catch_src + "\n"
        "}\n"
        "run().then(function () {\n"
        "  process.stdout.write(JSON.stringify({ rethrew: false, calls: calls }));\n"
        "}).catch(function (e) {\n"
        "  process.stdout.write(JSON.stringify({\n"
        "    rethrew: true, calls: calls, message: String(e && e.message),\n"
        "  }));\n"
        "});\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_err_rethrows_a_pass_through_error_without_degrading() -> None:
    """The empty-answer error (askWorker, passThrough = true) must reach
    send()'s outer catch -- t('somethingWrong') -- and must NOT trigger
    degraded mode's "the AI answer service is unreachable" message, which
    would be false: the backend answered, just with nothing usable."""
    err_js = (
        "(function () { "
        "var e = new Error('the model returned an empty answer'); "
        "e.passThrough = true; return e; "
        "})()"
    )
    got = _run_chat_err_branch(err_js)

    assert got["rethrew"] is True
    assert got["calls"] == [], "must rethrow before logging or calling degradedTurn"
    assert got["message"] == "the model returned an empty answer"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_err_still_degrades_on_a_generic_worker_error() -> None:
    """Complement: an ordinary askWorker failure (e.g. `worker 500`, no
    passThrough) must still fall into degraded mode -- a test that only
    checked the pass-through path would pass on a handler that always
    rethrows."""
    got = _run_chat_err_branch("new Error('worker 500')")

    assert got["rethrew"] is False
    assert "degradedTurn" in got["calls"]
    assert any(c.startswith("logTurn:") for c in got["calls"]), (
        "a /chat failure that falls to degraded mode must still be logged"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_err_rate_limit_behavior_is_unchanged_by_the_mechanism_switch() -> None:
    """The rate-limit error is being converted from the old
    `indexOf('rate limited') === 0` string check to the same `passThrough`
    flag as the empty-answer case. This pins that the visible behavior --
    still reaches the outer catch, still the same message -- did not move
    when the mechanism did."""
    err_js = (
        "(function () { "
        "var e = new Error('rate limited — please wait a minute and try again'); "
        "e.passThrough = true; return e; "
        "})()"
    )
    got = _run_chat_err_branch(err_js)

    assert got["rethrew"] is True
    assert got["calls"] == []
    assert got["message"] == "rate limited — please wait a minute and try again"


# ── Task 9 / Task 10: don't let the first message pay for a cold container ──
#
# functions/tencent/index.py's main() loads three ONNX models and the chunk
# index BEFORE binding its port, so a cold container's first request waits
# out the whole load. Task 9 added a longer per-attempt deadline
# (COLD_START_BUDGET_MS + embedDeadline()) as insurance against that wait
# outlasting EMBED_TIMEOUT_MS. Task 10 removed both: production measured the
# real cold start at 2.5-2.6s ("Init Report Coldstart: 2552ms" / "...2645ms"),
# well inside EMBED_TIMEOUT_MS's 20000ms, so the extra budget was guarding a
# failure mode that never happens -- its only live effect was letting a
# genuinely dead backend take ~96s to reach degraded mode instead of ~41s.
# embedQuery now uses one flat EMBED_TIMEOUT_MS for both attempts;
# prewarm() (still a real win -- see its own comment in chat-widget.js)
# still spends the visitor's role-picking time on the 2.5s spin-up.


def test_warming_notice_fires_well_before_the_embed_deadline_gives_up() -> None:
    """Values, not behavior -- but a notice that fires after the deadline
    already gave up on the request is dead code."""
    src = _widget_src()
    embed_ms = int(extract_js_var(src, "EMBED_TIMEOUT_MS"))
    warming_ms = int(extract_js_var(src, "WARMING_NOTICE_MS"))

    assert warming_ms < embed_ms, "the warming notice must fire well before the deadline gives up"


def _run_prewarm(worker_url_js: str, times: int) -> dict:
    src = _widget_src()
    script = (
        "var calls = [];\n"
        "var WORKER_URL = " + worker_url_js + ";\n"
        "var EMBED_TIMEOUT_MS = " + extract_js_var(src, "EMBED_TIMEOUT_MS") + ";\n"
        "var state = { prewarmed: false, backendWarm: false };\n"
        "function fetchWithTimeout(url, opts, ms) {\n"
        "  calls.push({ url: url, ms: ms });\n"
        "  return Promise.resolve({ res: { ok: true }, done: function () {} });\n"
        "}\n"
        + extract_js_function(src, "function prewarm(") + "\n"
        + "\n".join(["prewarm();"] * times) + "\n"
        "process.stdout.write(JSON.stringify({ count: calls.length, calls: calls }));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_prewarm_fires_at_most_once_per_page_load() -> None:
    """toggle() can be opened, closed and reopened repeatedly in one page
    load; prewarm must not re-fire the request each time."""
    got = _run_prewarm("'http://example.invalid'", 3)

    assert got["count"] == 1, "repeated opens must not re-fire the prewarm request"
    assert got["calls"][0]["url"] == "http://example.invalid/"
    assert got["calls"][0]["ms"] == int(extract_js_var(_widget_src(), "EMBED_TIMEOUT_MS")), (
        "prewarm's fetchWithTimeout deadline must be EMBED_TIMEOUT_MS -- the "
        "separate COLD_START_BUDGET_MS it used to reference was removed "
        "(Task 10) once production measured cold starts at 2.5-2.6s"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_prewarm_does_nothing_without_a_configured_worker() -> None:
    got = _run_prewarm("''", 3)

    assert got["count"] == 0, "no backend configured means there is nothing to prewarm"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_prewarm_swallows_a_rejected_fetch_without_an_unhandled_rejection() -> None:
    """Prewarming is a hint to the platform, not a request the widget needs
    answered -- a failed prewarm must never surface anywhere, including as an
    unhandled promise rejection (node exits non-zero on those, which is
    exactly what would make this test fail if the .catch were missing)."""
    src = _widget_src()
    script = (
        "var WORKER_URL = 'http://example.invalid';\n"
        "var EMBED_TIMEOUT_MS = " + extract_js_var(src, "EMBED_TIMEOUT_MS") + ";\n"
        "var state = { prewarmed: false, backendWarm: false };\n"
        "function fetchWithTimeout() { return Promise.reject(new Error('down')); }\n"
        + extract_js_function(src, "function prewarm(") + "\n"
        "prewarm();\n"
        "setTimeout(function () { process.stdout.write(JSON.stringify({ ok: true })); }, 50);\n"
    )
    got = run_node_json(script)

    assert got["ok"] is True, "a rejected prewarm fetch must not crash the page"


def test_warming_notice_exists_in_both_languages_without_a_double_ellipsis() -> None:
    """The ycchat-dots class the thinking bubble keeps on throughout appends
    an animated ellipsis via CSS ::after; a string that also ends in one
    would render two."""
    src = _widget_src()
    assert src.count("warming:") == 2, "expected one `warming` string in STR.en and STR.zh"
    for m in re.finditer(r"warming:\s*'([^']*)'", src):
        text = m.group(1)
        assert not text.endswith("…") and not text.endswith("..."), (
            f"{text!r} ends in an ellipsis; ycchat-dots already animates one via ::after"
        )


def _run_embed_query_deadlines() -> dict:
    """Drive the REAL embedQuery against a fetchWithTimeout stub that always
    fails (status 500, not 429) so both retry attempts run, recording the
    `ms` deadline passed on each. localModelMatchesIndex is stubbed false so
    the function settles (rejects) once the retries are exhausted -- the
    settlement itself isn't what this test is about, only which deadline
    each attempt was given."""
    src = _widget_src()
    script = (
        "var WORKER_URL = 'http://example.invalid';\n"
        "var EMBED_TIMEOUT_MS = " + extract_js_var(src, "EMBED_TIMEOUT_MS") + ";\n"
        "var state = { remoteEmbedDownAt: null, backendWarm: false };\n"
        "function remoteEmbedDown() { return false; }\n"
        "function logTurn() {}\n"
        "function localModelMatchesIndex() { return false; }\n"
        "var calls = [];\n"
        "function fetchWithTimeout(url, opts, ms) {\n"
        "  calls.push(ms);\n"
        "  return Promise.resolve({ res: { ok: false, status: 500 }, done: function () {} });\n"
        "}\n"
        + extract_js_function(src, "async function embedQuery(") + "\n"
        "embedQuery('q', 'q').then(function () {\n"
        "  process.stdout.write(JSON.stringify({ threw: false, calls: calls }));\n"
        "}).catch(function (e) {\n"
        "  process.stdout.write(JSON.stringify({\n"
        "    threw: true, calls: calls, message: String(e && e.message),\n"
        "  }));\n"
        "});\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_embed_query_uses_a_flat_deadline_for_both_attempts() -> None:
    """Task 9 gave the first attempt a longer COLD_START_BUDGET_MS via
    embedDeadline() (state.backendWarm chose between the two); Task 10
    removed both once production measured real cold starts at 2.5-2.6s (see
    EMBED_TIMEOUT_MS's comment in chat-widget.js) -- nowhere near enough to
    justify a separate, much longer per-attempt budget. Both retry attempts
    now get the same EMBED_TIMEOUT_MS regardless of state.backendWarm."""
    got = _run_embed_query_deadlines()

    embed_ms = int(extract_js_var(_widget_src(), "EMBED_TIMEOUT_MS"))
    assert len(got["calls"]) == 2, "embedQuery must still make exactly two attempts"
    assert got["calls"] == [embed_ms, embed_ms]


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_embed_query_marks_the_backend_warm_on_a_successful_response() -> None:
    """A successful /embed is the signal that gates the warming notice
    (WARMING_NOTICE_MS) for the rest of the session."""
    src = _widget_src()
    script = (
        "var WORKER_URL = 'http://example.invalid';\n"
        "var EMBED_TIMEOUT_MS = " + extract_js_var(src, "EMBED_TIMEOUT_MS") + ";\n"
        "var state = { remoteEmbedDownAt: null, backendWarm: false };\n"
        "function remoteEmbedDown() { return false; }\n"
        "function logTurn() {}\n"
        "function fetchWithTimeout(url, opts, ms) {\n"
        "  return Promise.resolve({\n"
        "    res: { ok: true, json: function () { return Promise.resolve({ gate: null, rid: 'r1' }); } },\n"
        "    done: function () {},\n"
        "  });\n"
        "}\n"
        + extract_js_function(src, "async function embedQuery(") + "\n"
        "embedQuery('q', 'q').then(function (r) {\n"
        "  process.stdout.write(JSON.stringify({ backendWarm: state.backendWarm, rid: r.rid }));\n"
        "});\n"
    )
    got = run_node_json(script)

    assert got["backendWarm"] is True
    assert got["rid"] == "r1"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_wire_body_is_what_chat_request_body_built() -> None:
    """The bytes askWorker actually hands to fetch(), not the builder in
    isolation.

    _run_ask_worker's mock used to accept (url, opts) and discard opts, so
    nothing in this repo read the wire body -- and the comment in the harness
    claimed the opposite. Every other request-shape test calls chatRequestBody()
    standalone, which cannot see askWorker dropping it. This closes that: it
    parses what was sent and checks the two fields page-awareness added."""
    got = _run_ask_worker(200, "return Promise.resolve({ answer: 'x', sources: [] });")

    import json as _json

    assert got.get("sentBody"), "askWorker sent no request body"
    sent = _json.loads(got["sentBody"])

    assert sent["page"] == {"url": "pages/skills.html"}, (
        "the page url must reach the wire in the index's site-relative form"
    )
    assert set(sent["page"]) == {"url"}, "only the url crosses the wire"
    assert sent["question"] == "who is YC"
    for field in ("session", "role", "lang", "history"):
        assert field in sent, f"the wire body is missing {field!r}"


def _chip_for(pathname: str) -> dict:
    """Run the widget's REAL pageAction() against a stand-in page.

    Extracted, not reimplemented. An earlier version of this helper rewrote the
    hub branch in its own node script, which meant a mutation to the widget's
    real branch could not turn it red -- retyped JS drifting from the file it
    mirrors, the exact failure this module's docstring exists to refuse. It was
    caught by mutating `onHub` and watching this stay green."""
    import json as _json

    src = _widget_src()
    script = (
        "var HUB_URLS = " + extract_js_var(src, "HUB_URLS") + ";\n"
        f"var window = {{ location: {{ pathname: {_json.dumps(pathname)} }} }};\n"
        + extract_js_function(src, "function currentPageUrl(") + "\n"
        + extract_js_function(src, "function pageAction(") + "\n"
        "process.stdout.write(JSON.stringify(pageAction()));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_the_chip_asks_a_different_question_on_the_hub_pages() -> None:
    """index.html carries two chunks and projects.html three, so "summarize this
    page" there is barely shorter than the page. HUB_URLS already marks both --
    it suppresses them as source cards for the same reason."""
    import json as _json

    for hub in ("/", "/index.html", "/pages/projects.html"):
        got = _chip_for(hub)
        assert got["intent"] == "top_projects", f"{hub} is a hub page"

    for page in ("/pages/gyrotris.html", "/pages/skills.html", "/pages/chat-agent.html"):
        got = _chip_for(page)
        assert got["intent"] == "summarize_page", f"{page} is an ordinary page"


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_a_source_card_shows_a_score_only_when_one_was_measured() -> None:
    """A recommendation card reports no similarity, because nothing was ranked
    to produce it. Faking 1.00 would make the number meaningless on the
    retrieval cards too."""
    import json as _json

    src = _widget_src()

    def render(source: dict) -> str:
        script = (
            "var PREFIX = '';\n"
            "var out = [];\n"
            "function h(tag, cls, txt) { return { tag: tag, txt: txt, appendChild: function(){}, setAttribute: function(){} }; }\n"
            + extract_js_function(src, "function resultsFromSources(") + "\n"
            "var r = resultsFromSources([" + _json.dumps(source) + "])[0];\n"
            "var tail = typeof r.score === 'number' ? '\u2026  (' + r.score.toFixed(2) + ')' : '\u2026';\n"
            "process.stdout.write(JSON.stringify({ tail: tail, score: r.score }));\n"
        )
        return run_node_json(script)

    base = {"id": "a", "url": "pages/gyrotris.html", "anchor": "top",
            "page_title": "Gyrotris", "section_title": "Gyrotris", "text": "a puzzle game"}

    retrieved = render({**base, "score": 0.4237})
    assert "(0.42)" in retrieved["tail"], "a real retrieval score must still be shown"

    recommended = render(base)
    assert "(" not in recommended["tail"], (
        "a recommendation card has no score to report -- it must not render one"
    )
