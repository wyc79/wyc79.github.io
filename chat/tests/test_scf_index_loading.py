"""Tests for functions/tencent/index.py's startup-path loading -- _load_index()
(Task 29, Important 1's defense-in-depth half -- fix round 1 introduced it, fix
round 2 corrected its signal) and _load_embedder()'s gate-file reads.

main() calls both BEFORE binding :9000, so anything either of them raises takes
the whole function down (every route, not just the feature that failed). The
final whole-branch review found two ways that could happen against a corrupt or
mis-bundled artifact; the tests at the bottom of this file pin both.

build_package.py (see test_build_package.py) refuses to BUNDLE a
mismatched-preset chunks file, but a packaged zip can also be assembled by
hand (or by a future refactor of that script) without going through it.
_load_index() independently re-checks the index it's actually given.

Fix round 1 compared the index's own query_prefix against this
deployment's QUERY_PREFIX env var. Fix round 2 review found that check
conflated two very different failure classes under one all-or-nothing
refusal: a genuine wrong-preset index (a real bug, and already fully
caught at the bundling side regardless of this check) versus a correctly
built deployment where an operator simply forgot the manual QUERY_PREFIX
console step (previously a silent quality regression; under the
QUERY_PREFIX check it became a permanent /chat 503 for an otherwise-fine
deployment -- worse than what it replaced).

The check now compares the index's own `model_preset` against
BUILD_INFO["preset"] -- the exact string build_package.py's
make_build_info() stamps into build_info.json when it packages a zip,
already loaded at module scope into BUILD_INFO, already used for
build_id/health reporting. No environment variable, no operator step to
forget. When that signal itself is absent (an old or hand-assembled zip
with no usable build_info.json) the check must NOT refuse -- an unknown
signal is not evidence of a mismatch -- it loads anyway and logs a loud
warning instead.

index.py is loaded fresh via importlib per test (mirrors
test_retrieval_sync.py); _load_index() reads Path(__file__).with_name(
"chunks.json") relative to the REAL index.py file on disk (Task 29 Part 2
renamed this zip-internal artifact from "index.json", matching the
chat/data/ source rename), so these tests write/remove a real (temporary)
functions/tencent/chunks.json -- guarded by a fixture that asserts nothing
was already there and always cleans up (also now gitignored, belt-and-
braces). BUILD_INFO itself is set directly on the freshly imported module
object per test (mod.BUILD_INFO = {...}) -- no need to also write a
temporary build_info.json on disk.
"""

import importlib.util
import json

import pytest

from portfolio_rag.config import settings

BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"
_SIDE_BY_SIDE_CHUNKS_JSON = BACKEND_PATH.with_name("chunks.json")


def _load_index_module():
    spec = importlib.util.spec_from_file_location("_scf_index_loading_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def no_stray_chunks_json():
    """functions/tencent/chunks.json is a build ARTIFACT (written straight
    into the zip by build_package.py, never left loose in the source tree) --
    assert none is already there before writing a temporary one, and always
    remove it after, so this test can never leave a stray file behind or
    silently test against someone else's leftover."""
    assert not _SIDE_BY_SIDE_CHUNKS_JSON.exists(), (
        f"unexpected file at {_SIDE_BY_SIDE_CHUNKS_JSON} -- remove it before running this test"
    )
    yield _SIDE_BY_SIDE_CHUNKS_JSON
    _SIDE_BY_SIDE_CHUNKS_JSON.unlink(missing_ok=True)


def _write_index(path, model_preset: str) -> None:
    path.write_text(json.dumps({
        "model_preset": model_preset,
        "query_prefix": "query: " if model_preset == "e5" else "",
        "chunks": [{"id": "c1", "url": "pages/x.html", "text": "hello", "vector": [1.0, 0.0]}],
    }), encoding="utf-8")


def _recording_log(mod) -> list:
    """Replace mod.log with a stub that records every call instead of
    printing, so a test can assert a specific warning WAS emitted (not just
    that loading succeeded) without scraping stdout."""
    calls: list = []
    mod.log = lambda record: calls.append(record)
    return calls


def test_matching_preset_loads_the_index_with_no_warning(no_stray_chunks_json) -> None:
    _write_index(no_stray_chunks_json, "e5")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}
    calls = _recording_log(mod)

    mod._load_index()

    assert mod._index["matrix"] is not None
    assert mod._index["error"] is None
    assert len(mod._index["chunks"]) == 1
    assert not any(c.get("type") == "index_preset_unverifiable" for c in calls), (
        "a genuine preset match must not log the 'unverifiable' warning"
    )
    assert any(c.get("type") == "index_loaded" for c in calls)


def test_mismatched_preset_refuses_to_load(no_stray_chunks_json) -> None:
    # An e5 index packaged (by hand, or by a stale/mismatched zip) into a
    # build whose build_info.json declares minilm. Must refuse, not
    # silently retrieve garbage.
    _write_index(no_stray_chunks_json, "e5")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "minilm"}

    mod._load_index()

    assert mod._index["matrix"] is None, "a mismatched index must not load"
    assert mod._index["chunks"] is None
    assert mod._index["error"] is not None
    assert "e5" in mod._index["error"] and "minilm" in mod._index["error"], (
        "the error must name both the index's own preset and the "
        "package's declared preset"
    )


def test_mismatched_index_makes_retrieval_report_unavailable(
    no_stray_chunks_json, monkeypatch
) -> None:
    """End-to-end within the module: a mismatched index must flip both the
    GET / health flag and /chat's own availability guard -- not just leave an
    internal error string nobody reads.

    This test previously claimed exactly that and did not do it. Its body
    called neither handler; its final line was `assert (mod._index["matrix"]
    is not None) is False`, a re-typed copy of the production expression and
    logically identical to the line above it. The final whole-branch review
    proved it inert with two mutations that both stayed green (5 passed):
    do_GET reporting `"retrieval": True` unconditionally, and _chat dropping
    the index-availability clause from its guard. Both are live-chat failures
    -- the first makes the health endpoint lie to whoever is checking a
    deployment, the second lets a wrong-embedding-space index serve real
    answers. It now drives the real do_GET and the real _chat (see
    _StubHandler), and both mutations were re-run and are red."""
    _write_index(no_stray_chunks_json, "minilm")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}  # deployment expects e5

    mod._load_index()

    assert mod._index["matrix"] is None
    # The embedder is deliberately made AVAILABLE: _chat's guard is
    # `_embed["session"] is None or _index["matrix"] is None`, so an absent
    # embedder would produce the same 503 and prove nothing about the index
    # half -- the half this test is named for.
    mod._embed["session"] = object()
    monkeypatch.setenv("LLM_API_KEY", "not-used-the-guard-fires-first")

    assert _health_payload(mod)["retrieval"] is False, (
        "GET / reported retrieval as available with a mismatched index -- the "
        "health endpoint is what an operator checks after a deploy"
    )
    assert _health_payload(mod)["retrieval_error"] is not None

    handler = _StubHandler(path="/chat")
    mod.Handler._chat(handler)
    assert handler.responses == [(503, {"error": "retrieval not available"})], (
        "/chat did not 503 on a mismatched index -- a wrong-embedding-space "
        f"index would serve real answers. Got {handler.responses!r}"
    )


def test_unknown_build_preset_loads_anyway_and_logs_a_warning(no_stray_chunks_json) -> None:
    """The critical fix-round-2 case: build_info.json has no usable
    'preset' (an old or hand-assembled zip -- BUILD_INFO defaults to
    {"build_id": "unknown", "built_at": None} with no "preset" key at all
    when build_info.json is missing/unreadable, see _load_build_info()).
    This must NOT be treated as a mismatch -- an unknown signal is not
    evidence of one. The index must load, and a warning must be logged
    (asserted directly on the log call, not inferred from the load
    succeeding)."""
    _write_index(no_stray_chunks_json, "e5")
    mod = _load_index_module()
    mod.BUILD_INFO = {"build_id": "unknown", "built_at": None}  # no "preset" key
    calls = _recording_log(mod)

    mod._load_index()

    assert mod._index["matrix"] is not None, (
        "an unverifiable preset must load anyway -- refusing here would "
        "recreate the exact 'correct deployment, missing signal -> total "
        "outage' trap fix round 2 exists to avoid"
    )
    assert mod._index["error"] is None
    warnings = [c for c in calls if c.get("type") == "index_preset_unverifiable"]
    assert len(warnings) == 1, f"expected exactly one warning log call, got {calls!r}"
    assert warnings[0]["index_model_preset"] == "e5"


def test_missing_chunks_json_still_refuses_as_before(no_stray_chunks_json) -> None:
    """No regression: the "not packaged at all" case must still behave as
    it did before this fix, regardless of BUILD_INFO."""
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}

    mod._load_index()

    assert mod._index["matrix"] is None
    assert mod._index["error"] == "chunks.json not packaged"


def test_a_zero_chunk_index_reports_itself_unavailable(no_stray_chunks_json) -> None:
    """A chunks.json with the right preset stamp but no chunks used to load as
    np.empty((0, 0)) -- which is NOT None, so it passed /chat's availability
    guard and GET /'s health flag while rank_chunks returned [] for every
    question: 200 + the canned refusal forever, the widget never falling to
    degraded mode (200 is success), and health reporting "retrieval": true.

    Reachable via chat/data/meta.json bundled as chunks.json (same
    model_preset, no chunks) -- see
    test_build_package.py::test_a_preset_matching_file_with_no_chunks_is_not_a_retrieval_corpus
    for the bundling-side half of the same fix."""
    no_stray_chunks_json.write_text(
        json.dumps({"model_preset": "e5", "query_prefix": "query: ", "chunks": []}), encoding="utf-8"
    )
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}
    calls = _recording_log(mod)

    mod._load_index()

    assert mod._index["matrix"] is None, (
        "an empty index must be indistinguishable from a missing one at the "
        "availability check -- both mean retrieval cannot work"
    )
    assert mod._index["error"] == "chunks.json has no chunks"
    assert not any(c.get("type") == "index_loaded" for c in calls), (
        "an empty index must not log itself as loaded"
    )
    assert any(c.get("type") == "index_load_failed" for c in calls)


# ── driving the real handlers ───────────────────────────────────────────────


class _StubHandler:
    """The minimal `self` do_GET and _chat need. Both are called as plain
    functions against it (mod.Handler.do_GET(stub)), so the REAL handler body
    runs -- no socket, no server. BaseHTTPRequestHandler.__init__ would try to
    read from a live connection, and the branches under test all return before
    touching rfile/wfile, so a stub is both sufficient and closer to what is
    being asserted.

    _json is the sink: every handler reply goes through it, so recording it
    captures the real status and the real payload dict the handler built."""

    def __init__(self, path: str = "/", origin: str = "https://wyc79.github.io"):
        self.path = path
        self._origin_value = origin
        self.responses: list = []

    def _origin(self):
        return self._origin_value

    def _json(self, status: int, obj: dict) -> None:
        self.responses.append((status, obj))

    # _chat only reaches these two if its availability guard has ALREADY let
    # it through. They exist so that a build without that guard fails on a
    # readable assertion (a 400 "question required" where a 503 was expected)
    # instead of an AttributeError deep inside the handler.
    def _ip(self) -> str:
        return "203.0.113.7"

    def _read_body(self, cap: int = 64 * 1024) -> bytes:
        return b""


def _health_payload(mod) -> dict:
    """Whatever the real GET / handler answers, unwrapped."""
    handler = _StubHandler(path="/")
    mod.Handler.do_GET(handler)
    assert len(handler.responses) == 1, f"do_GET replied {len(handler.responses)} times"
    status, payload = handler.responses[0]
    assert status == 200, f"health check returned {status}: {payload!r}"
    return payload


# ── QUERY_PREFIX observability ──────────────────────────────────────────────
#
# QUERY_PREFIX is a manual console step that silently governs the ONLY
# production retrieval path. The branch deliberately chose not to hard-fail on
# a missing prefix (a hard fail would take down an otherwise-correct
# deployment over one forgotten console step) -- but then left the tolerated
# failure invisible: not in embedder_loaded, not in the startup line, not in
# GET /. The golden harness evaluates runtime.py, which always applies the
# prefix, so the measurement apparatus could not see it either.
#
# The cross-check compares the effective prefix against chunks.json's OWN
# declared query_prefix -- the authoritative statement of what the vectors in
# that same file were built with -- not against a preset->prefix table copied
# in from portfolio_rag.config (this module is stdlib-only by contract).


def test_the_health_payload_reports_the_effective_and_declared_query_prefix(
    no_stray_chunks_json,
) -> None:
    _write_index(no_stray_chunks_json, "e5")  # declares query_prefix "query: "
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}
    mod._embed["prefix"] = "query: "
    mod._load_index()

    payload = _health_payload(mod)

    assert payload["query_prefix"] == "query: ", (
        "a deployment that forgot the QUERY_PREFIX console step must be "
        "distinguishable from a correct one via the health endpoint"
    )
    assert payload["index_query_prefix"] == "query: "


def test_a_forgotten_query_prefix_warns_and_still_serves(no_stray_chunks_json) -> None:
    """The operator omission this tolerates: chunks.json's vectors were built
    with "query: " but the env var is unset. Must log loudly and keep
    serving -- refusing here would turn one console step into a permanent
    /chat outage, which is worse than the degraded retrieval it replaces."""
    _write_index(no_stray_chunks_json, "e5")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}
    mod._embed["prefix"] = ""  # operator forgot the console step
    mod._load_index()
    calls = _recording_log(mod)

    mod._check_query_prefix()

    warnings = [c for c in calls if c.get("type") == "query_prefix_unexpected"]
    assert len(warnings) == 1, f"expected exactly one warning, got {calls!r}"
    assert warnings[0]["effective"] == ""
    assert warnings[0]["index_declares"] == "query: "
    assert mod._index["matrix"] is not None, "the mismatch must never make retrieval unavailable"
    assert _health_payload(mod)["retrieval"] is True


def test_a_matching_query_prefix_logs_no_warning(no_stray_chunks_json) -> None:
    _write_index(no_stray_chunks_json, "e5")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}
    mod._embed["prefix"] = "query: "
    mod._load_index()
    calls = _recording_log(mod)

    mod._check_query_prefix()

    assert not any(c.get("type") == "query_prefix_unexpected" for c in calls), (
        "a correct deployment must not cry wolf"
    )


# ── _load_embedder(): a corrupt gate file must not take the function down ──


_GATE_FILES = {lang: BACKEND_PATH.with_name(name) for lang, name in
               (("en", "gate_en_minilm.json"), ("zh", "gate_zh_bge.json"))}


@pytest.fixture()
def no_stray_gate_json():
    """Same discipline as no_stray_chunks_json: the gate files are build
    artifacts written straight into the zip, never left loose in the source
    tree. Assert none is already there, always clean up."""
    for path in _GATE_FILES.values():
        assert not path.exists(), (
            f"unexpected file at {path} -- remove it before running this test"
        )
    yield _GATE_FILES
    for path in _GATE_FILES.values():
        path.unlink(missing_ok=True)


def _write_gate(path, threshold: float = 0.2) -> None:
    path.write_text(json.dumps({
        "gate_stat": "top",
        "gate_threshold": threshold,
        "query_prefix": "",
        "pooling": "mean",
        "vectors": [[1.0, 0.0], [0.0, 1.0]],
    }), encoding="utf-8")


def test_a_truncated_gate_file_degrades_instead_of_killing_the_function(no_stray_gate_json) -> None:
    """The JSON read of a gate file used to sit OUTSIDE its try, so a truncated
    or malformed gate_*.json raised JSONDecodeError out of _load_embedder().
    main() calls _load_embedder() BEFORE ThreadingHTTPServer(...).serve_forever(),
    so that killed the process before :9000 was ever bound -- /chat, /embed,
    /log and health all down, for a file whose only honest blast radius is "no
    gate for this language".

    A missing KEY in the same file was already caught cleanly; only the read
    itself was on the wrong side of the try. The package is a 160-200 MB zip
    uploaded by hand and gate_zh_bge.json is regenerated on every build and
    unrecoverable by git, so a partial artifact is a realistic failure mode.

    _load_model is stubbed: ONNX session creation is not what is under test
    (and the model dirs are not in the source tree), and stubbing it is what
    lets _load_embedder() get past the retrieval embedder to the gate loop."""
    _write_gate(no_stray_gate_json["en"])
    no_stray_gate_json["zh"].write_text('{"gate_stat": "top", "gate_thre', encoding="utf-8")

    mod = _load_index_module()
    mod._load_model = lambda dir_name: (object(), object())
    calls = _recording_log(mod)

    mod._load_embedder()  # must not raise

    assert any(c.get("type") == "embedder_loaded" and "query_prefix" in c for c in calls), (
        "the effective QUERY_PREFIX must be logged at startup, not only applied"
    )
    assert mod._gates["en"] is not None, "a valid en gate must still load"
    assert mod._gates["zh"] is None, "a corrupt zh gate must be absent, not half-built"
    failures = [c for c in calls if c.get("type") == "gate_load_failed"]
    assert [c["lang"] for c in failures] == ["zh"], (
        f"expected exactly one gate_load_failed, for zh; got {calls!r}"
    )
    # The whole point: with no zh gate, gate_decision falls back to cjk_bypass
    # rather than refusing every Chinese visitor -- the documented degradation.
    assert mod.gate_decision("你好，请介绍一下这个项目") == {
        "pass": True, "value": None, "reason": "cjk_bypass",
    }
