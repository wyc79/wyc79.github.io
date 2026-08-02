"""Tests for functions/tencent/index.py's _load_index() (Task 29, Important
1's defense-in-depth half -- fix round 1 introduced it, fix round 2
corrected its signal).

build_package.py (see test_build_package.py) refuses to BUNDLE a
mismatched-preset index.json, but a packaged zip can also be assembled by
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
"index.json") relative to the REAL index.py file on disk, so these tests
write/remove a real (temporary) functions/tencent/index.json -- guarded by
a fixture that asserts nothing was already there and always cleans up
(also now gitignored, belt-and-braces). BUILD_INFO itself is set directly
on the freshly imported module object per test (mod.BUILD_INFO = {...}) --
no need to also write a temporary build_info.json on disk.
"""

import importlib.util
import json

import pytest

from portfolio_rag.config import settings

BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"
_SIDE_BY_SIDE_INDEX_JSON = BACKEND_PATH.with_name("index.json")


def _load_index_module():
    spec = importlib.util.spec_from_file_location("_scf_index_loading_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def no_stray_index_json():
    """functions/tencent/index.json is a build ARTIFACT (written straight
    into the zip by build_package.py, never left loose in the source tree) --
    assert none is already there before writing a temporary one, and always
    remove it after, so this test can never leave a stray file behind or
    silently test against someone else's leftover."""
    assert not _SIDE_BY_SIDE_INDEX_JSON.exists(), (
        f"unexpected file at {_SIDE_BY_SIDE_INDEX_JSON} -- remove it before running this test"
    )
    yield _SIDE_BY_SIDE_INDEX_JSON
    _SIDE_BY_SIDE_INDEX_JSON.unlink(missing_ok=True)


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


def test_matching_preset_loads_the_index_with_no_warning(no_stray_index_json) -> None:
    _write_index(no_stray_index_json, "e5")
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


def test_mismatched_preset_refuses_to_load(no_stray_index_json) -> None:
    # An e5 index packaged (by hand, or by a stale/mismatched zip) into a
    # build whose build_info.json declares minilm. Must refuse, not
    # silently retrieve garbage.
    _write_index(no_stray_index_json, "e5")
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


def test_mismatched_index_makes_retrieval_report_unavailable(no_stray_index_json) -> None:
    """End-to-end within the module: a mismatched index must flip both the
    GET / health flag and /chat's own availability guard -- not just leave
    an internal error string nobody reads."""
    _write_index(no_stray_index_json, "minilm")
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}  # deployment expects e5

    mod._load_index()

    assert mod._index["matrix"] is None
    # This is exactly the condition do_GET's health payload and _chat's
    # availability guard both read.
    assert (mod._index["matrix"] is not None) is False


def test_unknown_build_preset_loads_anyway_and_logs_a_warning(no_stray_index_json) -> None:
    """The critical fix-round-2 case: build_info.json has no usable
    'preset' (an old or hand-assembled zip -- BUILD_INFO defaults to
    {"build_id": "unknown", "built_at": None} with no "preset" key at all
    when build_info.json is missing/unreadable, see _load_build_info()).
    This must NOT be treated as a mismatch -- an unknown signal is not
    evidence of one. The index must load, and a warning must be logged
    (asserted directly on the log call, not inferred from the load
    succeeding)."""
    _write_index(no_stray_index_json, "e5")
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


def test_missing_index_json_still_refuses_as_before(no_stray_index_json) -> None:
    """No regression: the "not packaged at all" case must still behave as
    it did before this fix, regardless of BUILD_INFO."""
    mod = _load_index_module()
    mod.BUILD_INFO = {"preset": "e5"}

    mod._load_index()

    assert mod._index["matrix"] is None
    assert mod._index["error"] == "index.json not packaged"
