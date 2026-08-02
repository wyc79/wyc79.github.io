"""Tests for functions/tencent/index.py's _load_index() (Task 29 fix round
1, Important 1's defense-in-depth half).

build_package.py (see test_build_package.py) now refuses to BUNDLE a
mismatched-preset index.json, but a packaged zip can also be assembled by
hand (or by a future refactor of that script) without going through it.
_load_index() independently re-checks the index it's actually given against
this deployment's QUERY_PREFIX env var -- the one signal already required to
be configured correctly for embedding itself to work, and one that
identifies the preset 1:1 (e5="query: ", minilm=""). A mismatch must leave
retrieval unavailable (_index["matrix"] stays None -> /chat 503s), not load
a document/query embedding-space mismatch silently.

index.py is loaded fresh via importlib per test (mirrors
test_retrieval_sync.py); _load_index() reads Path(__file__).with_name(
"index.json") relative to the REAL index.py file on disk, so these tests
write/remove a real (temporary) functions/tencent/index.json -- guarded by
a fixture that asserts nothing was already there and always cleans up.
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


def _write_index(path, model_preset: str, query_prefix: str) -> None:
    path.write_text(json.dumps({
        "model_preset": model_preset,
        "query_prefix": query_prefix,
        "chunks": [{"id": "c1", "url": "pages/x.html", "text": "hello", "vector": [1.0, 0.0]}],
    }), encoding="utf-8")


def test_matching_query_prefix_loads_the_index(no_stray_index_json, monkeypatch) -> None:
    _write_index(no_stray_index_json, "e5", "query: ")
    monkeypatch.setenv("QUERY_PREFIX", "query: ")
    mod = _load_index_module()

    mod._load_index()

    assert mod._index["matrix"] is not None
    assert mod._index["error"] is None
    assert len(mod._index["chunks"]) == 1


def test_mismatched_query_prefix_refuses_to_load(no_stray_index_json, monkeypatch) -> None:
    # An e5 index (query_prefix "query: ") packaged into a deployment whose
    # QUERY_PREFIX env var says minilm ("") -- e.g. a hand-assembled or
    # stale zip. Must refuse, not silently retrieve garbage.
    _write_index(no_stray_index_json, "e5", "query: ")
    monkeypatch.setenv("QUERY_PREFIX", "")
    mod = _load_index_module()

    mod._load_index()

    assert mod._index["matrix"] is None, "a mismatched index must not load"
    assert mod._index["chunks"] is None
    assert mod._index["error"] is not None
    assert "e5" in mod._index["error"]
    assert "query_prefix" in mod._index["error"]


def test_mismatched_index_makes_retrieval_report_unavailable(no_stray_index_json, monkeypatch) -> None:
    """End-to-end within the module: a mismatched index must flip both the
    GET / health flag and /chat's own availability guard -- not just leave
    an internal error string nobody reads."""
    _write_index(no_stray_index_json, "minilm", "")
    monkeypatch.setenv("QUERY_PREFIX", "query: ")  # deployment expects e5
    mod = _load_index_module()

    mod._load_index()

    assert mod._index["matrix"] is None
    # This is exactly the condition do_GET's health payload and _chat's
    # availability guard both read.
    assert (mod._index["matrix"] is not None) is False


def test_missing_index_json_still_refuses_as_before(no_stray_index_json) -> None:
    """No regression: the "not packaged at all" case (no env involved) must
    still behave as it did before this fix."""
    mod = _load_index_module()

    mod._load_index()

    assert mod._index["matrix"] is None
    assert mod._index["error"] == "index.json not packaged"
