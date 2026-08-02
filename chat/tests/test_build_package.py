"""Tests for functions/tencent/build_package.py's preset-match gating
(Task 29 fix round 1, Important 1; renamed alongside the Task 29 Part 2 file
split).

Before this fix, build_zip() bundled chat/data/index.json into the package
unconditionally whenever the file existed, with no check against the preset
being packaged. A stale chunks file left on disk from a build with a
DIFFERENT preset (e.g. an e5 chunks file while `--preset minilm` packages)
then shipped silently: index.py's rank_chunks() can't tell "wrong embedding
space" apart from "no good match" once vectors are just numbers, so /chat
never 503'd -- it just always fell through to the canned refusal.

chunks_preset_status() (formerly index_preset_status(), renamed when
Task 29 Part 2 retired chat/data/index.json for chat/data/chunks_{preset}.
json) is the extracted, unit-testable piece of that check (factored out of
build_zip so this doesn't need real model/wheel files to test). Loaded via
importlib since functions/tencent isn't a package under portfolio_rag --
mirrors how test_retrieval_sync.py loads index.py.
"""

import importlib.util
import json
from pathlib import Path

from portfolio_rag.config import settings

BUILD_PACKAGE_PATH = settings.chat_root / "functions" / "tencent" / "build_package.py"


def _load_build_package():
    spec = importlib.util.spec_from_file_location("_build_package_under_test", str(BUILD_PACKAGE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chunks_payload(model_preset: str) -> str:
    """A minimal but STRUCTURALLY REAL chunks file: a retrieval corpus is a
    model_preset plus a non-empty chunks list. The fixtures used to carry the
    preset alone, which is also true of chat/data/meta.json -- see
    test_a_preset_matching_file_with_no_chunks_is_not_a_retrieval_corpus."""
    return json.dumps({
        "model_preset": model_preset,
        "chunks": [{"id": "c1", "url": "pages/x.html", "text": "hello", "vector": [1.0, 0.0]}],
    })


def test_matching_preset_is_ok_to_bundle(tmp_path: Path) -> None:
    mod = _load_build_package()
    chunks_file = tmp_path / "chunks_e5.json"
    chunks_file.write_text(_chunks_payload("e5"), encoding="utf-8")

    ok, reason = mod.chunks_preset_status(chunks_file, "e5")
    assert ok is True
    assert reason is None


def test_mismatched_preset_is_not_ok_and_names_both_presets(tmp_path: Path) -> None:
    mod = _load_build_package()
    chunks_file = tmp_path / "chunks_e5.json"
    chunks_file.write_text(_chunks_payload("e5"), encoding="utf-8")

    ok, reason = mod.chunks_preset_status(chunks_file, "minilm")
    assert ok is False
    assert reason is not None
    assert "e5" in reason and "minilm" in reason


def test_a_preset_matching_file_with_no_chunks_is_not_a_retrieval_corpus(tmp_path: Path) -> None:
    """The mis-bundling this guard exists to catch is not only "wrong preset".
    chat/data/meta.json sits beside chunks_e5.json, also declares
    "model_preset": "e5", and carries no chunks at all -- so under a
    preset-only check it was bundle-able as chunks.json, and index.py then
    loaded it as a zero-row matrix that is not None and therefore reported
    itself as available (200 + the canned refusal for every visitor, forever,
    with GET / saying "retrieval": true).

    Uses the REAL committed data/meta.json rather than a synthetic stand-in,
    so this test fails if meta.json ever stops being the shape that made the
    hole reachable."""
    mod = _load_build_package()
    real_meta = settings.resolve_path(settings.meta_path)
    assert real_meta.exists(), f"expected the committed sidecar at {real_meta}"
    assert json.loads(real_meta.read_text(encoding="utf-8")).get("model_preset") == settings.model_preset, (
        "premise of this test: meta.json carries the same model_preset as the chunks file, "
        "which is why a preset-only check could not tell them apart"
    )

    ok, reason = mod.chunks_preset_status(real_meta, settings.model_preset)
    assert ok is False, "meta.json must never be bundle-able as the retrieval corpus"
    assert reason is not None and "chunks" in reason


def test_an_empty_chunks_list_is_not_a_retrieval_corpus(tmp_path: Path) -> None:
    """A file that is otherwise a correct chunks file but whose chunk list is
    empty is a failed build, not a corpus with nothing in it."""
    mod = _load_build_package()
    chunks_file = tmp_path / "chunks_e5.json"
    chunks_file.write_text(json.dumps({"model_preset": "e5", "chunks": []}), encoding="utf-8")

    ok, reason = mod.chunks_preset_status(chunks_file, "e5")
    assert ok is False
    assert reason is not None and "chunks" in reason


def test_missing_chunks_file_is_not_ok_with_no_reason(tmp_path: Path) -> None:
    mod = _load_build_package()
    chunks_file = tmp_path / "does-not-exist.json"

    ok, reason = mod.chunks_preset_status(chunks_file, "e5")
    assert ok is False
    assert reason is None, "a missing file has nothing to explain -- caller prints its own message"


def test_chunks_source_path_matches_settings_resolve_chunks_path(monkeypatch) -> None:
    """chunks_source_path() reproduces Settings.resolve_chunks_path()'s
    default derivation without importing portfolio_rag (this script must
    stay stdlib-only at module scope) -- pin the two together by calling
    the REAL settings.resolve_chunks_path() (with model_preset monkeypatched
    to each preset in turn), not a re-hardcoded format string that could
    silently stop matching the real derivation.

    Fix round 1 review, Important 2: the original version of this test
    compared against `settings.resolve_path(f"data/chunks_{preset_name}.json")`
    -- a SECOND, independently-typed copy of the same format string, not a
    call into resolve_chunks_path() itself. That made the test inert: the
    reviewer proved it by monkeypatching Settings.resolve_chunks_path to a
    different derivation (`data/{preset}/chunks.json`) and the old test kept
    passing, because it never called the method it claimed to pin. Calling
    the real method here is what makes that drift (build_index writes one
    location, chunks_source_path reads another, chunks_preset_status
    returns (False, None), the zip ships with no chunks.json, /chat 503s)
    an immediate test failure instead of a silent divergence.
    """
    mod = _load_build_package()
    for preset_name in ("e5", "minilm"):
        monkeypatch.setattr(settings, "model_preset", preset_name)
        assert mod.chunks_source_path(preset_name) == settings.resolve_chunks_path()


def test_the_real_committed_chunks_file_matches_its_own_declared_preset() -> None:
    """Sanity check against the real artifact: chat/data/chunks_e5.json's
    model_preset must equal settings.model_preset (e5, per chat/.env) --
    otherwise a plain `python build_package.py` run today would already hit
    the mismatch warning instead of bundling the chunks file it just
    rebuilt."""
    mod = _load_build_package()
    chunks_file = mod.chunks_source_path(settings.model_preset)
    ok, reason = mod.chunks_preset_status(chunks_file, settings.model_preset)
    assert ok, reason
