"""Tests for functions/tencent/build_package.py's preset-match gating
(Task 29 fix round 1, Important 1).

Before this fix, build_zip() bundled chat/data/index.json into the package
unconditionally whenever the file existed, with no check against the preset
being packaged. A stale index.json left on disk from a build with a
DIFFERENT preset (e.g. an e5 index while `--preset minilm` packages) then
shipped silently: index.py's rank_chunks() can't tell "wrong embedding
space" apart from "no good match" once vectors are just numbers, so /chat
never 503'd -- it just always fell through to the canned refusal.

index_preset_status() is the extracted, unit-testable piece of that check
(factored out of build_zip so this doesn't need real model/wheel files to
test). Loaded via importlib since functions/tencent isn't a package under
portfolio_rag -- mirrors how test_retrieval_sync.py loads index.py.
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


def test_matching_preset_is_ok_to_bundle(tmp_path: Path) -> None:
    mod = _load_build_package()
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"model_preset": "e5"}), encoding="utf-8")

    ok, reason = mod.index_preset_status(index_file, "e5")
    assert ok is True
    assert reason is None


def test_mismatched_preset_is_not_ok_and_names_both_presets(tmp_path: Path) -> None:
    mod = _load_build_package()
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"model_preset": "e5"}), encoding="utf-8")

    ok, reason = mod.index_preset_status(index_file, "minilm")
    assert ok is False
    assert reason is not None
    assert "e5" in reason and "minilm" in reason


def test_missing_index_file_is_not_ok_with_no_reason(tmp_path: Path) -> None:
    mod = _load_build_package()
    index_file = tmp_path / "does-not-exist.json"

    ok, reason = mod.index_preset_status(index_file, "e5")
    assert ok is False
    assert reason is None, "a missing file has nothing to explain -- caller prints its own message"


def test_the_real_committed_index_matches_its_own_declared_preset() -> None:
    """Sanity check against the real artifact: chat/data/index.json's
    model_preset must equal settings.model_preset (e5, per chat/.env) --
    otherwise a plain `python build_package.py` run today would already hit
    the mismatch warning instead of bundling the index it just rebuilt."""
    mod = _load_build_package()
    index_file = settings.resolve_path(settings.index_path)
    ok, reason = mod.index_preset_status(index_file, settings.model_preset)
    assert ok, reason
