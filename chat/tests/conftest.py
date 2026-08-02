"""Session-wide recurrence guard for Task 18: a test that writes through
`settings` to a real, committed/gitignored data artifact instead of a
tmp_path-patched one silently corrupts every measurement taken afterwards.

See `chat/tests/test_index_builder.py`'s `tiny_site` fixture: it used to
monkeypatch only `settings.index_path` and `settings.roles_path`, so a build
run under it wrote its 5-chunk toy calibration straight over the real,
hand-calibrated `chat/data/gate_vectors.json` (gitignored, unrecoverable by
git) and `chat/data/fallback_vectors.json`. That specific hole is closed, but
the bug class — some future fixture or test doing the same thing — is not.
This fixture content-hashes the known real artifacts before and after the
whole test session and fails loudly, naming any file that changed.

Task 29 Part 2 split index.json/gate_vectors.json/fallback_vectors.json into
four single-purpose files (see chat/README.md) — the watch list below tracks
their replacements. `chunks_e5.json` is watched by its real, derived name
(this repo's committed `chat/.env` pins `RAG_MODEL_PRESET=e5`) rather than a
preset-agnostic pattern, matching how the retired `index.json` was watched by
its one real name; a fixture that switches presets without going through
`tiny_site`'s monkeypatching would need its own coverage regardless, not a
change to this list. `gate_zh_bge.json` is gitignored (same status
`gate_vectors.json` had) and unrecoverable by git the same way.
"""

import hashlib
from pathlib import Path

import pytest

_CHAT_ROOT = Path(__file__).resolve().parents[1]

# Files build_index() (or anything else exercised by the suite) can write.
# roles.json is also written by build_index() but is committed, low-risk,
# checked-out-from-git content (unlike gate_en_minilm.json/gate_zh_bge.json,
# one of which is gitignored and had to be hand-reconstructed once already)
# and is deliberately left off this watch list — see task-18-report.md for
# the full reasoning.
_WATCHED_FILES = [
    _CHAT_ROOT / "data" / "chunks_e5.json",
    _CHAT_ROOT / "data" / "meta.json",
    _CHAT_ROOT / "data" / "gate_en_minilm.json",
    _CHAT_ROOT / "data" / "gate_zh_bge.json",
    _CHAT_ROOT / "data" / "chunks_en_minilm.json",
    _CHAT_ROOT / "data" / "eval_baseline.json",
]


def _hash_file(path: Path) -> str | None:
    """SHA-256 of the file's bytes, or None if it doesn't exist (skip, don't error)."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _guard_real_data_artifacts_unmutated():
    before = {path: _hash_file(path) for path in _WATCHED_FILES}
    yield
    after = {path: _hash_file(path) for path in _WATCHED_FILES}
    mutated = [str(path) for path in _WATCHED_FILES if after[path] != before[path]]
    if mutated:
        pytest.fail(
            "A test in this session wrote through `settings` to a real data "
            "artifact instead of a tmp_path-patched path. Mutated file(s): "
            + ", ".join(mutated)
            + ". If this is gate_zh_bge.json, restore it immediately (it is "
            "gitignored and NOT recoverable via git) and find the fixture "
            "that failed to monkeypatch its settings path onto tmp_path.",
            pytrace=False,
        )
