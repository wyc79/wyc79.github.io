"""scripts/run_eval.py's --update-baseline refusals.

data/eval_baseline.json is the frozen number tests/test_golden.py's
test_no_metric_regressed compares against -- it is the branch's own go/no-go
criterion. --update-baseline already refused to capture it from a FILTERED run
(--role/--lang), because a partial run is not a measurement of the system. The
final whole-branch review found the second way to capture a baseline from a
state that does not represent the system, and it only warned:

  Runtime.stale_knowledge_headings detects that the committed chunks file's
  baked-in knowledge headings no longer match chat/knowledge/about_*.md on
  disk. Its docstring measures the consequence -- renaming 5 about_en.md
  headings with no rebuild moved knowledge_chunk_ids 108 -> 103 and
  hit_at_4_page_only 59/96 -> 56/96, a 3-case move in an unpredictable
  direction. run_eval.py printed a stderr note about exactly this and then,
  44 lines later, wrote the baseline anyway.

These tests drive the real main() with a stubbed runtime and a tmp_path
BASELINE_PATH (never the real file -- tests/conftest.py's session guard watches
it, and writing it here would corrupt the regression gate for every later
test). run_eval.py is a script, not a package module, so it is loaded by
importlib the way test_build_package.py loads build_package.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from portfolio_rag.config import settings

RUN_EVAL_PATH = settings.chat_root / "scripts" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("_run_eval_under_test", str(RUN_EVAL_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubCase:
    """Only the three fields main()'s --role/--lang filter reads. Chosen to
    MATCH both filters, so a filtered run still reaches the --update-baseline
    block instead of exiting early at "no cases matched the filter"."""

    type = "positive"
    role = "combat_design_recruiter"
    lang = "en"


class _StubRuntime:
    """Only the attributes main() reads before/while updating the baseline."""

    def __init__(self, stale: set[str]):
        self.retrieval_available = True
        self.zh_gate_available = True
        self.stale_knowledge_headings = stale
        self.gate_meta = {}


def _stub_pipeline(mod, monkeypatch, tmp_path: Path, stale: set[str]) -> Path:
    """Point every I/O edge of main() at stubs/tmp_path and return the tmp
    baseline path. Nothing real is read or written."""
    baseline = tmp_path / "eval_baseline.json"
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline)
    monkeypatch.setattr(mod, "load_runtime", lambda: _StubRuntime(stale))
    monkeypatch.setattr(mod, "load_cases", lambda path: [_StubCase()])
    monkeypatch.setattr(mod, "run_cases", lambda rt, cases: [])
    monkeypatch.setattr(mod, "aggregate", lambda results: {})
    monkeypatch.setattr(mod, "build_baseline", lambda rt, cells: {"stub": True})
    return baseline


def test_update_baseline_refuses_while_the_index_is_known_stale(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    mod = _load_run_eval()
    baseline = _stub_pipeline(mod, monkeypatch, tmp_path, stale={"Prime Engine", "Combat"})
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--update-baseline"])

    rc = mod.main()

    assert rc == 2, "a stale index must make --update-baseline exit non-zero, not warn and write"
    assert not baseline.exists(), (
        "the baseline must not be written from a state Runtime.stale_knowledge_headings "
        "flags -- it is what test_no_metric_regressed compares against"
    )
    err = capsys.readouterr().err
    assert "refusing to write a baseline" in err
    assert "--allow-stale-index" in err, "the refusal must name the deliberate override"


def test_update_baseline_writes_when_the_index_is_current(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The complement, so the refusal above cannot be satisfied by simply
    never writing a baseline."""
    mod = _load_run_eval()
    baseline = _stub_pipeline(mod, monkeypatch, tmp_path, stale=set())
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--update-baseline"])

    rc = mod.main()

    assert rc == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"stub": True}


def test_allow_stale_index_is_a_deliberate_override(tmp_path: Path, monkeypatch) -> None:
    mod = _load_run_eval()
    baseline = _stub_pipeline(mod, monkeypatch, tmp_path, stale={"Prime Engine"})
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--update-baseline", "--allow-stale-index"])

    assert mod.main() == 0
    assert baseline.exists(), "the override must still work -- this is a guard, not a ban"


@pytest.mark.parametrize("flag", ["--role", "--lang"])
def test_update_baseline_still_refuses_a_filtered_run(
    tmp_path: Path, monkeypatch, flag: str
) -> None:
    """No regression on the precedent this guard was modelled on."""
    mod = _load_run_eval()
    baseline = _stub_pipeline(mod, monkeypatch, tmp_path, stale=set())
    value = "combat_design_recruiter" if flag == "--role" else "en"
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--update-baseline", flag, value])

    assert mod.main() == 2
    assert not baseline.exists()
