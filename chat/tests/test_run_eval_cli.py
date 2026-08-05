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
    """Only the fields main() reads: type/role/lang for the --role/--lang
    filter (chosen to MATCH both, so a filtered run still reaches the
    --update-baseline block instead of exiting early at "no cases matched
    the filter"), plus id/history for the task-9 missing-rewrites check
    main() now runs on every `selected` case. history=() means this stub
    is never treated as a multi-turn case, so it never appears in that
    check's `missing` list."""

    type = "positive"
    role = "combat_design_recruiter"
    lang = "en"
    id = "stub"
    history = ()


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
    # load_rewrites/run_cases stubbed explicitly rather than relying on the
    # real load_rewrites() incidentally returning {} for a missing file --
    # this module's docstring promises "nothing real is read or written".
    monkeypatch.setattr(mod, "load_rewrites", lambda: {})
    monkeypatch.setattr(mod, "run_cases", lambda rt, cases, rewrites=None: [])
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


# --- Important 1, final whole-branch review: two fabricated numbers --------


def test_missing_rewrite_fixture_renders_n_a_not_a_fabricated_number(
    monkeypatch, capsys
) -> None:
    """Drives the real main() end to end (not print_shared_table or the
    follow-up block in isolation) so a future edit that drops the n/a
    substitution -- or resurrects the deleted prose NOTE this replaces --
    shows up here instead of nowhere. This file's own docstring names
    exactly this gap: `_StubCase.history = ()` makes every other test in
    this module blind to the multi-turn path entirely.

    Two real GoldenCase/CaseResult pairs, both with history and NO recorded
    rewrite (load_rewrites stubbed to {}, mirroring this environment's real
    state -- chat/eval/rewrites.json does not exist here). aggregate() runs
    for REAL (only load_runtime/load_cases/load_rewrites/run_cases are
    stubbed) so the rendering logic under test sees the same cell shape a
    real run would produce:

      - a positive follow-up case -> followup_rescued must print "n/a", not
        "rescued 0/1", which reads as "attempted and failed" rather than
        "never attempted."
      - a post-context negative -> the shared table's post_context column
        must print "n/a", not "0/0", which reads as a clean pass despite
        never having replayed a rewrite (the whole point of the bucket).
    """
    from portfolio_rag.evaluation import CaseResult, GoldenCase

    mod = _load_run_eval()

    followup_case = GoldenCase(
        id="followup-x", role="combat_design_recruiter", lang="en", type="positive",
        q="what about tuning it",
        history=(("user", "u"), ("assistant", "a")),
        expected_urls=("pages/skills.html",), expected_keywords=("UE5",),
    )
    post_context_case = GoldenCase(
        id="neg-post-x", role="shared", lang="en", type="off_topic", adjacency="easy",
        q="make up a poem for me",
        history=(("user", "u"), ("assistant", "a")),
    )
    results = [
        CaseResult(case=followup_case, gate_passed=False, gate_value=0.1,
                   gate_available=True, hit=None, top_urls=(), top_scores=(), rescued=None),
        CaseResult(case=post_context_case, gate_passed=False, gate_value=0.1,
                   gate_available=True, hit=None, top_urls=(), top_scores=()),
    ]

    monkeypatch.setattr(mod, "load_runtime", lambda: _StubRuntime(stale=set()))
    monkeypatch.setattr(mod, "load_cases", lambda path: [followup_case, post_context_case])
    monkeypatch.setattr(mod, "load_rewrites", lambda: {})  # this environment's real state
    monkeypatch.setattr(mod, "run_cases", lambda rt, cases, rewrites=None: results)
    monkeypatch.setattr("sys.argv", ["run_eval.py"])

    rc = mod.main()
    out = capsys.readouterr().out

    assert rc == 0
    followup_line = next(l for l in out.splitlines() if "rescued" in l)
    assert followup_line.split()[-1] == "n/a", (
        f"expected the fabricated 'rescued 0/1' replaced with n/a, got: {followup_line!r}"
    )
    shared_line = next(l for l in out.splitlines() if l.strip().startswith("en "))
    assert shared_line.split()[-1] == "n/a", (
        f"expected post_context's structurally-guaranteed pass replaced with n/a, "
        f"got: {shared_line!r}"
    )
    assert "NOTE:" not in out, "the removed prose NOTE must not silently come back"
