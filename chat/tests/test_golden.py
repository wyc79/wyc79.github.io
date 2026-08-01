"""Golden-set tests.

Three tests, not 160: well-formedness and disjointness are fast and need no
models; the regression gate (added in Task 8) runs the whole set once.
"""

import json
from collections import Counter

import pytest

from portfolio_rag.config import settings
from portfolio_rag.evaluation import (
    ADJACENCY,
    BASELINE_PATH,
    CASE_TYPES,
    GOLDEN_PATH,
    NEGATIVES_PER_LANG,
    POSITIVES_PER_CELL,
    SHARED_ROLE,
    aggregate,
    load_cases,
    run_cases,
)
from portfolio_rag.gate_calibration import OFF_TOPIC, OFF_TOPIC_ZH, ON_TOPIC, ON_TOPIC_ZH
from portfolio_rag.roles import ROLES
from portfolio_rag.runtime import load_runtime
from tests.test_gate import OFF_TOPIC as GATE_TEST_OFF_TOPIC


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" ?？。!！,.")


@pytest.fixture(scope="module")
def cases():
    return load_cases(GOLDEN_PATH)


@pytest.fixture(scope="module")
def rt():
    runtime = load_runtime()
    if not runtime.retrieval_available:
        pytest.skip("e5 retrieval model not present")
    return runtime


def test_ids_are_unique(cases) -> None:
    dupes = [cid for cid, n in Counter(c.id for c in cases).items() if n > 1]
    assert not dupes, f"duplicate case ids: {dupes}"


def test_roles_and_types_are_valid(cases) -> None:
    assert {c.role for c in cases} <= set(ROLES), "case names a role absent from roles.json"
    assert {c.type for c in cases} <= set(CASE_TYPES)
    assert {c.lang for c in cases} <= {"en", "zh"}


def test_positives_have_expectations_and_others_do_not(cases) -> None:
    for c in cases:
        if c.type == "positive":
            assert c.expected_urls, f"{c.id}: positive with no expected_urls"
            assert c.expected_keywords, f"{c.id}: positive with no expected_keywords"
            assert len(c.expected_keywords) <= 4, f"{c.id}: more than 4 keywords"
        else:
            assert not c.expected_urls, f"{c.id}: {c.type} must not carry expected_urls"
            assert not c.expected_keywords, f"{c.id}: {c.type} must not carry keywords"


def test_keywords_are_distinctive_enough_to_match_safely(cases) -> None:
    """A one- or two-character Latin keyword matches too much to mean anything.
    CJK is exempt: two Chinese characters are a real word."""
    weak = [
        f"{c.id}: {kw!r}"
        for c in cases
        for kw in c.expected_keywords
        if kw.isascii() and len(kw.strip()) < 3
    ]
    assert not weak, f"keywords too short to be distinctive: {weak}"


def test_expected_urls_point_at_real_pages(cases) -> None:
    missing = [
        f"{c.id} -> {url}"
        for c in cases
        for url in c.expected_urls
        if not (settings.site_root / url).exists()
    ]
    assert not missing, f"expected_urls naming files that do not exist: {missing}"


def test_cases_are_disjoint_from_fit_data(cases) -> None:
    """The calibration sets PICK the gate threshold and the starters are already
    asserted by test_gate.py. Reusing either scores the system against itself."""
    forbidden = {_normalize(s) for s in ON_TOPIC + OFF_TOPIC + ON_TOPIC_ZH + OFF_TOPIC_ZH}
    forbidden |= {_normalize(s) for s in GATE_TEST_OFF_TOPIC}
    for role in ROLES.values():
        forbidden |= {_normalize(s) for s in role["starters"]}
        forbidden |= {_normalize(s) for s in role.get("zh", {}).get("starters", [])}

    collisions = [c.id for c in cases if _normalize(c.q) in forbidden]
    assert not collisions, f"golden cases reuse fit-on data: {collisions}"


def test_positive_cells_have_the_right_composition(cases) -> None:
    """Checks every positive cell PRESENT in the file, so the suite stays
    green while the dataset is being built out cell by cell."""
    by_cell: dict[str, Counter] = {}
    for c in cases:
        if c.type == "positive":
            by_cell.setdefault(c.cell, Counter())["positive"] += 1
    wrong = {
        cell: dict(counts)
        for cell, counts in by_cell.items()
        if dict(counts) != {"positive": POSITIVES_PER_CELL}
    }
    assert not wrong, f"positive cells with the wrong count (want {POSITIVES_PER_CELL}): {wrong}"


def test_shared_negative_pool_has_the_right_composition(cases) -> None:
    """Negatives share ONE pool per language, keyed by SHARED_ROLE, split by
    off_topic's adjacency (easy/adjacent) plus injection. An off_topic case
    with no valid adjacency buckets under 'off_topic_' + whatever it carries
    (including '' on a not-yet-migrated case), which cannot match
    NEGATIVES_PER_LANG and so surfaces as a failure instead of silently
    miscounting."""
    by_lang: dict[str, Counter] = {}
    for c in cases:
        if c.type == "positive":
            continue
        assert c.cell == f"{SHARED_ROLE}/{c.lang}", f"{c.id}: negative not keyed to the shared pool"
        bucket = "injection" if c.type == "injection" else f"off_topic_{c.adjacency}"
        by_lang.setdefault(c.lang, Counter())[bucket] += 1
    wrong = {
        lang: dict(counts)
        for lang, counts in by_lang.items()
        if dict(counts) != NEGATIVES_PER_LANG
    }
    assert not wrong, (
        f"shared negative pools with the wrong composition (want {NEGATIVES_PER_LANG}): {wrong}"
    )


def test_all_eight_positive_cells_are_present(cases) -> None:
    """Unlike the composition test above (which only checks cells that
    already exist), this insists all 8 -- one per (role, lang) -- actually
    are present, so a future edit cannot silently delete a whole cell and
    still read as "nothing wrong here" from the per-cell composition check."""
    want = {f"{role}/{lang}" for role in ROLES for lang in ("en", "zh")}
    have = {c.cell for c in cases if c.type == "positive"}
    missing = want - have
    assert not missing, f"positive cells missing entirely: {sorted(missing)}"


def test_off_topic_adjacency_is_valid(cases) -> None:
    """adjacency drives NEGATIVES_PER_LANG bucketing (see above), so it is
    required on off_topic and forbidden everywhere else."""
    bad = [
        f"{c.id}: adjacency {c.adjacency!r} not in {ADJACENCY}"
        for c in cases
        if c.type == "off_topic" and c.adjacency not in ADJACENCY
    ]
    bad += [
        f"{c.id}: {c.type} must not carry adjacency (has {c.adjacency!r})"
        for c in cases
        if c.type != "off_topic" and c.adjacency
    ]
    assert not bad, f"invalid adjacency: {bad}"


# Task 14 replaced the single per-role "refusal" metric with a shared pool
# split into refusal_easy / refusal_adjacent / refusal_injection, and the
# baseline on disk predates that split entirely (it has no shared/lang cells
# at all -- see eval/README.md and .superpowers/sdd/EVAL_PLAN/task-14-report.md
# for why it was left stale). A metric or cell present on only ONE side is a
# SCHEMA change, not a regression, so it is skipped rather than compared; a
# metric present on BOTH sides (gate_pass/hit_at_4 on a positive cell whose
# role/lang key hasn't moved) is still compared strictly.
_GATE_METRICS = {"gate_pass", "refusal_easy", "refusal_adjacent", "refusal_injection"}
_REGRESSION_METRICS = (
    "gate_pass", "hit_at_4", "refusal_easy", "refusal_adjacent", "refusal_injection",
)


def _find_regressions(baseline_cells: dict, current_cells: dict) -> list[str]:
    """Pure comparison, no fixtures: kept separate from test_no_metric_regressed
    so the gate-availability logic can be exercised directly with synthetic
    cells (see test_gate_metric_skipped_unless_available_on_both_sides)
    instead of only indirectly through a real baseline + a real runtime."""
    drops = []
    for cell, want in baseline_cells.items():
        got = current_cells.get(cell)
        if got is None:
            drops.append(f"{cell}: missing from the dataset entirely")
            continue
        # A cjk_bypass decision means the gate is unmeasured, not perfect --
        # on a cell where EITHER side has no gate, gate_pass reads as an
        # inflated (always-True) pass rate and refusal_* reads as a deflated
        # (always-zero) refusal rate. Either direction is a false regression
        # waiting to happen, not a real one, so a gate-derived metric is only
        # ever compared when BOTH sides measured it for real. Non-gate
        # metrics (hit_at_4, keyword coverage below) are retrieval-only and
        # never skipped.
        gate_available_both = want.get("gate_available", True) and got.get("gate_available", True)
        for metric in _REGRESSION_METRICS:
            if metric not in want or metric not in got:
                continue  # metric renamed/restructured since the baseline; not comparable
            if metric in _GATE_METRICS and not gate_available_both:
                continue
            if got[metric] < want[metric]:
                drops.append(f"{cell}.{metric}: {want[metric]} -> {got[metric]}")
        if "keywords_total" in want and "keywords_total" in got:
            # Keyword coverage compares as a RATIO. Each positive carries 1-4
            # keywords, so the denominator genuinely moves when a case is edited —
            # raw found-counts would flag a shrinking keyword list as a regression.
            want_cov = want["keywords_found"] / max(want["keywords_total"], 1)
            got_cov = got["keywords_found"] / max(got["keywords_total"], 1)
            if got_cov < want_cov - 1e-9:
                drops.append(f"{cell}.keyword_coverage: {want_cov:.1%} -> {got_cov:.1%}")
    return drops


def test_no_metric_regressed(cases, rt) -> None:
    if not BASELINE_PATH.exists():
        pytest.skip("no baseline yet — run scripts/run_eval.py --update-baseline")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = aggregate(run_cases(rt, cases))

    if baseline.get("index_built_at") != rt.index_built_at:
        print(f"\nNOTE: baseline measured against index built "
              f"{baseline.get('index_built_at')}, current index is "
              f"{rt.index_built_at} — comparison is cross-build.")

    drops = _find_regressions(baseline["cells"], current)
    assert not drops, "metrics regressed against the baseline:\n  " + "\n  ".join(drops)


def test_gate_metric_skipped_unless_available_on_both_sides() -> None:
    """Direct coverage of _find_regressions's gate-availability rule, with no
    dependency on a real baseline or runtime.

    A cjk_bypass decision reports gate_passed=True unconditionally, which
    inflates gate_pass and deflates refusal_* to 0. Comparing either of those
    against a real (gate-available) measurement on the OTHER side is a false
    regression waiting to happen, in both directions:
      - want gate-available, got bypass: refusal_* reads 0 on the got side ->
        looks like a regression against a real want>0, even though nothing
        regressed -- just measured differently (this is the case the
        reviewer caught: a baseline built with a real zh gate, compared on a
        machine with none).
      - want bypass, got gate-available: gate_pass reads inflated (=n) on the
        want side -> a real drop on the got side could beat a fabricated
        ceiling and hide a genuine regression.
    So a gate-derived metric must be skipped whenever EITHER side lacks a
    real gate, and compared strictly only when BOTH have one. hit_at_4 is
    gate-free and must never be skipped, in any combination.
    """
    def cell(gate_available: bool, **metrics) -> dict:
        return {"gate_available": gate_available, "hit_at_4": 7, **metrics}

    # (want gate_available, got gate_available) -> expect gate metric skipped?
    for want_available, got_available, gate_metric_skipped in [
        (True, True, False),    # both real -> compare strictly
        (True, False, True),    # got bypassed -> skip (the reviewer's case)
        (False, True, True),    # want was bypassed -> skip
        (False, False, True),   # neither real -> skip
    ]:
        baseline_cells = {
            "shared/zh": cell(want_available, refusal_easy=3, gate_pass=7),
        }
        current_cells = {
            # gate_pass/refusal_easy deliberately "worse" than the baseline
            # so a real strict comparison WOULD flag them; hit_at_4 also
            # "worse" so it must be flagged regardless of gate availability.
            "shared/zh": cell(got_available, refusal_easy=0, gate_pass=6, hit_at_4=5),
        }
        drops = _find_regressions(baseline_cells, current_cells)
        gate_drops = [d for d in drops if "refusal_easy" in d or "gate_pass" in d]
        hit_drops = [d for d in drops if "hit_at_4" in d]

        case = f"want_available={want_available} got_available={got_available}"
        if gate_metric_skipped:
            assert not gate_drops, f"{case}: gate metric should be skipped, got {gate_drops}"
        else:
            assert gate_drops, f"{case}: gate metric should be compared strictly, got none"
        assert hit_drops, f"{case}: hit_at_4 (gate-free) must never be skipped"
