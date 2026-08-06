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
    FOLLOWUP_POSITIVES_PER_LANG,
    GOLDEN_PATH,
    NEGATIVES_PER_LANG,
    POSITIVES_PER_CELL,
    REWRITES_PATH,
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
    """Exact-equality normalization for test_cases_are_disjoint_from_fit_data.

    The strip set includes the full-width CJK punctuation "，、；" (the
    long-deferred Task 8 minor). It stayed theoretical until the final
    whole-branch review demonstrated the exploit: a held-out golden zh question
    copied verbatim into ON_TOPIC_ZH with its trailing "？" swapped for "，"
    left the entire suite green.

    This is the SMALLER half of that fix -- widening the strip set only defeats
    punctuation tricks, and the same review's other mutation ("Tell me: " +
    a verbatim golden EN question) needs no punctuation trick at all. The real
    fix is test_disjointness.py's LCS comparator now running on the
    calibration <-> golden edge; this remains the cheap exact-match backstop."""
    return " ".join(text.lower().split()).strip(" ?？。!！,.，、；")


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
    """Positives name a real role.json role; negatives belong to no role and
    must carry the literal SHARED_ROLE marker instead (see evaluation.cell) --
    a legacy per-role name here would silently re-create the per-role negative
    pools the shared-pool migration collapsed."""
    positive_roles = {c.role for c in cases if c.type == "positive"}
    assert positive_roles <= set(ROLES), "positive case names a role absent from roles.json"
    negative_roles = {c.role for c in cases if c.type != "positive"}
    assert negative_roles <= {SHARED_ROLE}, f"negative case(s) with a non-shared role: {negative_roles}"
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
    green while the dataset is being built out cell by cell.

    History-bearing (multi-turn "follow-up") cases are excluded from this
    count -- they measure the escalated rewrite-and-regate path, not
    retrieval-in-isolation, so folding one in would silently grow a cell's
    hit@4/keyword-coverage denominator for a reason that has nothing to do
    with retrieval quality (see evaluation.FOLLOWUP_POSITIVES_PER_LANG's
    comment). They are not merely exempted, though: see
    test_followup_positive_pool_has_the_right_composition below, which holds
    them to their own declared quota so this exemption can't quietly become
    an unbounded, unbalanced pile of follow-up cases in one cell."""
    by_cell: dict[str, Counter] = {}
    for c in cases:
        if c.type == "positive" and not c.history:
            by_cell.setdefault(c.cell, Counter())["positive"] += 1
    wrong = {
        cell: dict(counts)
        for cell, counts in by_cell.items()
        if dict(counts) != {"positive": POSITIVES_PER_CELL}
    }
    assert not wrong, f"positive cells with the wrong count (want {POSITIVES_PER_CELL}): {wrong}"


def test_followup_positive_pool_has_the_right_composition(cases) -> None:
    """The declared quota multi-turn positives are held to, in place of the
    POSITIVES_PER_CELL count the composition test above excludes them from --
    same "declared, not silently unprotected" contract as
    NEGATIVES_PER_LANG's post_context entry. Without this, a case dropped
    into one cell's history-bearing pile with no balance check of its own
    could grow without bound and no test would notice.

    FOLLOWUP_POSITIVES_PER_LANG is a per-language MAPPING, not a scalar,
    because the languages are not symmetric here: for a long time zh was
    declared 0 by measured finding (essentially every natural zh follow-up
    phrasing cleared the gate standalone, so there was no rescue phenomenon
    to measure), then later moved to 2 once a narrower phrasing shape
    (narrative-continuation fillers, not the elided-argument shape that
    dominates the EN side) was found to clear the bar with a real margin --
    see the constant's own comment and KNOWN_ISSUES.md Finding Q for the
    full record of both. Asserting the full mapping means either value
    change -- zh moving off 0, or any count changing at all -- must update
    this constant to justify itself, not just add a case and hope; it is
    enforced, not merely documented."""
    by_lang = Counter(c.lang for c in cases if c.type == "positive" and c.history)
    got = {lang: by_lang.get(lang, 0) for lang in FOLLOWUP_POSITIVES_PER_LANG}
    assert got == FOLLOWUP_POSITIVES_PER_LANG, (
        f"follow-up positive pool composition (want {FOLLOWUP_POSITIVES_PER_LANG}): {got}"
    )
    undeclared = set(by_lang) - set(FOLLOWUP_POSITIVES_PER_LANG)
    assert not undeclared, f"follow-up positives for an undeclared language: {undeclared}"


def test_shared_negative_pool_has_the_right_composition(cases) -> None:
    """Negatives share ONE pool per language, keyed by SHARED_ROLE, split by
    off_topic's adjacency (easy/adjacent), injection, and -- ahead of both,
    mirroring evaluation.aggregate()'s own bucket selection -- whether a
    conversation precedes the question (post_context). A post-context
    negative tests a different failure mode (the rewriter smuggling context
    into a question that should still be refused) than plain adjacency does,
    so it must not be silently absorbed into off_topic_easy/adjacent's count.
    An off_topic case with no valid adjacency buckets under 'off_topic_' +
    whatever it carries, which cannot match NEGATIVES_PER_LANG and so
    surfaces as a failure instead of silently miscounting. (load_cases now
    rejects that case outright, so this is the second line of defence rather
    than the first -- it still fires on a case constructed directly in a
    test, bypassing the loader.)"""
    by_lang: dict[str, Counter] = {}
    for c in cases:
        if c.type == "positive":
            continue
        assert c.cell == f"{SHARED_ROLE}/{c.lang}", f"{c.id}: negative not keyed to the shared pool"
        if c.type == "injection":
            bucket = "injection"
        elif c.history:
            bucket = "post_context"
        else:
            bucket = f"off_topic_{c.adjacency}"
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

# Metrics whose BASELINE value is 0, i.e. already at the floor. _find_regressions
# flags only `got < want`, so a count baselined at 0 cannot go lower and those
# comparisons are structurally incapable of ever failing: nothing in chat/tests/
# fails if the zh gate lets every adjacent off-topic probe and every injection
# through. "No metric regressed" therefore reads stronger than what it verifies,
# which is the pattern this project has been bitten by before (a count check
# reporting green while the property it protects sits at its floor).
#
# Ruling: they are DECLARED here rather than gated. An absolute floor
# assertion ("the zh gate must refuse at least one adjacent probe") would be a
# new, unmet requirement, not a regression guard -- the current measured state
# is genuinely 0/4, and inventing a threshold to make it fail is not the same
# as protecting it. Declaring the set makes the gap a pinned, reviewable fact:
# the test below fails if a metric silently JOINS this set (a real regression
# the gate cannot see) and equally if one LEAVES it (the zh gate improved and
# a new baseline was taken -- at which point it becomes genuinely protected and
# must be removed from here). Same precedent as
# test_hit_at_4_page_only_is_not_a_regression_metric below: state the exclusion
# behaviourally instead of leaving it silent.
_UNPROTECTED_AT_THE_BASELINE_FLOOR = frozenset({
    "shared/zh.refusal_adjacent",   # 0/4 -- the zh gate refuses no adjacent off-topic probe
    "shared/zh.refusal_injection",  # 0/4 -- nor any injection probe
})


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


# --- Task 24 review, Critical 1: hit_at_4_page_only is a diagnostic, -------
# --- never a pass/fail regression metric ------------------------------------


def test_hit_at_4_page_only_is_not_a_regression_metric() -> None:
    """The task-24 review explicitly warned against wiring the new page-only
    hit@4 diagnostic (evaluation.aggregate's hit_at_4_page_only) into this
    regression gate without first thinking through the gate-availability
    interaction that test_gate_metric_skipped_unless_available_on_both_sides
    (above) exists to cover. It was deliberately left out -- this is a
    guardrail against a future edit accidentally adding it to
    _REGRESSION_METRICS without that consideration, mirroring how
    build_baseline's docstring protects gate_margin the same way."""
    assert "hit_at_4_page_only" not in _REGRESSION_METRICS


def test_a_page_only_drop_alone_does_not_fail_the_regression_check() -> None:
    """Direct behavioural proof, not just absence from the tuple: a cell
    whose hit_at_4_page_only got WORSE, with every _REGRESSION_METRICS field
    unchanged or better, must report zero drops. If a future edit ever wires
    this diagnostic into the comparison by mistake, this test starts
    failing immediately instead of the regression silently going live."""
    baseline_cells = {
        "visitor/en": {
            "gate_available": True, "gate_pass": 12, "hit_at_4": 8,
            "hit_at_4_page_only": 7, "keywords_found": 10, "keywords_total": 20,
        },
    }
    current_cells = {
        "visitor/en": {
            "gate_available": True, "gate_pass": 12, "hit_at_4": 8,
            "hit_at_4_page_only": 3,  # dropped hard; every other metric held
            "keywords_found": 10, "keywords_total": 20,
        },
    }
    assert _find_regressions(baseline_cells, current_cells) == []


# --- Final whole-branch review, 4.3: the comparisons the gate cannot make ---


def test_metrics_at_their_baseline_floor_are_declared_not_silently_unprotected() -> None:
    """Which of the regression gate's comparisons are structurally incapable
    of failing must be a stated fact, checked against the real baseline.

    Measured on the current baseline: 22 live comparisons, 28 schema-skips
    (structurally correct -- positive cells carry no refusal_*, shared cells
    carry no gate_pass/hit_at_4), and exactly the two below at the floor.

    Fails in BOTH directions, which is what makes it a guard rather than a
    comment: a metric that newly drops to 0 has regressed in a way
    _find_regressions cannot report, and a metric that climbs off 0 is now
    genuinely protected and must be struck from the list."""
    if not BASELINE_PATH.exists():
        pytest.skip("no baseline yet — run scripts/run_eval.py --update-baseline")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    at_floor = {
        f"{cell}.{metric}"
        for cell, want in baseline["cells"].items()
        for metric in _REGRESSION_METRICS
        if want.get(metric) == 0
    }

    assert at_floor == set(_UNPROTECTED_AT_THE_BASELINE_FLOOR), (
        "the set of regression comparisons that cannot fire has changed.\n"
        f"  now at the floor:      {sorted(at_floor)}\n"
        f"  declared in this file: {sorted(_UNPROTECTED_AT_THE_BASELINE_FLOOR)}\n"
        "A metric that JOINED the set regressed to 0 without the gate being able "
        "to say so -- investigate before re-baselining. A metric that LEFT it is "
        "now genuinely protected: delete it from "
        "_UNPROTECTED_AT_THE_BASELINE_FLOOR."
    )


def test_a_metric_at_the_floor_genuinely_cannot_fire() -> None:
    """Behavioural proof of the claim above, pure and synthetic: a cell whose
    zh refusal counts are baselined at 0 reports no drop even when the current
    run also refuses nothing. This is not an argument about the comparator's
    code, it is the comparator's actual answer -- so if a future change ever
    makes a zero-baselined count reportable, this test fails and
    _UNPROTECTED_AT_THE_BASELINE_FLOOR can be retired."""
    baseline_cells = {
        "shared/zh": {
            "gate_available": True, "refusal_easy": 3, "n_easy": 4,
            "refusal_adjacent": 0, "n_adjacent": 4,
            "refusal_injection": 0, "n_injection": 4,
        },
    }
    current_cells = {
        "shared/zh": {
            "gate_available": True, "refusal_easy": 3, "n_easy": 4,
            "refusal_adjacent": 0, "n_adjacent": 4,   # still refuses nothing
            "refusal_injection": 0, "n_injection": 4,
        },
    }
    assert _find_regressions(baseline_cells, current_cells) == []

    # ...and the control: the one shared metric NOT at the floor does fire.
    worse = {"shared/zh": {**current_cells["shared/zh"], "refusal_easy": 2}}
    assert _find_regressions(baseline_cells, worse) == ["shared/zh.refusal_easy: 3 -> 2"]


def test_followup_metrics_are_excluded_from_the_regression_gate_only_while_no_fixture_exists() -> None:
    """followup_rescued and refusal_post_context (evaluation.aggregate) are
    deliberately absent from _REGRESSION_METRICS -- correct today, because
    chat/eval/rewrites.json does not exist (no LLM_API_KEY in this
    environment, so scripts/refresh_rewrites.py has never run) and every
    multi-turn case therefore scores off the raw question alone, never a
    replayed rewrite (see evaluation.load_rewrites's docstring and
    scripts/run_eval.py's _fixture_coverage). Wiring either metric into the
    gate today would protect a number that measures nothing.

    Same shape as test_hit_at_4_page_only_is_not_a_regression_metric above:
    an exclusion that is correct only conditionally must be tied to the
    condition that makes it correct, pinned by an assertion, not left as an
    omission indistinguishable from an oversight. The moment REWRITES_PATH
    exists, this test fails and forces a decision instead of the exclusion
    silently continuing to hold past the point it stopped being justified.

    THE TRAP for whoever wires this in once a fixture lands: adding these
    two names to _REGRESSION_METRICS is not sufficient by itself for
    refusal_post_context. It is gate-derived exactly like refusal_easy/
    refusal_adjacent/refusal_injection (all come from rt.gate()), so it must
    ALSO join _GATE_METRICS (line 246) -- otherwise a fresh clone, where
    data/gate_zh_bge.json is gitignored and so has no zh gate, will compare
    it STRICTLY on zh cells and report a false regression every time. This
    is exactly the bug test_gate_metric_skipped_unless_available_on_both_sides
    exists to prevent, and _find_regressions cannot warn about it on its own
    -- only a human wiring in the metric can remember. followup_rescued is
    NOT gate-derived the same way (it also requires a successful retrieval
    hit, see CaseResult.rescued), so it needs its own judgment call about
    gate-availability skipping, not a copy-paste of this rule.
    """
    assert "followup_rescued" not in _REGRESSION_METRICS
    assert "refusal_post_context" not in _REGRESSION_METRICS
    assert not REWRITES_PATH.exists(), (
        "chat/eval/rewrites.json now exists in this environment -- "
        "followup_rescued and refusal_post_context must be deliberately "
        "wired into the regression gate (see this test's docstring for the "
        "refusal_post_context/_GATE_METRICS trap), not left excluded by an "
        "assumption (no fixture, so nothing to protect) that no longer holds"
    )


def test_history_defaults_to_empty_and_round_trips(tmp_path) -> None:
    from portfolio_rag.evaluation import load_cases

    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "a", "role": "visitor", "lang": "en", "type": "positive",
                    "q": "what engines", "expected_urls": ["pages/skills.html"],
                    "expected_keywords": ["UE5"]}) + "\n"
        + json.dumps({"id": "b", "role": "visitor", "lang": "en", "type": "positive",
                      "q": "is there any optimization work",
                      "history": [{"role": "user", "content": "tell me about prime engine"},
                                  {"role": "assistant", "content": "a custom C++ renderer"}],
                      "expected_urls": ["pages/prime-engine.html"],
                      "expected_keywords": ["Prime Engine"]}) + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].history == ()
    assert cases[1].history == (
        ("user", "tell me about prime engine"),
        ("assistant", "a custom C++ renderer"),
    )
    assert cases[1].messages == [
        {"role": "user", "content": "tell me about prime engine"},
        {"role": "assistant", "content": "a custom C++ renderer"},
    ]


def test_a_case_with_history_stays_hashable() -> None:
    """GoldenCase is frozen, so it generates __hash__ from its fields. A dict
    or list history field would make every multi-turn case unhashable and blow
    up anything that puts cases in a set."""
    from portfolio_rag.evaluation import GoldenCase

    assert hash(GoldenCase(id="a", role="visitor", lang="en", type="positive", q="q",
                           history=(("user", "hi"),)))


def test_history_must_alternate_and_end_on_an_assistant_turn(tmp_path) -> None:
    """A case whose history ends on a user turn is not a conversation the
    widget could ever have produced -- state.history is only ever appended to
    in user/assistant pairs."""
    from portfolio_rag.evaluation import load_cases

    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "a", "role": "visitor", "lang": "en", "type": "positive",
                    "q": "q", "expected_urls": ["pages/skills.html"],
                    "expected_keywords": ["UE5"],
                    "history": [{"role": "user", "content": "hi"}]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assistant"):
        load_cases(path)


def test_history_rejects_a_bad_role(tmp_path) -> None:
    from portfolio_rag.evaluation import load_cases

    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "a", "role": "visitor", "lang": "en", "type": "positive",
                    "q": "q", "expected_urls": ["pages/skills.html"],
                    "expected_keywords": ["UE5"],
                    "history": [{"role": "system", "content": "hi"},
                                {"role": "assistant", "content": "yo"}]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="role"):
        load_cases(path)


# --- Task 8: the recorded-rewrite fixture ------------------------------------


def test_score_case_is_untouched_for_a_single_turn_case(rt) -> None:
    """The overwhelming majority of the set. A case with no history must score
    exactly as it did before rewrites existed."""
    from portfolio_rag.evaluation import GoldenCase, score_case

    case = GoldenCase(id="x", role="visitor", lang="en", type="positive",
                      q="what game engines has he used",
                      expected_urls=("pages/skills.html",), expected_keywords=("UE5",))

    with_fixture = score_case(rt, case, rewrites={"x": "should be ignored"})
    without = score_case(rt, case)

    assert with_fixture.gate_passed == without.gate_passed
    assert with_fixture.top_urls == without.top_urls
    assert with_fixture.rewrite_used is None
    assert with_fixture.rescued is None


def test_a_followup_that_already_passes_the_gate_is_not_rewritten(rt) -> None:
    """The rewrite is LAZY: it only runs when the raw question fails. A
    follow-up that stands on its own must not pay for it, and must not be
    scored as a rescue."""
    from portfolio_rag.evaluation import GoldenCase, score_case

    case = GoldenCase(id="x", role="visitor", lang="en", type="positive",
                      q="what game engines has he used",
                      history=(("user", "tell me about prime engine"), ("assistant", "...")),
                      expected_urls=("pages/skills.html",), expected_keywords=("UE5",))

    result = score_case(rt, case, rewrites={"x": "a rewrite that should never be used"})

    assert result.rewrite_used is None
    assert result.rescued is None


def test_a_missing_fixture_entry_leaves_the_case_refused_rather_than_inventing_one(rt) -> None:
    """A case added to golden.jsonl without re-running refresh_rewrites.py must
    score as "still refused", never as a silent pass. An absent measurement is
    never a favourable one."""
    from portfolio_rag.evaluation import GoldenCase, score_case

    case = GoldenCase(id="missing", role="visitor", lang="en", type="off_topic",
                      adjacency="easy", q="write me a poem",
                      history=(("user", "tell me about prime engine"), ("assistant", "...")))

    result = score_case(rt, case, rewrites={})

    assert result.rewrite_used is None
    assert result.gate_passed is False


def test_load_rewrites_of_an_absent_file_is_empty(tmp_path) -> None:
    from portfolio_rag.evaluation import load_rewrites

    assert load_rewrites(tmp_path / "nope.json") == {}


# --- Task 9: follow-up rescue and post-context refusal metrics --------------


def test_followup_rescue_is_counted_only_for_multi_turn_positives() -> None:
    """A multi-turn positive counts ONLY toward n_followup/followup_rescued,
    never toward n_positive (or gate_pass/hit_at_4/keywords_*, exercised
    directly by test_followup_positives_do_not_contaminate_single_turn_metrics
    below). Folding it into n_positive too would grow that cell's hit@4 and
    keyword-coverage DENOMINATOR every time a follow-up case is added, which
    is exactly the contamination this task's own review caught empirically:
    adding four follow-up positives moved two existing cells'
    keyword_coverage against an unchanged baseline, with no change to either
    cell's single-turn cases."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    single = GoldenCase(id="a", role="visitor", lang="en", type="positive", q="q",
                        expected_urls=("pages/skills.html",), expected_keywords=("UE5",))
    multi = GoldenCase(id="b", role="visitor", lang="en", type="positive", q="q",
                       history=(("user", "u"), ("assistant", "a")),
                       expected_urls=("pages/skills.html",), expected_keywords=("UE5",))
    results = [
        CaseResult(case=single, gate_passed=True, gate_value=0.4, gate_available=True,
                   hit=True, top_urls=(), top_scores=()),
        # rewrite_used set: a real rescue is only ever reachable through a
        # REPLAYED rewrite (score_case only sets rescued once rewrite_used is
        # not None), and n_followup now keys off exactly that signal -- see
        # aggregate()'s refused_standalone comment. Omitting it here would
        # describe a state score_case can never actually produce (rescued=True
        # with no rewrite ever replayed) and, post-fix, would no longer count
        # toward n_followup at all.
        CaseResult(case=multi, gate_passed=True, gate_value=0.4, gate_available=True,
                   hit=True, top_urls=(), top_scores=(), rewrite_used="a rewrite", rescued=True),
    ]

    cell = aggregate(results)["visitor/en"]

    assert cell["n_positive"] == 1, "the multi-turn case must not inflate n_positive"
    assert cell["n_followup"] == 1, "only the multi-turn case counts toward rescue"
    assert cell["followup_rescued"] == 1


def test_followup_positives_do_not_contaminate_single_turn_metrics() -> None:
    """Direct behavioural proof, not just n_positive: a multi-turn positive
    that GATE-PASSES and HITS on its own (i.e. would inflate every single-turn
    metric if it were folded in) must move none of gate_pass/hit_at_4/
    hit_at_4_page_only/keywords_found/keywords_total/retrieved_langs -- only
    n_followup/followup_rescued."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    single = GoldenCase(id="a", role="visitor", lang="en", type="positive", q="q",
                        expected_urls=("pages/skills.html",), expected_keywords=("UE5",))
    multi = GoldenCase(id="b", role="visitor", lang="en", type="positive", q="q",
                       history=(("user", "u"), ("assistant", "a")),
                       expected_urls=("pages/skills.html",), expected_keywords=("UE5", "Godot"))
    only_single = aggregate([
        CaseResult(case=single, gate_passed=True, gate_value=0.4, gate_available=True,
                   hit=True, top_urls=("pages/skills.html",), top_scores=(0.9,),
                   keywords_found=("UE5",), retrieved_langs={"en": 1}),
    ])["visitor/en"]
    with_multi = aggregate([
        CaseResult(case=single, gate_passed=True, gate_value=0.4, gate_available=True,
                   hit=True, top_urls=("pages/skills.html",), top_scores=(0.9,),
                   keywords_found=("UE5",), retrieved_langs={"en": 1}),
        # rewrite_used set for the same reason as the sibling test above: a
        # rescue is only reachable through a replayed rewrite, and n_followup
        # now keys off that signal rather than bare case.history.
        CaseResult(case=multi, gate_passed=True, gate_value=0.4, gate_available=True,
                   hit=True, top_urls=("pages/skills.html",), top_scores=(0.9,),
                   keywords_found=("UE5", "Godot"), retrieved_langs={"en": 1},
                   rewrite_used="a rewrite", rescued=True),
    ])["visitor/en"]

    for metric in ("gate_pass", "hit_at_4", "hit_at_4_page_only",
                   "keywords_found", "keywords_total", "retrieved_langs"):
        assert with_multi[metric] == only_single[metric], (
            f"{metric} moved from {only_single[metric]} to {with_multi[metric]} "
            "when a follow-up positive was added -- it must be invisible to "
            "every single-turn metric"
        )
    assert with_multi["n_followup"] == 1 and with_multi["followup_rescued"] == 1


def test_an_unrescued_followup_does_not_inflate_followup_rescued() -> None:
    """The mirror of test_a_post_context_negative_that_passes_is_counted_as_a_
    failure_to_refuse, on the positive side: every existing followup_rescued
    assertion in this file used rescued=True, so nothing pinned the counter
    to 0 when a follow-up was NOT rescued. Mutating aggregate() to
    `cell["followup_rescued"] += 1` unconditionally (dropping the
    `int(bool(r.rescued))` guard, i.e. crediting every follow-up as rescued
    regardless of CaseResult.rescued) left the whole suite green before this
    test existed -- caught by task-9 review, not by any test.

    Covers both ways a follow-up can be UNrescued: rescued=None (no rewrite
    was ever attempted -- no fixture entry, or the raw question already
    passed the gate) and rescued=False (a rewrite WAS attempted and replayed
    but still failed to clear the gate/land the page). Both must count
    toward n_followup (the case was still a follow-up) but neither toward
    followup_rescued."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    never_attempted = GoldenCase(id="a", role="visitor", lang="en", type="positive", q="q",
                                 history=(("user", "u"), ("assistant", "a")),
                                 expected_urls=("pages/skills.html",), expected_keywords=("UE5",))
    attempted_and_failed = GoldenCase(id="b", role="visitor", lang="en", type="positive", q="q",
                                      history=(("user", "u"), ("assistant", "a")),
                                      expected_urls=("pages/skills.html",), expected_keywords=("UE5",))
    results = [
        CaseResult(case=never_attempted, gate_passed=False, gate_value=0.1, gate_available=True,
                   hit=None, top_urls=(), top_scores=(), rescued=None),
        CaseResult(case=attempted_and_failed, gate_passed=False, gate_value=0.3, gate_available=True,
                   hit=False, top_urls=(), top_scores=(), rewrite_used="a rewrite", rescued=False),
    ]

    cell = aggregate(results)["visitor/en"]

    assert cell["n_followup"] == 2, "both cases are still follow-ups"
    assert cell["followup_rescued"] == 0, "neither case was actually rescued"


def test_a_followup_that_passes_the_gate_standalone_does_not_inflate_n_followup() -> None:
    """n_followup must count cases that were actually REFUSED standalone, not
    every case carrying history. The metric is defined (KNOWN_ISSUES.md
    Finding Q / eval/README.md) as "of the context-dependent follow-ups --
    those that fail the gate standing alone -- what fraction clear the gate
    after rewrite." A follow-up whose raw question already passes the gate on
    its own never reaches score_case's `if case.history and not
    decision.passed:` guard, so rewrite_used stays None and gate_passed is
    the ORIGINAL passing decision, not a replayed one -- rescue was never
    attempted for it. Counting it here would grow the denominator with a case
    structurally incapable of ever incrementing the numerator, permanently
    depressing the rate.

    It must not spill into n_positive either (mirroring
    test_followup_rescue_is_counted_only_for_multi_turn_positives): a
    multi-turn case that passed standalone is not a context-dependent
    follow-up by the metric's own definition, so it belongs in neither
    bucket -- it is simply not what this measurement is about."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    passed_standalone = GoldenCase(
        id="a", role="visitor", lang="en", type="positive", q="q",
        history=(("user", "u"), ("assistant", "a")),
        expected_urls=("pages/skills.html",), expected_keywords=("UE5",),
    )
    result = CaseResult(
        case=passed_standalone, gate_passed=True, gate_value=0.4, gate_available=True,
        hit=True, top_urls=("pages/skills.html",), top_scores=(0.9,),
        rewrite_used=None, rescued=None,
    )

    cell = aggregate([result])["visitor/en"]

    assert cell["n_followup"] == 0, (
        "a case that passed the gate standalone was never a rescue attempt"
    )
    assert cell["n_positive"] == 0, "nor is it a single-turn positive"
    assert cell["followup_rescued"] == 0


def test_a_post_context_negative_gets_its_own_bucket() -> None:
    """Bucketed by whether it has history, NOT by adjacency: adjacency measures
    how close the QUESTION is to the domain, which is orthogonal to whether the
    conversation could lend it topicality. Blending them would hide exactly the
    regression this bucket exists to catch."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    easy = GoldenCase(id="n1", role="shared", lang="en", type="off_topic",
                      adjacency="easy", q="what's the weather")
    post = GoldenCase(id="n2", role="shared", lang="en", type="off_topic",
                      adjacency="easy", q="write me a poem",
                      history=(("user", "tell me about prime engine"), ("assistant", "...")))
    results = [
        CaseResult(case=easy, gate_passed=False, gate_value=0.1, gate_available=True,
                   hit=None, top_urls=(), top_scores=()),
        CaseResult(case=post, gate_passed=False, gate_value=0.1, gate_available=True,
                   hit=None, top_urls=(), top_scores=()),
    ]

    cell = aggregate(results)["shared/en"]

    assert cell["n_easy"] == 1 and cell["refusal_easy"] == 1
    assert cell["n_post_context"] == 1 and cell["refusal_post_context"] == 1


def test_a_post_context_negative_that_passes_is_counted_as_a_failure_to_refuse() -> None:
    """The sticky-gate regression, stated as an assertion: if the rewriter
    lends a weather question Prime Engine's topicality, this number drops."""
    from portfolio_rag.evaluation import CaseResult, GoldenCase, aggregate

    post = GoldenCase(id="n2", role="shared", lang="en", type="off_topic",
                      adjacency="easy", q="write me a poem",
                      history=(("user", "tell me about prime engine"), ("assistant", "...")))
    results = [CaseResult(case=post, gate_passed=True, gate_value=0.4, gate_available=True,
                          hit=None, top_urls=(), top_scores=())]

    cell = aggregate(results)["shared/en"]

    assert cell["n_post_context"] == 1
    assert cell["refusal_post_context"] == 0


def test_a_replayed_rewrite_that_clears_the_gate_is_scored_as_a_rescue(rt) -> None:
    """Task 8's four score_case tests (above) never exercise a SUCCESSFUL
    replay: every one of them asserts an ABSENCE (rewrite_used is None, or the
    gate stays refused). Neutering the consultation block outright --
    `if case.history and not decision.passed:` replaced with
    `if False and ...` -- still leaves all four green. This closes that gap.

    No real fixture is needed: score_case never calls an LLM itself, it only
    replays whatever `rewrites` hands it, so a hand-written dict exercises
    exactly the same replay code a real chat/eval/rewrites.json entry would.
    Mirrors golden.jsonl's real followup-en-01 case, whose whole premise is
    that the raw question cannot stand alone -- verified below rather than
    assumed, so a future gate recalibration that happens to let it through
    fails loudly here instead of silently invalidating the test.
    """
    from portfolio_rag.evaluation import GoldenCase, score_case

    case = GoldenCase(
        id="followup-en-01", role="client_dev_recruiter", lang="en", type="positive",
        q="what about tuning it",
        history=(("user", "tell me about prime engine"),
                  ("assistant", "Prime Engine is a custom C++ rendering engine he built.")),
        expected_urls=("pages/prime-engine.html",), expected_keywords=("Prime Engine",),
    )
    rewrite = "Is there any tuning or optimization work on Prime Engine?"

    raw = score_case(rt, case)
    assert not raw.gate_passed, (
        "fixture assumption: the raw follow-up must fail the gate on its own "
        "or this test cannot distinguish a real rescue from a case that never "
        "needed one -- see followup-en-01's note in golden.jsonl"
    )

    result = score_case(rt, case, rewrites={case.id: rewrite})

    assert result.rewrite_used == rewrite
    assert result.gate_passed, "the replayed rewrite should clear the gate"
    assert result.top_urls != raw.top_urls, (
        "retrieval must actually be re-run against the rewrite, not just labeled"
    )
    assert result.hit, "the replayed rewrite's retrieval should land the expected page"
    assert result.rescued is True
