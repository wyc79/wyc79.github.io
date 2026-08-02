"""Rounding-safety tests for portfolio_rag.gate_calibration.compute_gate.

A code-review finding on Task 27: ordinary round() can round a threshold UP
past the true (unrounded) on-topic floor it is derived from --
round(0.41065001, 4) == 0.4107, which is ABOVE 0.41065001. If that value is
`hi` (the zero-catch branch) or a midpoint just barely under it (the normal
branch, in an unrealistically tight margin), the STORED threshold can exceed
the exact score of the calibration query that defines the on-topic floor --
so that very query would be refused at eval time, contradicting the
zero-false-refusal guarantee compute_gate's policy is built around (see its
module docstring). `_floor_threshold` rounds DOWN instead, and both branches
use it -- these tests pin that guarantee directly, independent of the real
ONNX models or any committed artifact.

A rank-1 (1-corpus-row, 1-dim) fake embedder + matrix makes stat_value("top",
...) return exactly the fake score assigned to each query, with GATE_STATS
patched to ("top",) so there is no cross-statistic tie-break to reason
about -- the test is entirely about the rounding arithmetic, not about which
statistic compute_gate picks.
"""

import numpy as np

from portfolio_rag import gate_calibration
from portfolio_rag.gate_calibration import _floor_threshold, compute_gate


class _FakeEmbedder:
    """embed_query(text) returns a 1-dim vector holding a pre-assigned score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([self._scores[text]], dtype=np.float64)


_MATRIX = np.array([[1.0]], dtype=np.float32)  # 1 corpus row, 1 dim: matrix @ v == v


def test_floor_threshold_rounds_down_not_to_nearest() -> None:
    """Direct pin of the helper: the reviewer's exact example."""
    value = 0.41065001
    assert round(value, 4) == 0.4107, "sanity: ordinary round() rounds this UP"
    assert round(value, 4) > value, "sanity: that rounded value exceeds the input"
    assert _floor_threshold(value) == 0.4106
    assert _floor_threshold(value) <= value


def test_floor_threshold_tolerates_float_noise_around_a_clean_boundary() -> None:
    """A value meant to be exactly 0.2006 can be stored as 0.20059999999999998
    (ordinary float representation noise). The 1e-9 epsilon must still land
    it on 0.2006, not floor it an extra step down to 0.2005."""
    noisy = 0.2006 - 1e-16 * 3  # nudge below the clean value, within float ULP noise
    assert _floor_threshold(0.2006) == 0.2006
    assert _floor_threshold(noisy) == 0.2006


def test_zero_catch_branch_threshold_never_exceeds_the_on_topic_floor(monkeypatch) -> None:
    """Reviewer's exact scenario: hi = 0.41065001, nothing caught (the
    off-topic score sits above hi, so the zero-catch branch fires). Before
    the fix, threshold = round(hi, 4) = 0.4107 > hi -- the on-topic query
    that defines the floor would itself be refused. After the fix,
    threshold must be <= hi.
    """
    monkeypatch.setattr(gate_calibration, "GATE_STATS", ("top",))
    hi = 0.41065001
    off_score = 0.5  # above hi -- not caught, so this exercises the zero-catch branch
    embedder = _FakeEmbedder({"on": hi, "off": off_score})

    result = compute_gate(embedder, _MATRIX, on=["on"], off=["off"])

    assert result["margin"] <= 0, "expected the zero-catch branch (nothing caught)"
    assert result["threshold"] <= hi, (
        f"threshold {result['threshold']} exceeds the on-topic floor {hi} -- "
        "the calibration query defining the floor would be refused"
    )
    assert result["threshold"] == 0.4106


def test_midpoint_branch_threshold_never_exceeds_the_on_topic_floor(monkeypatch) -> None:
    """The same rounding mistake, in the OTHER branch: an unrealistically
    tight margin (hi - lo ~= 0.00004, far tighter than either shipped gate,
    which sit at multi-percent margins) where the midpoint's naive rounding
    also crosses hi. Confirms the fix was applied consistently to both
    branches, not just the zero-catch one.
    """
    monkeypatch.setattr(gate_calibration, "GATE_STATS", ("top",))
    hi = 0.500071
    lo = 0.500031  # caught (lo < hi), so this exercises the midpoint branch
    midpoint = (lo + hi) / 2
    assert round(midpoint, 4) > hi, "sanity: naive rounding of this midpoint crosses hi"
    embedder = _FakeEmbedder({"on": hi, "off": lo})

    result = compute_gate(embedder, _MATRIX, on=["on"], off=["off"])

    assert result["margin"] > 0, "expected the midpoint (catch) branch, not zero-catch"
    assert result["threshold"] <= hi, (
        f"threshold {result['threshold']} exceeds the on-topic floor {hi} -- "
        "the calibration query defining the floor would be refused"
    )


def test_floor_threshold_never_exceeds_input_for_a_range_of_values() -> None:
    """Property-style sweep, not just the two hand-picked cases above: for
    many values whose 4th decimal digit sits right at or past the rounding
    boundary, the floored result must never exceed the original value --
    the one property this function exists to guarantee. round() is used as
    the point of comparison to confirm the sweep actually includes cases
    where the two functions disagree (otherwise the sweep would prove
    nothing)."""
    disagreements = 0
    for i in range(1, 10000):
        value = 0.1 + i * 1e-5  # sweeps through every possible 5th-decimal digit
        floored = _floor_threshold(value)
        assert floored <= value, f"{floored} > {value}"
        if round(value, 4) != floored:
            disagreements += 1
    assert disagreements > 0, "sweep never hit a case where round() and floor differ"
