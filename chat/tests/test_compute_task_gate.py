"""Direct tests for gate_calibration.compute_task_gate / MIN_FLAGGED_ON_TOPIC
(Task 28's go/no-go discipline), independent of a full build.

Code review finding: before this file existed, compute_task_gate was only
ever exercised indirectly through tests/test_index_builder.py's full
build_index() runs, and every one of those runs (both real corpora, both
languages) happened to clear MIN_FLAGGED_ON_TOPIC -- so the None-return path
(the actual discipline the brief cares about: "do not ship the second tier
for a language with too little evidence") had NO test coverage at all. These
tests pin that path directly, plus the filtering behaviour (unflagged
entries must not leak into the flagged-subset threshold math) that a full
build's real numbers can hide by coincidence.

Reuses test_gate_calibration.py's rank-1 fake embedder: matrix @ v == v, so
stat_value("top", ...) returns exactly the score assigned to each query.
GATE_STATS is irrelevant here (compute_task_gate takes `stat` directly, it
never picks one), so it is not patched.
"""

from portfolio_rag import gate_calibration
from portfolio_rag.gate_calibration import MIN_FLAGGED_ON_TOPIC, compute_task_gate
from tests.test_gate_calibration import _MATRIX, _FakeEmbedder

# Real trigger phrases (not synthetic tokens): TASK_REQUEST_RE must actually
# flag these for the tests below to mean anything.
_FLAGGED_ON = [
    "give me a summary",
    "write me a paragraph",
    "help me understand this",
    "translate this for me",
]
_UNFLAGGED_ON = ["what did he study", "does he know Unity"]


def test_min_flagged_on_topic_is_four() -> None:
    """Pin the constant: the tests below assume this exact floor."""
    assert MIN_FLAGGED_ON_TOPIC == 4


def test_returns_none_one_below_the_floor() -> None:
    """3 flagged on-topic entries, floor is 4 -- the discipline's core
    promise: too little evidence means "don't ship," not "derive anyway.\""""
    on = _FLAGGED_ON[:3] + _UNFLAGGED_ON
    off = ["tell me a joke"]
    scores = {q: 0.5 for q in on}
    scores["tell me a joke"] = 0.1
    embedder = _FakeEmbedder(scores)

    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is None


def test_derives_a_threshold_exactly_at_the_floor() -> None:
    """Exactly 4 flagged on-topic entries must be enough -- the boundary
    the discipline rule actually draws (`< MIN_FLAGGED_ON_TOPIC`, not
    `<=`)."""
    on = _FLAGGED_ON + _UNFLAGGED_ON
    off = ["tell me a joke"]
    scores = {
        "give me a summary": 0.50,
        "write me a paragraph": 0.60,
        "help me understand this": 0.55,
        "translate this for me": 0.52,
        # Deliberately extreme and UNFLAGGED: if the filtering step were
        # broken and this leaked into the "on" pool, hi would collapse to
        # 0.02 and every assertion below would fail loudly instead of
        # silently passing.
        "what did he study": 0.02,
        "does he know Unity": 0.98,
        "tell me a joke": 0.20,
    }
    embedder = _FakeEmbedder(scores)

    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is not None
    assert result["n_flagged_on"] == 4
    assert result["n_flagged_off"] == 1
    # hi = min(flagged-on) = 0.50 ("give me a summary"); the unflagged 0.02
    # must NOT have won that min.
    assert result["hi"] == 0.50
    assert result["n_caught"] == 1  # 0.20 < 0.50
    # midpoint(0.20, 0.50) = 0.35
    assert result["threshold"] == 0.35
    assert result["threshold"] <= result["hi"], "must never exceed the flagged on-topic floor"


def test_unflagged_off_topic_entries_are_excluded_from_the_off_pool() -> None:
    """An off-topic entry TASK_REQUEST_RE does not flag must not count
    toward n_flagged_off or affect the catch computation, even when its
    score would otherwise be the most informative one."""
    on = _FLAGGED_ON
    off = [
        "tell me a joke",           # flagged
        "what's the weather today",  # NOT flagged -- must be ignored
    ]
    scores = {q: 0.5 for q in on}
    scores["tell me a joke"] = 0.3
    # Lower than the flagged one -- if this leaked in, it would (wrongly)
    # become the caught/lo value instead of "tell me a joke".
    scores["what's the weather today"] = 0.05

    embedder = _FakeEmbedder(scores)
    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is not None
    assert result["n_flagged_off"] == 1
    assert result["lo"] == 0.3


def test_zero_catch_branch_when_no_off_topic_is_flagged() -> None:
    """Mirrors compute_gate's own zero-catch branch: flagged on-topic
    evidence clears the floor, but NOTHING in `off` is task-phrased at all
    -- still zero false refusals (threshold pinned to hi), zero measured
    benefit (margin 0, n_caught 0), same signal as a non-separating gate."""
    on = _FLAGGED_ON
    off = ["what's the weather today", "who won the world cup"]
    scores = {q: 0.5 for q in on}
    scores["what's the weather today"] = 0.1
    scores["who won the world cup"] = 0.2

    embedder = _FakeEmbedder(scores)
    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is not None
    assert result["n_flagged_off"] == 0
    assert result["n_caught"] == 0
    assert result["margin"] == 0.0
    assert result["threshold"] == 0.5  # floor(hi), all four "on" scores tie at 0.5


def test_a_flagged_off_topic_value_at_or_above_hi_is_not_caught() -> None:
    """The same "cannot catch without also refusing an on-topic query" rule
    compute_gate enforces for the base gate: an off-topic value >= hi is
    excluded from the catch, not pinned to as a threshold anchor."""
    on = _FLAGGED_ON
    off = ["tell me a joke", "write me an essay"]
    scores = {q: 0.5 for q in on}  # hi = 0.5
    scores["tell me a joke"] = 0.3       # caught (0.3 < 0.5)
    scores["write me an essay"] = 0.7    # NOT caught (0.7 >= 0.5)

    embedder = _FakeEmbedder(scores)
    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is not None
    assert result["n_flagged_off"] == 2
    assert result["n_caught"] == 1
    assert result["lo"] == 0.3
    assert result["threshold"] == 0.4  # midpoint(0.3, 0.5)


def test_returns_none_when_on_is_entirely_unflagged() -> None:
    """Zero flagged on-topic entries (not just "too few") must still take
    the None branch, not a division-by-empty-list crash."""
    on = _UNFLAGGED_ON
    off = ["tell me a joke"]
    scores = {q: 0.5 for q in on}
    scores["tell me a joke"] = 0.1
    embedder = _FakeEmbedder(scores)

    result = compute_task_gate(embedder, _MATRIX, "top", on, off)

    assert result is None
