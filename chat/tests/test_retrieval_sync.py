"""Cross-implementation sync test for retrieval ranking (Task 29).

Three implementations of retrieval now exist: portfolio_rag.runtime
(imported normally here), scripts/chat-widget.js (the browser widget --
source of truth for visitor-facing behaviour), and functions/tencent/index.py
(the SCF backend, which must stay stdlib-only at module scope and so cannot
import runtime.py). Shared logic has already silently diverged three times in
this project's history (a literal-alternation typo in a Chinese regex, a
Python-Unicode-\\w vs JavaScript-ASCII-\\w mismatch, and a gate that scored
the raw question instead of gateForm()'s output) -- see runtime.py's and
chat-widget.js's own comments. This test exists so a fourth divergence, this
time in the ranking arithmetic itself, does not repeat that silently: it
EXECUTES all three real files (not a reimplementation of the ranking in test
code, which could drift from all three the same way a fourth copy would)
against one shared fixture and asserts they agree on ranked ids and scores.

The JS copy is run for real via `node` (present on this machine; the test
SKIPS, not fails, when node is unavailable on PATH -- a missing dev tool is a
different claim than "the rankings disagree," so it must not read as a
failure). It is extracted verbatim from chat-widget.js's own scoreChunks
function and TOP_K/MIN_SCORE constants, not retyped, so it cannot drift from
the file it mirrors. The index.py copy is loaded via importlib from its
actual file on disk; its module-level imports are stdlib-only (numpy is
deferred into function bodies), so a plain module load is safe and cheap
here -- see functions/tencent/index.py's own module docstring.

Fix round 1 review corrections (both folded into this fixture, not a
separate file):

- "floor before top-k" (the mutation the original task brief suggested for
  proving this test bites) is a MATHEMATICAL NO-OP, not a real bug: on a
  single descending sort, the set of candidates clearing a same-key
  threshold is always a PREFIX of the sorted list, so `slice(K).filter(f)`
  and `filter(f).slice(K)` produce identical output for any tie-free,
  same-key threshold filter -- there is no valid construction where they
  diverge. (Proof sketch: if rank K fails the floor, every rank > K has an
  equal-or-lower score by the sort itself, so it also fails; nothing beyond
  K can ever be "promoted" in by removing a failing candidate ahead of it.)
  That was the task brief's error, not an implementation gap -- confirmed by
  mutating exactly that step in both chat-widget.js and index.py: the test
  stayed green in both directions, as it must.
- The REAL bugs this fixture missed were two DIFFERENT ordering questions:
  (a) tie-breaking -- np.argsort's default is not stable, so two candidates
  with EXACTLY equal scores could rank differently than JS's
  Array.prototype.sort (stable since ES2019, which chat-widget.js relies on
  implicitly). Not hypothetical: chat/data/index.json holds several groups
  of exact-duplicate chunk vectors (identical sections indexed under more
  than one anchor/language), so ties are real fixture material, not a
  contrived edge case. (b) floor-vs-rounding order -- both Python copies
  compared MIN_SCORE against the 4-decimal-ROUNDED score, not the raw one,
  so a raw score in the ~5e-5 band just under 0.18 (e.g. 0.179960) rounded
  up to 0.1800 and incorrectly cleared the floor; chat-widget.js's
  scoreChunks never rounds internally, so it was already correct and the
  Python copies were the ones that had drifted. Both are now fixed in
  runtime.py/index.py (stable sort, raw-score floor comparison) and both are
  covered below by dedicated queries ("ties", "floor_epsilon").
"""

import importlib.util
import json

import numpy as np
import pytest

from portfolio_rag.config import settings
from portfolio_rag.runtime import MIN_SCORE, TOP_K, rank_hits
from tests._node_harness import extract_js_function, extract_js_var, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"

# 9 chunks in a 7-dimensional "one-hot" space, so chunk_i . query == query[i]
# directly for c1-c6 -- every score in the fixture can be chosen by hand
# without a real embedder. The ranking functions under test only ever take a
# raw dot product; unit-normalizing vectors is the embedder's job, not the
# ranker's, so the fixture deliberately doesn't bother.
#
# c7, c8 and c9 are LITERAL duplicates of each other (all one-hot at dim 6)
# -- a genuine tie under any query with a nonzero 7th component, mirroring
# the real committed index's exact-duplicate chunk groups. c1-c6 (dims 0-5)
# are unaffected by dim 6, so the pre-existing "floor"/"clean" queries below
# (which set dim 6 to 0) rank exactly as before this fixture grew.
#
# A three-way tie, not two: np.argsort's default (introsort/quicksort) is
# NOT guaranteed stable, but it can still happen to preserve order on a
# SPECIFIC small array by chance (this fixture's earlier two-way-tie
# revision did, and so did not catch the bug it was meant to catch -- a
# false negative found only by re-testing after the fix, not trusted on
# faith). This exact three-way-tie construction was verified directly (not
# assumed) to make numpy's default kind disagree with kind="stable" on THIS
# numpy build/version: default puts c8 in the 4th slot, stable puts c7.
# Chosen deliberately, over random fuzzing, so the divergence is
# reproducible and the fixture stays readable.
CHUNK_IDS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
MATRIX = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],  # c7
        [0, 0, 0, 0, 0, 0, 1],  # c8 -- exact duplicate of c7
        [0, 0, 0, 0, 0, 0, 1],  # c9 -- exact duplicate of c7/c8
    ],
    dtype=np.float32,
)
CHUNK_META = [{"id": cid, "url": f"pages/{cid}.html", "lang": "en"} for cid in CHUNK_IDS]

# Four query vectors (7 components: dims 0-5 score c1-c6 directly, dim 6
# scores c7/c8/c9 identically):
#  - "floor": the raw top-4 by score includes a 4th-place chunk (c4, score
#    0.10) BELOW MIN_SCORE (0.18) -- dropped, leaving 3 hits.
#  - "clean": 4 chunks clear the floor outright, and a 5th chunk (c5, score
#    0.22) is INDIVIDUALLY above the floor too but ranked 5th -- it must
#    stay excluded for being outside the top-4 by rank, proving the floor
#    does not turn into "everything above MIN_SCORE."
#  - "ties": c1/c2/c3 (0.90/0.70/0.60) clearly outrank the c7/c8/c9 trio
#    (structurally identical vectors, all score 0.55), but only ONE of that
#    tied trio fits in the 4th and last top-4 slot -- which one is exactly
#    the tie-break question. The correct, stable answer is c7 (first by
#    original fixture order); a non-stable sort can return c8 or c9 instead
#    (see the MATRIX comment above -- verified directly for c8 on this
#    numpy build). This changes which chunk id comes back, not just an
#    internal reordering of an already-agreed set.
#  - "floor_epsilon": c4's score is 0.179960 -- strictly below MIN_SCORE
#    (0.18) but rounds UP to 0.1800 at 4 decimal places. Exercises the
#    floor-vs-rounding order: correct only when the comparison runs on the
#    raw score, not the rounded one.
QUERIES = {
    "floor": [0.90, 0.50, 0.30, 0.10, 0.05, 0.02, 0.0],
    "clean": [0.80, 0.60, 0.40, 0.25, 0.22, 0.01, 0.0],
    "ties": [0.90, 0.70, 0.60, 0.50, 0.40, 0.10, 0.55],
    "floor_epsilon": [0.90, 0.50, 0.30, 0.179960, 0.05, 0.01, 0.0],
}


def test_fixture_exercises_floor_drop_ties_and_floor_epsilon() -> None:
    """Sanity check on the fixture itself (using runtime.py's own rank_hits,
    the same precedent test_task_request_re_sync.py's fixture-sanity test
    sets): if any of these four cases didn't actually exercise what its name
    claims, the agreement tests below could pass trivially without proving
    anything about ordering."""
    floor = rank_hits(MATRIX, CHUNK_META, np.asarray(QUERIES["floor"], dtype=np.float32))
    assert [h.chunk_id for h in floor.hits] == ["c1", "c2", "c3"], (
        "fixture assumption: 'floor' must drop exactly the raw rank-4 "
        "candidate (c4, below MIN_SCORE), leaving 3 hits"
    )
    assert floor.dropped_by_floor == 1

    clean = rank_hits(MATRIX, CHUNK_META, np.asarray(QUERIES["clean"], dtype=np.float32))
    assert [h.chunk_id for h in clean.hits] == ["c1", "c2", "c3", "c4"], (
        "fixture assumption: 'clean' must return exactly TOP_K hits"
    )
    assert "c5" not in [h.chunk_id for h in clean.hits], (
        "c5 clears MIN_SCORE on its own but is ranked 5th -- it must not "
        "appear just because it individually clears the floor"
    )

    ties = rank_hits(MATRIX, CHUNK_META, np.asarray(QUERIES["ties"], dtype=np.float32))
    assert [h.chunk_id for h in ties.hits] == ["c1", "c2", "c3", "c7"], (
        "fixture assumption: 'ties' must fill its 4th (last) slot with c7 "
        "-- the first-by-original-order member of the tied c7/c8/c9 trio "
        "(all score 0.55, structurally identical vectors) -- not c8 or c9"
    )

    epsilon = rank_hits(MATRIX, CHUNK_META, np.asarray(QUERIES["floor_epsilon"], dtype=np.float32))
    assert [h.chunk_id for h in epsilon.hits] == ["c1", "c2", "c3"], (
        "fixture assumption: 'floor_epsilon' must drop c4 (raw score "
        "0.179960 < MIN_SCORE, even though it ROUNDS UP to 0.1800) -- the "
        "floor must compare the raw score, not round() then compare"
    )
    assert epsilon.dropped_by_floor == 1


def _runtime_py_results() -> dict:
    out = {}
    for name, q in QUERIES.items():
        result = rank_hits(MATRIX, CHUNK_META, np.asarray(q, dtype=np.float32))
        out[name] = [(h.chunk_id, h.score) for h in result.hits]
    return out


def _index_py_rank_chunks():
    """Load functions/tencent/index.py fresh and return its real
    rank_chunks -- not a reimplementation, the module's own compiled
    function object."""
    spec = importlib.util.spec_from_file_location("_scf_index_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rank_chunks


def _index_py_results() -> dict:
    rank_chunks = _index_py_rank_chunks()
    chunks = [{"id": cid} for cid in CHUNK_IDS]
    out = {}
    for name, q in QUERIES.items():
        hits = rank_chunks(MATRIX, chunks, q)
        out[name] = [(h["chunk"]["id"], h["score"]) for h in hits]
    return out


def test_index_py_agrees_with_runtime_py() -> None:
    runtime_results = _runtime_py_results()
    backend_results = _index_py_results()
    assert backend_results == runtime_results


def _extract_widget_score_chunks_source() -> str:
    """Pull scoreChunks, and the TOP_K/MIN_SCORE constants it closes over,
    verbatim out of the widget source -- not retyped, so this cannot drift
    from the file it mirrors the way a reimplementation could."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    fn_src = extract_js_function(src, "function scoreChunks(")
    return (
        f"var TOP_K = {extract_js_var(src, 'TOP_K')};\n"
        f"var MIN_SCORE = {extract_js_var(src, 'MIN_SCORE')};\n"
        f"{fn_src}\n"
    )


def _widget_results() -> dict:
    """Execute the widget's own scoreChunks in a real Node process against
    the shared fixture, returning {query_name: [(id, score), ...]}."""
    fn_src = _extract_widget_score_chunks_source()
    script = (
        fn_src + "\n"
        "const matrix = " + json.dumps(MATRIX.tolist()) + ";\n"
        "const ids = " + json.dumps(CHUNK_IDS) + ";\n"
        "const queries = " + json.dumps(QUERIES) + ";\n"
        "const out = {};\n"
        "for (const name in queries) {\n"
        "  const q = queries[name];\n"
        "  const r = scoreChunks(matrix.length, q, function (i) { return matrix[i]; }, "
        "function (i) { return { id: ids[i] }; });\n"
        "  out[name] = r.results.map(function (x) { return [x.chunk.id, +x.score.toFixed(4)]; });\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    raw = run_node_json(script)
    return {name: [tuple(pair) for pair in rows] for name, rows in raw.items()}


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py() -> None:
    widget_results = _widget_results()
    runtime_results = _runtime_py_results()
    assert widget_results == runtime_results


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_index_py() -> None:
    """All three, not just JS-vs-Python-mirror: the SCF backend and the
    widget must agree DIRECTLY too, since a visitor's real request is judged
    by whichever of these two actually runs (widget locally/degraded, backend
    when reachable) -- runtime.py is the test harness's own mirror, not
    something a visitor's request ever touches."""
    widget_results = _widget_results()
    backend_results = _index_py_results()
    assert widget_results == backend_results
