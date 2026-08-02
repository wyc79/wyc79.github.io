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
"""

import importlib.util
import json
import shutil
import subprocess

import numpy as np
import pytest

from portfolio_rag.config import settings
from portfolio_rag.runtime import MIN_SCORE, TOP_K, rank_hits

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"

# 6 chunks in 6-dimensional "one-hot" space, so chunk_i . query == query[i]
# directly -- every score in the fixture can be chosen by hand without a real
# embedder. The ranking functions under test only ever take a raw dot
# product; unit-normalizing vectors is the embedder's job, not the ranker's,
# so the fixture deliberately doesn't bother.
CHUNK_IDS = ["c1", "c2", "c3", "c4", "c5", "c6"]
MATRIX = np.eye(6, dtype=np.float32)
CHUNK_META = [{"id": cid, "url": f"pages/{cid}.html", "lang": "en"} for cid in CHUNK_IDS]

# Two query vectors (component i is chunk c{i+1}'s score):
#  - "floor": the raw top-4 by score includes a 4th-place chunk (c4, score
#    0.10) BELOW MIN_SCORE (0.18) -- exercises floor-after-top-k exactly as
#    the brief asks: the 4th-best chunk must be dropped, leaving 3 hits, not
#    4 and not backfilled from a lower-ranked chunk.
#  - "clean": 4 chunks clear the floor outright, and a 5th chunk (c5, score
#    0.22) is INDIVIDUALLY above the floor too but ranked 5th -- it must
#    stay excluded for being outside the top-4 by rank, proving the floor
#    does not turn into "everything above MIN_SCORE."
QUERIES = {
    "floor": [0.90, 0.50, 0.30, 0.10, 0.05, 0.02],
    "clean": [0.80, 0.60, 0.40, 0.25, 0.22, 0.01],
}


def test_fixture_exercises_both_a_floor_drop_and_a_clean_top_k() -> None:
    """Sanity check on the fixture itself (using runtime.py's own rank_hits,
    the same precedent test_task_request_re_sync.py's fixture-sanity test
    sets): if 'floor' didn't actually drop a raw top-4 candidate, or 'clean'
    didn't actually exclude an individually-above-floor 5th candidate, the
    agreement tests below could pass trivially without exercising the
    ordering this test exists to check."""
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


def _node_available() -> bool:
    return shutil.which("node") is not None


def _extract_widget_score_chunks_source() -> str:
    """Pull scoreChunks, and the TOP_K/MIN_SCORE constants it closes over,
    verbatim out of the widget source -- not retyped, so this cannot drift
    from the file it mirrors the way a reimplementation could."""
    src = WIDGET_PATH.read_text(encoding="utf-8")

    def _const(name: str) -> str:
        marker = f"var {name} = "
        start = src.index(marker) + len(marker)
        end = src.index(";", start)
        return src[start:end]

    fn_marker = "function scoreChunks("
    fn_start = src.index(fn_marker)
    brace_start = src.index("{", fn_start)
    depth, i = 0, brace_start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    fn_src = src[fn_start : i + 1]
    return f"var TOP_K = {_const('TOP_K')};\nvar MIN_SCORE = {_const('MIN_SCORE')};\n{fn_src}\n"


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
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    raw = json.loads(result.stdout)
    return {name: [tuple(pair) for pair in rows] for name, rows in raw.items()}


@pytest.mark.skipif(not _node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py() -> None:
    widget_results = _widget_results()
    runtime_results = _runtime_py_results()
    assert widget_results == runtime_results


@pytest.mark.skipif(not _node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_index_py() -> None:
    """All three, not just JS-vs-Python-mirror: the SCF backend and the
    widget must agree DIRECTLY too, since a visitor's real request is judged
    by whichever of these two actually runs (widget locally/degraded, backend
    when reachable) -- runtime.py is the test harness's own mirror, not
    something a visitor's request ever touches."""
    widget_results = _widget_results()
    backend_results = _index_py_results()
    assert widget_results == backend_results
