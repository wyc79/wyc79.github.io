"""Regression test for chat-widget.js's degraded-mode retrieval (Task 29
Part 2 fix round 1 review, Important 4).

`retrieveFallback`/`runOfflineSearch` are the exact code Task 24 silently
broke and that stayed broken for five tasks: `retrieveFallback` used to map
`fb.vectors[i]` (one array, the curated gate corpus) to
`state.index.chunks[i]` (a DIFFERENT array, the full chunk index) --
positionally, by array index alone, across two arrays that no longer agreed
in size or order. Task 29 Part 2 fixed it by giving degraded mode its own
real, id-carrying retrieval corpus (chunks_en_minilm.json) and having
`retrieveFallback` index `vecAt(i)`/`chunkAt(i)` into the SAME array at the
SAME position. Before this file, nothing in `chat/tests/` executed
`retrieveFallback` at all -- a hand-run proof (real MiniLM model, real
committed data, all 48 English golden questions) is what fix round 1's
review reproduced independently, but a hand-run proof expires the moment
the session ends. This is the committed, always-run version of that check.

`retrieveFallback` and its dependency `scoreChunks` (plus the TOP_K/
MIN_SCORE constants they close over) are extracted VERBATIM from
scripts/chat-widget.js and executed in a real `node` subprocess --
mirroring test_retrieval_sync.py's extraction of `scoreChunks` and
test_implementation_sync.py's extraction of `stripName`/`gateForm`, both
via chat/tests/_node_harness.py. A reimplementation in Python could drift
from the file it's meant to mirror the same way a fourth copy would.

Deliberately a SMALL, synthetic fixture (five one-hot-ish chunk records),
not the real 192-chunk chunks_en_minilm.json -- the point of this test is
to pin the ID-RESOLUTION CONTRACT (results carry ids/fields from the exact
array `retrieveFallback` was handed), not to re-measure retrieval quality
against the real corpus, which fix round 1's report already did by hand
with the real MiniLM model. The node script also defines a `state.chunks`
stand-in populated with DIFFERENT records at the same positions (disjoint
ids from the real fixture) -- unused by the current, correct
`retrieveFallback` implementation, but present so a regression that
reintroduces exactly the historical `state.chunks[i]`-style cross-array
lookup would read from it and get caught by the assertions below, not
silently pass because the stand-in didn't exist.
"""

import json

import pytest

from portfolio_rag.config import settings
from tests._node_harness import extract_js_function, extract_js_var, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"

# Five records, deliberately not real page content -- vectors are one-hot
# (plus one blended row) in a 4-dim space so every score is exactly one
# query component and hand-verifiable. c1-c4 all clear MIN_SCORE (0.18) and
# fit in TOP_K (4); c5's blended vector scores 0.24 -- individually above
# the floor, but ranked 5th, so it must NOT appear (same "floor does not
# mean everything above MIN_SCORE" property test_retrieval_sync.py's
# "clean" query already covers for scoreChunks directly; repeated here
# because retrieveFallback is the thing actually under test).
CHUNKS = [
    {"id": "c1", "url": "pages/a.html", "anchor": "sec1", "page_title": "A",
     "section_title": "A1", "text": "chunk c1 text", "vector": [1, 0, 0, 0]},
    {"id": "c2", "url": "pages/b.html", "anchor": "sec2", "page_title": "B",
     "section_title": "B1", "text": "chunk c2 text", "vector": [0, 1, 0, 0]},
    {"id": "c3", "url": "pages/c.html", "anchor": "sec3", "page_title": "C",
     "section_title": "C1", "text": "chunk c3 text", "vector": [0, 0, 1, 0]},
    {"id": "c4", "url": "pages/d.html", "anchor": "sec4", "page_title": "D",
     "section_title": "D1", "text": "chunk c4 text", "vector": [0, 0, 0, 1]},
    {"id": "c5", "url": "pages/e.html", "anchor": "sec5", "page_title": "E",
     "section_title": "E1", "text": "chunk c5 text", "vector": [0.1, 0.1, 0.1, 0.1]},
]
QUERY = [0.9, 0.7, 0.5, 0.3]

# A DIFFERENT array, same length as the TOP_K slice (4), disjoint ids --
# stands in for the historical bug's second array (originally
# state.index.chunks, later state.chunks). Only reachable if
# retrieveFallback is mutated to reference it instead of its own `chunks`
# parameter.
DECOY_CHUNKS = [
    {"id": "decoy1", "url": "pages/decoy1.html"},
    {"id": "decoy2", "url": "pages/decoy2.html"},
    {"id": "decoy3", "url": "pages/decoy3.html"},
    {"id": "decoy4", "url": "pages/decoy4.html"},
]


def _extract_retrieve_fallback_source() -> str:
    src = WIDGET_PATH.read_text(encoding="utf-8")
    score_chunks_src = extract_js_function(src, "function scoreChunks(")
    retrieve_fallback_src = extract_js_function(src, "function retrieveFallback(")
    return (
        f"var TOP_K = {extract_js_var(src, 'TOP_K')};\n"
        f"var MIN_SCORE = {extract_js_var(src, 'MIN_SCORE')};\n"
        f"{score_chunks_src}\n"
        f"{retrieve_fallback_src}\n"
    )


def _run_retrieve_fallback():
    fn_src = _extract_retrieve_fallback_source()
    script = (
        fn_src + "\n"
        # Stand-in for the browser's module-level `state` object, carrying a
        # DIFFERENT chunks array at the same length -- present but unused by
        # a correct retrieveFallback (which never references `state` at
        # all). See the module docstring for why this exists.
        "var state = { chunks: " + json.dumps(DECOY_CHUNKS) + " };\n"
        "const chunks = " + json.dumps(CHUNKS) + ";\n"
        "const query = " + json.dumps(QUERY) + ";\n"
        "const r = retrieveFallback(chunks, query);\n"
        "process.stdout.write(JSON.stringify(r.results.map(function (x) {\n"
        "  return { id: x.chunk.id, url: x.chunk.url, anchor: x.chunk.anchor,\n"
        "           page_title: x.chunk.page_title, section_title: x.chunk.section_title,\n"
        "           text: x.chunk.text, score: +x.score.toFixed(4) };\n"
        "})));\n"
    )
    return run_node_json(script)


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_fixture_exercises_a_top_k_truncation_and_a_floor_exclusion() -> None:
    """Sanity check on the fixture itself (mirrors test_retrieval_sync.py's
    own fixture-sanity test): if this didn't actually exercise TOP_K
    truncation (c5 excluded despite clearing MIN_SCORE) the assertions below
    could pass trivially."""
    results = _run_retrieve_fallback()
    assert [r["id"] for r in results] == ["c1", "c2", "c3", "c4"], (
        "fixture assumption: the top 4 by score must be exactly c1-c4, in "
        "that order -- c5 (score 0.24, individually above MIN_SCORE 0.18) "
        "must be excluded for ranking 5th"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_retrieve_fallback_resolves_every_field_from_the_array_it_was_given() -> None:
    """The actual regression guard: every returned record's id AND every
    other field (url/anchor/page_title/section_title/text) must come from
    the SAME chunk record in the SAME `chunks` array `retrieveFallback` was
    called with -- never a position borrowed from a different array (the
    Task 24 bug class), and never a coherent-looking record stitched
    together from mismatched fields."""
    results = _run_retrieve_fallback()
    by_id = {c["id"]: c for c in CHUNKS}
    decoy_ids = {c["id"] for c in DECOY_CHUNKS}

    assert len(results) == 4
    for r in results:
        assert r["id"] not in decoy_ids, (
            f"result carries decoy id {r['id']!r} -- retrieveFallback read "
            "from a different array than the one it was handed"
        )
        expected = by_id.get(r["id"])
        assert expected is not None, f"result id {r['id']!r} is not in the input array at all"
        for field in ("url", "anchor", "page_title", "section_title", "text"):
            assert r[field] == expected[field], (
                f"{r['id']}: field {field!r} is {r[field]!r}, expected "
                f"{expected[field]!r} -- record fields do not all trace back "
                "to the same source chunk"
            )
