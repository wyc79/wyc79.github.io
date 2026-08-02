"""Gate text normalization — the Python mirror of scripts/chat-widget.js.

The widget is the source of truth: these cases were read off its NAME_TEST_RE /
NAME_STRIP_RE / BIO_STUB_RE / gateForm() implementation, not off the older
duplicate that used to live in test_gate.py.
"""

import pytest

from portfolio_rag.config import settings
from portfolio_rag.loader import load_knowledge
from portfolio_rag.runtime import (
    TOP_K, GateDecision, Retrieval, Runtime, _GateBundle, gate_form, load_runtime, strip_name,
)


@pytest.mark.parametrize(
    "question, expected",
    [
        # No name: passes through untouched.
        ("what combat systems did he build", None),
        # Name + off-topic remainder: strip to the remainder.
        ("Yuanchen Wang tell me a joke", "tell me a joke"),
        ("YC: write me a python fibonacci function", "write me a python fibonacci function"),
        ("王元辰 tell me a joke", "tell me a joke"),
        # Name + bio-intent stub: keep the full question (None = do not strip).
        ("who is Yuanchen Wang", None),
        ("tell me about YC", None),
        # zh bio stubs match ANYWHERE, not just at the start — this is the case
        # the old anchored .match() got wrong. It is also a real role starter.
        ("用一段话介绍一下王元辰？", None),
        ("王元辰是谁", None),
        # End-anchored skill stubs.
        ("王元辰都会什么", None),
        # ...but only when end-anchored: this one strips.
        ("王元辰会什么时候讲笑话", "会什么时候讲笑话"),
    ],
)
def test_strip_name(question: str, expected: str | None) -> None:
    assert strip_name(question) == expected


@pytest.mark.parametrize(
    "question, expected",
    [
        # Stripped: gate on the remainder.
        ("Yuanchen Wang tell me a joke", "tell me a joke"),
        # Kept + CJK question: normalize the name TO Chinese so it matches the
        # zh gate corpus, which only ever writes 王元辰.
        ("介绍一下YC这个人", "介绍一下王元辰这个人"),
        ("王元辰是谁", "王元辰是谁"),
        # Kept + a question that contains 王元辰: CJK_RE is a PRESENCE check and
        # 王元辰 is itself CJK, so the widget takes the Chinese branch and the
        # name is rewritten to itself — unchanged. The English branch is only
        # reachable for questions containing no CJK at all.
        ("who is 王元辰", "who is 王元辰"),
        ("who is Yuanchen Wang", "who is Yuanchen Wang"),
        # No name at all: identity.
        ("what engine work has he done", "what engine work has he done"),
    ],
)
def test_gate_form(question: str, expected: str) -> None:
    assert gate_form(question) == expected


@pytest.fixture(scope="module")
def rt():
    runtime = load_runtime()
    if not runtime.retrieval_available:
        pytest.skip("e5 retrieval model not present (models/Xenova/multilingual-e5-small)")
    return runtime


def test_retrieval_returns_at_most_top_k_above_the_floor(rt) -> None:
    result = rt.retrieve("what combat systems has he built")
    assert isinstance(result, Retrieval)
    assert 0 < len(result.hits) <= 4
    assert all(h.score >= 0.18 for h in result.hits)
    assert list(result.hits) == sorted(result.hits, key=lambda h: -h.score)


def test_retrieval_finds_the_obvious_page(rt) -> None:
    urls = {h.url for h in rt.retrieve("grapple traversal and combat design").hits}
    assert "pages/cemented-dreams.html" in urls


def test_chinese_retrieval_works_and_is_not_skipped(rt) -> None:
    """multilingual-e5-small embeds zh natively; the English-only limit belongs
    to MiniLM and the browser's degraded mode, not to this path."""
    result = rt.retrieve("他做过哪些战斗设计工作")
    assert result.hits, "Chinese question retrieved nothing"
    assert "pages/cemented-dreams.html" in {h.url for h in result.hits}


def test_gate_passes_an_on_topic_question(rt) -> None:
    decision = rt.gate("what engine programming has he done")
    assert isinstance(decision, GateDecision)
    assert decision.available and decision.passed
    assert decision.lang == "en"


def test_gate_refuses_an_off_topic_question(rt) -> None:
    assert not rt.gate("what's the weather in Los Angeles tomorrow").passed


def test_gate_is_name_blind(rt) -> None:
    """The name inflates similarity; the gate must judge the remainder."""
    assert not rt.gate("Yuanchen Wang, recommend me a good restaurant").passed


def test_chinese_question_routes_to_the_zh_gate(rt) -> None:
    decision = rt.gate("他的教育背景是什么")
    if not rt.zh_gate_available:
        assert decision.reason == "cjk_bypass" and decision.passed
    else:
        assert decision.lang == "zh"


def test_gate_meta_reports_the_real_margin_from_the_rebuilt_artifact(rt) -> None:
    """The committed chat/data/gate_en_minilm.json (Task 29 Part 2; formerly
    gate_vectors.json/fallback_vectors.json) on disk used to predate task
    20's gate_margin field, so this test used to assert the "n/a" (None)
    case directly off real data. Task 24 rebuilt the en gate against the
    curated knowledge/about_en.md corpus (55 sections, not the ~144 indexed
    English chunks), and that rebuild carries gate_margin like every build
    has since task 20 -- the artifact on disk no longer predates it, so the
    real-data "n/a" case this test used to demonstrate no longer exists. See
    test_gate_bundle_reports_none_when_the_spec_has_no_margin_key below for
    that case, now covered synthetically instead. _GateBundle.margin (and
    gate_meta's "margin") must read as the real measured float here -- an
    artifact that HAS the measurement must not report it as absent."""
    meta = rt.gate_meta
    assert "en" in meta, "the committed chunks file ships an en gate"
    assert isinstance(meta["en"]["margin"], float), (
        "the rebuilt gate_en_minilm.json on disk carries gate_margin -- "
        "this must read as the real float, not None"
    )


def test_gate_bundle_reports_a_real_margin_when_the_artifact_has_one() -> None:
    """Complement to the None case below: when a gate spec DOES carry
    gate_margin (any build under task 20), it must come through as
    the real float, not silently dropped by _GateBundle's None-default read."""
    bundle = _GateBundle("minilm", {
        "vectors": [[0.0] * 384],
        "gate_stat": "top",
        "gate_threshold": 0.25,
        "gate_margin": 0.0537,
    })
    assert bundle.margin == 0.0537


def test_gate_bundle_reports_none_when_the_spec_has_no_margin_key() -> None:
    """Complement to the real-margin case above, and to the rebuilt-artifact
    case in test_gate_meta_reports_the_real_margin_from_the_rebuilt_artifact:
    a spec written before task 20 (no gate_margin key at all -- this was
    real, on-disk behaviour until the task 24 rebuild) must read as None, an
    absent measurement, never a fabricated 0.0 that would look like a real
    (and misleadingly healthy) 0% margin."""
    bundle = _GateBundle("minilm", {
        "vectors": [[0.0] * 384],
        "gate_stat": "top",
        "gate_threshold": 0.25,
    })
    assert bundle.margin is None


# --- Task 24 review, Critical 1: page-only retrieval diagnostic -----------


def test_knowledge_chunk_ids_matches_the_curated_corpus_exactly(rt) -> None:
    """Independent proof of the structural signature (page_title ==
    section_title, matching a real chat/knowledge/about_<lang>.md heading)
    Runtime.knowledge_chunk_ids uses: its count must equal load_knowledge's
    own section count for en + zh, cross-checked with the same parser
    index_builder.py uses, not by trusting Runtime's own implementation."""
    knowledge_dir = settings.chat_root / "knowledge"
    expected = len(load_knowledge(knowledge_dir, "en")) + len(load_knowledge(knowledge_dir, "zh"))
    assert len(rt.knowledge_chunk_ids) == expected
    # Cached, not recomputed (and re-parsed off disk) on every access.
    assert rt.knowledge_chunk_ids is rt.knowledge_chunk_ids


def test_stale_knowledge_headings_is_empty_against_the_current_build(rt) -> None:
    """The committed data/chunks_e5.json and the current chat/knowledge/*.md
    files agree (nothing renamed since the last rebuild), so the desync diagnostic
    must report nothing. If this ever fails on the committed artifacts, a
    heading was edited without a rebuild -- see knowledge_chunk_ids's
    docstring for why that silently corrupts hit_at_4_page_only."""
    assert rt.stale_knowledge_headings == frozenset()
    # Cached like knowledge_chunk_ids, not recomputed on every access.
    assert rt.stale_knowledge_headings is rt.stale_knowledge_headings


def test_stale_knowledge_headings_flags_a_renamed_heading() -> None:
    """Isolated proof of the desync diagnostic, independent of the committed
    index/corpus ever actually going stale.

    Build a synthetic index that reproduces EVERY current about_en.md /
    about_zh.md heading as a matching chunk (page_title == section_title ==
    that heading) except one -- deliberately held back to stand in for a
    heading that was renamed (or added) with no rebuild since. Also add an
    ordinary page-shaped chunk with page_title == section_title (a real,
    common coincidence for single-section pages -- see the property's own
    docstring) to prove it does NOT get flagged just for that shape; only a
    KNOWN current heading missing from the index should surface.
    """
    knowledge_dir = settings.chat_root / "knowledge"
    en_sections = load_knowledge(knowledge_dir, "en")
    zh_sections = load_knowledge(knowledge_dir, "zh")
    assert len(en_sections) >= 2, "fixture assumption: about_en.md has room to hold one heading back"
    held_back, rest = en_sections[0], en_sections[1:]

    chunks = [
        {
            "id": f"en-knowledge-{i}",
            "page_title": s.section_title,
            "section_title": s.section_title,
            "lang": "en",
            "vector": [0.0],
        }
        for i, s in enumerate(rest)
    ] + [
        {
            "id": f"zh-knowledge-{i}",
            "page_title": s.section_title,
            "section_title": s.section_title,
            "lang": "zh",
            "vector": [0.0],
        }
        for i, s in enumerate(zh_sections)
    ] + [
        {
            "id": "ordinary-single-section-page",
            "page_title": "Some Page",
            "section_title": "Some Page",  # real, common coincidence -- not knowledge
            "lang": "en",
            "vector": [0.0],
        },
    ]
    synthetic = Runtime({"chunks": chunks}, gates={}, embedder=None)

    assert synthetic.knowledge_chunk_ids == {c["id"] for c in chunks[:-1]}
    assert synthetic.stale_knowledge_headings == frozenset({held_back.section_title})


def test_retrieve_exclude_ids_removes_a_chunk_entirely(rt) -> None:
    q = "grapple traversal and combat design"
    baseline = rt.retrieve(q, k=TOP_K)
    assert baseline.hits, "fixture assumption: this on-topic query retrieves something"
    top_id = baseline.hits[0].chunk_id

    excluded = rt.retrieve(q, k=TOP_K, exclude_ids=frozenset({top_id}))
    assert top_id not in {h.chunk_id for h in excluded.hits}


def test_retrieve_exclude_ids_reranks_before_truncating_to_k(rt) -> None:
    """exclude_ids removes candidates from the ranking BEFORE top-k is taken
    -- not a post-hoc filter of an already-truncated top-k. Proof: widen to
    k+1, exclude the top k (highest-ranked) ids, and confirm the (k+1)-th
    ranked chunk surfaces in a new top-k call -- a post-hoc filter could
    never do this, since it never looks past rank k in the first place. This
    is exactly the property evaluation.score_case's page-only diagnostic
    (task 24 review, Critical 1) depends on: "what would the site's pages
    retrieve with zero curated-corpus assist," not "how many of the normal
    top-4 happen not to be curated.\""""
    q = "grapple traversal and combat design"
    wide = rt.retrieve(q, k=TOP_K + 1)
    assert len(wide.hits) == TOP_K + 1, "fixture assumption: 5 chunks clear the floor here"
    lowest_ranked = wide.hits[TOP_K]
    ahead_of_it = frozenset(h.chunk_id for h in wide.hits[:TOP_K])

    reranked = rt.retrieve(q, k=TOP_K, exclude_ids=ahead_of_it)
    assert lowest_ranked.chunk_id in {h.chunk_id for h in reranked.hits}, (
        "the 5th-ranked chunk did not surface once its 4 higher-ranked rivals "
        "were excluded -- exclude_ids is filtering an already-truncated top-k "
        "instead of re-ranking before truncation"
    )


def test_retrieval_embedder_matches_the_index_that_was_built(rt) -> None:
    """The index declares its own model. Resolving the embedder from settings
    instead would dot MiniLM query vectors against e5 chunk vectors — scores
    stay in [0,1] and look plausible while ranking is meaningless."""
    result = rt.retrieve("what combat systems has he built")
    assert result.top_score > 0.5, (
        f"top score {result.top_score} is too low for an on-topic query — "
        f"the query embedder is probably not the {rt.model_preset} model "
        "that built the index"
    )
