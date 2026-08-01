"""Gate text normalization — the Python mirror of scripts/chat-widget.js.

The widget is the source of truth: these cases were read off its NAME_TEST_RE /
NAME_STRIP_RE / BIO_STUB_RE / gateForm() implementation, not off the older
duplicate that used to live in test_gate.py.
"""

import pytest

from portfolio_rag.runtime import (
    GateDecision, Retrieval, _GateBundle, gate_form, load_runtime, strip_name,
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


def test_gate_meta_reports_margin_as_none_when_the_artifact_predates_it(rt) -> None:
    """The committed chat/data/gate_vectors.json (or fallback_vectors.json)
    on disk right now was built before task 20 added gate_margin, so it
    carries no such key. _GateBundle.margin (and gate_meta's "margin") must
    read as None -- an absent measurement, never a fabricated 0.0 that would
    look like a real (and misleadingly healthy) 0% margin. This is a direct,
    real-data proof of the exact "n/a" scenario the task report has to
    demonstrate; see also evaluation.format_margin, which renders this None
    as the literal string "n/a" for the printed table."""
    meta = rt.gate_meta
    assert "en" in meta, "the committed index ships an en gate"
    assert meta["en"]["margin"] is None, (
        "the real gate_vectors.json/fallback_vectors.json on disk predates "
        "gate_margin -- this must read as None, not 0 or 0.0"
    )


def test_gate_bundle_reports_a_real_margin_when_the_artifact_has_one() -> None:
    """Complement to the None case above: when a gate spec DOES carry
    gate_margin (any future rebuild under task 20), it must come through as
    the real float, not silently dropped by _GateBundle's None-default read."""
    bundle = _GateBundle("minilm", {
        "vectors": [[0.0] * 384],
        "gate_stat": "top",
        "gate_threshold": 0.25,
        "gate_margin": 0.0537,
    })
    assert bundle.margin == 0.0537


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
