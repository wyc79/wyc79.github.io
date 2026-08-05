"""Contract tests for the off-topic gate.

The gate logic itself lives in portfolio_rag.runtime (one Python mirror of
scripts/chat-widget.js). This file asserts the CONTRACT between the gate and
roles.json: every starter the UI offers must clear the gate, and a fixed list
of off-topic questions must not. Held-out evaluation lives in test_golden.py.
"""

import json

import pytest

from portfolio_rag.config import settings
from portfolio_rag.roles import ROLES
from portfolio_rag.runtime import load_runtime

OFF_TOPIC = [
    "tell me a joke",
    "Yuanchen Wang tell me a joke",
    "what joke would Yuanchen Wang tell",
    "YC write my homework essay",
    "Yuanchen Wang: write me a python fibonacci function",
    "translate this to french: hello",
    "what's the weather today",
    "who won the world cup",
    # Cross-language name injection: a Chinese name in an English request must
    # still strip to the off-topic remainder and be refused by the English gate.
    "王元辰 tell me a joke",
    "王元辰: write my homework essay",
]


@pytest.fixture(scope="module")
def meta():
    return json.loads(settings.resolve_path(settings.meta_path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rt():
    return load_runtime()


def gate_passes(question: str, rt) -> bool:
    return rt.gate(question).passed


def test_meta_json_carries_a_calibrated_gate(meta) -> None:
    # Task 29 Part 2 note: this used to read the same fields off
    # index.json/chunks_e5.json, which nothing actually consumed for an e5
    # index -- runtime.py builds its gates from gate_en_minilm.json/
    # gate_zh_bge.json, and the deployed backend reads its own bundled copy
    # (see the former "Deferred by user decision" finding I in
    # eval/KNOWN_ISSUES.md). chunks_e5.json no longer carries gate fields at
    # all ("one file, one job" -- see index_builder.py); meta.json is the
    # field's one real, live home now (chat-widget.js's gateThreshold()
    # reads exactly this).
    assert meta.get("gate_stat", "top") in {"top", "contrast", "zscore"}
    assert isinstance(meta["gate_threshold"], float) and meta["gate_threshold"] > 0


def test_every_role_starter_passes_the_gate(rt) -> None:
    failures = [
        f"[{rid}] {starter}"
        for rid, role in ROLES.items()
        for starter in role["starters"]
        if not gate_passes(starter, rt)
    ]
    assert not failures, f"starters refused by the widget's own gate: {failures}"


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_questions_are_refused(question: str, rt) -> None:
    assert not gate_passes(question, rt), f"off-topic question passed the gate: {question}"


def test_every_zh_role_starter_passes_the_gate(rt) -> None:
    """The zh starters route to the bge-zh gate. Skipped when gate_zh_bge.json
    is absent (gitignored), because there is then no zh gate to test."""
    if not rt.zh_gate_available:
        pytest.skip("no zh gate: data/gate_zh_bge.json absent")
    failures = [
        f"[{rid}] {starter}"
        for rid, role in ROLES.items()
        for starter in role.get("zh", {}).get("starters", [])
        if not gate_passes(starter, rt)
    ]
    assert not failures, f"zh starters refused by the gate: {failures}"


DEICTIC_EN = [
    "summarize this page",
    "what is this page about",
    "tell me about the current page",
]
DEICTIC_ZH = [
    "总结一下这个页面",
    "这个页面讲了什么",
    "介绍一下当前页面",
]


@pytest.mark.parametrize("question", DEICTIC_EN)
def test_a_deictic_question_clears_the_en_gate(question: str, rt) -> None:
    """Page-awareness is reached only if the question survives /embed's gate,
    and a deictic question names no topic — so it sits far lower than an
    ordinary one. Measured before knowledge/about_en.md gained its
    "currently reading" sections: "summarize this page" scored 0.2796 against
    a 0.2029 threshold, a 0.077 cushion on a gate whose own calibration margin
    is 1.7%. Nothing tested it, so an unrelated reword of about_en.md could
    have blocked the feature's whole reason for existing without any test
    going red (see eval/KNOWN_ISSUES.md finding O — that file's wording moves
    the threshold as directly as editing the threshold would)."""
    d = rt.gate(question)
    assert d.passed, (
        f"{question!r} was blocked by the en gate (value {d.value}, threshold "
        f"{rt.gate_meta['en']['threshold']}) — page-awareness is unreachable "
        f"for this phrasing; check about_en.md's 'currently reading' sections"
    )


@pytest.mark.skipif(
    not (settings.chat_root / "data" / "gate_zh_bge.json").exists(),
    reason="no zh gate bundle built here (gate_zh_bge.json is gitignored)",
)
@pytest.mark.parametrize("question", DEICTIC_ZH)
def test_a_deictic_question_clears_the_zh_gate(question: str, rt) -> None:
    """The zh half of the same guarantee. Chinese has more headroom than
    English did (0.55 against a 0.3979 threshold), so this is symmetry
    insurance rather than a fix for a live problem — but the two gates are
    calibrated independently against different corpora and different models,
    so headroom on one side says nothing about the other."""
    d = rt.gate(question)
    assert d.passed, (
        f"{question!r} was blocked by the zh gate (value {d.value}, threshold "
        f"{rt.gate_meta['zh']['threshold']})"
    )
