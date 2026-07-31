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
def index():
    return json.loads(settings.resolve_path(settings.index_path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rt():
    return load_runtime()


def gate_passes(question: str, rt) -> bool:
    return rt.gate(question).passed


def test_index_carries_a_calibrated_gate(index) -> None:
    assert index.get("gate_stat", "top") in {"top", "contrast", "zscore"}
    assert isinstance(index["gate_threshold"], float) and index["gate_threshold"] > 0


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
    """The zh starters route to the bge-zh gate. Skipped when gate_vectors.json
    is absent (gitignored), because there is then no zh gate to test."""
    if not rt.zh_gate_available:
        pytest.skip("no zh gate: data/gate_vectors.json absent")
    failures = [
        f"[{rid}] {starter}"
        for rid, role in ROLES.items()
        for starter in role.get("zh", {}).get("starters", [])
        if not gate_passes(starter, rt)
    ]
    assert not failures, f"zh starters refused by the gate: {failures}"
