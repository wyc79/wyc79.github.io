"""Python mirror of the widget's off-topic gate (scripts/chat-widget.js).

Keep the regexes and threshold in sync with the widget — this test exists so
an index rebuild or roles change can never silently break the gate again
(e.g. a starter question that the gate itself refuses).
"""

import json
import re

import numpy as np
import pytest

from portfolio_rag.config import settings
from portfolio_rag.embedder import get_embedder
from portfolio_rag.gate_calibration import stat_value
from portfolio_rag.roles import ROLES

FALLBACK_GATE = 0.22  # widget fallback when index carries no threshold
NAME_RE = re.compile(r"\b(yuanchen|wang|yc)(?:'s)?\b", re.I)
BIO_STUB_RE = re.compile(
    r"^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b|^$",
    re.I,
)

OFF_TOPIC = [
    "tell me a joke",
    "Yuanchen Wang tell me a joke",
    "what joke would Yuanchen Wang tell",
    "YC write my homework essay",
    "Yuanchen Wang: write me a python fibonacci function",
    "translate this to french: hello",
    "what's the weather today",
    "who won the world cup",
]


@pytest.fixture(scope="module")
def index():
    return json.loads(settings.resolve_path(settings.index_path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrix(index):
    return np.array([c["vector"] for c in index["chunks"]], dtype=np.float32)


@pytest.fixture(scope="module")
def gate(index):
    return {
        "stat": index.get("gate_stat", "top"),
        "threshold": index.get("gate_threshold", FALLBACK_GATE),
    }


def gate_passes(question: str, matrix: np.ndarray, gate: dict) -> bool:
    emb = get_embedder()
    text = question
    if NAME_RE.search(question):
        stripped = re.sub(r"\s+", " ", NAME_RE.sub(" ", question)).strip().strip(":;,.!?—- ").strip()
        if not BIO_STUB_RE.match(stripped):
            text = stripped
    scores = matrix @ emb.embed_query(text)
    return stat_value(scores, gate["stat"]) >= gate["threshold"]


def test_index_carries_a_calibrated_gate(index) -> None:
    assert index.get("gate_stat", "top") in {"top", "contrast", "zscore"}
    assert isinstance(index["gate_threshold"], float) and index["gate_threshold"] > 0


def test_every_role_starter_passes_the_gate(matrix, gate) -> None:
    failures = [
        f"[{rid}] {starter}"
        for rid, role in ROLES.items()
        for starter in role["starters"]
        if not gate_passes(starter, matrix, gate)
    ]
    assert not failures, f"starters refused by the widget's own gate: {failures}"


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_questions_are_refused(question: str, matrix, gate) -> None:
    assert not gate_passes(question, matrix, gate), f"off-topic question passed the gate: {question}"
