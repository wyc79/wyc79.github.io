"""Python mirror of the widget's off-topic gate (scripts/chat-widget.js).

Keep the regexes and threshold in sync with the widget — this test exists so
an index rebuild or roles change can never silently break the gate again
(e.g. a starter question that the gate itself refuses).
"""

import json
import re

import numpy as np
import pytest

from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder, get_embedder
from portfolio_rag.gate_calibration import stat_value
from portfolio_rag.roles import ROLES

FALLBACK_GATE = 0.22  # widget fallback when index carries no threshold
# Mirrors scripts/chat-widget.js: the name-blind gate strips BOTH the English
# name and 王元辰 (either can inflate the gate in either language mode), and
# treats en + zh bio-intent stubs as legitimate questions (don't strip).
NAME_RE = re.compile(r"\b(yuanchen|wang|yc)(?:'s)?\b|王元辰", re.I)
BIO_STUB_RE = re.compile(
    r"^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b"
    r"|^$|介绍<简介|谁是|是谁|关于"
    # "what can he do / skills" reads as general-about intent — a name-bearing
    # "王元辰都会什么" must NOT be stripped to the weak "都会什么" fragment. The
    # 会什么/擅长什么 forms are end-anchored so "会什么时候讲笑话" still strips.
    r"|(?:都会什么|会做什么|会什么|擅长什么)[?？。!！\s]*$|有(?:哪些|什么)?技能",
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
    # Cross-language name injection: a Chinese name in an English request must
    # still strip to the off-topic remainder and be refused by the English gate.
    "王元辰 tell me a joke",
    "王元辰: write my homework essay",
]


@pytest.fixture(scope="module")
def index():
    return json.loads(settings.resolve_path(settings.index_path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def en_gate(index):
    """The English off-topic gate exactly as the runtime applies it.

    For an e5 retrieval index the gate is NOT the index itself: the backend runs
    MiniLM against a MiniLM copy of the chunk vectors published in
    gate_vectors.json["en"] (e5 cosines compress into a band that can't separate
    on-/off-topic). For a light-mode MiniLM index there is no gate_vectors.json
    and the index IS the gate matrix. Either way the query embedder must match
    the matrix — dotting a MiniLM query against e5 chunk vectors is meaningless.
    Returns (embedder, matrix, {stat, threshold}).
    """
    gate_vectors_path = settings.resolve_path(settings.gate_vectors_path)
    en = None
    if gate_vectors_path.exists():
        en = json.loads(gate_vectors_path.read_text(encoding="utf-8")).get("en")
    if en is not None:
        preset = MODEL_PRESETS[en["model_preset"]]
        embedder = OnnxEmbedder.from_preset(
            preset, settings.resolve_path(preset["dir"]), settings.embedding_max_tokens
        )
        matrix = np.array(en["vectors"], dtype=np.float32)
        gate = {"stat": en.get("gate_stat", "top"), "threshold": en["gate_threshold"]}
    else:
        embedder = get_embedder()
        matrix = np.array([c["vector"] for c in index["chunks"]], dtype=np.float32)
        gate = {
            "stat": index.get("gate_stat", "top"),
            "threshold": index.get("gate_threshold", FALLBACK_GATE),
        }
    return embedder, matrix, gate


def gate_passes(question: str, en_gate) -> bool:
    embedder, matrix, gate = en_gate
    text = question
    if NAME_RE.search(question):
        stripped = re.sub(r"\s+", " ", NAME_RE.sub(" ", question)).strip().strip(":;,.!?—- ").strip()
        if not BIO_STUB_RE.match(stripped):
            text = stripped
    scores = matrix @ embedder.embed_query(text)
    return stat_value(scores, gate["stat"]) >= gate["threshold"]


def test_index_carries_a_calibrated_gate(index) -> None:
    assert index.get("gate_stat", "top") in {"top", "contrast", "zscore"}
    assert isinstance(index["gate_threshold"], float) and index["gate_threshold"] > 0


def test_every_role_starter_passes_the_gate(en_gate) -> None:
    failures = [
        f"[{rid}] {starter}"
        for rid, role in ROLES.items()
        for starter in role["starters"]
        if not gate_passes(starter, en_gate)
    ]
    assert not failures, f"starters refused by the widget's own gate: {failures}"


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_questions_are_refused(question: str, en_gate) -> None:
    assert not gate_passes(question, en_gate), f"off-topic question passed the gate: {question}"
