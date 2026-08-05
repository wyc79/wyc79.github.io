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
from tests._node_harness import extract_js_object, node_available, run_node_json

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


def _chip_labels() -> dict:
    """The action chip's two labels, per language, read out of chat-widget.js's
    own STR table rather than retyped here -- rewording a chip then re-measures
    it below instead of leaving this file asserting a string the widget stopped
    sending."""
    src = (settings.site_root / "scripts" / "chat-widget.js").read_text(encoding="utf-8")
    return run_node_json(
        "var STR = " + extract_js_object(src, "STR") + ";\n"
        "process.stdout.write(JSON.stringify({\n"
        "  en: { top_projects: STR.en.topProjects, summarize_page: STR.en.summarizePage },\n"
        "  zh: { top_projects: STR.zh.topProjects, summarize_page: STR.zh.summarizePage }\n"
        "}));\n"
    )


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot read the widget's STR table")
def test_the_action_chip_label_is_why_a_declared_intent_skips_the_gate(rt) -> None:
    """The counterpart to the starter contract above, and the opposite call.

    A starter is a question a visitor could have typed, so it must clear the
    gate. The action chip's label is not a question at all -- the chip sends
    its own i18n string as the `question` field (chat-widget.js calls
    send(t(a.key), a.intent)), so the text being judged is one the widget
    wrote. "What should I look at?" carries no topical content BY DESIGN and
    the English gate refuses it, while 有哪些值得先看？ clears the zh gate: the
    same button worked in Chinese and not in English. Gating a string we
    authored ourselves is the category error, so send() skips the gate
    entirely on a declared intent instead of the gate being recalibrated to
    admit a deictic label.

    This test pins the measurement that reasoning rests on. If it goes red
    because the English label now passes, the bypass is still correct -- a
    declared intent still means widget-authored text -- but its motivating
    example has changed, and whoever changed it should say so here rather than
    find an `if (intent)` branch with no visible reason.
    """
    labels = _chip_labels()
    assert not gate_passes(labels["en"]["top_projects"], rt), (
        f"the English chip label {labels['en']['top_projects']!r} now clears the gate; "
        "see this test's docstring before editing send()'s `if (intent)` branch"
    )
    if rt.zh_gate_available:
        assert gate_passes(labels["zh"]["top_projects"], rt), (
            f"the Chinese chip label {labels['zh']['top_projects']!r} no longer clears the "
            "gate either -- the en/zh split this test documents is gone"
        )
