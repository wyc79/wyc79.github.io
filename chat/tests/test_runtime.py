# -*- coding: utf-8 -*-
"""Gate text normalization — the Python mirror of scripts/chat-widget.js.

The widget is the source of truth: these cases were read off its NAME_TEST_RE /
NAME_STRIP_RE / BIO_STUB_RE / gateForm() implementation, not off the older
duplicate that used to live in test_gate.py.
"""

import pytest

from portfolio_rag.runtime import gate_form, strip_name


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
        # Kept + English question: normalize 王元辰 to "YC" for the English gate.
        ("who is 王元辰", "who is YC"),
        ("who is Yuanchen Wang", "who is Yuanchen Wang"),
        # No name at all: identity.
        ("what engine work has he done", "what engine work has he done"),
    ],
)
def test_gate_form(question: str, expected: str) -> None:
    assert gate_form(question) == expected
