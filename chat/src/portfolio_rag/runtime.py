"""Python mirror of the widget's read path (scripts/chat-widget.js).

One implementation, imported by tests/test_gate.py and the evaluation harness.
The WIDGET is the source of truth for behaviour — it is what visitors actually
hit — so when this file and chat-widget.js disagree, chat-widget.js wins and
this file is wrong.

functions/tencent/index.py deliberately keeps its own inlined copy: the SCF
package must stay stdlib-only, so it cannot import this module.
"""

import re

# Mirrors chat-widget.js TOP_K / MIN_SCORE / OFFTOPIC_GATE.
TOP_K = 4
MIN_SCORE = 0.18  # per-source display floor; below it a chunk never becomes context
OFFTOPIC_GATE = 0.22  # fallback only, for indexes built before gate calibration

# Name-dropping inflates similarity, so a question mentioning YC is gated on the
# question WITHOUT the name — unless what remains is a bio-intent stub, which is
# a legitimate question about him. Mirrors NAME_TEST_RE / NAME_STRIP_RE (one
# pattern suffices here: re.sub replaces every occurrence by default).
# In Python, CJK characters are \w, so we use negative lookaround instead of \b
# to allow matching Latin names surrounded by CJK (e.g., "介绍YC这个人").
NAME_RE = re.compile(r"(?<![a-zA-Z0-9_])(yuanchen|wang|yc)(?:'s)?(?![a-zA-Z0-9_])|王元辰", re.I)

# English stubs anchor at the start; the zh stubs match ANYWHERE, so
# "用一段话介绍一下" and "…是谁" survive. Used with re.search to mirror the
# widget's .test() — an anchored .match() would strip zh bio questions.
BIO_STUB_RE = re.compile(
    r"^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b"
    r"|^$|介绍|简介|谁是|是谁|关于"
    r"|(?:都会什么|会做什么|会什么|擅长什么)[?？。!！\s]*$"
    r"|有(?:哪些|什么)?技能",
    re.I,
)

# Hiragana/katakana + CJK unified + compatibility ideographs. Mirrors the
# widget's CJK_RE and index.py's CJK_RE. Written with EXPLICIT \u escapes,
# NOT literal characters: the compatibility-range start char is a homoglyph
# of the unified-range one and is easy to mistype invisibly (chat-widget.js:66
# carries the same warning). The string is NOT raw, so Python reads them.
CJK_RE = re.compile("[\\u3040-\\u30ff\\u3400-\\u9fff\\uf900-\\ufaff]")

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s:;,.!?—-]+|[\s:;,.!?—-]+$")


def strip_name(question: str) -> str | None:
    """The name-stripped remainder, or None when the name should be kept.

    None means "gate on the whole question": either it never mentioned YC, or
    what remained after removing the name was a bio-intent stub.
    """
    if not NAME_RE.search(question):
        return None
    remainder = _EDGE_PUNCT_RE.sub("", _WS_RE.sub(" ", NAME_RE.sub(" ", question)).strip())
    return None if BIO_STUB_RE.search(remainder) else remainder


def gate_form(question: str) -> str:
    """Exactly the text chat-widget.js sends as gate_text.

    When the name was stripped, gate on the remainder. When it was KEPT, the
    name is normalized to the gate's own language — each gate corpus is
    single-language, so a Chinese question saying "YC" (or an English one
    saying 王元辰) would otherwise miss it. Retrieval always uses the original.
    """
    stripped = strip_name(question)
    if stripped is not None:
        return stripped
    # Detect language by checking if majority of content is CJK.
    cjk_chars = len(CJK_RE.findall(question))
    is_cjk_question = (cjk_chars * 2 > len(question))  # > 50% CJK
    if is_cjk_question:
        return NAME_RE.sub("王元辰", question)
    return question.replace("王元辰", "YC")
