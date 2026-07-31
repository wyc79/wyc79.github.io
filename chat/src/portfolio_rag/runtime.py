"""Python mirror of the widget's read path (scripts/chat-widget.js).

One implementation, imported by tests/test_gate.py and the evaluation harness.
The WIDGET is the source of truth for behaviour — it is what visitors actually
hit — so when this file and chat-widget.js disagree, chat-widget.js wins and
this file is wrong.

functions/tencent/index.py deliberately keeps its own inlined copy: the SCF
package must stay stdlib-only, so it cannot import this module.
"""

import json
import re
from dataclasses import dataclass

import numpy as np

from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder

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
# Like NAME_RE, use ASCII-only lookahead instead of \b to avoid Python's CJK
# classification issue.
BIO_STUB_RE = re.compile(
    r"^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)(?![a-zA-Z0-9_])"
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

    The language branch is determined by PRESENCE of CJK characters (not ratio),
    mirroring the widget's CJK_RE.test() presence check. This must not be
    changed to a majority vote, as that silently changes which questions get
    refused by the gate.
    """
    stripped = strip_name(question)
    if stripped is not None:
        return stripped
    # Presence check, not majority vote: any CJK character routes to Chinese branch.
    if CJK_RE.search(question):
        return NAME_RE.sub("王元辰", question)
    return question.replace("王元辰", "YC")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    url: str
    lang: str
    score: float


@dataclass(frozen=True)
class Retrieval:
    hits: tuple[Hit, ...]
    dropped_by_floor: int
    top_score: float


@dataclass(frozen=True)
class GateDecision:
    available: bool
    passed: bool
    value: float | None
    lang: str
    threshold: float | None
    reason: str | None = None


def _stat_value(scores: np.ndarray, kind: str) -> float:
    """Mirrors gate_calibration.stat_value and the widget's statValue()."""
    top = float(np.max(scores))
    if kind == "top":
        return top
    mean = float(np.mean(scores))
    if kind == "contrast":
        return top - mean
    if kind == "zscore":
        return (top - mean) / (float(np.std(scores)) + 1e-6)
    raise ValueError(f"unknown gate stat {kind!r}")


class _GateBundle:
    """One language's gate: an embedder, a chunk matrix, a stat and a threshold."""

    def __init__(self, preset_name: str, spec: dict) -> None:
        preset = MODEL_PRESETS[preset_name]
        self.embedder = OnnxEmbedder(
            settings.resolve_path(preset["dir"]),
            max_tokens=settings.embedding_max_tokens,
            query_prefix=spec.get("query_prefix", preset["query_prefix"]),
            passage_prefix="",
            pooling=spec.get("pooling", preset.get("pooling", "mean")),
        )
        self.matrix = np.array(spec["vectors"], dtype=np.float32)
        self.stat = spec.get("gate_stat", "top")
        self.threshold = float(spec.get("gate_threshold", OFFTOPIC_GATE))

    def judge(self, text: str) -> tuple[bool, float]:
        scores = self.matrix @ self.embedder.embed_query(text)
        value = _stat_value(scores, self.stat)
        return value >= self.threshold, round(value, 4)


class Runtime:
    """The widget's read path, reproduced locally.

    Reproduces the NORMAL path (server-side e5 embedding + MiniLM/bge gates),
    not the browser's degraded mode — that runs MiniLM over English-only
    fallback vectors and refuses CJK outright, which is a different system.
    """

    def __init__(self, index: dict, gates: dict, embedder) -> None:
        self._index = index
        self._gates = gates
        self._embedder = embedder
        self._matrix = (
            np.array([c["vector"] for c in index["chunks"]], dtype=np.float32)
            if index["chunks"]
            else np.empty((0, 0), dtype=np.float32)
        )

    @property
    def retrieval_available(self) -> bool:
        return self._embedder is not None

    @property
    def zh_gate_available(self) -> bool:
        return self._gates.get("zh") is not None

    @property
    def index_built_at(self) -> str:
        return self._index.get("built_at", "")

    @property
    def gate_meta(self) -> dict:
        return {
            lang: {"stat": bundle.stat, "threshold": bundle.threshold}
            for lang, bundle in self._gates.items()
            if bundle is not None
        }

    def chunk_text(self, chunk_id: str) -> str:
        """The indexed text of one chunk. Used to check whether a retrieved
        passage actually carries the fact the answer needs."""
        if not hasattr(self, "_by_id"):
            self._by_id = {c["id"]: c["text"] for c in self._index["chunks"]}
        return self._by_id.get(chunk_id, "")

    def retrieve(self, question: str, k: int = TOP_K) -> Retrieval:
        if self._embedder is None:
            raise RuntimeError(
                "retrieval unavailable: the e5 model directory "
                f"({MODEL_PRESETS['e5']['dir']}) is not present — it is gitignored, "
                "so a fresh clone must download it before evaluating"
            )
        scores = self._matrix @ self._embedder.embed_query(question)
        order = np.argsort(-scores)[:k]
        chunks = self._index["chunks"]
        top = [
            Hit(
                chunk_id=chunks[i]["id"],
                url=chunks[i]["url"],
                lang=chunks[i].get("lang", "en"),
                score=round(float(scores[i]), 4),
            )
            for i in order
        ]
        kept = [h for h in top if h.score >= MIN_SCORE]
        return Retrieval(
            hits=tuple(kept),
            dropped_by_floor=len(top) - len(kept),
            top_score=top[0].score if top else 0.0,
        )

    def gate(self, question: str) -> GateDecision:
        text = gate_form(question)
        if CJK_RE.search(text):
            bundle = self._gates.get("zh")
            if bundle is None:
                # No zh gate: let CJK through to the LLM prompt guard rather
                # than refusing every Chinese visitor. Mirrors index.py.
                return GateDecision(True, True, None, "zh", None, reason="cjk_bypass")
            lang = "zh"
        else:
            bundle, lang = self._gates.get("en"), "en"
            if bundle is None:
                return GateDecision(False, False, None, "en", None, reason="no_en_gate")
        passed, value = bundle.judge(text)
        return GateDecision(True, passed, value, lang, bundle.threshold)


def load_runtime() -> Runtime:
    """Load index + gate bundles + retrieval embedder, degrading explicitly."""
    index = json.loads(
        settings.resolve_path(settings.index_path).read_text(encoding="utf-8")
    )

    gates: dict = {"en": None, "zh": None}
    gate_path = settings.resolve_path(settings.gate_vectors_path)
    if gate_path.exists():
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
        for lang in ("en", "zh"):
            spec = payload.get(lang)
            if spec:
                gates[lang] = _GateBundle(spec["model_preset"], spec)
    else:
        # gate_vectors.json is gitignored. fallback_vectors.json IS committed
        # and carries the same MiniLM vectors + stat + threshold, so the en
        # gate always survives. There is no committed zh equivalent.
        fallback = settings.resolve_path(settings.fallback_vectors_path)
        if fallback.exists():
            gates["en"] = _GateBundle("minilm", json.loads(fallback.read_text(encoding="utf-8")))

    # Retrieval is hardcoded to e5, independent of settings.model_preset (which
    # defaults to "minilm" for local/default index builds — see
    # test_index_builder.py). data/index.json is built and committed with e5
    # (scripts/build_index.py --model e5); dot-producting its vectors against a
    # differently-trained embedder is a silent, wrong answer, not an explicit
    # unavailability, so this must never go through get_embedder()'s ambient,
    # process-global cache keyed on the current settings.model_preset.
    e5_preset = MODEL_PRESETS["e5"]
    model_dir = settings.resolve_path(e5_preset["dir"])
    embedder = (
        OnnxEmbedder.from_preset(e5_preset, model_dir, settings.embedding_max_tokens)
        if model_dir.is_dir()
        else None
    )
    return Runtime(index, gates, embedder)
