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
from portfolio_rag.loader import load_knowledge

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
        # None (not 0.0) on any artifact built before task 20 -- gate_margin
        # is a reported diagnostic, and an absent measurement must render as
        # unavailable, never as an implied-healthy 0% margin. Only build_index
        # (index_builder.py) ever writes this key.
        margin = spec.get("gate_margin")
        self.margin = None if margin is None else float(margin)

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

    def __init__(self, index: dict, gates: dict, embedder, model_dir=None) -> None:
        self._index = index
        self._gates = gates
        self._embedder = embedder
        self._model_dir = model_dir
        self._matrix = (
            np.array([c["vector"] for c in index["chunks"]], dtype=np.float32)
            if index["chunks"]
            else np.empty((0, 0), dtype=np.float32)
        )
        self._id_to_row = {c["id"]: i for i, c in enumerate(index["chunks"])}
        self._knowledge_chunk_ids: frozenset[str] | None = None  # lazy, see property
        self._stale_knowledge_headings: frozenset[str] | None = None  # lazy, see property

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
    def model_preset(self) -> str:
        """Which embedding model built the index this runtime is serving."""
        return self._index.get("model_preset", "")

    @property
    def gate_meta(self) -> dict:
        return {
            lang: {"stat": bundle.stat, "threshold": bundle.threshold, "margin": bundle.margin}
            for lang, bundle in self._gates.items()
            if bundle is not None
        }

    def chunk_text(self, chunk_id: str) -> str:
        """The indexed text of one chunk. Used to check whether a retrieved
        passage actually carries the fact the answer needs."""
        if not hasattr(self, "_by_id"):
            self._by_id = {c["id"]: c["text"] for c in self._index["chunks"]}
        return self._by_id.get(chunk_id, "")

    @property
    def knowledge_chunk_ids(self) -> frozenset[str]:
        """Ids of index chunks sourced from chat/knowledge/about_<lang>.md's
        curated sections, as opposed to the site's own pages.

        Task 24 review, Critical 1: the headline hit@4 gain from that task's
        rebuild (66/96 -> 90/96) is largely the corpus retrieving curated text
        that was authored by reading the very pages golden.jsonl names in
        expected_urls -- an answer key partly scoring against itself. This is
        the "means that survives re-chunking" the review asked for, used by
        evaluation.score_case to report a page-only hit@4 alongside the full
        number (see Runtime.retrieve's exclude_ids).

        Structural signature, not position or a hardcoded count:
        load_knowledge() (loader.py) sets a curated section's page_title AND
        section_title to the exact `## heading` text, verbatim, which real
        page sections essentially never do by coincidence (a real page's own
        title/heading pair matching one of these long, distinctive authored
        headings character-for-character is not a realistic collision this
        project has hit). chunk_text may split one curated section into
        several chunk dicts; every piece keeps that same page_title/
        section_title pair, so each piece is still correctly identified after
        re-chunking -- PROVIDED the index chunks and the on-disk
        about_<lang>.md files still agree on heading text.

        They do not always. The CHUNKS are frozen at build time: page_title/
        section_title is whatever heading text was on disk when index.json
        was last written. The heading SET this is compared against, by
        contrast, is read from the CURRENT about_<lang>.md files on every
        call (via load_knowledge, the same parser index_builder.py uses) --
        not a snapshot from that same build. A heading rename or removal with
        no rebuild desyncs the two: the stale chunk still carries the OLD
        heading text, which is no longer in the live set, so it silently
        drops out of this frozenset and leaks into
        hit_at_4_page_only's "page-only" candidate pool as if it were real
        page content. Demonstrated directly (final whole-branch review,
        follow-up to Critical 1): renaming 5 about_en.md headings with no
        rebuild moved knowledge_chunk_ids from 108 to 103 and
        hit_at_4_page_only from 59/96 to 56/96 -- a 3-case, unpredictable-
        direction move baked straight into data/eval_baseline.json if a run
        happens to update it in that state.

        So: this check is correct only against the SAME build the index
        chunks came from. A rebuild is required after any about_<lang>.md
        heading edit to keep it accurate -- it is not merely nice-to-have.
        `stale_knowledge_headings` (below) is a best-effort, non-blocking
        diagnostic for exactly this desync.
        """
        self._compute_knowledge_signature()
        return self._knowledge_chunk_ids

    @property
    def stale_knowledge_headings(self) -> frozenset[str]:
        """Headings present in the CURRENT chat/knowledge/about_<lang>.md
        files that have no matching chunk in the index (no index chunk with
        page_title == section_title == that exact heading text, in that
        heading's language).

        Deliberately checked in this direction -- from the live corpus
        outward, not by scanning the index for orphaned chunks -- because
        page_title == section_title is NOT itself a rare coincidence for
        real page chunks (many single-section project pages have a page
        title that equals their only section's heading, e.g. "Skills",
        "Prime Engine" -- verified against the current index: 141 chunks
        satisfy that alone). What IS reliably rare is one of these long,
        sentence-shaped, hand-authored headings (e.g. "Who this portfolio
        site is about") coincidentally matching a real page's short title
        character-for-character -- the same assumption knowledge_chunk_ids's
        matching already leans on. Scanning from the known heading text
        keeps that assumption intact and avoids false positives from the
        page-title coincidence above.

        A non-empty result means the index and the live corpus have
        desynchronized -- either a heading was renamed or removed in
        about_<lang>.md with no rebuild since (the stale chunk keeps the OLD
        text, which is no longer on disk, while the NEW text has no chunk
        yet -- both look identical from here: "this current heading isn't in
        the index"), or a heading was added and no rebuild has happened yet.
        Either way the fix is the same: rebuild. See knowledge_chunk_ids's
        docstring for why leaving it stale silently moves
        hit_at_4_page_only.

        This is a REPORTED diagnostic only: scripts/run_eval.py prints a
        note when it is non-empty, and it is deliberately NOT raised as an
        error and NOT compared by tests/test_golden.py::test_no_metric_regressed
        -- a stale corpus should never block a run, only stop being silent
        about it.
        """
        self._compute_knowledge_signature()
        return self._stale_knowledge_headings

    def _compute_knowledge_signature(self) -> None:
        if self._knowledge_chunk_ids is not None:
            return
        knowledge_dir = settings.chat_root / "knowledge"
        headings: dict[str, set[str]] = {"en": set(), "zh": set()}
        if knowledge_dir.is_dir():
            for lang in headings:
                headings[lang] = {s.section_title for s in load_knowledge(knowledge_dir, lang)}
        ids: set[str] = set()
        built_headings: dict[str, set[str]] = {"en": set(), "zh": set()}
        for c in self._index["chunks"]:
            if c.get("page_title") != c.get("section_title"):
                continue
            lang = c.get("lang", "en")
            if c["page_title"] in headings.get(lang, ()):
                ids.add(c["id"])
                built_headings.setdefault(lang, set()).add(c["page_title"])
        stale: set[str] = set()
        for lang, current in headings.items():
            stale |= current - built_headings.get(lang, set())
        self._knowledge_chunk_ids = frozenset(ids)
        self._stale_knowledge_headings = frozenset(stale)

    def retrieve(
        self, question: str, k: int = TOP_K, exclude_ids: frozenset[str] | None = None
    ) -> Retrieval:
        """exclude_ids removes candidates from the ranking BEFORE top-k is
        taken (not a post-hoc filter of an already-truncated top-k) -- so a
        chunk that would have ranked 5th among all candidates but 4th once
        excluded ids are removed still gets its place. Used by
        evaluation.score_case with knowledge_chunk_ids to answer "what would
        the site's own pages retrieve without any curated-corpus assist,"
        which a post-hoc filter of the normal top-4 cannot answer."""
        if self._embedder is None:
            raise RuntimeError(
                f"retrieval unavailable: the model directory {self._model_dir} "
                f"declared by index.json (model_preset="
                f"{self._index.get('model_preset')!r}) is not present — it is "
                "gitignored, so a fresh clone must download it before evaluating"
            )
        scores = self._matrix @ self._embedder.embed_query(question)
        if exclude_ids:
            rows = [self._id_to_row[i] for i in exclude_ids if i in self._id_to_row]
            if rows:
                scores = scores.copy()
                scores[rows] = -np.inf
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
            if np.isfinite(scores[i])
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
            fallback_spec = json.loads(fallback.read_text(encoding="utf-8"))
            gates["en"] = _GateBundle(
                # fallback_vectors.json records no model identity today, so
                # "minilm" is the documented default. Read the key if a future
                # build starts writing it, rather than silently mismatching the
                # way the settings-based embedder lookup did.
                fallback_spec.get("model_preset", "minilm"),
                fallback_spec,
            )

    # The INDEX declares which model built it — not settings, and not a
    # hardcoded preset. settings.model_preset defaults to "minilm" while the
    # committed index.json was built with e5, so resolving from settings would
    # dot MiniLM query vectors against e5 chunk vectors and return
    # plausible-looking garbage (measured: top score ~0.10 instead of ~0.86).
    # get_embedder()'s module-level cache is shared with the other test modules,
    # so build a dedicated instance rather than reusing it.
    preset_name = index.get("model_preset", settings.model_preset)
    preset = MODEL_PRESETS[preset_name]
    model_dir = settings.resolve_path(preset["dir"])
    embedder = (
        OnnxEmbedder.from_preset(preset, model_dir, settings.embedding_max_tokens)
        if model_dir.is_dir()
        else None
    )
    return Runtime(index, gates, embedder, model_dir=model_dir)
