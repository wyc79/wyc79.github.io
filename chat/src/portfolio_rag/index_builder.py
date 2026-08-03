"""Write path of the pipeline: site HTML → chunks → vectors →
data/chunks_{model_preset}.json (+ data/gate_en_minilm.json,
data/gate_zh_bge.json, data/chunks_en_minilm.json for a preset that
delegates gating -- see Settings.resolve_chunks_path).

The outputs are static files served by GitHub Pages; normal mode's retrieval
happens server-side (Task 29 Part 1), and the browser widget fetches a
chunks file itself only for light mode or the degraded-mode fallback, doing
retrieval (dot product over normalized vectors) client-side in that case.
Chunk ids are deterministic ({url}#{anchor}:{i}) so rebuilds are stable
diffs.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from portfolio_rag.chunker import chunk_text
from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder
from portfolio_rag.gate_calibration import OFF_TOPIC_ZH, ON_TOPIC_ZH, compute_gate
from portfolio_rag.loader import load_knowledge, load_site
from portfolio_rag.roles import roles_payload

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ZhGateResult:
    """Outcome of one call to `_build_zh_gate`. `gate` is None for four
    different reasons (see that function's docstring), and the caller's
    only *correct* reaction to a stale on-disk `gate_zh_bge.json` depends on
    WHICH one -- so the distinction is carried here, on the return value,
    instead of being re-derived at the call site (which has no way to tell
    "the bge_zh model directory is missing" apart from "calibration ran and
    the margin did not clear" once all it sees is `gate is None`).

    `calibrated` is True only when `compute_gate()` actually ran THIS build
    and measured a margin, whether or not that margin cleared the bar.
    False means calibration never ran at all this build (model directory
    missing, chat/knowledge/ missing, or about_zh.md has no zh sections
    yet) -- an incomplete environment, not evidence about the gate's
    quality. A fresh clone (bge_zh model not yet downloaded) is exactly
    this case, which is why the caller must NOT unlink an existing gate
    file when `calibrated` is False: this build simply couldn't check it,
    so treating "couldn't check" the same as "checked and it failed" would
    delete a previously-good, separating gate for no evidence-based reason.
    """

    gate: dict | None
    calibrated: bool


def _build_zh_gate(preset: dict, ndigits: int) -> ZhGateResult:
    """Chinese first-pass gate: bge-zh over the hand-written
    knowledge/about_zh.md corpus. Evidence-gated — only enabled if calibration
    on the zh query sets actually separates (otherwise the backend keeps the
    CJK bypass). Set RAG_ZH_GATE_FORCE=1 to write it despite overlap (testing).

    Returns a ZhGateResult, never a bare dict-or-None: `.gate` is never
    written here — the caller writes it to settings.gate_zh_bge_path,
    gitignored, see config.py — the caller decides whether to write it at
    all. `.calibrated` tells the caller WHY `.gate` is None when it is,
    which matters because `.gate is None` happens for four different
    reasons and only one of them (calibration ran and the margin did not
    clear) is evidence that a stale on-disk gate_zh_bge.json should be
    removed:
      1. this preset has no gate_model_zh at all (not e5) — calibrated=False
      2. the bge_zh model directory is not present on this machine (e.g. a
         fresh clone before functions/tencent/build_package.py's first
         download) — calibrated=False
      3. chat/knowledge/ is not present — calibrated=False
      4. about_zh.md has no zh sections yet — calibrated=False
      5. calibration ran and the margin did not clear — calibrated=True
    A machine that is simply missing the model (2) looks, from the OLD
    "gate is None" signal alone, identical to a calibration that genuinely
    stopped separating (5) — but only (5) is real evidence about gate
    quality; (2)-(4) mean this build never got to check at all. Conflating
    them was the over-fire a reviewer demonstrated by pointing
    MODEL_PRESETS["bge_zh"]["dir"] at a nonexistent path (case 2) and
    watching a planted, still-separating gate_zh_bge.json get deleted
    anyway.

    There is deliberately no "chunks_zh_minilm" retrieval counterpart to
    pair with this gate the way chunks_en_minilm.json pairs with the en
    gate. MiniLM — the ONLY model this project ever ships to the browser —
    cannot embed Chinese at all, and the widget already handles CJK
    questions in degraded mode by showing `degradedCJK` plus static page
    links (the SUGGESTED_PAGES list in scripts/chat-widget.js), never by
    attempting local Chinese retrieval. A symmetric-looking
    "chunks_zh_minilm.json" is the tempting mistake someone will eventually
    reach for BECAUSE gate_en_minilm.json has a retrieval counterpart and
    this file looks like it "should" too — it should not, and building one
    would silently violate the "no Chinese degraded mode" contract with no
    error to catch it."""
    zh_model = preset.get("gate_model_zh")
    if not zh_model:
        return ZhGateResult(gate=None, calibrated=False)
    zh_preset = MODEL_PRESETS[zh_model]
    model_dir = settings.resolve_path(zh_preset["dir"])
    corpus_dir = settings.chat_root / "knowledge"
    if not model_dir.is_dir() or not corpus_dir.is_dir():
        logger.info("zh gate: skipped (%s missing)", "model" if not model_dir.is_dir() else "knowledge/")
        return ZhGateResult(gate=None, calibrated=False)

    sections = load_knowledge(corpus_dir, "zh")
    if not sections:
        logger.info("zh gate: skipped (knowledge/about_zh.md has no sections yet)")
        return ZhGateResult(gate=None, calibrated=False)
    embedder = OnnxEmbedder.from_preset(zh_preset, model_dir, settings.embedding_max_tokens)
    vecs = np.round(embedder.embed_documents([s.text for s in sections]).astype(float), ndigits)
    gate = compute_gate(embedder, vecs.astype(np.float32), on=ON_TOPIC_ZH, off=OFF_TOPIC_ZH)
    if gate["margin"] <= 0 and not os.environ.get("RAG_ZH_GATE_FORCE"):
        logger.warning("zh gate: calibration does not separate (margin %.1f%%) — NOT enabled; "
                       "CJK queries will bypass the gate", gate["margin"] * 100)
        return ZhGateResult(gate=None, calibrated=True)
    logger.info("zh gate: enabled (%s >= %s, margin %.1f%%, %d sections)",
                gate["stat"], gate["threshold"], gate["margin"] * 100, len(sections))
    return ZhGateResult(
        gate={
            "model": zh_preset["name"],
            "model_preset": zh_model,
            "query_prefix": zh_preset["query_prefix"],
            "pooling": zh_preset.get("pooling", "mean"),
            "lang": "zh",
            "corpus": "knowledge/about_zh.md",
            "gate_stat": gate["stat"],
            "gate_threshold": gate["threshold"],
            "gate_margin": gate["margin"],
            "vectors": vecs.tolist(),
        },
        calibrated=True,
    )


def _check_en_gate_margin(gate: dict) -> None:
    """Refuse to ship an en gate calibration that doesn't clear the floor.

    compute_gate() already logs a WARNING and ships anyway when nothing
    separates on-/off-topic — and unlike the zh gate (_build_zh_gate returns
    None and the backend keeps the CJK bypass), the en gate was written
    unconditionally: there's no "quietly disable" fallback for English the
    way there is for zh. That asymmetry is how a live false-refusal bug
    reached production silently. This raises instead of warning, so it
    cannot be scrolled past in a build log.

    Raising here must not leave data/ half-updated — that would be worse than
    the silent-ship bug it replaces. This used to be an ordering RULE stated in
    this docstring ("call before ANY of gate_en_minilm.json, gate_zh_bge.json,
    chunks_en_minilm.json, roles.json or chunks_{model_preset}.json are
    written"), and the rule was already broken: build_index() wrote
    gate_en_minilm.json and then ran two more failable ONNX passes
    (_build_zh_gate's model load + inference, and the 192-chunk degraded-corpus
    embedding) before writing anything else. It is now a STRUCTURAL property
    instead: build_index() accumulates every payload and flushes them in one
    pass at the very end (see _flush_build_outputs), so no raise from the
    build's COMPUTATION phase — this check, an OOM, a corrupt ONNX file — can
    leave a partial data/. There is no ordering constraint left for a caller
    to honour.

    Scope, stated precisely because this comment is load-bearing: that covers
    everything up to the flush, which is every failable computation in the
    build. It does NOT cover the flush itself — an IO error partway through
    _flush_build_outputs's write loop can still leave data/ half-updated, and
    that function's own docstring says so. Closing the remaining window needs
    write-to-.tmp-then-rename and is deliberately not claimed here.

    The floor is RAG_MIN_GATE_MARGIN (default 0.0), not a hardcoded sign
    check: +0.5% margin isn't meaningfully healthier than -0.5%, and the
    floor should move once there's evidence about what a healthy margin
    looks like for this corpus. RAG_ALLOW_NEGATIVE_MARGIN=1 is the
    deliberate-ship opt-out, mirroring RAG_ALLOW_PRESET_CHANGE above and
    RAG_ZH_GATE_FORCE below.
    """
    floor = float(os.environ.get("RAG_MIN_GATE_MARGIN", "0.0"))
    if gate["margin"] >= floor or os.environ.get("RAG_ALLOW_NEGATIVE_MARGIN"):
        return
    raise ValueError(
        f"en gate calibration does not clear the margin floor: stat={gate['stat']} "
        f"margin={gate['margin']:.4f} (off-topic max={gate['lo']:.4f}, "
        f"on-topic min={gate['hi']:.4f}) < floor={floor:.4f}. Shipping this index "
        "would write an en gate with no evidence it separates on-/off-topic "
        "queries. Set RAG_ALLOW_NEGATIVE_MARGIN=1 to build anyway, or "
        "RAG_MIN_GATE_MARGIN=<value> to change the floor."
    )


def _flush_build_outputs(writes: dict[Path, str], deletes: list[Path]) -> None:
    """The build's ONE write phase. build_index() computes every payload first
    and calls this last, so that any failure during the build — a gate margin
    below the floor, an OOM in one of the three ONNX passes, a corrupt model
    file — leaves data/ exactly as it was rather than half-updated. That
    matters more than usual here because one of the outputs
    (gate_zh_bge.json) is gitignored and unrecoverable by git.

    This closes the failable-computation window, not the disk-IO one: an IO
    error partway through the loop below can still leave a partial data/. That
    residual would need write-to-.tmp-then-rename; it is deliberately not
    claimed here."""
    for path in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in writes.items():
        path.write_text(text, encoding="utf-8")
    for path in deletes:
        path.unlink(missing_ok=True)


def build_index(site_root: Path | None = None) -> dict:
    t0 = time.time()
    site_root = site_root or settings.site_root
    preset = settings.preset

    # Every output this build will produce, accumulated and flushed exactly
    # once at the end -- see _flush_build_outputs and _check_en_gate_margin's
    # docstring. Nothing below this line touches the filesystem until then.
    writes: dict[Path, str] = {}
    deletes: list[Path] = []

    # Guard against silently rebuilding the retrieval corpus in a different
    # embedding space. gate_en_minilm.json, gate_zh_bge.json,
    # chunks_en_minilm.json and the deployed Tencent function all depend on
    # the model_preset the committed artifacts were built with (see
    # .env.example) — a mismatch here means settings.model_preset drifted
    # from that (e.g. a fresh clone missing chat/.env fell back to the
    # "minilm" default). Checked before any work is done, not after, so a
    # refused build doesn't waste an embedding pass.
    #
    # ANCHORED ON meta.json, not on the chunks file alone. An earlier version
    # checked only the chunks file and reasoned that a genuine preset switch
    # writes to a different filename entirely, "which is fine: chunks_e5.json
    # and chunks_minilm.json coexisting is not a desync, it's two builds'
    # outputs living side by side." True of the chunks files. FALSE of
    # meta.json and roles.json, which are single-valued and overwritten
    # unconditionally by every build. Once chunks_path became preset-derived
    # (config.py's resolve_chunks_path), a `--model minilm` build resolved to a
    # not-yet-existing chunks_minilm.json and so SKIPPED this guard entirely,
    # then repointed the committed meta.json to model_preset="minilm",
    # gate_remote=false, chunks_file="chunks_minilm.json" and a gate_threshold
    # calibrated against all 192 site chunks instead of the 55 curated
    # about_en.md sections. _check_en_gate_margin does not stop that either:
    # the minilm self-gate over the real 192 en chunks measures margin +2.52%,
    # so both guards cleared. chat-widget.js reads meta.json as the sole
    # authority for which mode to run and at what threshold, so committing that
    # flips EVERY visitor into light mode -- a ~23 MB in-browser model download
    # plus exactly the site-growth-coupled gate this branch exists to remove.
    # (This supersedes Task 34 Part C's "no corruption hole" conclusion, which
    # was reached against the chunks-file-only version of this guard.)
    #
    # meta.json is the right anchor precisely because it is the one file every
    # preset always writes. The chunks file is still checked as well, for the
    # stale-file-at-the-current-preset's-own-path case. RAG_ALLOW_PRESET_CHANGE=1
    # remains the deliberate-switch opt-out.
    if not os.environ.get("RAG_ALLOW_PRESET_CHANGE"):
        for existing in (settings.resolve_path(settings.meta_path), settings.resolve_chunks_path()):
            if not existing.exists():
                continue
            existing_preset = json.loads(existing.read_text(encoding="utf-8")).get("model_preset")
            if existing_preset is not None and existing_preset != settings.model_preset:
                raise ValueError(
                    f"existing {existing.name} at {existing} was built with "
                    f"model_preset={existing_preset!r}, but settings.model_preset is "
                    f"{settings.model_preset!r}. Rebuilding would desync meta.json "
                    "(which chat-widget.js reads to decide which mode every visitor "
                    "runs), gate_en_minilm.json, gate_zh_bge.json, "
                    "chunks_en_minilm.json and the deployed backend. If this is a "
                    "fresh clone, you likely just need chat/.env: copy it from "
                    "chat/.env.example (RAG_MODEL_PRESET=e5) so settings.model_preset "
                    "matches the committed artifacts. Only set "
                    "RAG_ALLOW_PRESET_CHANGE=1 if you deliberately mean to switch "
                    "presets and rebuild everything downstream."
                )

    # A multilingual retrieval model (e5) gets a DE-INTERLEAVED bilingual index:
    # clean English-only sections then clean Chinese-only sections (en first),
    # instead of the en+zh-interleaved chunks the bilingual pages would produce
    # under one get_text(). Two payoffs: (1) each chunk vector is monolingual,
    # so retrieval isn't muddied; (2) the MiniLM en gate + degraded fallback
    # (below) can cover just the English prefix — MiniLM can't embed zh, and
    # mixing zh in poisons the gate. A monolingual model (minilm) keeps the
    # original single-view build and id scheme (no lang segment).
    if preset["multilingual"]:
        tagged = [(s, "en") for s in load_site(site_root, "en")] + [
            (s, "zh") for s in load_site(site_root, "zh")
        ]
    else:
        # A monolingual model (minilm) is English-only — take the English view
        # of the (now bilingual) pages so its chunks aren't en+zh mush that the
        # model can't embed. Tag None so ids keep the original scheme (no lang
        # segment): a single-language index has no en/zh collision to break.
        tagged = [(s, None) for s in load_site(site_root, "en")]

    chunks: list[dict] = []
    section_ordinal: dict[tuple, int] = defaultdict(int)  # per (page, lang) counter
    for sec, lang in tagged:
        ordinal = section_ordinal[(sec.url, lang)]
        section_ordinal[(sec.url, lang)] += 1
        # Anchor-less sections fall back to their per-page ordinal so two of
        # them on the same page can never share an id. In the bilingual build
        # the en/zh copies of a section share url+anchor, so a lang segment in
        # the id keeps them distinct.
        anchor_part = sec.anchor or f"sec{ordinal}"
        for i, piece in enumerate(chunk_text(sec.text, settings.chunk_size, settings.chunk_overlap)):
            cid = (
                f"{sec.url}#{anchor_part}:{i}"
                if lang is None
                else f"{sec.url}#{anchor_part}:{lang}:{i}"
            )
            chunk = {
                "id": cid,
                "url": sec.url,
                "anchor": sec.anchor,
                "page_title": sec.page_title,
                "section_title": sec.section_title,
                "text": piece,
            }
            if lang is not None:
                chunk["lang"] = lang
            chunks.append(chunk)

    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate chunk ids: {dupes[:5]}")

    # A dedicated instance built from the local `preset`, not the process-wide
    # get_embedder() cache: that cache is keyed on whichever preset first
    # requested it and is shared across test modules (and any other caller
    # in-process), so a build running after something else already resolved
    # the cache under a different preset would silently embed with the wrong
    # model while still labeling the index with settings.model_preset. This
    # is the same failure mode runtime.py's index-declares-its-own-model fix
    # exists to avoid (see the comment above Runtime's embedder resolution) —
    # apply the same fix here, where it matters even more: this is what
    # produces the vectors actually written to the committed
    # chunks_{model_preset}.json.
    embedder = OnnxEmbedder.from_preset(
        preset, settings.resolve_path(preset["dir"]), settings.embedding_max_tokens
    )
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    ndigits = settings.vector_round_decimals
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = [round(float(v), ndigits) for v in vector]

    # Off-topic gate (Task 29 Part 2: "one file, one job" — this whole branch
    # writes THREE separate single-purpose files instead of the old
    # gate_vectors.json+fallback_vectors.json pair that conflated a gate
    # corpus with a retrieval corpus, which is why Task 24 silently broke
    # degraded-mode source links). If the preset delegates gating to another
    # model (e5 can't separate on/off-topic), embed the gate corpus AGAIN
    # with the gate model; otherwise calibrate the retrieval model itself for
    # the widget's local gate.
    gate_model = preset.get("gate_model")
    if gate_model:
        gate_preset = MODEL_PRESETS[gate_model]
        gate_embedder = OnnxEmbedder.from_preset(
            gate_preset,
            settings.resolve_path(gate_preset["dir"]),
            settings.embedding_max_tokens,
        )
        # First-pass gate corpus: the curated knowledge/about_en.md sections —
        # NOT every indexed English chunk — symmetric with _build_zh_gate
        # below (task 24; see task 21's diagnosis). With stat="top" the gate
        # statistic is a max over its corpus: a corpus that grows with the
        # SITE (every page/section added) hands off-topic queries an ever
        # growing number of chances at a spurious match, while on-topic
        # queries already sit near their ceiling. Measured before this
        # change: retrieval improved on a rebuild (hit@4 64/96 -> 66/96)
        # while the en gate margin fell -- gate quality and site growth were
        # coupled in the wrong direction. A small, curated, question-shaped
        # corpus decouples them, the same way the zh gate already does.
        # Whole sections are embedded (like _build_zh_gate's `s.text for s in
        # sections`), not chunk_text pieces: these are short, single-topic
        # entries by construction and don't need splitting.
        corpus_dir = settings.chat_root / "knowledge"
        en_sections = load_knowledge(corpus_dir, "en")
        if not en_sections:
            # The en gate has no quiet-disable fallback (see
            # _check_en_gate_margin's docstring) -- an empty corpus must fail
            # the build loudly, not silently revert to gating against every
            # indexed chunk (the exact bug this change fixes).
            raise ValueError(
                f"en gate: no usable sections in {corpus_dir}/about_en.md — "
                "cannot build an en gate corpus. Not falling back to indexed "
                "chunks: that would silently reintroduce the site-growth "
                "coupling this corpus exists to remove."
            )
        gate_vecs = gate_embedder.embed_documents([s.text for s in en_sections])
        gate_vecs = np.round(gate_vecs.astype(float), ndigits)
        gate = compute_gate(gate_embedder, gate_vecs.astype(np.float32),
                            multilingual=gate_preset["multilingual"])
        _check_en_gate_margin(gate)
        # Symmetric with the zh line below. The en gate is unconditional (e5
        # can't self-gate), so this always prints; a negative margin only warns
        # (compute_gate still picks a threshold just above the off-topic max).
        logger.info("en gate: enabled (%s >= %s, margin %.1f%%, %d sections)",
                    gate["stat"], gate["threshold"], gate["margin"] * 100, len(en_sections))
        # gate_*.json keys ("one file, one job" -- Task 29 Part 2 spec §2.2):
        # model, model_preset, query_prefix, pooling, lang, corpus, gate_stat,
        # gate_threshold, gate_margin, vectors. No chunk ids, no chunk text,
        # no retrieval fields -- a gate corpus is not a retrieval corpus, and
        # conflating the two (the old gate_vectors.json / fallback_vectors.json
        # pair) is the exact bug this file split exists to prevent. This is
        # now the ONLY copy of the en gate (replaces both gate_vectors.json's
        # "en" entry and the fallback_vectors.json duplicate the widget's
        # degraded mode used to read for its gate decision), and it is
        # COMMITTED — e5 can never self-gate, so this file must always exist,
        # unlike gate_zh_bge.json below.
        gate_en = {
            "model": gate_preset["name"],
            "model_preset": gate_model,
            "query_prefix": gate_preset["query_prefix"],
            "pooling": gate_preset.get("pooling", "mean"),
            "lang": "en",
            "corpus": "knowledge/about_en.md",
            "gate_stat": gate["stat"],
            "gate_threshold": gate["threshold"],
            "gate_margin": gate["margin"],
            "vectors": gate_vecs.tolist(),
        }
        writes[settings.resolve_path(settings.gate_en_minilm_path)] = json.dumps(
            gate_en, ensure_ascii=False
        )

        zh_result = _build_zh_gate(preset, ndigits)
        gate_zh_path = settings.resolve_path(settings.gate_zh_bge_path)
        if zh_result.gate:
            # Gitignored (settings.gate_zh_bge_path / chat/.gitignore) — the
            # ONLY enforcement that this never reaches a browser which does
            # not depend on the widget behaving correctly. See
            # _build_zh_gate's own docstring for why there is deliberately no
            # "chunks_zh_minilm.json" retrieval counterpart to pair it with.
            writes[gate_zh_path] = json.dumps(zh_result.gate, ensure_ascii=False)
        elif zh_result.calibrated:
            # Fix round 1 review, Important 1: the OLD gate_vectors.json was
            # rewritten WHOLE on every build (one combined file, "en" and
            # "zh" keys both written every time), so a zh gate that stopped
            # separating simply vanished from that file the next rebuild.
            # Splitting it into its own file made that implicit "clear on
            # rebuild" behavior stop happening for free -- a STALE
            # gate_zh_bge.json from an earlier build (when calibration DID
            # separate) would otherwise survive untouched forever, silently
            # outliving the "NOT enabled" log line _build_zh_gate just
            # printed above. runtime.py would load it, run_eval.py would
            # score against it, and build_package.py would bundle it into
            # the deployed zip -- shipping a gate this exact build run just
            # decided should not exist. The unlink is queued with the writes
            # and performed by _flush_build_outputs (missing_ok=True there: a
            # machine that never had one -- the common case, gitignored and
            # not on a fresh clone -- must not raise), so a later failure in
            # this build cannot delete it and then abort.
            #
            # Guarded on zh_result.calibrated (Task 34, Part B — the previous
            # round's unconditional `else: unlink` over-fired): compute_gate()
            # genuinely ran THIS build and measured a margin that did not
            # clear, so this build has real evidence the file is stale.
            deletes.append(gate_zh_path)
        else:
            # zh_result.calibrated is False: calibration never ran this
            # build at all (bge_zh model not on this machine, chat/knowledge/
            # missing, or about_zh.md has no zh sections yet -- see
            # _build_zh_gate's docstring). A fresh clone is exactly this
            # case (the bge_zh model is gitignored, never checked out), so
            # unlinking here on the strength of "gate is None" alone would
            # delete a previously-good, separating gate_zh_bge.json for no
            # evidence-based reason -- the over-fire a reviewer demonstrated
            # by pointing MODEL_PRESETS["bge_zh"]["dir"] at a nonexistent
            # path. Leave any existing file untouched, and say so explicitly
            # -- silence here is exactly what made the original stale-gate
            # bug invisible in the first place.
            if gate_zh_path.exists():
                logger.info(
                    "zh gate: environment incomplete this build (calibration did not "
                    "run) -- leaving existing %s untouched", gate_zh_path.name,
                )

        # Degraded-mode RETRIEVAL corpus (Task 29 Part 2 — the actual fix for
        # chat-widget.js's degraded-mode source links, not just a rename).
        # Task 24 broke those links by pointing the gate at a curated
        # 55-section corpus that no longer order-aligned with the full
        # chunk index, while chat-widget.js's retrieveFallback kept mapping
        # fb.vectors[i] -> state.index.chunks[i] positionally -- once the
        # gate corpus and the chunk index diverged in size/order, that
        # mapping named an unrelated chunk, so the links were suppressed
        # rather than shown wrong. The fix is to give degraded mode its own
        # real, retrieval-shaped corpus instead of repurposing the gate's:
        # the SAME 192 English chunks as this build's own lang=="en" entries
        # -- same ids, same order, same text (filtered from `chunks`, not
        # rebuilt, so tests/test_data_file_layout.py's alignment check holds
        # by construction) -- re-embedded with MiniLM (reusing gate_embedder,
        # not a fresh model load). chat-widget.js's retrieveFallback resolves
        # each result from THIS array's own chunk records, never by
        # borrowing a position from gate_en_minilm.json or state.chunks.
        # NOT "all pages": 137 page chunks + 55 curated about_en chunks --
        # populating it with pages alone would take degraded-mode hit@4 from
        # 90/96 to 59/96 (measured; see
        # .superpowers/sdd/EVAL_PLAN/task-29b-brief.md §1).
        en_chunks = [c for c in chunks if c.get("lang") == "en"]
        if en_chunks:
            en_minilm_vectors = gate_embedder.embed_documents([c["text"] for c in en_chunks])
        else:
            en_minilm_vectors = np.empty((0, 0), dtype=np.float32)
        chunks_en_minilm = {
            "schema_version": SCHEMA_VERSION,
            "model": gate_preset["name"],
            "model_preset": gate_model,
            "query_prefix": gate_preset["query_prefix"],
            "pooling": gate_preset.get("pooling", "mean"),
            "dim": int(en_minilm_vectors.shape[1]) if en_chunks else 0,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunks": [
                {**{k: v for k, v in c.items() if k != "vector"},
                 "vector": [round(float(v), ndigits) for v in vec]}
                for c, vec in zip(en_chunks, en_minilm_vectors)
            ],
        }
        writes[settings.resolve_path(settings.chunks_en_minilm_path)] = json.dumps(
            chunks_en_minilm, ensure_ascii=False
        )

        index_gate = {
            "gate_remote": True, "gate_stat": gate["stat"], "gate_threshold": gate["threshold"],
            "gate_margin": gate["margin"],
        }
    else:
        matrix = np.array([c["vector"] for c in chunks], dtype=np.float32)
        gate = compute_gate(embedder, matrix, multilingual=preset["multilingual"])
        # Same guard, same "before any write" placement, for the branch
        # where the retrieval model gates itself (no gate_en_minilm.json at
        # all -- chunks_{model_preset}.json's OWN gate_stat/threshold, folded
        # into meta.json below, ARE the gate).
        _check_en_gate_margin(gate)
        index_gate = {
            "gate_remote": False, "gate_stat": gate["stat"], "gate_threshold": gate["threshold"],
            "gate_margin": gate["margin"],
        }

    # chunks_*.json keys ("one file, one job" -- Task 29 Part 2 spec §2.2):
    # schema_version, model, model_preset, query_prefix, pooling, dim,
    # built_at, chunk_size, chunk_overlap, chunks. No gate fields, no
    # "multilingual" flag -- gate thresholds live in the gate files and in
    # meta.json below, not duplicated here.
    chunks_payload = {
        "schema_version": SCHEMA_VERSION,
        "model": preset["name"],
        "model_preset": settings.model_preset,
        "query_prefix": preset["query_prefix"],
        "pooling": preset.get("pooling", "mean"),
        "dim": int(vectors.shape[1]) if len(chunks) else 0,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks": chunks,
    }

    # Filename is DERIVED from model_preset, not hardcoded -- see
    # Settings.resolve_chunks_path(). In production this is chunks_e5.json; a
    # light `--model minilm` build writes chunks_minilm.json.
    chunks_path = settings.resolve_chunks_path()
    writes[chunks_path] = json.dumps(chunks_payload, ensure_ascii=False)

    # Small metadata sidecar (Task 29 Part 1, extended Part 2): the same
    # values already computed above for chunks_payload, minus the (multi-MB)
    # chunks array, plus chunks_file (Part 2) naming the retrieval corpus so
    # the widget's light mode can fetch the right file without inferring a
    # name from `model` itself -- see chat-widget.js's loadCore. Degraded
    # mode never reads chunks_file: it always fetches chunks_en_minilm.json
    # by its fixed name, regardless of what preset built this meta.json --
    # see chat-widget.js's retrieveFallback.
    meta = {
        "schema_version": chunks_payload["schema_version"],
        "model": chunks_payload["model"],
        "model_preset": chunks_payload["model_preset"],
        "query_prefix": chunks_payload["query_prefix"],
        "chunks_file": chunks_path.name,
        "gate_remote": index_gate["gate_remote"],
        "gate_stat": index_gate["gate_stat"],
        "gate_threshold": index_gate["gate_threshold"],
        "gate_margin": index_gate["gate_margin"],
        "built_at": chunks_payload["built_at"],
        "dim": chunks_payload["dim"],
        "chunk_count": len(chunks),
    }
    writes[settings.resolve_path(settings.meta_path)] = json.dumps(
        meta, ensure_ascii=False, indent=2
    )

    writes[settings.resolve_path(settings.roles_path)] = json.dumps(
        roles_payload(), ensure_ascii=False, indent=2
    )

    # Everything above is pure computation. This is the only line in the build
    # that mutates data/.
    _flush_build_outputs(writes, deletes)

    pages = {c["url"] for c in chunks}
    return {
        "pages": len(pages),
        "sections": len(tagged),
        "chunks": len(chunks),
        "chunks_kb": round(chunks_path.stat().st_size / 1024, 1),
        "gate_stat": gate["stat"],
        "gate_threshold": gate["threshold"],
        "gate_margin": gate["margin"],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
