"""Write path of the pipeline: site HTML → chunks → vectors → data/index.json.

The output is a static file served by GitHub Pages; the browser widget
fetches it once and does retrieval (dot product over normalized vectors)
entirely client-side. Chunk ids are deterministic ({url}#{anchor}:{i}) so
rebuilds are stable diffs.
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from portfolio_rag.chunker import chunk_text
from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder, get_embedder
from portfolio_rag.gate_calibration import OFF_TOPIC_ZH, ON_TOPIC_ZH, compute_gate
from portfolio_rag.loader import load_knowledge, load_site
from portfolio_rag.roles import roles_payload

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


def _build_zh_gate(preset: dict, ndigits: int) -> dict | None:
    """Chinese first-pass gate: bge-zh over the hand-written
    knowledge/about_zh.md corpus. Evidence-gated — only enabled if calibration
    on the zh query sets actually separates (otherwise the backend keeps the
    CJK bypass). Set RAG_ZH_GATE_FORCE=1 to write it despite overlap (testing)."""
    zh_model = preset.get("gate_model_zh")
    if not zh_model:
        return None
    zh_preset = MODEL_PRESETS[zh_model]
    model_dir = settings.resolve_path(zh_preset["dir"])
    corpus_dir = settings.chat_root / "knowledge"
    if not model_dir.is_dir() or not corpus_dir.is_dir():
        logger.info("zh gate: skipped (%s missing)", "model" if not model_dir.is_dir() else "knowledge/")
        return None

    sections = load_knowledge(corpus_dir, "zh")
    if not sections:
        logger.info("zh gate: skipped (knowledge/about_zh.md has no sections yet)")
        return None
    embedder = OnnxEmbedder(
        model_dir,
        max_tokens=settings.embedding_max_tokens,
        query_prefix=zh_preset["query_prefix"],
        passage_prefix=zh_preset["passage_prefix"],
        pooling=zh_preset.get("pooling", "mean"),
    )
    vecs = np.round(embedder.embed_documents([s.text for s in sections]).astype(float), ndigits)
    gate = compute_gate(embedder, vecs.astype(np.float32), on=ON_TOPIC_ZH, off=OFF_TOPIC_ZH)
    if gate["margin"] <= 0 and not os.environ.get("RAG_ZH_GATE_FORCE"):
        logger.warning("zh gate: calibration does not separate (margin %.1f%%) — NOT enabled; "
                       "CJK queries will bypass the gate", gate["margin"] * 100)
        return None
    logger.info("zh gate: enabled (%s >= %s, margin %.1f%%, %d sections)",
                gate["stat"], gate["threshold"], gate["margin"] * 100, len(sections))
    return {
        "model": zh_preset["name"],
        "model_preset": zh_model,
        "query_prefix": zh_preset["query_prefix"],
        "pooling": zh_preset.get("pooling", "mean"),
        "gate_stat": gate["stat"],
        "gate_threshold": gate["threshold"],
        "vectors": vecs.tolist(),
    }


def build_index(site_root: Path | None = None) -> dict:
    t0 = time.time()
    site_root = site_root or settings.site_root
    preset = settings.preset

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

    embedder = get_embedder()
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    ndigits = settings.vector_round_decimals
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = [round(float(v), ndigits) for v in vector]

    # Off-topic gate. If the preset delegates gating to another model (e5
    # can't separate on/off-topic), embed the chunks AGAIN with the gate
    # model and write data/gate_vectors.json for the backend; otherwise
    # calibrate the retrieval model itself for the widget's local gate.
    gate_model = preset.get("gate_model")
    if gate_model:
        gate_preset = MODEL_PRESETS[gate_model]
        gate_embedder = OnnxEmbedder(
            settings.resolve_path(gate_preset["dir"]),
            max_tokens=settings.embedding_max_tokens,
            query_prefix=gate_preset["query_prefix"],
            passage_prefix=gate_preset["passage_prefix"],
            pooling=gate_preset.get("pooling", "mean"),
        )
        # The MiniLM en gate (and the degraded fallback) cover ONLY the English
        # chunks: MiniLM can't embed Chinese, so including zh chunks re-adds the
        # off-topic hubness the gate exists to avoid. In the bilingual build the
        # en chunks are the index PREFIX (built first), so fallback.vectors[i]
        # still lines up with index.chunks[i] and the widget's English-only
        # degraded loop naturally stops before the zh chunks. For a monolingual
        # build every chunk is "en" here, so this is a no-op.
        en_chunks = [c for c in chunks if c.get("lang") != "zh"]
        gate_vecs = gate_embedder.embed_documents([c["text"] for c in en_chunks])
        gate_vecs = np.round(gate_vecs.astype(float), ndigits)
        gate = compute_gate(gate_embedder, gate_vecs.astype(np.float32),
                            multilingual=gate_preset["multilingual"])
        # Symmetric with the zh line below. The en gate is unconditional (e5
        # can't self-gate), so this always prints; a negative margin only warns
        # (compute_gate still picks a threshold just above the off-topic max).
        logger.info("en gate: enabled (%s >= %s, margin %.1f%%, %d chunks)",
                    gate["stat"], gate["threshold"], gate["margin"] * 100, len(en_chunks))
        gate_payload = {
            "en": {
                "model": gate_preset["name"],
                "model_preset": gate_model,
                "query_prefix": gate_preset["query_prefix"],
                "pooling": gate_preset.get("pooling", "mean"),
                "gate_stat": gate["stat"],
                "gate_threshold": gate["threshold"],
                "chunk_ids": [c["id"] for c in en_chunks],
                "vectors": gate_vecs.tolist(),
            }
        }
        zh_gate = _build_zh_gate(preset, ndigits)
        if zh_gate:
            gate_payload["zh"] = zh_gate
        gate_path = settings.resolve_path(settings.gate_vectors_path)
        gate_path.write_text(json.dumps(gate_payload, ensure_ascii=False), encoding="utf-8")

        # Published fallback for the widget's degraded mode (backend down):
        # MiniLM vectors of the English chunks, order-aligned with the English
        # prefix of index.json (see en_chunks note above).
        fallback = {
            "model": gate_preset["name"],
            "query_prefix": gate_preset["query_prefix"],
            "gate_stat": gate["stat"],
            "gate_threshold": gate["threshold"],
            "vectors": gate_vecs.tolist(),
        }
        settings.resolve_path(settings.fallback_vectors_path).write_text(
            json.dumps(fallback, ensure_ascii=False), encoding="utf-8"
        )
        index_gate = {"gate_remote": True, "gate_stat": gate["stat"], "gate_threshold": gate["threshold"]}
    else:
        matrix = np.array([c["vector"] for c in chunks], dtype=np.float32)
        gate = compute_gate(embedder, matrix, multilingual=preset["multilingual"])
        index_gate = {"gate_remote": False, "gate_stat": gate["stat"], "gate_threshold": gate["threshold"]}

    index = {
        "schema_version": SCHEMA_VERSION,
        "model": preset["name"],
        "model_preset": settings.model_preset,
        "query_prefix": preset["query_prefix"],
        "multilingual": preset["multilingual"],
        **index_gate,
        "dim": int(vectors.shape[1]) if len(chunks) else 0,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks": chunks,
    }

    index_path = settings.resolve_path(settings.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    roles_path = settings.resolve_path(settings.roles_path)
    roles_path.write_text(
        json.dumps(roles_payload(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pages = {c["url"] for c in chunks}
    return {
        "pages": len(pages),
        "sections": len(tagged),
        "chunks": len(chunks),
        "index_kb": round(index_path.stat().st_size / 1024, 1),
        "gate_stat": gate["stat"],
        "gate_threshold": gate["threshold"],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
