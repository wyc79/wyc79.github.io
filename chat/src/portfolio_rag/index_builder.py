"""Write path of the pipeline: site HTML → chunks → vectors → data/index.json.

The output is a static file served by GitHub Pages; the browser widget
fetches it once and does retrieval (dot product over normalized vectors)
entirely client-side. Chunk ids are deterministic ({url}#{anchor}:{i}) so
rebuilds are stable diffs.
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from portfolio_rag.chunker import chunk_text
from portfolio_rag.config import settings
from portfolio_rag.embedder import get_embedder
from portfolio_rag.loader import load_site
from portfolio_rag.roles import roles_payload

SCHEMA_VERSION = 1


def build_index(site_root: Path | None = None) -> dict:
    t0 = time.time()
    site_root = site_root or settings.site_root
    sections = load_site(site_root)

    chunks: list[dict] = []
    section_ordinal: dict[str, int] = defaultdict(int)  # per-page section counter
    for sec in sections:
        ordinal = section_ordinal[sec.url]
        section_ordinal[sec.url] += 1
        # Anchor-less sections fall back to their per-page ordinal so two of
        # them on the same page can never share an id.
        anchor_part = sec.anchor or f"sec{ordinal}"
        for i, piece in enumerate(chunk_text(sec.text, settings.chunk_size, settings.chunk_overlap)):
            chunks.append(
                {
                    "id": f"{sec.url}#{anchor_part}:{i}",
                    "url": sec.url,
                    "anchor": sec.anchor,
                    "page_title": sec.page_title,
                    "section_title": sec.section_title,
                    "text": piece,
                }
            )

    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate chunk ids: {dupes[:5]}")

    vectors = get_embedder().embed_documents([c["text"] for c in chunks])
    ndigits = settings.vector_round_decimals
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = [round(float(v), ndigits) for v in vector]

    index = {
        "schema_version": SCHEMA_VERSION,
        "model": "Xenova/all-MiniLM-L6-v2 (quantized ONNX, mean pooling, normalized)",
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

    pages = {s.url for s in sections}
    return {
        "pages": len(pages),
        "sections": len(sections),
        "chunks": len(chunks),
        "index_kb": round(index_path.stat().st_size / 1024, 1),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
