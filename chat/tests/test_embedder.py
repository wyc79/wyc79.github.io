import numpy as np
import pytest

from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder, get_embedder


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def _embedder_for(preset_name: str) -> OnnxEmbedder:
    # A dedicated OnnxEmbedder for a named preset, independent of
    # get_embedder()'s module-level cache (which is keyed on whatever
    # settings.model_preset happened to be at first call, and is shared
    # across test modules — mutating it here has poisoned other tests before).
    preset = MODEL_PRESETS[preset_name]
    return OnnxEmbedder.from_preset(
        preset, settings.resolve_path(preset["dir"]), settings.embedding_max_tokens
    )


def test_dimensions_and_unit_norm(embedder) -> None:
    vecs = embedder.embed_documents(["combat design", "engine programming in C++"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


def test_deterministic(embedder) -> None:
    a = embedder.embed_query("game design portfolio")
    b = embedder.embed_query("game design portfolio")
    assert np.allclose(a, b)


@pytest.mark.parametrize(
    "preset_name, min_gap",
    [
        # minilm spreads cosines widely (gap ~0.61 on this pair).
        ("minilm", 0.15),
        # e5 compresses cosines into ~0.7-0.9 (config.py) — related 0.882 vs
        # unrelated 0.748, gap ~0.134 — the same property that stops e5 from
        # self-gating (see MODEL_PRESETS["e5"]["gate_model"]). 0.15 is a
        # minilm-specific number; use a smaller margin that still proves the
        # (weaker but real) separation, with headroom below the measured gap.
        ("e5", 0.10),
    ],
)
def test_semantic_neighbors_beat_strangers(preset_name, min_gap) -> None:
    embedder = _embedder_for(preset_name)
    query = embedder.embed_query("combat mechanics and fighting systems in games")
    related = embedder.embed_query("designing melee combat gameplay")
    unrelated = embedder.embed_query("grading policy for late homework submissions")
    assert float(query @ related) > float(query @ unrelated) + min_gap


def test_documents_and_query_share_one_code_path() -> None:
    # Docs are embedded one at a time, unpadded — the same way the browser
    # embeds queries — so both sides of the dot product are exact matches.
    # That equality holds only when query_prefix and passage_prefix are both
    # empty: true of minilm, not of e5 ("query: "/"passage: "). This is a
    # minilm-specific guarantee, not a general embedder property, so pin the
    # preset explicitly rather than inherit the ambient RAG_MODEL_PRESET.
    embedder = _embedder_for("minilm")
    texts = [f"sample text number {i} about games" for i in range(3)]
    docs = embedder.embed_documents(texts)
    queries = np.vstack([embedder.embed_query(t) for t in texts])
    assert np.array_equal(docs, queries)
