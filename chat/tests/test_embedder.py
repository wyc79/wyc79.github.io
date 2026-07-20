import numpy as np
import pytest

from portfolio_rag.embedder import get_embedder


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def test_dimensions_and_unit_norm(embedder) -> None:
    vecs = embedder.embed_documents(["combat design", "engine programming in C++"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


def test_deterministic(embedder) -> None:
    a = embedder.embed_query("game design portfolio")
    b = embedder.embed_query("game design portfolio")
    assert np.allclose(a, b)


def test_semantic_neighbors_beat_strangers(embedder) -> None:
    query = embedder.embed_query("combat mechanics and fighting systems in games")
    related = embedder.embed_query("designing melee combat gameplay")
    unrelated = embedder.embed_query("grading policy for late homework submissions")
    assert float(query @ related) > float(query @ unrelated) + 0.15


def test_documents_and_query_share_one_code_path(embedder) -> None:
    # Docs are embedded one at a time, unpadded — the same way the browser
    # embeds queries — so both sides of the dot product are exact matches.
    texts = [f"sample text number {i} about games" for i in range(3)]
    docs = embedder.embed_documents(texts)
    queries = np.vstack([embedder.embed_query(t) for t in texts])
    assert np.array_equal(docs, queries)
