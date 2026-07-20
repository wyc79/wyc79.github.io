import pytest

from portfolio_rag.chunker import chunk_text

TEXT = "abcdefghij" * 30  # 300 chars, no whitespace to normalize


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_rejects_illegal_params(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text(TEXT, chunk_size, overlap)


def test_returns_empty_for_blank_text() -> None:
    assert chunk_text("", 100, 20) == []
    assert chunk_text("   \n\t ", 100, 20) == []


def test_returns_single_chunk_when_text_fits() -> None:
    assert chunk_text("hello world", 100, 20) == ["hello world"]


def test_collapses_whitespace() -> None:
    assert chunk_text("Prime\n\nEngine   demo", 100, 20) == ["Prime Engine demo"]


def test_chunks_cover_the_whole_text() -> None:
    chunks = chunk_text(TEXT, 100, 20)
    assert chunks[0] == TEXT[:100]
    assert TEXT.endswith(chunks[-1])
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_text(TEXT, 100, 20)
    for current, following in zip(chunks, chunks[1:]):
        assert current[-20:] == following[:20]
