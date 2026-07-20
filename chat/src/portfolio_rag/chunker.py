import re


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character chunker (same contract as the lesson04 chunker).

    Sections from the HTML loader are usually smaller than one chunk; this
    only kicks in for long project pages, where the overlap keeps sentences
    that straddle a boundary retrievable from both sides.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
