from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_PRESETS: dict[str, dict] = {
    "minilm": {
        "dir": "models/Xenova/all-MiniLM-L6-v2",
        "name": "Xenova/all-MiniLM-L6-v2 (quantized ONNX, mean pooling, normalized)",
        "hf_repo": "Xenova/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "multilingual": False,
    },
    "e5": {
        "dir": "models/Xenova/multilingual-e5-small",
        "name": "Xenova/multilingual-e5-small (quantized ONNX, mean pooling, normalized, query:/passage: prefixes)",
        "hf_repo": "Xenova/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "multilingual": True,
        # e5 compresses cosines into ~0.7-0.9 and cannot separate on/off-topic
        # (measured: negative margins on every statistic), so the off-topic
        # gate keeps using MiniLM — served by the backend, which gets a
        # MiniLM copy of the curated on-topic corpus + threshold via
        # gate_en_minilm_path (data/gate_en_minilm.json).
        "gate_model": "minilm",
        # Chinese first-pass gate: bge-zh scores zh queries against the
        # hand-written zh gate corpus (knowledge/about_zh.md). Enabled by the build
        # only if its calibration actually separates on/off-topic.
        "gate_model_zh": "bge_zh",
    },
    # Gate-only model (never used for the retrieval index): zh-specialized,
    # CLS pooling, bge query instruction. dim 512.
    "bge_zh": {
        "dir": "models/Xenova/bge-small-zh-v1.5",
        "name": "Xenova/bge-small-zh-v1.5 (quantized ONNX, cls pooling, normalized)",
        "hf_repo": "Xenova/bge-small-zh-v1.5",
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
        "passage_prefix": "",
        "pooling": "cls",
        "multilingual": False,
    },
}


class Settings(BaseSettings):
    """Build-time settings. Env vars use the RAG_ prefix (see .env.example)."""

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # Which embedding model the whole system uses. Query vectors (browser
    # widget or Tencent function) and document vectors (built here) must come
    # from the same model — presets keep dir/prefixes/flags consistent.
    #   minilm — self-hosted in the browser, English (the static-site default)
    #   e5     — multilingual-e5-small, served by the Tencent function
    model_preset: str = "minilm"
    embedding_max_tokens: int = 256

    # Retrieval corpus (Task 29 Part 2): one file, one job -- chunks + vectors
    # for the model_preset actually configured, nothing else (no gate fields).
    # None (the default) DERIVES "data/chunks_{model_preset}.json" from
    # model_preset -- see resolve_chunks_path() -- so the filename can never
    # silently mismatch the preset that built it, the exact bug class that
    # motivated splitting this file out in the first place. Set explicitly
    # (like every other path below) so tests can confine writes to tmp_path.
    chunks_path: str | None = None
    roles_path: str = "data/roles.json"
    # Small metadata sidecar (Task 29 Part 1): the fields the widget used to
    # read off index.json (gate_threshold, gate_stat, gate_remote, model,
    # query_prefix) so it can load ~1KB instead of the multi-MB chunk index on
    # every visit when it doesn't need the chunks themselves (normal mode,
    # backend up). Task 29 Part 2 adds chunks_file, naming the retrieval
    # corpus above so light mode can fetch it without inferring a filename
    # from `model` itself.
    meta_path: str = "data/meta.json"
    # English off-topic gate (Task 29 Part 2): MiniLM vectors over
    # knowledge/about_en.md's curated sections ONLY -- no chunk ids, no chunk
    # text, no retrieval fields. Committed: e5 can't self-gate, so this file
    # must always be present, unlike gate_zh_bge_path below. Replaces both
    # gate_vectors.json's "en" entry and fallback_vectors.json (which used to
    # duplicate it for the widget's degraded mode).
    gate_en_minilm_path: str = "data/gate_en_minilm.json"
    # Chinese off-topic gate: bge-zh vectors over knowledge/about_zh.md.
    # Gitignored (see chat/.gitignore) -- MUST NEVER be served to a browser
    # (MiniLM, the only in-browser model, cannot embed Chinese at all; the
    # widget's degraded mode is English-only by design, see chat-widget.js's
    # degradedCJK). Gitignoring it is the enforcement that does not depend on
    # widget behaviour, mirroring why gate_vectors.json was gitignored before
    # it split into this file and gate_en_minilm_path above.
    gate_zh_bge_path: str = "data/gate_zh_bge.json"
    # Degraded-mode RETRIEVAL corpus (Task 29 Part 2): the SAME 192 English
    # chunks as chunks_{e5}.json's lang=="en" entries (same ids, same order,
    # same text -- see index_builder.py), re-embedded with MiniLM so the
    # browser's self-hosted model can score them when the e5-dependent
    # backend is unreachable. Committed, unlike gate_zh_bge_path -- this is
    # exactly what fixes degraded mode's source links (see chat-widget.js's
    # retrieveFallback): resolve real chunk records by id, never by a
    # position borrowed from an unrelated array.
    chunks_en_minilm_path: str = "data/chunks_en_minilm.json"
    vector_round_decimals: int = 6

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="RAG_"
    )

    @property
    def preset(self) -> dict:
        return MODEL_PRESETS[self.model_preset]

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.model_preset not in MODEL_PRESETS:
            raise ValueError(f"unknown model_preset {self.model_preset!r}; options: {list(MODEL_PRESETS)}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def chat_root(self) -> Path:
        """The chat/ directory (this package lives at chat/src/portfolio_rag)."""
        return Path(__file__).resolve().parents[2]

    @property
    def site_root(self) -> Path:
        """The repository root, which is also the GitHub Pages web root."""
        return self.chat_root.parent

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.chat_root / candidate

    def resolve_chunks_path(self) -> Path:
        """The retrieval corpus's real path: chunks_path if explicitly set
        (tests use this to confine writes to tmp_path), otherwise DERIVED
        from model_preset -- "data/chunks_{model_preset}.json" -- so a light
        `--model minilm` build writes chunks_minilm.json and the production
        e5 build writes chunks_e5.json, never a name that could disagree with
        what's actually inside (see tests/test_data_file_layout.py's
        name-matches-contents checks)."""
        return self.resolve_path(self.chunks_path or f"data/chunks_{self.model_preset}.json")


settings = Settings()
