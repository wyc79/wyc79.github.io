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
        # MiniLM copy of the chunk vectors + threshold via gate_vectors.json.
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

    index_path: str = "data/index.json"
    roles_path: str = "data/roles.json"
    gate_vectors_path: str = "data/gate_vectors.json"
    # Published (committed) MiniLM copy of the chunk vectors: the widget's
    # degraded mode retrieves against these locally when the backend that an
    # e5 index depends on is unreachable.
    fallback_vectors_path: str = "data/fallback_vectors.json"
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


settings = Settings()
