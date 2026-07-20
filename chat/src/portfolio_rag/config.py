from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Build-time settings. Env vars use the RAG_ prefix (see .env.example)."""

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # The same ONNX export the browser widget loads — this is what guarantees
    # that document vectors (built here) and query vectors (computed in the
    # visitor's browser by transformers.js) live in the same embedding space.
    embedding_model_dir: str = "models/Xenova/all-MiniLM-L6-v2"
    embedding_max_tokens: int = 256

    index_path: str = "data/index.json"
    roles_path: str = "data/roles.json"
    vector_round_decimals: int = 6

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="RAG_"
    )

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
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
