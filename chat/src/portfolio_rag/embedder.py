"""ONNX embedding service.

Runs the exact model file the browser widget uses (models/Xenova/
all-MiniLM-L6-v2/onnx/model_quantized.onnx) through onnxruntime, with the
same mean-pooling + L2-normalize recipe transformers.js applies with
{pooling: "mean", normalize: true}. Same weights + same pooling = document
and query vectors are directly comparable by dot product.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_embedder = None


class OnnxEmbedder:
    def __init__(self, model_dir: Path, max_tokens: int = 256) -> None:
        logger.info("Loading ONNX embedding model from %s", model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _run_one(self, text: str) -> np.ndarray:
        # One text per run, no padding — deliberately. The model is dynamically
        # quantized, so padded batches shift activation quantization scales and
        # produce slightly different vectors than the browser widget, which
        # embeds queries one at a time. Single-text runs keep both sides exact.
        enc = self.tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        (last_hidden,) = self.session.run(["last_hidden_state"], feed)

        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (last_hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        mean_pooled = summed / counts
        norms = np.clip(np.linalg.norm(mean_pooled, axis=1, keepdims=True), 1e-12, None)
        return (mean_pooled / norms).astype(np.float32)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return np.vstack([self._run_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._run_one(text)


def get_embedder() -> OnnxEmbedder:
    global _embedder
    if _embedder is None:
        from portfolio_rag.config import settings

        _embedder = OnnxEmbedder(
            settings.resolve_path(settings.embedding_model_dir),
            max_tokens=settings.embedding_max_tokens,
        )
    return _embedder
