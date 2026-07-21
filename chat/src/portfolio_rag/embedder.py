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
    def __init__(
        self,
        model_dir: Path,
        max_tokens: int = 256,
        query_prefix: str = "",
        passage_prefix: str = "",
        pooling: str = "mean",  # "mean" (MiniLM/e5) or "cls" (bge family)
    ) -> None:
        logger.info("Loading ONNX embedding model from %s (pooling=%s)", model_dir, pooling)
        # e5-style models are trained with asymmetric prefixes; MiniLM uses none.
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        # Deterministic, reproducible embeddings. Dynamically-quantized MatMul is
        # otherwise nondeterministic run-to-run (CPU memory-arena reuse can corrupt
        # a few activations), which flakes gate calibration and tests/test_gate.py
        # near the threshold. This module runs only at build/test time on tiny
        # single-text inputs, so single-threaded + no arena costs nothing.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"),
            so,
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

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        else:
            mask = attention_mask[:, :, None].astype(np.float32)
            summed = (last_hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return (pooled / norms).astype(np.float32)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return np.vstack([self._run_one(self.passage_prefix + t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._run_one(self.query_prefix + text)

    @classmethod
    def from_preset(cls, preset: dict, model_dir, max_tokens: int) -> "OnnxEmbedder":
        """Build from a MODEL_PRESETS entry — the one place prefixes/pooling are unpacked."""
        return cls(
            model_dir,
            max_tokens=max_tokens,
            query_prefix=preset["query_prefix"],
            passage_prefix=preset["passage_prefix"],
            pooling=preset.get("pooling", "mean"),
        )


def get_embedder() -> OnnxEmbedder:
    global _embedder
    if _embedder is None:
        from portfolio_rag.config import settings

        preset = settings.preset
        _embedder = OnnxEmbedder.from_preset(
            preset,
            settings.resolve_path(preset["dir"]),
            settings.embedding_max_tokens,
        )
    return _embedder
