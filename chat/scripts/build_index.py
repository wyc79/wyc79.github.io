"""Rebuild the static retrieval index after editing site content.

    cd chat && python scripts/build_index.py [--model minilm|e5]

--model e5 requires the multilingual model at chat/models/Xenova/
multilingual-e5-small (fetched by functions/tencent/build_package.py).
Outputs data/index.json (chunks + vectors + calibrated gate threshold) and
data/roles.json, both committed to the repo and served by GitHub Pages.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HEADER = f"{'pages':>6}{'sections':>10}{'chunks':>8}{'index_kb':>10}{'gate':>8}{'seconds':>9}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["minilm", "e5"], default=None,
                        help="embedding model preset (default: RAG_MODEL_PRESET env or minilm)")
    args = parser.parse_args()
    if args.model:
        os.environ["RAG_MODEL_PRESET"] = args.model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from portfolio_rag.config import settings  # noqa: E402 (after env is set)
    from portfolio_rag.embedder import get_embedder  # noqa: E402
    from portfolio_rag.index_builder import build_index  # noqa: E402

    print(f"model preset: {settings.model_preset} ({settings.preset['name']})")
    get_embedder()  # load the model up front so timing reflects the build only
    stats = build_index()
    print(HEADER)
    print("-" * len(HEADER))
    print(
        f"{stats['pages']:>6}{stats['sections']:>10}{stats['chunks']:>8}"
        f"{stats['index_kb']:>10}{stats['gate_threshold']:>8}{stats['elapsed_seconds']:>9.3f}"
    )
    print(f"off-topic gate: {stats['gate_stat']} >= {stats['gate_threshold']}")


if __name__ == "__main__":
    main()
