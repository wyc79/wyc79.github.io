"""Rebuild the static retrieval index after editing site content.

    cd chat && python scripts/build_index.py

Outputs data/index.json (chunks + vectors) and data/roles.json (personas),
both committed to the repo and served by GitHub Pages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_rag.embedder import get_embedder  # noqa: E402
from portfolio_rag.index_builder import build_index  # noqa: E402

HEADER = f"{'pages':>6}{'sections':>10}{'chunks':>8}{'index_kb':>10}{'seconds':>9}"


def main() -> None:
    get_embedder()  # load the model up front so timing reflects the build only
    stats = build_index()
    print(HEADER)
    print("-" * len(HEADER))
    print(
        f"{stats['pages']:>6}{stats['sections']:>10}{stats['chunks']:>8}"
        f"{stats['index_kb']:>10}{stats['elapsed_seconds']:>9.3f}"
    )


if __name__ == "__main__":
    main()
