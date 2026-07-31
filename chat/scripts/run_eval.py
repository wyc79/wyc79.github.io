"""Run the golden set and report gate + retrieval metrics.

    python scripts/run_eval.py
    python scripts/run_eval.py --role combat_design_recruiter --lang zh --verbose
    python scripts/run_eval.py --update-baseline
    python scripts/run_eval.py --json

This is evaluation, not testing: it prints scores. The pass/fail gate lives in
tests/test_golden.py, which compares these numbers against data/eval_baseline.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from portfolio_rag.evaluation import (
    BASELINE_PATH, GOLDEN_PATH, aggregate, load_cases, run_cases,
)
from portfolio_rag.runtime import TOP_K, load_runtime


def build_baseline(rt, cells: dict) -> dict:
    return {
        "k": TOP_K,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index_built_at": rt.index_built_at,
        "gate": rt.gate_meta,
        "cells": cells,
    }


def print_table(cells: dict) -> None:
    print(f"\n{'cell':<40} {'gate':>7} {'refuse':>7} {'hit@4':>7} {'keywords':>9}  retrieved")
    print("-" * 96)
    totals = {"gate_pass": 0, "n_positive": 0, "refusal": 0, "n_negative": 0,
              "hit_at_4": 0, "keywords_found": 0, "keywords_total": 0}
    for name in sorted(cells):
        c = cells[name]
        for key in totals:
            totals[key] += c[key]
        gate = "n/a" if not c["gate_available"] else f"{c['gate_pass']}/{c['n_positive']}"
        refuse = "n/a" if not c["gate_available"] else f"{c['refusal']}/{c['n_negative']}"
        hits = f"{c['hit_at_4']}/{c['n_positive']}"
        kw = f"{c['keywords_found']}/{c['keywords_total']}"
        langs = " ".join(f"{k}:{v}" for k, v in sorted(c["retrieved_langs"].items()))
        print(f"{name:<40} {gate:>7} {refuse:>7} {hits:>7} {kw:>9}  {langs}")
    print("-" * 96)
    print(f"{'TOTAL':<40} "
          f"{totals['gate_pass']}/{totals['n_positive']:<5} "
          f"{totals['refusal']}/{totals['n_negative']:<5} "
          f"{totals['hit_at_4']}/{totals['n_positive']:<5} "
          f"{totals['keywords_found']}/{totals['keywords_total']}")
    print("\nMetrics are reported side by side on purpose — a blended score cannot")
    print("distinguish a gate problem from a retrieval problem, and hit@4 passing")
    print("while keywords fail means the right PAGE came back with the wrong chunk.\n")


def print_verbose(results: list) -> None:
    for r in results:
        c = r.case
        mark = "OK  " if r.ok else "FAIL"
        gate = "bypass" if r.gate_value is None else f"{r.gate_value:.4f}"
        print(f"\n[{mark}] {c.id}  ({c.type})")
        print(f"  q:    {c.q}")
        print(f"  gate: {'pass' if r.gate_passed else 'REFUSED'} (value {gate})")
        if c.type == "positive":
            print(f"  want: {', '.join(c.expected_urls)}")
            for url, score in zip(r.top_urls, r.top_scores):
                print(f"        {score:.4f}  {url}")
            if r.dropped_by_floor:
                print(f"        ({r.dropped_by_floor} more dropped below the 0.18 floor)")
            print(f"  keys: found {list(r.keywords_found)}")
            if r.keywords_missing:
                print(f"        MISSING {list(r.keywords_missing)}"
                      f"{'  <- right page, wrong chunk' if r.hit else ''}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role")
    ap.add_argument("--lang", choices=("en", "zh"))
    ap.add_argument("--verbose", action="store_true", help="per-case detail")
    ap.add_argument("--json", action="store_true", help="machine-readable cells")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    rt = load_runtime()
    if not rt.retrieval_available:
        print("retrieval model missing (see index.json model_preset) — "
              "cannot evaluate", file=sys.stderr)
        return 2
    if not rt.zh_gate_available:
        print("note: no zh gate (data/gate_vectors.json absent) — Chinese gate "
              "metrics report n/a. Chinese hit@4 is unaffected.", file=sys.stderr)

    cases = load_cases(GOLDEN_PATH)
    selected = [c for c in cases
                if (not args.role or c.role == args.role)
                and (not args.lang or c.lang == args.lang)]
    if not selected:
        print("no cases matched the filter", file=sys.stderr)
        return 2

    results = run_cases(rt, selected)
    cells = aggregate(results)

    if args.verbose:
        print_verbose(results)
    if args.json:
        print(json.dumps(cells, ensure_ascii=False, indent=2))
    else:
        print_table(cells)

    if args.update_baseline:
        if args.role or args.lang:
            print("refusing to write a baseline from a filtered run — "
                  "run without --role/--lang", file=sys.stderr)
            return 2
        BASELINE_PATH.write_text(
            json.dumps(build_baseline(rt, cells), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"baseline written to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
