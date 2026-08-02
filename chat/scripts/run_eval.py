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

from portfolio_rag.evaluation import (
    BASELINE_PATH, GOLDEN_PATH, SHARED_ROLE, aggregate, build_baseline, format_margin,
    load_cases, run_cases,
)
from portfolio_rag.runtime import load_runtime

def print_gate_summary(gate_meta: dict) -> None:
    """Build-time gate calibration, per language -- stat/threshold/margin.

    A language with no bundle at all (e.g. no zh gate) is simply absent from
    gate_meta and prints no row here; that's covered separately by the
    "note: no zh gate" message in main(). margin specifically renders "n/a"
    (never 0 or 0.0%) whenever the underlying artifact predates task 20's
    gate_margin field -- see evaluation.format_margin.
    """
    if not gate_meta:
        return
    print(f"\n{'gate calibration':<10} {'stat':>10} {'threshold':>10} {'margin':>10}")
    print("-" * 44)
    for lang in sorted(gate_meta):
        meta = gate_meta[lang]
        print(f"{lang:<10} {meta['stat']:>10} {meta['threshold']:>10.4f} "
              f"{format_margin(meta['margin']):>10}")


def print_positive_table(cells: dict) -> None:
    """role/lang cells: gate-pass and retrieval health. No refusal column --
    negatives no longer live here, see print_shared_table.

    hit@4(pg) is the task-24-review page-only diagnostic: hit@4 re-ranked
    with chat/knowledge/about_<lang>.md's curated sections excluded from the
    candidate pool (see Runtime.knowledge_chunk_ids), i.e. "what the site's
    OWN pages would retrieve with zero curated-corpus assist." Printed next
    to hit@4 so a corpus change that inflates the full number by retrieving
    its own authored content cannot pass unnoticed -- it is a reported
    diagnostic, never folded into hit@4, and not compared by
    tests/test_golden.py's regression check (see aggregate()'s comment).
    """
    print(f"\n{'positive cells':<40} {'gate':>7} {'hit@4':>7} {'hit@4(pg)':>9} "
          f"{'keywords':>9}  retrieved")
    print("-" * 100)
    totals = {"gate_pass": 0, "n_positive": 0, "hit_at_4": 0, "hit_at_4_page_only": 0,
              "keywords_found": 0, "keywords_total": 0}
    # Gate column is summed separately, over gate_available cells only: a
    # cjk_bypass decision always reports gate_passed=True, so blending it into
    # the total would silently pass off meaningless data as a real number.
    gate_totals = {"gate_pass": 0, "n_positive": 0}
    n_available, n_excluded = 0, 0
    for name in sorted(cells):
        c = cells[name]
        for key in totals:
            totals[key] += c[key]
        if c["gate_available"]:
            n_available += 1
            for key in gate_totals:
                gate_totals[key] += c[key]
        else:
            n_excluded += 1
        gate = "n/a" if not c["gate_available"] else f"{c['gate_pass']}/{c['n_positive']}"
        hits = f"{c['hit_at_4']}/{c['n_positive']}"
        hits_pg = f"{c['hit_at_4_page_only']}/{c['n_positive']}"
        kw = f"{c['keywords_found']}/{c['keywords_total']}"
        langs = " ".join(f"{k}:{v}" for k, v in sorted(c["retrieved_langs"].items()))
        print(f"{name:<40} {gate:>7} {hits:>7} {hits_pg:>9} {kw:>9}  {langs}")
    print("-" * 100)
    if n_available == 0:
        gate_total = "n/a"
    else:
        mark = "*" if n_excluded else ""
        gate_total = f"{gate_totals['gate_pass']}/{gate_totals['n_positive']}{mark}"
    hits_total = f"{totals['hit_at_4']}/{totals['n_positive']}"
    hits_pg_total = f"{totals['hit_at_4_page_only']}/{totals['n_positive']}"
    kw_total = f"{totals['keywords_found']}/{totals['keywords_total']}"
    print(f"{'TOTAL':<40} {gate_total:>7} {hits_total:>7} {hits_pg_total:>9} {kw_total:>9}")
    if n_available and n_excluded:
        print(f"* gate totals exclude {n_excluded} cell(s) with no gate available "
              "(see n/a rows)")


def print_shared_table(cells: dict) -> None:
    """shared/lang cells: refusal broken out by adjacency bucket. Never
    blended with the positive table above, and the three buckets are never
    blended with each other -- a gate that refuses easy off-topic but not
    adjacent off-topic is broken in a completely different way than one that
    refuses nothing."""
    print(f"\n{'shared negatives':<16} {'off_topic/easy':>16} {'off_topic/adjacent':>20} "
          f"{'injection':>12}")
    print("-" * 68)
    for name in sorted(cells):
        c = cells[name]
        if c["gate_available"]:
            easy = f"{c['refusal_easy']}/{c['n_easy']}"
            adjacent = f"{c['refusal_adjacent']}/{c['n_adjacent']}"
            injection = f"{c['refusal_injection']}/{c['n_injection']}"
        else:
            easy = adjacent = injection = "n/a"
        print(f"{c['lang']:<16} {easy:>16} {adjacent:>20} {injection:>12}")
    print("-" * 68)


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
    ap.add_argument(
        "--allow-stale-index", action="store_true",
        help="write the baseline even though the committed chunks file's knowledge-corpus "
             "headings no longer match chat/knowledge/about_*.md on disk (see "
             "Runtime.stale_knowledge_headings). Only for deliberately freezing a "
             "known-stale state.",
    )
    args = ap.parse_args()

    rt = load_runtime()
    if not rt.retrieval_available:
        print("retrieval model missing (see the chunks file's model_preset) -- "
              "cannot evaluate", file=sys.stderr)
        return 2
    if not rt.zh_gate_available:
        print("note: no zh gate (data/gate_zh_bge.json absent) -- Chinese gate "
              "metrics report n/a. Chinese hit@4 is unaffected.", file=sys.stderr)
    stale_headings = rt.stale_knowledge_headings
    if stale_headings:
        shown = ", ".join(sorted(stale_headings)[:5])
        more = f" (+{len(stale_headings) - 5} more)" if len(stale_headings) > 5 else ""
        print(
            f"note: {len(stale_headings)} knowledge-corpus heading(s) baked into "
            "the committed chunks file no longer match chat/knowledge/about_*.md "
            "on disk -- hit@4(pg) may be understating the real page-only number "
            f"(see Runtime.stale_knowledge_headings). Rebuild the index. "
            f"Stale: {shown}{more}",
            file=sys.stderr,
        )

    if args.role:
        # Negatives carry no role (SHARED_ROLE) -- --role filters positive
        # cells only. Applying it to negatives too would print a "shared"
        # block that is not the shared pool at all, just the slice of it the
        # filter happened to keep, under a heading that says otherwise --
        # worse than not filtering it.
        print(f"note: --role={args.role} filters positive cells only; negatives have no "
              "role and the shared block below always covers the full pool for the "
              "selected language(s).", file=sys.stderr)

    cases = load_cases(GOLDEN_PATH)
    selected = [c for c in cases
                if (not args.role or c.type != "positive" or c.role == args.role)
                and (not args.lang or c.lang == args.lang)]
    if not selected:
        print("no cases matched the filter", file=sys.stderr)
        return 2

    results = run_cases(rt, selected)
    cells = aggregate(results)
    positive_cells = {k: v for k, v in cells.items() if v["role"] != SHARED_ROLE}
    shared_cells = {k: v for k, v in cells.items() if v["role"] == SHARED_ROLE}

    if args.verbose:
        print_verbose(results)
    if args.json:
        print(json.dumps(cells, ensure_ascii=False, indent=2))
    else:
        if positive_cells:
            print_positive_table(positive_cells)
        if shared_cells:
            print_shared_table(shared_cells)
        print_gate_summary(rt.gate_meta)
        print("\nMetrics are reported side by side on purpose -- a blended score cannot")
        print("distinguish a gate problem from a retrieval problem, and hit@4 passing")
        print("while keywords fail means the right PAGE came back with the wrong chunk.")
        print("The shared negatives block is never blended with positives, and its three")
        print("adjacency buckets are never blended with each other -- see eval/README.md.")
        print("hit@4(pg) excludes chat/knowledge/*.md's curated sections from retrieval --")
        print("a gap between hit@4 and hit@4(pg) means the corpus, not the site's own")
        print("pages, is answering the question (task 24 review, Critical 1).\n")

    if args.update_baseline:
        if args.role or args.lang:
            print("refusing to write a baseline from a filtered run -- "
                  "run without --role/--lang", file=sys.stderr)
            return 2
        # Same refusal, same reason, for the other way a baseline can be
        # captured from a state that does not represent the system. The stale
        # note printed above (:171-182) was advisory, and a printed warning is
        # not a guard when the thing being warned about is a write:
        # Runtime.stale_knowledge_headings' own docstring measures the damage
        # -- renaming 5 about_en.md headings with no rebuild moved
        # knowledge_chunk_ids 108 -> 103 and hit_at_4_page_only 59/96 -> 56/96,
        # a 3-case move in an unpredictable direction. data/eval_baseline.json
        # is what tests/test_golden.py::test_no_metric_regressed compares
        # against, so freezing it here silently re-anchors the project's own
        # success criterion to a broken measurement.
        if stale_headings and not args.allow_stale_index:
            print(
                f"refusing to write a baseline while {len(stale_headings)} knowledge-corpus "
                "heading(s) are stale (see the note above) -- the committed chunks file no "
                "longer matches chat/knowledge/about_*.md, so hit@4(pg) is measuring a state "
                "that does not exist. Rebuild the index first: python scripts/build_index.py. "
                "Pass --allow-stale-index to freeze this state deliberately.",
                file=sys.stderr,
            )
            return 2
        BASELINE_PATH.write_text(
            json.dumps(build_baseline(rt, cells), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"baseline written to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
