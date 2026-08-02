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

# Every "n_*" key a shared cell can carry WITHOUT a not-yet-migrated negative
# in it: the two bookkeeping counts, the three adjacency buckets, and (Task
# 28) each bucket's n_<bucket>_intent sub-count (see evaluation.aggregate).
# Anything ELSE starting with "n_" is a dynamic leftover bucket for a
# negative with no valid adjacency -- reported as a count of excluded cases
# rather than silently folded into a bucket it doesn't belong to. The
# n_*_intent keys must be listed here too: they are a legitimate mechanism
# sub-count, not an unmigrated case, and would otherwise misreport as one.
_SHARED_N_KEYS = {
    "n_positive", "n_negative", "n_easy", "n_adjacent", "n_injection",
    "n_easy_intent", "n_adjacent_intent", "n_injection_intent",
}


def print_gate_summary(gate_meta: dict) -> None:
    """Build-time gate calibration, per language -- stat/threshold/margin,
    plus (Task 28) the second-tier task_threshold/task_margin, side by side
    with the base columns rather than blended into them.

    A language with no bundle at all (e.g. no zh gate) is simply absent from
    gate_meta and prints no row here; that's covered separately by the
    "note: no zh gate" message in main(). margin/task_margin specifically
    render "n/a" (never 0 or 0.0%) whenever the underlying artifact predates
    task 20's gate_margin field, or (task_* columns) whenever this build's
    calibration had too few TASK_REQUEST_RE-flagged on-topic queries to place
    a second threshold honestly -- see evaluation.format_margin and
    gate_calibration.compute_task_gate's MIN_FLAGGED_ON_TOPIC. "n/a" here
    means "this gate has no second tier," never "it refuses nothing."
    """
    if not gate_meta:
        return
    print(f"\n{'gate calibration':<10} {'stat':>10} {'threshold':>10} {'margin':>10} "
          f"{'task_thr':>10} {'task_margin':>12}")
    print("-" * 68)
    for lang in sorted(gate_meta):
        meta = gate_meta[lang]
        task_threshold = meta.get("task_threshold")
        task_thr = "n/a" if task_threshold is None else f"{task_threshold:.4f}"
        print(f"{lang:<10} {meta['stat']:>10} {meta['threshold']:>10.4f} "
              f"{format_margin(meta['margin']):>10} {task_thr:>10} "
              f"{format_margin(meta.get('task_margin')):>12}")


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
    unaccounted = 0
    for name in sorted(cells):
        c = cells[name]
        if c["gate_available"]:
            easy = f"{c['refusal_easy']}/{c['n_easy']}"
            adjacent = f"{c['refusal_adjacent']}/{c['n_adjacent']}"
            injection = f"{c['refusal_injection']}/{c['n_injection']}"
        else:
            easy = adjacent = injection = "n/a"
        print(f"{c['lang']:<16} {easy:>16} {adjacent:>20} {injection:>12}")
        unaccounted += sum(v for k, v in c.items()
                            if k.startswith("n_") and k not in _SHARED_N_KEYS)
    print("-" * 68)
    if unaccounted:
        print(f"note: {unaccounted} negative case(s) have no valid adjacency (an off_topic "
              "case without 'easy'/'adjacent') and are excluded from the buckets above -- "
              "run with --json to see them.")


def print_mechanism_table(cells: dict) -> None:
    """Task 28: the SAME off_topic/easy, off_topic/adjacent, injection
    columns as print_shared_table above, cross-cut by refusal MECHANISM
    instead of totalled by it -- topic (refused by the ordinary threshold)
    vs intent (refused by the higher task_threshold; TASK_REQUEST_RE flagged
    this case's gated text). For any bucket, the topic and intent numerators
    sum to the total print_shared_table already showed; this table exists
    because "the gate can't see this as off-topic at all" and "the gate sees
    it as on-topic-shaped but ALSO a work request" are different findings
    with different fixes, and the totalled number above cannot distinguish
    them -- never blend, this project's standing rule (see
    print_shared_table's own docstring)."""
    print(f"\n{'shared negatives (mechanism)':<16} {'off_topic/easy':>16} "
          f"{'off_topic/adjacent':>20} {'injection':>12}")
    print("-" * 68)
    for name in sorted(cells):
        c = cells[name]
        if not c["gate_available"]:
            print(f"{c['lang']:<16} {'n/a':>16} {'n/a':>20} {'n/a':>12}")
            continue
        for mech in ("topic", "intent"):
            cols = []
            for bucket in ("easy", "adjacent", "injection"):
                n_intent = c.get(f"n_{bucket}_intent", 0)
                n = n_intent if mech == "intent" else c[f"n_{bucket}"] - n_intent
                refused = c.get(f"refusal_{bucket}_{mech}", 0)
                cols.append(f"{refused}/{n}")
            label = f"{c['lang']} {mech}"
            print(f"{label:<16} {cols[0]:>16} {cols[1]:>20} {cols[2]:>12}")
    print("-" * 68)
    print("topic + intent numerators sum to the totals in the table above; a bucket's")
    print("intent DENOMINATOR is how many of its cases TASK_REQUEST_RE flagged at all,")
    print("whether refused or not -- 'n/a' rows have no measured gate to break out.")


def print_verbose(results: list) -> None:
    for r in results:
        c = r.case
        mark = "OK  " if r.ok else "FAIL"
        gate = "bypass" if r.gate_value is None else f"{r.gate_value:.4f}"
        # Task 28: name the MECHANISM inline whenever the intent tier fired --
        # "REFUSED (intent)" and "REFUSED (topic)" point at different fixes,
        # and print_shared_table/print_mechanism_table already report this in
        # aggregate; --verbose surfaces it per-case for the same reason.
        reason_tag = f" [{r.reason}]" if r.reason else ""
        print(f"\n[{mark}] {c.id}  ({c.type})")
        print(f"  q:    {c.q}")
        print(f"  gate: {'pass' if r.gate_passed else 'REFUSED'} (value {gate}){reason_tag}")
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
        print("retrieval model missing (see index.json model_preset) -- "
              "cannot evaluate", file=sys.stderr)
        return 2
    if not rt.zh_gate_available:
        print("note: no zh gate (data/gate_vectors.json absent) -- Chinese gate "
              "metrics report n/a. Chinese hit@4 is unaffected.", file=sys.stderr)
    stale_headings = rt.stale_knowledge_headings
    if stale_headings:
        shown = ", ".join(sorted(stale_headings)[:5])
        more = f" (+{len(stale_headings) - 5} more)" if len(stale_headings) > 5 else ""
        print(
            f"note: {len(stale_headings)} knowledge-corpus heading(s) baked into "
            "data/index.json no longer match chat/knowledge/about_*.md on disk -- "
            "hit@4(pg) may be understating the real page-only number (see "
            f"Runtime.stale_knowledge_headings). Rebuild the index. Stale: {shown}{more}",
            file=sys.stderr,
        )

    if args.role:
        # Negatives carry no role (SHARED_ROLE) -- --role filters positive
        # cells only. Applying it to negatives too would show a "shared"
        # block that is not actually the shared pool, just whatever fraction
        # of it happens to still carry this role in the not-yet-migrated
        # dataset, which is worse than not filtering at all.
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
            print_mechanism_table(shared_cells)
        print_gate_summary(rt.gate_meta)
        print("\nMetrics are reported side by side on purpose -- a blended score cannot")
        print("distinguish a gate problem from a retrieval problem, and hit@4 passing")
        print("while keywords fail means the right PAGE came back with the wrong chunk.")
        print("The shared negatives block is never blended with positives, and its three")
        print("adjacency buckets are never blended with each other -- see eval/README.md.")
        print("The mechanism block (Task 28) is a second, orthogonal cut of the SAME")
        print("refusals -- topic (similarity alone) vs intent (the task-request tier) --")
        print("never blended with adjacency or with each other: 'refused as off-topic'")
        print("and 'refused because it asked me to do work' have different fixes.")
        print("hit@4(pg) excludes chat/knowledge/*.md's curated sections from retrieval --")
        print("a gap between hit@4 and hit@4(pg) means the corpus, not the site's own")
        print("pages, is answering the question (task 24 review, Critical 1).\n")

    if args.update_baseline:
        if args.role or args.lang:
            print("refusing to write a baseline from a filtered run -- "
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
