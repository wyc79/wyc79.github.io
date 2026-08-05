"""Record the rewrite each multi-turn golden case produces, for offline replay.

THE ONLY THING IN THIS REPO THAT CALLS A REAL LLM FOR A REWRITE. run_eval.py
and pytest never do -- they replay eval/rewrites.json, which is what keeps the
harness deterministic and offline (the same index and gate vectors produce
bit-identical numbers on every run, and CI needs no API key).

Re-run this ONLY when the rewrite prompt changes, and review the diff: a
rewrite that starts smuggling a project name into an off-topic follow-up is
exactly the regression the post-context negatives exist to catch, and it shows
up here first, in plain text.

Usage (from chat/, with LLM_API_KEY set):

    python scripts/refresh_rewrites.py            # rewrite every multi-turn case
    python scripts/refresh_rewrites.py --dry-run  # print, write nothing
"""

import argparse
import importlib.util
import json
import sys

from portfolio_rag.config import settings
from portfolio_rag.evaluation import REWRITES_PATH, load_cases

BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"


def _backend():
    """Load functions/tencent/index.py directly. It is stdlib-only and cannot be
    imported as a package, so this mirrors what the sync tests already do --
    and using the REAL rewrite_question is the point: a fixture recorded from a
    reimplementation would measure the reimplementation."""
    spec = importlib.util.spec_from_file_location("_scf_index_for_rewrites", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args()

    mod = _backend()
    cases = [c for c in load_cases() if c.history]
    if not cases:
        print("no multi-turn cases in golden.jsonl -- nothing to record")
        return 0

    out: dict[str, str] = {}
    for case in cases:
        text, outcome = mod.rewrite_question(case.messages, case.q)
        out[case.id] = text
        flag = "" if outcome in ("rewritten", "echoed") else f"  <-- {outcome.upper()}"
        print(f"{case.id:<24} {case.q!r}\n{'':<24} -> {text!r}  [{outcome}]{flag}")

    failed = [c.id for c in cases if out[c.id] == c.q and c.type == "positive"]
    if failed:
        print(
            f"\nWARNING: {len(failed)} positive case(s) echoed unchanged: {failed}\n"
            "A positive multi-turn case is one whose question CANNOT stand alone, so an "
            "echo means either the case is miswritten or the prompt stopped resolving.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    REWRITES_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(out)} rewrite(s) to {REWRITES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
