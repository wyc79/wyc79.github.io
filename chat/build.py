#!/usr/bin/env python3
"""Master build for the portfolio chat agent — one command, all artifacts.

    cd chat
    python build.py                # rebuild the SITE artifacts only (fast):
                                   #   data/chunks_e5.json, data/gate_en_minilm.json,
                                   #   data/gate_zh_bge.json, data/chunks_en_minilm.json,
                                   #   data/meta.json, data/roles.json
    python build.py --function     # ALSO (re)build the Tencent SCF zip — downloads
                                   #   any missing models first, then re-zips
    python build.py --model minilm # English-only light-mode index instead of e5

Without --function this runs scripts/build_index.py, which needs the models in
chat/models/ (build_package.py fetches them on the first --function run). With
--function it runs functions/tencent/build_package.py, which downloads models as
needed, rebuilds the retrieval corpus itself, and writes
tencent-function-<preset>.zip.

After building: if you built the function (--function), upload the zip and redeploy it
FIRST — the widget's /chat sends no `contexts`, and an old deployed function 400s
without it. Only then git add/commit/push the data/ files (and knowledge/, roles, etc.
if you changed them) and publish the site. The Next: block this script prints at the
end carries the same instruction plus what going out of order actually costs; that
block is the canonical statement — keep this paragraph in step with it.
"""

import argparse
import subprocess
import sys
from pathlib import Path

CHAT = Path(__file__).resolve().parent


def run(cmd: list) -> None:
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=CHAT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild every chat artifact; optionally repackage the cloud function."
    )
    ap.add_argument("--function", action="store_true",
                    help="also (re)build the Tencent SCF function zip (downloads missing models)")
    ap.add_argument("--model", choices=["e5", "minilm"], default="e5",
                    help="embedding preset (default: e5 — the deployed multilingual model)")
    ap.add_argument("--python-version", default="310",
                    help="SCF runtime ABI for --function wheels (default: 310)")
    args = ap.parse_args()

    if args.function:
        # build_package.py = download models (if missing) + build_index + zip.
        run([sys.executable, str(CHAT / "functions" / "tencent" / "build_package.py"),
             "--preset", args.model, "--python-version", args.python_version])
    else:
        # build_index.py = chunks_{model}.json + meta.json + (for e5)
        # gate_en_minilm.json/gate_zh_bge.json/chunks_en_minilm.json + roles.json.
        run([sys.executable, str(CHAT / "scripts" / "build_index.py"), "--model", args.model])

    site = f"data/chunks_{args.model}.json, data/meta.json, data/roles.json"
    if args.model == "e5":
        site = (
            f"data/chunks_{args.model}.json, data/meta.json, data/gate_en_minilm.json, "
            "data/gate_zh_bge.json (if calibrated), data/chunks_en_minilm.json, data/roles.json"
        )
    zipname = f"functions/tencent/tencent-function-{args.model}.zip"
    print("\n" + "=" * 60)
    print(f"[OK] site artifacts: {site}")
    if args.function:
        print(f"[OK] cloud function: {zipname}")
    print("Next:")
    if args.function:
        # Order is load-bearing (Task 29): the widget's /chat sends no
        # `contexts`, and the OLD deployed function 400s without it. Stated
        # precisely, because an overstated load-bearing comment is the kind
        # that gets discovered to be wrong and then distrusted wholesale --
        # this one used to say the wrong order "breaks live chat for every
        # visitor", and it does not. askWorker throws on the 400, send()
        # catches it and routes to degradedTurn, so the visitor gets the
        # offline-search consent prompt rather than a broken widget. The
        # conclusion is unchanged and still strictly better: function first,
        # always. Only the cost is different -- every visitor loses the LLM
        # answer and gets the offline path until the function catches up.
        print(f"  1. Upload {zipname} to the SCF console and redeploy the function FIRST "
              "-- an old function 400s on /chat's contexts-free request.")
        print("  2. THEN git add/commit/push the data/ files (chunks/meta/gate/roles) and publish the site.")
        print("     Out of order, every visitor loses the LLM answer and falls to the widget's")
        print("     offline search path (a ~23 MB in-browser model) until the function catches up.")
    else:
        print("  - git add/commit/push the data/ files (chunks/meta/gate/roles)")
        print("  - run with --function when you also need to redeploy the backend")


if __name__ == "__main__":
    main()
