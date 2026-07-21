#!/usr/bin/env python3
"""Master build for the portfolio chat agent — one command, all artifacts.

    cd chat
    python build.py                # rebuild the SITE artifacts only (fast):
                                   #   data/index.json, data/gate_vectors.json,
                                   #   data/fallback_vectors.json, data/roles.json
    python build.py --function     # ALSO (re)build the Tencent SCF zip — downloads
                                   #   any missing models first, then re-zips
    python build.py --model minilm # English-only light-mode index instead of e5

Without --function this runs scripts/build_index.py, which needs the models in
chat/models/ (build_package.py fetches them on the first --function run). With
--function it runs functions/tencent/build_package.py, which downloads models as
needed, rebuilds the index itself, and writes tencent-function-<preset>.zip.

After building: git add/commit/push the data/ files (and knowledge/, roles, etc.
if you changed them); if you built the function, re-upload the zip to SCF.
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
        # build_index.py = index + (for e5) gate/fallback vectors + roles.
        run([sys.executable, str(CHAT / "scripts" / "build_index.py"), "--model", args.model])

    site = "data/index.json, data/roles.json"
    if args.model == "e5":
        site = ("data/index.json, data/gate_vectors.json, "
                "data/fallback_vectors.json, data/roles.json")
    zipname = f"functions/tencent/tencent-function-{args.model}.zip"
    print("\n" + "=" * 60)
    print(f"[OK] site artifacts: {site}")
    if args.function:
        print(f"[OK] cloud function: {zipname}")
    print("Next:")
    print("  - git add/commit/push the data/ files (index/gate/fallback/roles)")
    if args.function:
        print(f"  - re-upload {zipname} to the SCF function, then redeploy")
    else:
        print("  - run with --function when you also need to redeploy the backend")


if __name__ == "__main__":
    main()
