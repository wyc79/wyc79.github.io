"""Build the stage-2 Tencent SCF package (chat backend + server-side embedding).

Run on your own machine (needs internet):

    cd chat/functions/tencent
    python build_package.py                # e5 multilingual (the real thing)
    python build_package.py --preset minilm --python-version 311   # CI/testing

Does four things:
1. Downloads Xenova/multilingual-e5-small (ONNX + tokenizer, ~135MB) from
   huggingface.co, falling back to hf-mirror.com — into chat/models/... so
   `python scripts/build_index.py --model e5` can rebuild the index with the
   SAME model the function serves.
2. Downloads Linux wheels for onnxruntime/tokenizers/numpy matching the SCF
   runtime (Python 3.10, manylinux) — regardless of your local OS.
3. Assembles index.py + scf_bootstrap + roles.json + model/ + deps.
4. Writes tencent-function-e5.zip (~160-200MB, under SCF's 350MB direct
   upload limit), with scf_bootstrap's executable bit set even on Windows.
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT = HERE.parents[1]
HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model_quantized.onnx",
]
PRESETS = {
    "e5": {"repo": "Xenova/multilingual-e5-small", "query_prefix": "query: "},
    "minilm": {"repo": "Xenova/all-MiniLM-L6-v2", "query_prefix": ""},
}


def git_short_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def make_build_info(preset: str, version: str | None) -> dict:
    """A dated, versioned stamp bundled into the package as build_info.json.
    code_sha is a hash of the exact index.py shipped, so a running function can
    be compared against local source without trusting the zip's filename."""
    now = datetime.datetime.now(datetime.timezone.utc)
    date = now.strftime("%Y%m%d")
    sha = git_short_sha()
    code_sha = hashlib.sha256((HERE / "index.py").read_bytes()).hexdigest()[:12]
    return {
        "version": version or date,
        "preset": preset,
        "build_id": f"{preset}-{date}-{sha}",
        "built_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": sha,
        "code_sha": code_sha,
    }


def download_model(repo: str, dest: Path) -> None:
    for rel in MODEL_FILES:
        target = dest / rel
        if target.exists() and target.stat().st_size > 0:
            print(f"  have {rel}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        last_err = None
        for host in HOSTS:
            url = f"{host}/{repo}/resolve/main/{rel}"
            try:
                print(f"  fetching {url}")
                with urllib.request.urlopen(url, timeout=60) as res, open(target, "wb") as out:
                    shutil.copyfileobj(res, out, length=1 << 20)
                last_err = None
                break
            except Exception as err:  # try next mirror
                last_err = err
                print(f"    failed ({err}); trying next mirror")
        if last_err:
            raise SystemExit(f"could not download {rel}: {last_err}")


def download_wheels(py_version: str, wheel_dir: Path) -> list[Path]:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    options = [
        "--dest", str(wheel_dir),
        "--python-version", py_version,
        "--implementation", "cp",
        "--only-binary", ":all:",
        "--no-deps",
    ]
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64"):
        options += ["--platform", platform]

    def pip_download(packages: list[str], required: bool) -> None:
        cmd = [sys.executable, "-m", "pip", "download", *packages, *options]
        subprocess.run(cmd, check=required)

    pip_download(["onnxruntime", "tokenizers", "numpy"], required=True)
    # onnxruntime's own runtime deps (--no-deps skips them; pure-python, small)
    pip_download(
        ["coloredlogs", "flatbuffers", "packaging", "protobuf", "sympy", "humanfriendly", "mpmath"],
        required=False,
    )
    return sorted(wheel_dir.glob("*.whl"))


def build_zip(preset: dict, model_src: Path, wheels: list[Path], out_zip: Path,
              gate_models: dict[str, Path] | None = None, build_info: dict | None = None) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # code + fresh roles fallback
        zf.write(HERE / "index.py", "index.py")
        # build stamp: index.py reads this at startup and logs build_id on
        # every line, so SCF logs say which build produced them.
        if build_info is not None:
            zf.writestr("build_info.json", json.dumps(build_info, ensure_ascii=False, indent=2))
        roles = (CHAT / "data" / "roles.json").read_text(encoding="utf-8")
        zf.writestr("roles.json", roles)
        # scf_bootstrap with the executable bit, portable across build OSes
        bootstrap = (HERE / "scf_bootstrap").read_text(encoding="utf-8").replace("\r\n", "\n")
        info = zipfile.ZipInfo("scf_bootstrap")
        info.external_attr = 0o755 << 16
        zf.writestr(info, bootstrap)
        # retrieval model
        for rel in MODEL_FILES:
            zf.write(model_src / rel, f"model/{rel}")
        # gate model(s) + gate vectors (server-side off-topic gate)
        for dir_name, src in (gate_models or {}).items():
            for rel in MODEL_FILES:
                zf.write(src / rel, f"{dir_name}/{rel}")
        gate_file = CHAT / "data" / "gate_vectors.json"
        if gate_models and gate_file.exists():
            zf.write(gate_file, "gate_vectors.json")
        # dependencies, extracted wheel contents at package root
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as wz:
                for name in wz.namelist():
                    if name.endswith("/"):
                        continue
                    zf.writestr(name, wz.read(name))
    print(f"\nwrote {out_zip}  ({out_zip.stat().st_size / 1e6:.1f} MB)")
    if out_zip.stat().st_size > 350e6:
        raise SystemExit("package exceeds SCF's 350MB direct-upload limit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="e5")
    ap.add_argument("--python-version", default="310", help="SCF runtime ABI (310 for Python 3.10)")
    ap.add_argument("--version", default=None,
                    help="build version label (default: UTC build date, e.g. 20260721)")
    args = ap.parse_args()
    preset = PRESETS[args.preset]

    build_info = make_build_info(args.preset, args.version)
    print(f"[build] {build_info['build_id']}  built_at={build_info['built_at']}  "
          f"code_sha={build_info['code_sha']}")

    model_dir = CHAT / "models" / preset["repo"]
    print(f"[1/4] model {preset['repo']} -> {model_dir}")
    download_model(preset["repo"], model_dir)

    # If the zh gate corpus has content, its model must exist BEFORE the
    # index build (which embeds + calibrates it). Empty corpus = zh gate off.
    # zh corpus lives in the shared knowledge/ folder as about_zh.md (English
    # is about_en.md); match *_zh.md so we never trip on the English file.
    zh_corpus_dir = CHAT / "knowledge"
    zh_wanted = args.preset == "e5" and any(
        line.startswith("## ")
        for md in zh_corpus_dir.glob("*_zh.md")
        for line in md.read_text(encoding="utf-8").splitlines()
    )
    if zh_wanted:
        download_model("Xenova/bge-small-zh-v1.5", CHAT / "models" / "Xenova" / "bge-small-zh-v1.5")

    print(f"[2/4] rebuild index with --model {args.preset} (also calibrates + writes gate vectors)")
    subprocess.run(
        [sys.executable, str(CHAT / "scripts" / "build_index.py"), "--model", args.preset],
        check=True, cwd=CHAT,
    )

    # Server-side gate artifacts (e5 delegates gating to MiniLM; a zh gate is
    # bundled too if build_index enabled one — see knowledge/about_zh.md).
    gate_models: dict[str, Path] = {}
    gate_file = CHAT / "data" / "gate_vectors.json"
    if gate_file.exists() and args.preset == "e5":
        payload = json.loads(gate_file.read_text(encoding="utf-8"))
        gate_models["gate_model"] = CHAT / "models" / "Xenova" / "all-MiniLM-L6-v2"
        if "zh" in payload:
            zh_dir = CHAT / "models" / "Xenova" / "bge-small-zh-v1.5"
            download_model("Xenova/bge-small-zh-v1.5", zh_dir)
            gate_models["gate_model_zh"] = zh_dir
        # Echo each gate the index build wrote (en is always present; zh only
        # when its calibration separated) so the package log is self-describing.
        for lang in ("en", "zh"):
            spec = payload.get(lang)
            print(
                f"  gate[{lang}]: {spec['gate_stat']} >= {spec['gate_threshold']}"
                if spec else f"  gate[{lang}]: not built (dormant)"
            )

    print(f"[3/4] linux wheels for cp{args.python_version}")
    wheels = download_wheels(args.python_version, HERE / "_wheels")
    print("  " + "\n  ".join(w.name for w in wheels))

    out = HERE / f"tencent-function-{args.preset}.zip"
    print(f"[4/4] packaging -> {out.name} (gates: {sorted(gate_models) or 'none'})")
    build_zip(preset, model_dir, wheels, out, gate_models, build_info)

    print(json.dumps({
        "build": build_info["build_id"],
        "next": [
            f"rebuild index: cd chat && python scripts/build_index.py --model {args.preset}"
            + (" (note the new gate threshold it prints)" if args.preset == "e5" else ""),
            f"console: 本地上传zip包 -> {out.name}; 内存 1024MB; 初始化超时 120s;"
            " env add MODEL_DIR=model, QUERY_PREFIX='" + preset["query_prefix"] + "'",
            "commit the rebuilt chat/data/index.json and publish the site",
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
