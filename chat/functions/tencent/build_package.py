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
              gate_models: dict[str, Path] | None = None) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # code + fresh roles fallback
        zf.write(HERE / "index.py", "index.py")
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
    args = ap.parse_args()
    preset = PRESETS[args.preset]

    model_dir = CHAT / "models" / preset["repo"]
    print(f"[1/4] model {preset['repo']} -> {model_dir}")
    download_model(preset["repo"], model_dir)

    # If the zh gate corpus has content, its model must exist BEFORE the
    # index build (which embeds + calibrates it). Empty corpus = zh gate off.
    zh_corpus_dir = CHAT / "knowledge_zh"
    zh_wanted = args.preset == "e5" and any(
        line.startswith("## ")
        for md in zh_corpus_dir.glob("*.md")
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
    # bundled too if build_index enabled one — see knowledge_zh/gate.md).
    gate_models: dict[str, Path] = {}
    gate_file = CHAT / "data" / "gate_vectors.json"
    if gate_file.exists() and args.preset == "e5":
        payload = json.loads(gate_file.read_text(encoding="utf-8"))
        gate_models["gate_model"] = CHAT / "models" / "Xenova" / "all-MiniLM-L6-v2"
        if "zh" in payload:
            zh_dir = CHAT / "models" / "Xenova" / "bge-small-zh-v1.5"
            download_model("Xenova/bge-small-zh-v1.5", zh_dir)
            gate_models["gate_model_zh"] = zh_dir

    print(f"[3/4] linux wheels for cp{args.python_version}")
    wheels = download_wheels(args.python_version, HERE / "_wheels")
    print("  " + "\n  ".join(w.name for w in wheels))

    out = HERE / f"tencent-function-{args.preset}.zip"
    print(f"[4/4] packaging -> {out.name} (gates: {sorted(gate_models) or 'none'})")
    build_zip(preset, model_dir, wheels, out, gate_models)

    print(json.dumps({
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
