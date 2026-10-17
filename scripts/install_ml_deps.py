#!/usr/bin/env python3
"""Install ML deps when pip DNS to files.pythonhosted.org fails — uses urllib (same as curl)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WHEEL_DIR = Path("/tmp/rcs_ml_wheels")
WHEEL_DIR.mkdir(parents=True, exist_ok=True)

# Package -> prefer py3-none-any unless binary needed
PACKAGES = [
    "transformers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "datasets",
    "tokenizers",
    "dill",
    "multiprocess",
    "fsspec",
    "filelock",
    "packaging",
    "regex",
    "tqdm",
    "pyyaml",
    "requests",
    "urllib3",
    "certifi",
    "charset-normalizer",
    "idna",
    "numpy",
    "pyarrow",
    "aiohttp",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "sniffio",
    "typing-extensions",
    "hf-xet",
    "xxhash",
]


def pick_wheel_url(package: str) -> str:
    out = subprocess.check_output(
        ["curl", "-sL", f"https://pypi.org/pypi/{package}/json"],
        text=True,
    )
    data = json.loads(out)
    wheels = [u for u in data["urls"] if u["packagetype"] == "bdist_wheel"]
    if not wheels:
        raise RuntimeError(f"no wheel for {package}")
    # Prefer universal wheel, else cp313 macosx arm64, else any cp313
    for pred in (
        lambda u: "py3-none-any" in u["filename"],
        lambda u: "cp313" in u["filename"] and "macosx" in u["filename"] and "arm64" in u["filename"],
        lambda u: "cp313" in u["filename"],
        lambda u: True,
    ):
        matches = [u for u in wheels if pred(u)]
        if matches:
            return matches[-1]["url"]
    return wheels[-1]["url"]


def download(package: str) -> Path:
    wheel_url = pick_wheel_url(package)
    dest = WHEEL_DIR / wheel_url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return dest
    print(f"  downloading {dest.name} ...")
    subprocess.check_call(["curl", "-sL", "-o", str(dest), wheel_url])
    return dest


def main() -> None:
    wheels: list[Path] = []
    for pkg in PACKAGES:
        try:
            wheels.append(download(pkg))
        except Exception as exc:
            print(f"  skip {pkg}: {exc}", file=sys.stderr)

    venv_python = Path(sys.executable)
    pip = venv_python.parent / "pip"
    cmd = [
        str(pip),
        "install",
        "--no-index",
        "--find-links",
        str(WHEEL_DIR),
        *PACKAGES,
    ]
    print("Installing from local wheels ...")
    subprocess.check_call(cmd)
    print("OK:", subprocess.check_output([str(venv_python), "-c", "import transformers,torch,datasets; print(transformers.__version__, torch.__version__)"]).decode())


if __name__ == "__main__":
    main()
