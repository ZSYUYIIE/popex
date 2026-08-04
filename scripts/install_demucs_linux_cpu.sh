#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PROFILE_DIR="$REPO_ROOT/runtimes/profiles/linux-cpu"
WORKER_DIR="${POPEX_DEMUCS_WORKER_DIR:-$REPO_ROOT/runtimes/demucs_worker}"
PYTHON_BIN="${POPEX_DEMUCS_PYTHON:-python3.13}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
RUNTIME_DIR="${POPEX_DEMUCS_LINUX_CPU_DIR:-$DATA_HOME/popex/runtimes/linux-x86_64-cpu-cpython313}"
VENV_DIR="$RUNTIME_DIR/venv"
HF_HOME="$RUNTIME_DIR/huggingface"
CREATED_RUNTIME=0

on_exit() {
    local status=$?
    if (( status != 0 )); then
        printf '\nLinux CPU runtime installation failed.\n' >&2
        if (( CREATED_RUNTIME == 1 )); then
            printf 'Review the error, then remove the incomplete isolated runtime with:\n  rm -rf %q\n' "$RUNTIME_DIR" >&2
        else
            printf 'No runtime directory was created.\n' >&2
        fi
    fi
    return "$status"
}
trap on_exit EXIT

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    echo "Refusing to install the optional runtime as root. Run as a normal user." >&2
    exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This profile supports Linux only." >&2
    exit 2
fi

case "$(uname -m)" in
    x86_64|amd64) ;;
    *)
        echo "This profile supports Linux x86-64 only." >&2
        exit 2
        ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "CPython 3.13 is required. Set POPEX_DEMUCS_PYTHON to its executable." >&2
    exit 2
fi

if ! "$PYTHON_BIN" - <<'PY'
import platform
import sys
raise SystemExit(0 if platform.python_implementation() == "CPython" and sys.version_info[:2] == (3, 13) else 1)
PY
then
    echo "This profile requires CPython >=3.13.0,<3.14.0." >&2
    exit 2
fi

if [[ ! -f "$WORKER_DIR/pyproject.toml" ]]; then
    echo "The local popex-demucs-worker source is missing at: $WORKER_DIR" >&2
    echo "Merge or check out the worker implementation before installing this profile." >&2
    exit 3
fi

for required in profile.json requirements.lock torch.lock worker-runtime-lock.json; do
    if [[ ! -f "$PROFILE_DIR/$required" ]]; then
        echo "Profile file is missing: $PROFILE_DIR/$required" >&2
        exit 3
    fi
done

if [[ -e "$RUNTIME_DIR" ]]; then
    echo "Refusing to overwrite the existing runtime directory: $RUNTIME_DIR" >&2
    echo "Remove it explicitly before reinstalling." >&2
    exit 3
fi

mkdir -p -- "$RUNTIME_DIR"
CREATED_RUNTIME=1
"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$VENV_PYTHON" -m pip install \
    --no-deps \
    --require-hashes \
    --only-binary=:all: \
    --index-url https://pypi.org/simple \
    -r "$PROFILE_DIR/requirements.lock"

"$VENV_PYTHON" -m pip install \
    --no-deps \
    --require-hashes \
    --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cpu \
    -r "$PROFILE_DIR/torch.lock"

"$VENV_PYTHON" -m pip check

"$VENV_PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import os

import demucs
import demucs.api
import demucs.hf
import einops
import huggingface_hub
import julius
import lameenc
import numpy
import safetensors
import sphn
import torch
import tqdm
import yaml

names = {dist.metadata["Name"].lower().replace("_", "-") for dist in metadata.distributions()}
for forbidden in ("torchaudio", "openunmix", "dora-search"):
    if forbidden in names:
        raise SystemExit(f"Forbidden package installed: {forbidden}")
if any(name.startswith("nvidia-") for name in names):
    raise SystemExit("An NVIDIA package was installed in the CPU profile.")
if torch.__version__ != "2.13.0+cpu" or torch.version.cuda is not None or torch.cuda.is_available():
    raise SystemExit("The installed PyTorch build is not the approved CPU-only build.")
if demucs.__version__ != "4.1.0" or numpy.__version__ != "2.5.1":
    raise SystemExit("The installed Demucs or NumPy version is not approved.")
if Path(os.environ["HF_HOME"]).exists():
    raise SystemExit("Package installation or import unexpectedly created a model cache.")
print(json.dumps({"dependencyImports": "ok", "torch": torch.__version__, "demucs": demucs.__version__}, sort_keys=True))
PY

"$VENV_PYTHON" -m pip install \
    --no-deps \
    --no-build-isolation \
    --no-index \
    "$WORKER_DIR"

"$VENV_PYTHON" -m pip check
cp -- "$PROFILE_DIR/worker-runtime-lock.json" "$RUNTIME_DIR/runtime-lock.json"

export POPEX_DEMUCS_RUNTIME_LOCK="$RUNTIME_DIR/runtime-lock.json"
"$VENV_DIR/bin/popex-demucs-worker" --protocol-version 1 runtime-probe
