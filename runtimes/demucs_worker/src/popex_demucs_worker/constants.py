from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Final

PROTOCOL_VERSION: Final = 1
WORKER_VERSION: Final = "1.0.0"
RUNTIME_LOCK_ENV: Final = "POPEX_DEMUCS_RUNTIME_LOCK"
UNPROVISIONED_PROFILE: Final = "unprovisioned"

MODEL_REPOSITORY: Final = "adefossez/HTDemucs"
MODEL_REVISION: Final = "bf35a81b663819a8255c8fefee17f9d812b786b5"
BAG_FILE: Final = "htdemucs.yaml"
BAG_SIGNATURE: Final = "955717e8"
CHECKPOINT_FILE: Final = "955717e8.safetensors"
CHECKPOINT_SIZE_BYTES: Final = 84025440
CHECKPOINT_SHA256: Final = (
    "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
)
DEMUCS_VERSION: Final = "4.1.0"
READINESS_RELATIVE_PATH: Final = "readiness/htdemucs-bf35a81b-v1.json"
EXPECTED_MODEL_CLASS: Final = "HTDemucs"
EXPECTED_AUDIO_CHANNELS: Final = 2
EXPECTED_SAMPLE_RATE: Final = 44100
EXPECTED_SOURCES: Final = ("drums", "bass", "other", "vocals")
OUTPUT_SOURCES: Final = ("vocals", "bass", "drums", "other")
ALLOWED_DEVICES: Final = frozenset({"cpu", "cuda", "mps"})

DISTRIBUTIONS: Final = {
    "demucs": "demucs",
    "torch": "torch",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "PyYAML": "PyYAML",
}


class RuntimeLockError(ValueError):
    pass


def _read_lock_text() -> str:
    external = os.environ.get(RUNTIME_LOCK_ENV)
    if external:
        if "\x00" in external:
            raise RuntimeLockError("Runtime lock path is invalid.")
        path = Path(external)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RuntimeLockError("Runtime lock path is invalid.")
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeLockError("Runtime lock cannot be read.") from exc
    return resources.files(__package__).joinpath("runtime-lock.json").read_text(
        encoding="utf-8"
    )


def load_runtime_lock() -> dict:
    try:
        payload = json.loads(_read_lock_text())
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise RuntimeLockError("Runtime lock metadata is invalid.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or not isinstance(payload.get("runtimeProfile"), str)
        or not payload["runtimeProfile"]
        or payload.get("workerVersion") != WORKER_VERSION
        or not isinstance(payload.get("packages"), dict)
    ):
        raise RuntimeLockError("Runtime lock metadata is invalid.")
    return payload
