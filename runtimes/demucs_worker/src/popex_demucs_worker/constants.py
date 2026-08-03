from __future__ import annotations

import json
from importlib import resources
from typing import Final

PROTOCOL_VERSION: Final = 1
WORKER_VERSION: Final = "1.0.0"

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

_DISTRIBUTIONS: Final = {
    "demucs": "demucs",
    "torch": "torch",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "PyYAML": "PyYAML",
}


def load_runtime_lock() -> dict:
    text = resources.files(__package__).joinpath("runtime-lock.json").read_text(
        encoding="utf-8"
    )
    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or not isinstance(payload.get("runtimeProfile"), str)
        or not isinstance(payload.get("packages"), dict)
    ):
        raise RuntimeError("Embedded runtime lock metadata is invalid.")
    return payload


RUNTIME_LOCK: Final = load_runtime_lock()
RUNTIME_PROFILE: Final = RUNTIME_LOCK["runtimeProfile"]
LOCKED_PACKAGE_VERSIONS: Final = dict(RUNTIME_LOCK["packages"])
DISTRIBUTIONS: Final = dict(_DISTRIBUTIONS)
