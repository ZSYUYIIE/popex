from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .constants import (
    BAG_FILE,
    BAG_SIGNATURE,
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROTOCOL_VERSION,
    READINESS_RELATIVE_PATH,
    WORKER_VERSION,
)
from .paths import atomic_write_json, relative_asset_path
from .protocol import EXIT_MODEL_VERIFICATION_FAILED, WorkerError
from .runtime_support import _load_bag_yaml

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise WorkerError(
            "CHECKPOINT_UNREADABLE",
            "The checkpoint could not be read completely.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    return digest.hexdigest()


def _verify_assets(cache_root: Path, bag_path: Path, checkpoint_path: Path) -> tuple[dict, str, str]:
    bag_relative = relative_asset_path(cache_root, bag_path)
    checkpoint_relative = relative_asset_path(cache_root, checkpoint_path)
    bag_data = _load_bag_yaml(bag_path)
    try:
        size = checkpoint_path.stat().st_size
    except OSError as exc:
        raise WorkerError(
            "CHECKPOINT_UNREADABLE",
            "The checkpoint is unavailable.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if size != CHECKPOINT_SIZE_BYTES:
        raise WorkerError(
            "CHECKPOINT_SIZE_MISMATCH",
            "The checkpoint size does not match the approved model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    digest = _sha256(checkpoint_path)
    if digest != CHECKPOINT_SHA256:
        raise WorkerError(
            "CHECKPOINT_HASH_MISMATCH",
            "The checkpoint digest does not match the approved model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return bag_data, bag_relative, checkpoint_relative


def _manifest(
    runtime_profile: str,
    versions: dict[str, str],
    bag_relative: str,
    checkpoint_relative: str,
    *,
    verified_at: str,
) -> dict:
    return {
        "schemaVersion": 1,
        "protocolVersion": PROTOCOL_VERSION,
        "runtimeProfile": runtime_profile,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": versions["torch"],
        "huggingfaceHubVersion": versions["huggingface_hub"],
        "packageVersions": versions,
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "bagFile": BAG_FILE,
        "bagModelSignatures": [BAG_SIGNATURE],
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "verifiedAt": verified_at,
        "cacheAssets": {
            "bag": bag_relative,
            "checkpoint": checkpoint_relative,
        },
        "offlineReady": True,
        "warnings": [],
    }


def _publish_manifest(cache_root: Path, payload: dict) -> Path:
    path = cache_root.joinpath(*READINESS_RELATIVE_PATH.split("/"))
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    atomic_write_json(path, encoded)
    return path


def _provenance(payload: dict) -> dict:
    return {
        "runtimeProfile": payload["runtimeProfile"],
        "workerVersion": payload["workerVersion"],
        "demucsVersion": payload["demucsVersion"],
        "torchVersion": payload["torchVersion"],
        "huggingfaceHubVersion": payload["huggingfaceHubVersion"],
        "modelRepository": payload["modelRepository"],
        "modelRevision": payload["modelRevision"],
        "bagFile": payload["bagFile"],
        "checkpointFile": payload["checkpointFile"],
        "checkpointSizeBytes": payload["checkpointSizeBytes"],
        "checkpointSha256": payload["checkpointSha256"],
        "verifiedAt": payload["verifiedAt"],
        "offlineReady": True,
        "readinessManifest": READINESS_RELATIVE_PATH,
    }


