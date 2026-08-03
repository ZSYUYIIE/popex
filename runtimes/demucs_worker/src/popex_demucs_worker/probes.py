from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .constants import (
    BAG_FILE,
    BAG_SIGNATURE,
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    DISTRIBUTIONS,
    LOCKED_PACKAGE_VERSIONS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROTOCOL_VERSION,
    READINESS_RELATIVE_PATH,
    RUNTIME_PROFILE,
    WORKER_VERSION,
)
from .paths import resolve_contained, trusted_root
from .protocol import (
    EXIT_MODEL_DOWNLOAD_REQUIRED,
    EXIT_MODEL_VERIFICATION_FAILED,
    EXIT_RUNTIME_INCOMPATIBLE,
    WorkerError,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, distribution in DISTRIBUTIONS.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def require_compatible_runtime() -> dict[str, str]:
    versions = installed_versions()
    mismatches = {
        name: (versions.get(name), expected)
        for name, expected in LOCKED_PACKAGE_VERSIONS.items()
        if versions.get(name) != expected
    }
    if versions.get("demucs") != DEMUCS_VERSION or mismatches:
        raise WorkerError(
            "RUNTIME_INCOMPATIBLE",
            "The installed optional runtime does not match its embedded lock profile.",
            EXIT_RUNTIME_INCOMPATIBLE,
        )
    return {key: value for key, value in versions.items() if value is not None}


def runtime_probe() -> dict:
    versions = require_compatible_runtime()
    return {
        "pythonVersion": ".".join(str(part) for part in sys.version_info[:3]),
        "workerVersion": WORKER_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
        "installedVersions": versions,
        "lockedVersions": dict(LOCKED_PACKAGE_VERSIONS),
        "compatible": True,
    }


def _parse_verified_at(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest has an invalid verification timestamp.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest has an invalid verification timestamp.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if parsed.tzinfo is None:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest has an invalid verification timestamp.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )


def load_readiness_manifest(cache_root: Path) -> tuple[dict, Path, Path, Path]:
    manifest_path = cache_root.joinpath(*READINESS_RELATIVE_PATH.split("/"))
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise WorkerError(
            "MODEL_DOWNLOAD_REQUIRED",
            "The verified htdemucs model is not available in this runtime.",
            EXIT_MODEL_DOWNLOAD_REQUIRED,
            retryable=True,
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    versions = require_compatible_runtime()
    required_exact = {
        "schemaVersion": 1,
        "protocolVersion": PROTOCOL_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": versions["torch"],
        "huggingfaceHubVersion": versions["huggingface_hub"],
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "bagFile": BAG_FILE,
        "bagModelSignatures": [BAG_SIGNATURE],
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "offlineReady": True,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required_exact.items()
    ):
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest does not match the approved runtime and model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    if payload.get("packageVersions") != versions:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest package summary is incompatible.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    if not isinstance(payload.get("warnings"), list):
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest warnings field is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    _parse_verified_at(payload.get("verifiedAt"))
    digest = payload.get("checkpointSha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The stored checkpoint digest is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    assets = payload.get("cacheAssets")
    if not isinstance(assets, dict) or set(assets) != {"bag", "checkpoint"}:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest asset map is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    bag_path = resolve_contained(
        cache_root,
        assets["bag"],
        require_regular_file=True,
        error_code="MODEL_ASSET_INVALID",
    )
    checkpoint_path = resolve_contained(
        cache_root,
        assets["checkpoint"],
        require_regular_file=True,
        error_code="MODEL_ASSET_INVALID",
    )
    try:
        size = checkpoint_path.stat().st_size
    except OSError as exc:
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            "The checkpoint file is unavailable.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if size != CHECKPOINT_SIZE_BYTES:
        raise WorkerError(
            "CHECKPOINT_SIZE_MISMATCH",
            "The cached checkpoint size does not match the approved model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return payload, manifest_path, bag_path, checkpoint_path


def model_probe(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text)
    payload, _, _, checkpoint_path = load_readiness_manifest(cache_root)
    return {
        "runtimeProfile": payload["runtimeProfile"],
        "workerVersion": payload["workerVersion"],
        "offlineReady": True,
        "modelRepository": payload["modelRepository"],
        "modelRevision": payload["modelRevision"],
        "checkpointFile": payload["checkpointFile"],
        "checkpointSizeBytes": checkpoint_path.stat().st_size,
        "checkpointSha256": payload["checkpointSha256"],
        "verifiedAt": payload["verifiedAt"],
        "readinessManifest": READINESS_RELATIVE_PATH,
    }
