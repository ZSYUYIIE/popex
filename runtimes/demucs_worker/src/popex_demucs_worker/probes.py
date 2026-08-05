from __future__ import annotations

import json
import os
import re
import stat
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
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROTOCOL_VERSION,
    READINESS_RELATIVE_PATH,
    RUNTIME_LOCK_ENV,
    UNPROVISIONED_PROFILE,
    WORKER_VERSION,
    RuntimeLockError,
    load_runtime_lock,
)
from .model_artifacts import _verify_assets
from .paths import resolve_contained, safe_posix_relative, trusted_root
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


def require_compatible_runtime() -> tuple[str, dict[str, str], dict[str, str]]:
    try:
        lock = load_runtime_lock()
    except RuntimeLockError as exc:
        raise WorkerError(
            "RUNTIME_PROFILE_INVALID",
            "The optional runtime profile lock metadata is invalid.",
            EXIT_RUNTIME_INCOMPATIBLE,
        ) from exc
    profile = lock["runtimeProfile"]
    locked = lock["packages"]
    if profile == UNPROVISIONED_PROFILE:
        raise WorkerError(
            "RUNTIME_PROFILE_UNPROVISIONED",
            "The optional runtime profile has not supplied an exact package lock.",
            EXIT_RUNTIME_INCOMPATIBLE,
        )
    if set(DISTRIBUTIONS) - set(locked) or any(
        not isinstance(locked.get(name), str) or not locked[name]
        for name in DISTRIBUTIONS
    ):
        raise WorkerError(
            "RUNTIME_PROFILE_INVALID",
            "The optional runtime profile lock metadata is incomplete.",
            EXIT_RUNTIME_INCOMPATIBLE,
        )
    if locked["demucs"] != DEMUCS_VERSION:
        raise WorkerError(
            "RUNTIME_INCOMPATIBLE",
            "The optional runtime profile does not approve Demucs 4.1.0.",
            EXIT_RUNTIME_INCOMPATIBLE,
        )
    installed = installed_versions()
    mismatches = {
        name: (installed.get(name), locked[name])
        for name in DISTRIBUTIONS
        if installed.get(name) != locked[name]
    }
    if mismatches:
        raise WorkerError(
            "RUNTIME_INCOMPATIBLE",
            "The installed optional runtime does not match its profile lock.",
            EXIT_RUNTIME_INCOMPATIBLE,
        )
    exact_installed = {name: installed[name] for name in DISTRIBUTIONS}
    exact_locked = {name: locked[name] for name in DISTRIBUTIONS}
    return profile, exact_installed, exact_locked  # type: ignore[return-value]


def runtime_probe() -> dict:
    profile, versions, locked = require_compatible_runtime()
    return {
        "pythonVersion": ".".join(str(part) for part in sys.version_info[:3]),
        "workerVersion": WORKER_VERSION,
        "runtimeProfile": profile,
        "runtimeLockSource": "profile" if RUNTIME_LOCK_ENV in os.environ else "bundled",
        "installedVersions": versions,
        "lockedVersions": locked,
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


def _validate_readiness_path(cache_root: Path, manifest_path: Path) -> Path:
    current = cache_root
    for part in Path(READINESS_RELATIVE_PATH).parts[:-1]:
        current = current / part
        try:
            value = os.lstat(current)
        except OSError as exc:
            raise WorkerError(
                "READINESS_MANIFEST_INVALID",
                "The model readiness manifest parent is unavailable.",
                EXIT_MODEL_VERIFICATION_FAILED,
            ) from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise WorkerError(
                "READINESS_MANIFEST_INVALID",
                "The model readiness manifest parent is unsafe.",
                EXIT_MODEL_VERIFICATION_FAILED,
            )
    try:
        lexical = os.lstat(manifest_path)
        resolved = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest is unavailable.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if (
        stat.S_ISLNK(lexical.st_mode)
        or not stat.S_ISREG(lexical.st_mode)
        or not resolved.is_relative_to(cache_root)
        or os.path.normcase(str(resolved)) != os.path.normcase(str(manifest_path))
    ):
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest is not safely contained.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return resolved


def load_readiness_manifest(cache_root: Path) -> tuple[dict, Path, Path, Path]:
    root = trusted_root(str(cache_root))
    manifest_path = root.joinpath(*READINESS_RELATIVE_PATH.split("/"))
    if not manifest_path.exists():
        raise WorkerError(
            "MODEL_DOWNLOAD_REQUIRED",
            "The verified htdemucs model is not available in this runtime.",
            EXIT_MODEL_DOWNLOAD_REQUIRED,
            retryable=True,
        )
    resolved_manifest = _validate_readiness_path(root, manifest_path)
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    runtime_profile, versions, _ = require_compatible_runtime()
    required_exact = {
        "schemaVersion": 1,
        "protocolVersion": PROTOCOL_VERSION,
        "runtimeProfile": runtime_profile,
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
    bag_relative = safe_posix_relative(assets["bag"], error_code="MODEL_ASSET_INVALID")
    checkpoint_relative = safe_posix_relative(
        assets["checkpoint"], error_code="MODEL_ASSET_INVALID"
    )
    # Keep the existing containment error behavior before the cryptographic pass.
    resolve_contained(
        root,
        assets["bag"],
        require_regular_file=True,
        error_code="MODEL_ASSET_INVALID",
    )
    resolve_contained(
        root,
        assets["checkpoint"],
        require_regular_file=True,
        error_code="MODEL_ASSET_INVALID",
    )
    bag_lexical = root.joinpath(*bag_relative.parts)
    checkpoint_lexical = root.joinpath(*checkpoint_relative.parts)
    verified = _verify_assets(root, bag_lexical, checkpoint_lexical)
    if (
        verified.bag_relative != assets["bag"]
        or verified.checkpoint_relative != assets["checkpoint"]
    ):
        raise WorkerError(
            "READINESS_MANIFEST_INVALID",
            "The model readiness manifest asset identity changed.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return (
        payload,
        resolved_manifest,
        verified.bag.resolved_path,
        verified.checkpoint.resolved_path,
    )


def model_probe(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text)
    payload, _, _, checkpoint_path = load_readiness_manifest(cache_root)
    return {
        "runtimeProfile": payload["runtimeProfile"],
        "workerVersion": payload["workerVersion"],
        "demucsVersion": payload["demucsVersion"],
        "torchVersion": payload["torchVersion"],
        "huggingfaceHubVersion": payload["huggingfaceHubVersion"],
        "modelRepository": payload["modelRepository"],
        "modelRevision": payload["modelRevision"],
        "checkpointFile": payload["checkpointFile"],
        "checkpointSizeBytes": checkpoint_path.stat().st_size,
        "checkpointSha256": CHECKPOINT_SHA256,
        "verifiedAt": payload["verifiedAt"],
        "offlineReady": True,
        "readinessManifest": READINESS_RELATIVE_PATH,
    }
