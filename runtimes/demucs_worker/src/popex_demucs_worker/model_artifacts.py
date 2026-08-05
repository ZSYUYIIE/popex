from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
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
from .paths import (
    AtomicPublication,
    FileIdentity,
    atomic_write_json,
    relative_asset_path,
    trusted_root,
)
from .protocol import EXIT_MODEL_VERIFICATION_FAILED, WorkerError
from .runtime_support import _load_bag_yaml

_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AssetIdentity:
    relative: str
    lexical_path: Path
    resolved_path: Path
    lexical_identity: FileIdentity
    resolved_identity: FileIdentity


@dataclass(frozen=True, slots=True)
class VerifiedAssets:
    bag_data: dict
    bag: AssetIdentity
    checkpoint: AssetIdentity
    checkpoint_digest: str

    @property
    def bag_relative(self) -> str:
        return self.bag.relative

    @property
    def checkpoint_relative(self) -> str:
        return self.checkpoint.relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise WorkerError(
            "CHECKPOINT_UNREADABLE",
            "The checkpoint could not be read completely.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    return digest.hexdigest()


def _capture_asset(cache_root: Path, path: Path, *, label: str) -> AssetIdentity:
    root = trusted_root(str(cache_root))
    relative = relative_asset_path(root, path)
    lexical = root.joinpath(*relative.split("/"))
    try:
        lexical_stat = os.lstat(lexical)
        resolved = lexical.resolve(strict=True)
        resolved_stat = os.lstat(resolved)
    except OSError as exc:
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            f"The cached {label} is unavailable.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if not resolved.is_relative_to(root):
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            f"The cached {label} escapes the trusted cache root.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    if not (
        stat.S_ISREG(lexical_stat.st_mode) or stat.S_ISLNK(lexical_stat.st_mode)
    ) or not stat.S_ISREG(resolved_stat.st_mode):
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            f"The cached {label} is not a safe regular file.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return AssetIdentity(
        relative=relative,
        lexical_path=lexical,
        resolved_path=resolved,
        lexical_identity=FileIdentity.from_stat(lexical_stat),
        resolved_identity=FileIdentity.from_stat(resolved_stat),
    )


def _asset_unchanged(before: AssetIdentity, after: AssetIdentity) -> bool:
    return (
        before.relative == after.relative
        and os.path.normcase(str(before.lexical_path))
        == os.path.normcase(str(after.lexical_path))
        and os.path.normcase(str(before.resolved_path))
        == os.path.normcase(str(after.resolved_path))
        and before.lexical_identity == after.lexical_identity
        and before.resolved_identity == after.resolved_identity
    )


def _verify_bag(cache_root: Path, bag_path: Path) -> tuple[dict, AssetIdentity]:
    before = _capture_asset(cache_root, bag_path, label="bag definition")
    yaml_was_loaded = "yaml" in sys.modules
    try:
        bag_data = _load_bag_yaml(before.resolved_path)
    finally:
        # Passive probing stays lazy from the caller's perspective. The locked
        # parser is imported only for this validation and can be re-imported by
        # later worker commands when it was not already part of the process.
        if not yaml_was_loaded:
            sys.modules.pop("yaml", None)
    after = _capture_asset(cache_root, before.lexical_path, label="bag definition")
    if not _asset_unchanged(before, after):
        raise WorkerError(
            "BAG_CHANGED_DURING_VERIFICATION",
            "The htdemucs bag changed while it was being verified.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return bag_data, after


def _verify_checkpoint(
    cache_root: Path,
    checkpoint_path: Path,
) -> tuple[str, AssetIdentity]:
    before = _capture_asset(cache_root, checkpoint_path, label="checkpoint")
    if before.resolved_identity.size != CHECKPOINT_SIZE_BYTES:
        raise WorkerError(
            "CHECKPOINT_SIZE_MISMATCH",
            "The checkpoint size does not match the approved model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    digest = _sha256(before.resolved_path)
    after = _capture_asset(cache_root, before.lexical_path, label="checkpoint")
    if not _asset_unchanged(before, after):
        raise WorkerError(
            "CHECKPOINT_CHANGED_DURING_VERIFICATION",
            "The checkpoint changed while its digest was being verified.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    if digest != CHECKPOINT_SHA256:
        raise WorkerError(
            "CHECKPOINT_HASH_MISMATCH",
            "The checkpoint digest does not match the approved model.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return digest, after


def _verify_assets(
    cache_root: Path,
    bag_path: Path,
    checkpoint_path: Path,
) -> VerifiedAssets:
    root = trusted_root(str(cache_root))
    bag_data, bag = _verify_bag(root, bag_path)
    digest, checkpoint = _verify_checkpoint(root, checkpoint_path)
    return VerifiedAssets(
        bag_data=bag_data,
        bag=bag,
        checkpoint=checkpoint,
        checkpoint_digest=digest,
    )


def _revalidate_assets(cache_root: Path, expected: VerifiedAssets) -> VerifiedAssets:
    current = _verify_assets(
        cache_root,
        expected.bag.lexical_path,
        expected.checkpoint.lexical_path,
    )
    if (
        not _asset_unchanged(expected.bag, current.bag)
        or not _asset_unchanged(expected.checkpoint, current.checkpoint)
        or current.checkpoint_digest != expected.checkpoint_digest
    ):
        raise WorkerError(
            "MODEL_ASSET_CHANGED",
            "The verified model assets changed before readiness publication completed.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return current


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


def _publish_manifest(cache_root: Path, payload: dict) -> AtomicPublication:
    root = trusted_root(str(cache_root))
    path = root.joinpath(*READINESS_RELATIVE_PATH.split("/"))
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return atomic_write_json(root, path, encoded)


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
