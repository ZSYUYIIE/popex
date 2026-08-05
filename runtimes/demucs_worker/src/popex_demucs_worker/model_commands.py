from __future__ import annotations

from datetime import datetime, timezone

from .constants import (
    BAG_FILE,
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    READINESS_RELATIVE_PATH,
    WORKER_VERSION,
)
from .model_artifacts import (
    VerifiedAssets,
    _manifest,
    _provenance,
    _publish_manifest,
    _revalidate_assets,
    _verify_assets,
)
from .paths import trusted_root
from .probes import load_readiness_manifest, require_compatible_runtime
from .protocol import EXIT_MODEL_VERIFICATION_FAILED, WorkerError
from .runtime_support import _download_exact_assets


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _not_ready_result(runtime_profile: str, versions: dict[str, str]) -> dict:
    """Return a fail-closed non-ready result for a detected in-flight replacement.

    The worker command has not published readiness. Base clients reject this result
    because ``offlineReady`` is false, while direct callers can inspect the state
    without mistaking an old manifest for authoritative readiness.
    """
    return {
        "runtimeProfile": runtime_profile,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": versions["torch"],
        "huggingfaceHubVersion": versions["huggingface_hub"],
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "bagFile": BAG_FILE,
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "verifiedAt": _utc_now(),
        "offlineReady": False,
        "readinessManifest": READINESS_RELATIVE_PATH,
    }


def _publish_verified_readiness(
    cache_root,
    runtime_profile: str,
    versions: dict[str, str],
    verified: VerifiedAssets,
) -> dict:
    # Re-hash and compare the exact candidate immediately before publication.
    verified = _revalidate_assets(cache_root, verified)
    payload = _manifest(
        runtime_profile,
        versions,
        verified.bag_relative,
        verified.checkpoint_relative,
        verified_at=_utc_now(),
    )
    publication = _publish_manifest(cache_root, payload)
    try:
        # Readiness is authoritative only while the same exact bag/checkpoint
        # identity and digest still exist after the atomic replacement.
        _revalidate_assets(cache_root, verified)
    except BaseException:
        removed = publication.remove()
        publication.close()
        if not removed:
            raise WorkerError(
                "READINESS_PUBLICATION_UNSAFE",
                "A changed checkpoint could not be detached from readiness safely.",
                EXIT_MODEL_VERIFICATION_FAILED,
            ) from None
        raise
    publication.close()
    return _provenance(payload)


def prepare_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text, create=True)
    runtime_profile, versions, _ = require_compatible_runtime()
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=False
    )
    verified = _verify_assets(cache_root, bag_path, checkpoint_path)
    return _publish_verified_readiness(
        cache_root,
        runtime_profile,
        versions,
        verified,
    )


def verify_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text)
    runtime_profile, versions, _ = require_compatible_runtime()
    try:
        load_readiness_manifest(cache_root)
    except WorkerError as exc:
        if exc.code == "CHECKPOINT_CHANGED_DURING_VERIFICATION":
            return _not_ready_result(runtime_profile, versions)
        raise
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=True
    )
    try:
        verified = _verify_assets(cache_root, bag_path, checkpoint_path)
    except WorkerError as exc:
        if exc.code == "CHECKPOINT_CHANGED_DURING_VERIFICATION":
            return _not_ready_result(runtime_profile, versions)
        raise
    return _publish_verified_readiness(
        cache_root,
        runtime_profile,
        versions,
        verified,
    )
