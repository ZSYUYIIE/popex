from __future__ import annotations

from datetime import datetime, timezone

from .model_artifacts import _manifest, _provenance, _publish_manifest, _verify_assets
from .paths import trusted_root
from .probes import load_readiness_manifest, require_compatible_runtime
from .runtime_support import _download_exact_assets, _import_optional, _load_bag_yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text, create=True)
    runtime_profile, versions, _ = require_compatible_runtime()
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=False
    )
    _, bag_relative, checkpoint_relative = _verify_assets(
        cache_root, bag_path, checkpoint_path
    )
    payload = _manifest(
        runtime_profile,
        versions,
        bag_relative,
        checkpoint_relative,
        verified_at=_utc_now(),
    )
    _publish_manifest(cache_root, payload)
    return _provenance(payload)


def verify_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text)
    runtime_profile, versions, _ = require_compatible_runtime()
    load_readiness_manifest(cache_root)
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=True
    )
    _, bag_relative, checkpoint_relative = _verify_assets(
        cache_root, bag_path, checkpoint_path
    )
    payload = _manifest(
        runtime_profile,
        versions,
        bag_relative,
        checkpoint_relative,
        verified_at=_utc_now(),
    )
    _publish_manifest(cache_root, payload)
    return _provenance(payload)


