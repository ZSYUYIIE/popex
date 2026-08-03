from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from .constants import BAG_FILE, BAG_SIGNATURE, CHECKPOINT_FILE, MODEL_REPOSITORY, MODEL_REVISION
from .protocol import (
    EXIT_MODEL_DOWNLOAD_FAILED,
    EXIT_MODEL_VERIFICATION_FAILED,
    EXIT_RUNTIME_INCOMPATIBLE,
    WorkerError,
)

_CREDENTIAL_ENV_KEYS = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "GOOGLE_APPLICATION_CREDENTIALS",
}

def _configure_environment(cache_root: Path, *, offline: bool) -> Path:
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["POPEX_DEMUCS_CACHE_ROOT"] = str(cache_root)
    os.environ["POPEX_DEMUCS_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["NO_COLOR"] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["POPEX_DISABLE_UPDATE_CHECKS"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    for key in list(os.environ):
        upper = key.upper()
        if key in _CREDENTIAL_ENV_KEYS or (
            not upper.startswith("POPEX_")
            and any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY"))
        ):
            os.environ.pop(key, None)
    return hub_cache


def _import_optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise WorkerError(
            "RUNTIME_INCOMPATIBLE",
            "The optional runtime is missing a required locked package.",
            EXIT_RUNTIME_INCOMPATIBLE,
        ) from exc


def _download_exact_assets(cache_root: Path, *, local_files_only: bool) -> tuple[Path, Path]:
    hub_cache = _configure_environment(cache_root, offline=local_files_only)
    hub = _import_optional("huggingface_hub")
    kwargs = {
        "repo_id": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "cache_dir": str(hub_cache),
        "token": False,
        "local_files_only": local_files_only,
    }
    try:
        bag = Path(hub.hf_hub_download(filename=BAG_FILE, **kwargs))
        checkpoint = Path(hub.hf_hub_download(filename=CHECKPOINT_FILE, **kwargs))
    except Exception as exc:
        if local_files_only:
            raise WorkerError(
                "MODEL_VERIFICATION_FAILED",
                "The exact cached model assets could not be resolved offline.",
                EXIT_MODEL_VERIFICATION_FAILED,
            ) from exc
        raise WorkerError(
            "MODEL_DOWNLOAD_FAILED",
            "The authorized model download could not be completed.",
            EXIT_MODEL_DOWNLOAD_FAILED,
            retryable=True,
        ) from exc
    return bag, checkpoint


def _load_bag_yaml(bag_path: Path) -> dict:
    yaml = _import_optional("yaml")
    try:
        payload = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WorkerError(
            "BAG_SCHEMA_INVALID",
            "The htdemucs bag definition is invalid.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if not isinstance(payload, dict) or payload.get("models") != [BAG_SIGNATURE]:
        raise WorkerError(
            "BAG_SIGNATURE_MISMATCH",
            "The htdemucs bag does not contain the approved model signature.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return payload


