from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_DEVICES,
    BAG_FILE,
    BAG_SIGNATURE,
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    EXPECTED_AUDIO_CHANNELS,
    EXPECTED_MODEL_CLASS,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_SOURCES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    OUTPUT_SOURCES,
    PROTOCOL_VERSION,
    READINESS_RELATIVE_PATH,
    RUNTIME_PROFILE,
    WORKER_VERSION,
)
from .paths import (
    atomic_write_json,
    relative_asset_path,
    trusted_root,
    validate_input_file,
    validate_output_directory,
)
from .probes import load_readiness_manifest, require_compatible_runtime
from .protocol import (
    EXIT_MODEL_DOWNLOAD_FAILED,
    EXIT_MODEL_VERIFICATION_FAILED,
    EXIT_RUNTIME_INCOMPATIBLE,
    EXIT_SEPARATION_FAILED,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    versions: dict[str, str],
    bag_relative: str,
    checkpoint_relative: str,
    *,
    verified_at: str,
) -> dict:
    return {
        "schemaVersion": 1,
        "protocolVersion": PROTOCOL_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
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


def prepare_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text, create=True)
    versions = require_compatible_runtime()
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=False
    )
    _, bag_relative, checkpoint_relative = _verify_assets(
        cache_root, bag_path, checkpoint_path
    )
    payload = _manifest(
        versions,
        bag_relative,
        checkpoint_relative,
        verified_at=_utc_now(),
    )
    _publish_manifest(cache_root, payload)
    return _provenance(payload)


def verify_model(cache_root_text: str) -> dict:
    cache_root = trusted_root(cache_root_text)
    versions = require_compatible_runtime()
    load_readiness_manifest(cache_root)
    bag_path, checkpoint_path = _download_exact_assets(
        cache_root, local_files_only=True
    )
    _, bag_relative, checkpoint_relative = _verify_assets(
        cache_root, bag_path, checkpoint_path
    )
    payload = _manifest(
        versions,
        bag_relative,
        checkpoint_relative,
        verified_at=_utc_now(),
    )
    _publish_manifest(cache_root, payload)
    return _provenance(payload)


def _validate_loaded_model(model: Any, *, require_family: bool = True) -> None:
    model_type = type(model)
    if require_family and (
        model_type.__name__ != EXPECTED_MODEL_CLASS
        or model_type.__module__ != "demucs.htdemucs"
    ):
        raise WorkerError(
            "MODEL_FAMILY_MISMATCH",
            "The loaded checkpoint is not the approved HTDemucs model family.",
            EXIT_SEPARATION_FAILED,
        )
    if getattr(model, "audio_channels", None) != EXPECTED_AUDIO_CHANNELS:
        raise WorkerError(
            "MODEL_CHANNEL_MISMATCH",
            "The loaded model does not accept stereo audio.",
            EXIT_SEPARATION_FAILED,
        )
    if getattr(model, "samplerate", None) != EXPECTED_SAMPLE_RATE:
        raise WorkerError(
            "MODEL_SAMPLE_RATE_MISMATCH",
            "The loaded model does not use the required sample rate.",
            EXIT_SEPARATION_FAILED,
        )
    if tuple(getattr(model, "sources", ())) != EXPECTED_SOURCES:
        raise WorkerError(
            "MODEL_SOURCE_MISMATCH",
            "The loaded model source order does not match the approved profile.",
            EXIT_SEPARATION_FAILED,
        )


def _separator_with_verified_bag(separator_class: type, bag: Any, device: str) -> Any:
    class VerifiedSeparator(separator_class):
        def __init__(self) -> None:
            self._popex_verified_bag = bag
            super().__init__(model="popex-verified-htdemucs", device=device, progress=False)

        def _load_model(self) -> None:
            self._model = self._popex_verified_bag
            self._audio_channels = self._model.audio_channels
            self._samplerate = self._model.samplerate

    return VerifiedSeparator()


def separate(
    cache_root_text: str,
    workspace_root_text: str,
    input_relative: str,
    output_relative: str,
    device: str,
) -> dict:
    if device not in ALLOWED_DEVICES:
        raise WorkerError(
            "INVALID_DEVICE",
            "The requested separation device is not supported.",
            30,
        )
    cache_root = trusted_root(cache_root_text)
    workspace_root = trusted_root(workspace_root_text)
    verify_model(str(cache_root))
    payload, _, bag_path, checkpoint_path = load_readiness_manifest(cache_root)
    input_path = validate_input_file(workspace_root, input_relative)
    output_path = validate_output_directory(workspace_root, output_relative)

    demucs_hf = _import_optional("demucs.hf")
    demucs_apply = _import_optional("demucs.apply")
    demucs_api = _import_optional("demucs.api")
    try:
        model = demucs_hf.load_safetensors_model(checkpoint_path)
        _validate_loaded_model(model)
        bag_data = _load_bag_yaml(bag_path)
        bag_kwargs: dict[str, Any] = {}
        if "weights" in bag_data:
            bag_kwargs["weights"] = bag_data["weights"]
        if "segment" in bag_data:
            bag_kwargs["segment"] = bag_data["segment"]
        bag = demucs_apply.BagOfModels([model], **bag_kwargs)
        _validate_loaded_model(bag, require_family=False)
        separator = _separator_with_verified_bag(demucs_api.Separator, bag, device)
        _, separated = separator.separate_audio_file(input_path)
        if not isinstance(separated, dict) or set(separated) != set(EXPECTED_SOURCES):
            raise WorkerError(
                "SEPARATION_OUTPUT_INVALID",
                "Demucs did not return the four approved sources.",
                EXIT_SEPARATION_FAILED,
            )
        for source in OUTPUT_SOURCES:
            demucs_api.save_audio(
                separated[source],
                output_path / f"{source}.wav",
                samplerate=EXPECTED_SAMPLE_RATE,
                bits_per_sample=16,
            )
    except (KeyboardInterrupt, TimeoutError):
        raise
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            "SEPARATION_FAILED",
            "Stem separation or output generation failed.",
            EXIT_SEPARATION_FAILED,
            retryable=True,
        ) from exc

    expected_files = [f"{source}.wav" for source in OUTPUT_SOURCES]
    actual = sorted(path.name for path in output_path.iterdir())
    if actual != sorted(expected_files) or any(
        not (output_path / name).is_file() or (output_path / name).is_symlink()
        for name in expected_files
    ):
        raise WorkerError(
            "SEPARATION_OUTPUT_INVALID",
            "The worker output directory does not contain exactly four valid stems.",
            EXIT_SEPARATION_FAILED,
        )
    return {
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": payload["torchVersion"],
        "huggingfaceHubVersion": payload["huggingfaceHubVersion"],
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSha256": CHECKPOINT_SHA256,
        "device": device,
        "outputs": expected_files,
    }
