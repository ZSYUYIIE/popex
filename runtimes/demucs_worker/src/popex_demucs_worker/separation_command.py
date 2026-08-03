from __future__ import annotations

from typing import Any

from .constants import (
    ALLOWED_DEVICES,
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
    WORKER_VERSION,
)
from .model_commands import verify_model
from .runtime_support import _import_optional, _load_bag_yaml
from .paths import trusted_root, validate_input_file, validate_output_directory
from .probes import load_readiness_manifest
from .protocol import EXIT_SEPARATION_FAILED, WorkerError

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
        "runtimeProfile": payload["runtimeProfile"],
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
