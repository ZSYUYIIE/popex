"""Run the manual Windows real-model FastAPI stem-separation validation.

This script is intentionally executed by the lightweight repository Python. The
optional Demucs/PyTorch runtime remains isolated behind the configured worker
executable and runtime lock.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import sys
import time
import tracemalloc
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app import db
from app.config import Settings
from app.main import create_app
from app.media import secure_job_dir
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_NAME,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    REQUIRED_STEM_KINDS,
    STEM_MANIFEST_RELATIVE_PATH,
    STEM_MANIFEST_SCHEMA_VERSION,
)
from app.separation_runtime import (
    CHECKPOINT_SIZE_BYTES,
    SeparationRuntimeClient,
)

RUNTIME_PROFILE = "windows-x86_64-cpu-cpython313"
WORKER_VERSION = "1.0.0"
PROTOCOL_VERSION = 1
SAMPLE_RATE = 44_100
CHANNELS = 2
AUDIO_DURATION_SECONDS = 3.0
READINESS_RELATIVE = PurePosixPath("readiness/htdemucs-bf35a81b-v1.json")
_SAFE_ERROR_RE = re.compile(r"(?i)(token|secret|password|authorization|api[_-]?key)")
_URL_RE = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s,;\"']+")
_SPACE_RE = re.compile(r"\s+")


class ValidationError(RuntimeError):
    """Safe top-level failure for the manual validation."""


def _existing_regular_file(raw: str, name: str) -> Path:
    path = _absolute_normalized(raw, name)
    try:
        lexical = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"{name} must be an existing regular file.") from exc
    if path.is_symlink() or not path.is_file() or not resolved.is_file():
        raise ValidationError(f"{name} must be an existing regular non-symlink file.")
    if os.path.normcase(str(path)) != os.path.normcase(str(resolved)):
        raise ValidationError(f"{name} must not contain symbolic-link components.")
    if not lexical.st_size:
        raise ValidationError(f"{name} must not be empty.")
    return resolved


def _empty_directory(raw: str, name: str) -> Path:
    path = _absolute_normalized(raw, name)
    _reject_symlink_components(path, name)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValidationError(f"{name} must be a directory and not a symlink.")
        if any(path.iterdir()):
            raise ValidationError(f"{name} must be new or empty.")
    else:
        path.mkdir(parents=True, exist_ok=False)
    resolved = path.resolve(strict=True)
    if os.path.normcase(str(path)) != os.path.normcase(str(resolved)):
        raise ValidationError(f"{name} must not contain symbolic-link components.")
    return resolved


def _absolute_normalized(raw: str, name: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValidationError(f"{name} must be a non-empty local path.")
    path = Path(raw)
    if not path.is_absolute():
        raise ValidationError(f"{name} must be absolute.")
    normalized = Path(os.path.normpath(str(path)))
    if os.path.normcase(str(path)) != os.path.normcase(str(normalized)):
        raise ValidationError(f"{name} must be normalized.")
    return normalized


def _reject_symlink_components(path: Path, name: str) -> None:
    current = path
    existing: list[Path] = []
    while True:
        if current.exists():
            existing.append(current)
        if current.parent == current:
            break
        current = current.parent
    if any(item.is_symlink() for item in existing):
        raise ValidationError(f"{name} must not contain symbolic-link components.")


def _paths_are_distinct(paths: Iterable[Path]) -> None:
    normalized = [os.path.normcase(str(path)) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ValidationError("Worker, runtime-lock, cache, and data paths must be distinct.")


def _configure_privacy_environment(cache_root: Path) -> None:
    for name in tuple(os.environ):
        upper = name.upper()
        if upper in {
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
        } or any(marker in upper for marker in ("PASSWORD", "SECRET", "API_KEY")):
            os.environ.pop(name, None)
    os.environ.update(
        {
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(cache_root / "hub"),
            "HF_XET_CACHE": str(cache_root / "xet"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_UPDATE_CHECK": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    os.environ.pop("HF_HUB_OFFLINE", None)


def _generate_synthetic_audio(path: Path) -> dict[str, Any]:
    frames = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    time_axis = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    envelope = np.minimum(1.0, time_axis * 8.0) * np.minimum(
        1.0, (AUDIO_DURATION_SECONDS - time_axis) * 8.0
    )
    left = (
        0.20 * np.sin(2 * np.pi * 220.0 * time_axis)
        + 0.08 * np.sin(2 * np.pi * 440.0 * time_axis)
    )
    right = (
        0.18 * np.sin(2 * np.pi * 329.6276 * time_axis)
        + 0.07 * np.sin(2 * np.pi * 659.2551 * time_axis)
    )
    audio = np.column_stack((left, right)) * envelope[:, None]
    transient_length = int(SAMPLE_RATE * 0.012)
    transient = np.hanning(transient_length) * 0.45
    for start_seconds in (0.25, 0.75, 1.25, 1.75, 2.25, 2.75):
        start = int(start_seconds * SAMPLE_RATE)
        end = min(frames, start + transient_length)
        pulse = transient[: end - start]
        audio[start:end, 0] += pulse
        audio[start:end, 1] -= pulse * 0.75
    audio = np.clip(audio, -0.95, 0.95).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    info = sf.info(str(path))
    if (
        str(info.format).upper() != "WAV"
        or int(info.samplerate) != SAMPLE_RATE
        or int(info.channels) != CHANNELS
        or int(info.frames) != frames
    ):
        raise ValidationError("Synthetic analysis audio failed validation.")
    return {
        "durationSeconds": float(info.frames / info.samplerate),
        "sampleRate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "sizeBytes": path.stat().st_size,
    }


def _settings(
    *,
    data_dir: Path,
    worker: Path,
    runtime_lock: Path,
    cache_root: Path,
    timeout_seconds: int,
) -> Settings:
    return Settings(
        data_dir=data_dir,
        allowed_hosts=("youtube.com",),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=500,
        audio_quality="192",
        audio_analysis_enabled=True,
        stem_separation_enabled=True,
        stem_separation_version="demucs-worker-v3",
        stem_separation_worker_executable=worker,
        stem_separation_runtime_lock=runtime_lock,
        stem_separation_cache_dir=cache_root,
        stem_separation_runtime_profile=RUNTIME_PROFILE,
        stem_separation_device="cpu",
        stem_separation_timeout_seconds=timeout_seconds,
    )


def _create_completed_job(settings: Settings, job_id: str) -> dict[str, Any]:
    record = db.create_job(
        settings.database_path,
        job_id,
        source_type="upload",
        original_filename="synthetic-validation.wav",
    )
    job_dir = secure_job_dir(settings, job_id, create=True)
    analysis_path = job_dir / "analysis.wav"
    metadata = _generate_synthetic_audio(analysis_path)
    (job_dir / "source.wav").write_bytes(analysis_path.read_bytes())
    (job_dir / "metadata.json").write_text(
        json.dumps({"synthetic": True, **metadata}, sort_keys=True),
        encoding="utf-8",
    )
    analysis_json = job_dir / "analysis" / "audio-analysis.json"
    analysis_json.parent.mkdir(parents=True, exist_ok=True)
    analysis_json.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "analysisVersion": "manual-real-model-e2e-v1",
                "synthetic": True,
                "audio": metadata,
                "warnings": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    db.update_job(
        settings.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        message="Synthetic audio preparation and analysis are complete.",
        source_file_name="source.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        preparation_status="completed",
        analysis_status="completed",
        analysis_version="manual-real-model-e2e-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-05T00:00:00+00:00",
    )
    return db.get_job(settings.database_path, job_id) or record


def _assert_no_model_before_consent(cache_root: Path) -> None:
    forbidden = [
        path
        for path in cache_root.rglob("*")
        if path.is_file()
        and (
            path.name == AUDITED_CHECKPOINT_FILE
            or path.name == READINESS_RELATIVE.name
            or path.suffix.lower() in {".safetensors", ".th", ".ckpt"}
        )
    ]
    if forbidden:
        raise ValidationError("Passive startup created model or readiness assets before consent.")


def _load_and_verify_readiness(cache_root: Path) -> dict[str, Any]:
    readiness_path = cache_root.joinpath(*READINESS_RELATIVE.parts)
    try:
        payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("The audited readiness manifest was not published.") from exc
    expected = {
        "protocolVersion": PROTOCOL_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": AUDITED_DEMUCS_VERSION,
        "modelRepository": AUDITED_MODEL_REPOSITORY,
        "modelRevision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "offlineReady": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValidationError(f"Readiness field {key} does not match the audited contract.")
    assets = payload.get("cacheAssets")
    if not isinstance(assets, dict) or not isinstance(assets.get("checkpoint"), str):
        raise ValidationError("The readiness manifest has no safe checkpoint asset.")
    relative = PurePosixPath(assets["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in assets["checkpoint"]:
        raise ValidationError("The readiness checkpoint path is unsafe.")
    checkpoint = cache_root.joinpath(*relative.parts).resolve(strict=True)
    cache_resolved = cache_root.resolve(strict=True)
    if cache_resolved not in checkpoint.parents or not checkpoint.is_file():
        raise ValidationError("The readiness checkpoint escaped the trusted cache.")
    if checkpoint.stat().st_size != CHECKPOINT_SIZE_BYTES:
        raise ValidationError("The prepared checkpoint size is incorrect.")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != AUDITED_CHECKPOINT_SHA256:
        raise ValidationError("The prepared checkpoint digest is incorrect.")
    legacy = [
        path
        for path in cache_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".th", ".ckpt"}
    ]
    if legacy:
        raise ValidationError("A legacy model checkpoint was created.")
    return payload


def _wait_for_terminal_job(client: TestClient, job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.get(f"/api/jobs/{job_id}")
        if response.status_code != 200:
            raise ValidationError("The completed job could not be read from the API.")
        payload = response.json()
        status = payload.get("separation", {}).get("status")
        if status in {"completed", "failed"}:
            return payload
        if time.monotonic() >= deadline:
            raise ValidationError("The real separation attempt exceeded the validation timeout.")
        time.sleep(0.5)


def _validate_manifest_and_endpoints(
    *,
    client: TestClient,
    settings: Settings,
    job_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    record = db.get_job(settings.database_path, job_id)
    if record is None:
        raise ValidationError("The completed SQLite job is missing.")
    if (
        record.get("separation_status") != "completed"
        or record.get("separation_stage") != "completed"
        or float(record.get("separation_progress") or 0) != 100.0
        or record.get("stem_manifest_file_name") != STEM_MANIFEST_RELATIVE_PATH
        or not record.get("separated_at")
    ):
        raise ValidationError("SQLite did not record canonical separation completion.")

    job_dir = secure_job_dir(settings, job_id)
    manifest_path = job_dir / STEM_MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("The schema-3 stem manifest is unavailable.") from exc
    if manifest.get("schemaVersion") != STEM_MANIFEST_SCHEMA_VERSION:
        raise ValidationError("The published stem manifest is not schema 3.")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValidationError("The stem manifest has no model provenance.")
    expected_model = {
        "name": AUDITED_MODEL_NAME,
        "packageVersion": AUDITED_DEMUCS_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "repository": AUDITED_MODEL_REPOSITORY,
        "revision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "weightsIdentifier": f"sha256:{AUDITED_CHECKPOINT_SHA256}",
        "device": "cpu",
    }
    for key, value in expected_model.items():
        if model.get(key) != value:
            raise ValidationError(f"Manifest model field {key} is incorrect.")
    if model.get("torchVersion") != "2.13.0+cpu":
        raise ValidationError("The manifest does not report the exact CPU-only Torch build.")

    details_response = client.get(f"/api/jobs/{job_id}/stems")
    if details_response.status_code != 200:
        raise ValidationError("The stem details endpoint did not return the published result.")
    details = details_response.json()
    stems = details.get("stems")
    if not isinstance(stems, list) or [item.get("kind") for item in stems] != list(REQUIRED_STEM_KINDS):
        raise ValidationError("The stem details endpoint did not return the four stable stem kinds.")
    public_payloads = [details_response.text]
    safe_stems: list[dict[str, Any]] = []
    for stem in stems:
        kind = stem["kind"]
        file_name = stem.get("fileName")
        if not isinstance(file_name, str):
            raise ValidationError("A stem detail has no safe relative filename.")
        relative = PurePosixPath(file_name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in file_name:
            raise ValidationError("A stem detail exposed an unsafe path.")
        stored = job_dir.joinpath(*relative.parts).resolve(strict=True)
        if job_dir.resolve(strict=True) not in stored.parents:
            raise ValidationError("A stem detail escaped the job directory.")
        expected_bytes = stored.read_bytes()
        preview = client.get(f"/api/jobs/{job_id}/stems/{kind}/preview")
        download = client.get(f"/api/jobs/{job_id}/stems/{kind}/download")
        if preview.status_code != 200 or download.status_code != 200:
            raise ValidationError(f"The {kind} preview or download endpoint failed.")
        if preview.content != expected_bytes or download.content != expected_bytes:
            raise ValidationError(f"The {kind} endpoint bytes differ from the manifest-backed file.")
        if not preview.headers.get("content-type", "").startswith("audio/wav"):
            raise ValidationError(f"The {kind} preview has the wrong media type.")
        if f"{kind}.wav" not in download.headers.get("content-disposition", ""):
            raise ValidationError(f"The {kind} download filename is incorrect.")
        safe_stems.append(
            {
                "kind": kind,
                "sizeBytes": stem["sizeBytes"],
                "durationSeconds": stem["durationSeconds"],
                "sampleRate": stem["sampleRate"],
                "channels": stem["channels"],
            }
        )
    return manifest, safe_stems, public_payloads


def _assert_public_text_is_safe(texts: Iterable[str], known_paths: Iterable[Path]) -> None:
    combined = "\n".join(texts)
    lowered = combined.lower()
    for path in known_paths:
        raw = str(path)
        variants = {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}
        if any(value and value.lower() in lowered for value in variants):
            raise ValidationError("A public API response exposed a machine-local path.")
    forbidden = (
        "traceback (most recent call last)",
        "popex_demucs_runtime_lock",
        "hf_token",
        "authorization:",
    )
    if any(item in lowered for item in forbidden):
        raise ValidationError("A public API response exposed an internal diagnostic or credential marker.")


def _process_peak_working_set_bytes() -> int | None:
    if platform.system() != "Windows":
        return None

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if ok else None


def _safe_error_message(exc: Exception, known_paths: Iterable[Path]) -> str:
    text = str(exc) or "Windows real-model validation failed."
    text = _URL_RE.sub("[redacted]", text)
    for path in sorted((str(item) for item in known_paths), key=len, reverse=True):
        if path:
            text = text.replace(path, "[redacted]")
    text = _WINDOWS_PATH_RE.sub("[redacted]", text)
    if _SAFE_ERROR_RE.search(text):
        text = "Windows real-model validation failed at a protected boundary."
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:239] + "…" if len(text) > 240 else text


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValidationError("This validation requires Windows x86-64.")
    if sys.version_info[:2] != (3, 13) or sys.implementation.name != "cpython":
        raise ValidationError("This validation requires CPython 3.13.")

    worker = _existing_regular_file(args.worker, "worker")
    runtime_lock = _existing_regular_file(args.runtime_lock, "runtime-lock")
    cache_root = _empty_directory(args.cache_root, "cache-root")
    data_dir = _empty_directory(args.data_dir, "data-dir")
    _paths_are_distinct((worker, runtime_lock, cache_root, data_dir))
    known_paths = (worker, runtime_lock, cache_root, data_dir)
    _configure_privacy_environment(cache_root)

    total_started = time.monotonic()
    tracemalloc.start()
    runtime_client = SeparationRuntimeClient(
        worker,
        cache_root,
        runtime_lock_path=runtime_lock,
        expected_runtime_profile=RUNTIME_PROFILE,
        command_timeouts={"separate": float(args.separation_timeout_seconds)},
    )
    probe_started = time.monotonic()
    runtime_probe = runtime_client.runtime_probe()
    runtime_probe_seconds = time.monotonic() - probe_started
    if (
        runtime_probe.runtime_profile != RUNTIME_PROFILE
        or runtime_probe.worker_version != WORKER_VERSION
        or runtime_probe.demucs_version != AUDITED_DEMUCS_VERSION
        or runtime_probe.torch_version != "2.13.0+cpu"
    ):
        raise ValidationError("The installed Windows runtime does not match the exact profile.")

    settings = _settings(
        data_dir=data_dir,
        worker=worker,
        runtime_lock=runtime_lock,
        cache_root=cache_root,
        timeout_seconds=args.separation_timeout_seconds,
    )
    app = create_app(settings=settings)
    public_payloads: list[str] = []
    startup_started = time.monotonic()
    with TestClient(app) as client:
        startup_seconds = time.monotonic() - startup_started
        _assert_no_model_before_consent(cache_root)
        job_id = uuid4().hex
        _create_completed_job(settings, job_id)
        before = client.get(f"/api/jobs/{job_id}")
        if before.status_code != 200:
            raise ValidationError("The prepared synthetic job could not be read.")
        public_payloads.append(before.text)
        before_payload = before.json()
        runtime_state = before_payload.get("separation", {}).get("runtime", {}).get("state")
        if runtime_state != "download_required":
            raise ValidationError("Passive startup did not report download_required before consent.")
        _assert_no_model_before_consent(cache_root)

        request_started = time.monotonic()
        start_response = client.post(
            f"/api/jobs/{job_id}/separate",
            json={"allowModelDownload": True},
        )
        request_seconds = time.monotonic() - request_started
        public_payloads.append(start_response.text)
        if start_response.status_code != 202:
            raise ValidationError("The exact Boolean consent request was not accepted.")
        completed = _wait_for_terminal_job(client, job_id, args.poll_timeout_seconds)
        public_payloads.append(json.dumps(completed, sort_keys=True))
        separation = completed.get("separation", {})
        if separation.get("status") != "completed":
            message = separation.get("error") or separation.get("message") or "unknown failure"
            raise ValidationError(f"Real stem separation did not complete: {message}")

        readiness = _load_and_verify_readiness(cache_root)
        manifest, stems, endpoint_payloads = _validate_manifest_and_endpoints(
            client=client,
            settings=settings,
            job_id=job_id,
        )
        public_payloads.extend(endpoint_payloads)

    _assert_public_text_is_safe(public_payloads, known_paths)
    job_dir = settings.exports_dir / job_id
    run_root = job_dir / "stems" / "runs"
    run_directories = [path for path in run_root.iterdir() if path.is_dir()]
    if len(run_directories) != 1:
        raise ValidationError("The API sequence did not publish exactly one separation run.")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_seconds = time.monotonic() - total_started

    summary: dict[str, Any] = {
        "status": "ok",
        "platform": "windows-x86_64",
        "pythonVersion": platform.python_version(),
        "runtimeProfile": runtime_probe.runtime_profile,
        "workerVersion": runtime_probe.worker_version,
        "protocolVersion": PROTOCOL_VERSION,
        "demucsVersion": runtime_probe.demucs_version,
        "torchVersion": runtime_probe.torch_version,
        "huggingfaceHubVersion": runtime_probe.huggingface_hub_version,
        "safetensorsVersion": runtime_probe.safetensors_version,
        "pyyamlVersion": runtime_probe.pyyaml_version,
        "model": {
            "repository": AUDITED_MODEL_REPOSITORY,
            "revision": AUDITED_MODEL_REVISION,
            "checkpointFile": AUDITED_CHECKPOINT_FILE,
            "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
            "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
            "offlineReady": readiness.get("offlineReady") is True,
        },
        "consent": {
            "startupState": "download_required",
            "explicitBooleanAccepted": True,
            "modelPreparationObserved": True,
            "apiStartRequests": 1,
            "publishedRunCount": 1,
        },
        "job": {
            "separationStatus": "completed",
            "manifestSchemaVersion": manifest["schemaVersion"],
            "stemCount": len(stems),
        },
        "stems": stems,
        "elapsedSeconds": {
            "runtimeProbe": round(runtime_probe_seconds, 3),
            "appStartup": round(startup_seconds, 3),
            "consentRequestAndBackgroundWork": round(request_seconds, 3),
            "total": round(total_seconds, 3),
        },
        "memory": {
            "pythonCurrentBytes": current,
            "pythonPeakBytes": peak,
            "processPeakWorkingSetBytes": _process_peak_working_set_bytes(),
        },
        "privacy": {
            "machinePathsExposed": False,
            "credentialsExposed": False,
            "tracebacksExposed": False,
            "artifactsUploaded": False,
        },
    }
    encoded = json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True)
    _assert_public_text_is_safe((encoded,), known_paths)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the exact Windows CPU real-model FastAPI separation path."
    )
    parser.add_argument("--worker", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--separation-timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-timeout-seconds", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    known: list[Path] = []
    for value in (args.worker, args.runtime_lock, args.cache_root, args.data_dir):
        try:
            known.append(Path(value))
        except TypeError:
            pass
    try:
        if args.separation_timeout_seconds <= 0 or args.poll_timeout_seconds <= 0:
            raise ValidationError("Validation timeouts must be positive.")
        summary = run_validation(args)
    except Exception as exc:
        error = {
            "status": "error",
            "code": "WINDOWS_REAL_MODEL_E2E_FAILED",
            "message": _safe_error_message(exc, known),
        }
        sys.stderr.write(json.dumps(error, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")
        return 1
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
