from __future__ import annotations

import logging
import math
import stat
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Protocol

from app import db
from app.config import Settings
from app.media import MediaProcessingError, friendly_error, secure_job_dir
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_NAME,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    STEM_MANIFEST_RELATIVE_PATH,
    SeparationOptions,
    StemSeparationError,
    StemSeparationResult,
    separate_stems,
)
from app.separation_capability import (
    STATE_DOWNLOAD_REQUIRED,
    STATE_READY,
    SeparationCapability,
    probe_separation_capability,
)
from app.separation_runtime import (
    RuntimeConfigurationError,
    SeparationRuntimeClient,
    SeparationRuntimeError,
    WorkerErrorDetail,
)


CANONICAL_SEPARATION_STAGES = frozenset(
    {
        "preparing_separation",
        "separating_stems",
        "validating_stems",
        "saving_stems",
    }
)
VALID_SEPARATION_STATUSES = frozenset(
    {"not_started", "processing", "completed", "failed"}
)
VALID_SEPARATION_STAGES = frozenset(
    {
        "not_started",
        *CANONICAL_SEPARATION_STAGES,
        "completed",
        "failed",
    }
)


class SeparationProcessor(Protocol):
    def __call__(
        self,
        job_id: str,
        settings: Settings,
        options: SeparationOptions,
        *,
        stage_callback: Callable[[str, str, float], None] | None = None,
    ) -> StemSeparationResult: ...


class SeparationStartConflict(RuntimeError):
    """A separation request is valid but cannot start in the current state."""


class SeparationJobNotFound(RuntimeError):
    """The requested persisted job does not exist."""


class _UnavailableRuntimeClient:
    def __init__(self, message: str):
        self._error = RuntimeConfigurationError(
            WorkerErrorDetail(
                code="RUNTIME_CONFIGURATION_INVALID",
                message=message,
                retryable=False,
            )
        )

    def runtime_probe(self):
        raise self._error

    def model_probe(self):
        raise self._error


class SeparationService:
    """Own optional-runtime lifecycle, serialization, and separation persistence."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_client: Any | None = None,
        processor: SeparationProcessor = separate_stems,
        capability_probe: Callable[..., SeparationCapability | None] = (
            probe_separation_capability
        ),
    ) -> None:
        self.settings = settings
        self._processor = processor
        self._capability_probe = capability_probe
        self._lock = RLock()
        self._capability: SeparationCapability | None = None
        self._runtime_client = (
            runtime_client
            if runtime_client is not None
            else self._construct_runtime_client()
        )

    @property
    def runtime_client(self) -> Any | None:
        return self._runtime_client

    @property
    def capability(self) -> SeparationCapability | None:
        if not self.settings.stem_separation_enabled:
            return None
        with self._lock:
            if self._capability is None:
                self._capability = self._probe_capability()
            return self._capability

    def initialize(self) -> SeparationCapability | None:
        if not self.settings.stem_separation_enabled:
            return None
        return self.refresh_capability()

    def refresh_capability(self) -> SeparationCapability | None:
        if not self.settings.stem_separation_enabled:
            return None
        capability = self._probe_capability()
        with self._lock:
            self._capability = capability
        return capability

    def serialize_job(self, job: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.settings.stem_separation_enabled:
            return None

        capability = self.capability
        assert capability is not None
        status = _safe_status(job.get("separation_status"))
        stage = _safe_stage(job.get("separation_stage"), status)
        progress = _clamp_progress(job.get("separation_progress"))
        message = _summary_message(job, capability, status)
        error = _safe_error_value(job.get("separation_error"), self.settings)
        manifest_available = (
            job.get("stem_manifest_file_name") == STEM_MANIFEST_RELATIVE_PATH
        )
        can_start = (
            status in {"not_started", "failed"}
            and job.get("preparation_status") == "completed"
            and self._analysis_audio_available(str(job.get("id") or ""))
            and capability.actionable
        )
        job_id = str(job.get("id") or "")
        return {
            "enabled": True,
            "status": status,
            "stage": stage,
            "progress": progress,
            "message": message,
            "model": job.get("separation_model") or AUDITED_MODEL_NAME,
            "version": (
                job.get("separation_version")
                or self.settings.stem_separation_version
            ),
            "separatedAt": job.get("separated_at"),
            "canStart": can_start,
            "startUrl": (
                f"/api/jobs/{job_id}/separate" if can_start else None
            ),
            "detailsUrl": (
                f"/api/jobs/{job_id}/stems" if manifest_available else None
            ),
            "error": error,
            "runtime": capability.runtime_payload(),
        }

    def request_start(
        self,
        job_id: str,
        *,
        allow_model_download: bool,
        schedule: Callable[..., Any],
    ) -> dict[str, Any]:
        if not self.settings.stem_separation_enabled:
            raise SeparationStartConflict(
                "Stem separation is disabled by configuration."
            )

        record = db.get_job(self.settings.database_path, job_id)
        if record is None:
            raise SeparationJobNotFound("Job not found")
        status = _safe_status(record.get("separation_status"))
        if status == "processing":
            raise SeparationStartConflict("Stem separation is already running.")
        if status == "completed":
            raise SeparationStartConflict("Stem separation is already complete.")
        if record.get("preparation_status") != "completed":
            raise SeparationStartConflict(
                "Source preparation must finish before stem separation can start."
            )
        if not self._analysis_audio_available(job_id):
            raise SeparationStartConflict(
                "Analysis audio is missing or unsafe for this job."
            )

        capability = self.refresh_capability()
        if capability is None or not capability.actionable:
            raise SeparationStartConflict(
                capability.message
                if capability is not None
                else "Stem separation is unavailable."
            )
        prepare_model = capability.state == STATE_DOWNLOAD_REQUIRED
        if prepare_model and allow_model_download is not True:
            raise SeparationStartConflict(
                "Explicit consent is required before the first local model download."
            )

        claimed = db.claim_separation_attempt(
            self.settings.database_path,
            job_id,
            separation_version=self.settings.stem_separation_version,
            separation_model=AUDITED_MODEL_NAME,
            message=(
                "Preparing the verified stem-separation model."
                if prepare_model
                else "Preparing stem separation."
            ),
        )
        if not claimed:
            current = db.get_job(self.settings.database_path, job_id)
            if current is None:
                raise SeparationJobNotFound("Job not found")
            raise SeparationStartConflict(
                "Stem separation could not be claimed because the job state changed."
            )

        try:
            schedule(self.run_attempt, job_id, prepare_model)
        except Exception:
            logging.exception("Could not schedule stem separation for job %s", job_id)
            self._record_failure(
                job_id,
                "Stem separation could not be scheduled. Try again.",
                safe_error="Stem separation could not be scheduled.",
            )
            raise SeparationStartConflict(
                "Stem separation could not be scheduled. Try again."
            ) from None

        current = db.get_job(self.settings.database_path, job_id)
        if current is None:
            raise SeparationJobNotFound("Job not found")
        return current

    def run_attempt(self, job_id: str, prepare_model: bool) -> None:
        try:
            client = self._runtime_client
            if client is None:
                raise StemSeparationError(
                    "The optional stem-separation runtime is unavailable."
                )
            if prepare_model:
                db.update_job(
                    self.settings.database_path,
                    job_id,
                    separation_status="processing",
                    separation_stage="preparing_separation",
                    separation_progress=3,
                    separation_message=(
                        "Preparing the verified local stem-separation model."
                    ),
                    separation_error=None,
                )
                client.prepare_model(allow_model_download=True)
                capability = self.refresh_capability()
                if capability is None or capability.state != STATE_READY:
                    raise StemSeparationError(
                        "The verified local stem-separation model is not ready."
                    )

            result = self._processor(
                job_id,
                self.settings,
                self._separation_options(client),
                stage_callback=lambda stage, message, progress: self._update_stage(
                    job_id, stage, message, progress
                ),
            )
        except (StemSeparationError, SeparationRuntimeError, MediaProcessingError) as exc:
            self._record_failure(
                job_id,
                "Stem separation could not be completed. Try again.",
                safe_error=_safe_exception(exc, self.settings),
            )
            self.refresh_capability()
        except Exception:
            logging.exception("Unexpected stem separation failure for job %s", job_id)
            self._record_failure(
                job_id,
                "Stem separation could not be completed. Try again.",
                safe_error="Unexpected stem separation failure. Check server logs.",
            )
            self.refresh_capability()
        else:
            db.update_job(
                self.settings.database_path,
                job_id,
                separation_status="completed",
                separation_stage="completed",
                separation_progress=100,
                separation_message="Stem separation complete.",
                separation_version=result.separation_version,
                separation_model=result.model_name,
                stem_manifest_file_name=result.manifest_file_name,
                separated_at=result.created_at,
                separation_error=None,
            )

    def _construct_runtime_client(self) -> Any | None:
        if not self.settings.stem_separation_enabled:
            return None
        worker = self.settings.stem_separation_worker_executable
        cache_root = self._cache_root()
        if worker is None:
            return None
        try:
            return SeparationRuntimeClient(
                worker,
                cache_root,
                runtime_lock_path=self.settings.stem_separation_runtime_lock,
                expected_runtime_profile=(
                    self.settings.stem_separation_runtime_profile
                ),
                command_timeouts={
                    "separate": float(
                        self.settings.stem_separation_timeout_seconds
                    )
                },
            )
        except Exception:
            logging.warning(
                "Stem separation is enabled but trusted runtime configuration is invalid."
            )
            return _UnavailableRuntimeClient(
                "The optional stem-separation runtime configuration is unavailable."
            )

    def _probe_capability(self) -> SeparationCapability:
        capability = self._capability_probe(
            self._runtime_client,
            enabled=self.settings.stem_separation_enabled,
            device=self.settings.stem_separation_device,
        )
        assert capability is not None
        return capability

    def _cache_root(self) -> Path:
        return (
            self.settings.stem_separation_cache_dir
            or self.settings.data_dir / "runtime-cache" / "demucs"
        )

    def _separation_options(self, client: Any) -> SeparationOptions:
        return SeparationOptions(
            separation_version=self.settings.stem_separation_version,
            worker_runner=client,
            cache_root=self._cache_root(),
            expected_model_repository=AUDITED_MODEL_REPOSITORY,
            expected_model_revision=AUDITED_MODEL_REVISION,
            expected_checkpoint_file=AUDITED_CHECKPOINT_FILE,
            expected_checkpoint_sha256=AUDITED_CHECKPOINT_SHA256,
            expected_demucs_version=AUDITED_DEMUCS_VERSION,
            expected_runtime_profile=(
                self.settings.stem_separation_runtime_profile
            ),
            device=self.settings.stem_separation_device,
            timeout_seconds=float(
                self.settings.stem_separation_timeout_seconds
            ),
        )

    def _analysis_audio_available(self, job_id: str) -> bool:
        if not job_id:
            return False
        try:
            job_dir = secure_job_dir(self.settings, job_id)
            path = job_dir / "analysis.wav"
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                return False
            return path.resolve(strict=True).parent == job_dir.resolve(strict=True)
        except (MediaProcessingError, OSError, RuntimeError):
            return False

    def _update_stage(
        self,
        job_id: str,
        stage: str,
        message: str,
        progress: float,
    ) -> None:
        if stage not in CANONICAL_SEPARATION_STAGES:
            raise StemSeparationError(
                "The stem-separation processor reported an unknown stage."
            )
        safe_progress = _clamp_processing_progress(progress)
        db.update_job(
            self.settings.database_path,
            job_id,
            separation_status="processing",
            separation_stage=stage,
            separation_progress=safe_progress,
            separation_message=_safe_message(message),
            separation_error=None,
        )

    def _record_failure(
        self,
        job_id: str,
        message: str,
        *,
        safe_error: str,
    ) -> None:
        current = db.get_job(self.settings.database_path, job_id) or {}
        progress = _clamp_processing_progress(
            current.get("separation_progress", 0)
        )
        db.update_job(
            self.settings.database_path,
            job_id,
            separation_status="failed",
            separation_stage="failed",
            separation_progress=progress,
            separation_message=_safe_message(message),
            separation_error=safe_error,
        )


def _safe_status(value: object) -> str:
    return value if value in VALID_SEPARATION_STATUSES else "not_started"


def _safe_stage(value: object, status: str) -> str:
    if value in VALID_SEPARATION_STAGES:
        return str(value)
    return "completed" if status == "completed" else (
        "failed" if status == "failed" else "not_started"
    )


def _clamp_progress(value: object) -> float:
    if isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return round(max(0.0, min(100.0, number)), 1)


def _clamp_processing_progress(value: object) -> float:
    return min(99.0, _clamp_progress(value))


def _summary_message(
    job: Mapping[str, Any],
    capability: SeparationCapability,
    status: str,
) -> str:
    value = job.get("separation_message")
    if isinstance(value, str) and value.strip():
        return _safe_message(value)
    if status == "completed":
        return "Stem separation complete."
    if status == "processing":
        return "Stem separation is in progress."
    if status == "failed":
        return "Stem separation could not be completed. Try again."
    return _safe_message(capability.message)


def _safe_message(value: object) -> str:
    if not isinstance(value, str):
        return "Stem separation is unavailable."
    message = " ".join(value.replace("\x00", " ").split()).strip()
    if not message:
        return "Stem separation is unavailable."
    return message[:239] + "…" if len(message) > 240 else message


def _safe_error_value(value: object, settings: Settings) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if "traceback (most recent call last)" in value.lower():
        return "Stem separation failed."
    return friendly_error(value, settings=settings)


def _safe_exception(exc: BaseException, settings: Settings) -> str:
    value = str(exc)
    if not value or "traceback (most recent call last)" in value.lower():
        return "Stem separation failed."
    return friendly_error(value, settings=settings)


__all__ = [
    "CANONICAL_SEPARATION_STAGES",
    "SeparationJobNotFound",
    "SeparationService",
    "SeparationStartConflict",
]
