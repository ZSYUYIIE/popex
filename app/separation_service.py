from __future__ import annotations

import logging
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from app import db
from app.config import Settings
from app.media import MediaProcessingError, secure_job_dir
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
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}")
_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(token|password|passwd|secret|authorization|api[_-]?key|"
    r"access[_-]?key|runtime[_-]?lock|cache[_-]?root)\b\s*[:=]\s*[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
_URL_RE = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s,;\"']+")
_POSIX_PATH_RE = re.compile(r"(?<![:\w])/(?:[^\s,;\"']+)")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_SPACE_RE = re.compile(r"\s+")
_RUNTIME_CONFIGURATION_MESSAGE = (
    "The optional stem-separation runtime configuration is unavailable."
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


@dataclass(frozen=True, slots=True)
class _ManifestSnapshot:
    had_pointer: bool
    payload: bytes | None


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
        self._worker_slot = Lock()
        self._capability: SeparationCapability | None = None
        self._cache_ready = False
        self._configuration_error = _runtime_settings_error(settings)
        if self._configuration_error is not None:
            self._runtime_client = _UnavailableRuntimeClient(
                _RUNTIME_CONFIGURATION_MESSAGE
            )
        else:
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
        self._prepare_probe_environment()
        with self._lock:
            capability = self._capability
        if capability is None:
            capability = self.refresh_capability()
        return capability

    def initialize(self) -> SeparationCapability | None:
        if not self.settings.stem_separation_enabled:
            return None
        self._prepare_probe_environment()
        return self.refresh_capability()

    def refresh_capability(self) -> SeparationCapability | None:
        if not self.settings.stem_separation_enabled:
            return None
        self._prepare_probe_environment()
        capability = self._probe_capability()
        with self._lock:
            self._capability = capability
        return capability

    def serialize_job(self, job: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.settings.stem_separation_enabled:
            return None

        capability = self.capability
        assert capability is not None
        persisted_status = _known_status(job.get("separation_status"))
        status_known = persisted_status is not None
        status = persisted_status or "failed"
        stage = _safe_stage(job.get("separation_stage"), status, status_known)
        progress = _clamp_progress(job.get("separation_progress"))
        message = _summary_message(
            job,
            capability,
            status,
            status_known=status_known,
            settings=self.settings,
        )
        error = _safe_error_value(job.get("separation_error"), self.settings)
        manifest_available = (
            job.get("stem_manifest_file_name") == STEM_MANIFEST_RELATIVE_PATH
        )
        can_start = (
            status_known
            and status in {"not_started", "failed"}
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
        status = _known_status(record.get("separation_status"))
        if status is None:
            raise SeparationStartConflict(
                "Stem separation state is unavailable and cannot be started."
            )
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
            self._best_effort_record_failure(
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
        """Run one claimed attempt without allowing task exceptions to escape."""
        try:
            self._run_attempt_serialized(job_id, prepare_model)
        except Exception:
            logging.exception(
                "Stem-separation background safety boundary caught an unexpected "
                "failure for job %s",
                job_id,
            )
            self._best_effort_record_failure(
                job_id,
                "Stem separation could not be completed. Try again.",
                safe_error="Unexpected stem separation failure. Check server logs.",
            )
            self._safe_refresh_capability()

    def _run_attempt_serialized(self, job_id: str, prepare_model: bool) -> None:
        acquired = self._worker_slot.acquire(blocking=False)
        if not acquired:
            try:
                db.update_job(
                    self.settings.database_path,
                    job_id,
                    separation_status="processing",
                    separation_stage="preparing_separation",
                    separation_message=(
                        "Waiting for the current local stem separation to finish."
                    ),
                    separation_error=None,
                )
            except Exception:
                logging.exception(
                    "Could not persist queued separation state for job %s", job_id
                )
            self._worker_slot.acquire()
        try:
            self._run_attempt_in_slot(job_id, prepare_model)
        finally:
            self._worker_slot.release()

    def _run_attempt_in_slot(self, job_id: str, prepare_model: bool) -> None:
        previous_manifest = self._capture_published_manifest(job_id)
        try:
            client = self._runtime_client
            if client is None:
                raise StemSeparationError(
                    "The optional stem-separation runtime is unavailable."
                )

            if prepare_model:
                # A second first-use request may have waited behind another job.
                # Re-probe inside the shared slot so the verified model is prepared
                # at most once.
                capability = self.refresh_capability()
                if capability is not None and capability.state == STATE_READY:
                    prepare_model = False
                elif capability is None or capability.state != STATE_DOWNLOAD_REQUIRED:
                    raise StemSeparationError(
                        "The verified local stem-separation model is unavailable."
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
            self._best_effort_record_failure(
                job_id,
                "Stem separation could not be completed. Try again.",
                safe_error=_safe_exception(exc, self.settings),
            )
            self._safe_refresh_capability()
            return
        except Exception:
            logging.exception("Unexpected stem separation failure for job %s", job_id)
            self._best_effort_record_failure(
                job_id,
                "Stem separation could not be completed. Try again.",
                safe_error="Unexpected stem separation failure. Check server logs.",
            )
            self._safe_refresh_capability()
            return

        try:
            self._record_completion(job_id, result)
        except Exception:
            logging.exception(
                "Could not persist completed stem separation for job %s", job_id
            )
            self._restore_published_manifest(job_id, previous_manifest)
            self._best_effort_record_failure(
                job_id,
                "Separated audio could not be recorded. Try again.",
                safe_error="Stem separation results could not be recorded.",
            )

    def _construct_runtime_client(self) -> Any | None:
        if not self.settings.stem_separation_enabled:
            return None
        worker = self.settings.stem_separation_worker_executable
        if worker is None:
            return None
        try:
            return SeparationRuntimeClient(
                worker,
                self._cache_root(),
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
            self._configuration_error = _RUNTIME_CONFIGURATION_MESSAGE
            return _UnavailableRuntimeClient(_RUNTIME_CONFIGURATION_MESSAGE)

    def _prepare_probe_environment(self) -> None:
        if not self.settings.stem_separation_enabled:
            return
        with self._lock:
            if self._cache_ready:
                return
            if self._configuration_error is not None:
                self._runtime_client = _UnavailableRuntimeClient(
                    _RUNTIME_CONFIGURATION_MESSAGE
                )
                self._cache_ready = True
                return
            try:
                _create_safe_cache_root(self._cache_root())
            except Exception:
                logging.warning(
                    "Stem separation cache configuration is unavailable.",
                    exc_info=True,
                )
                self._configuration_error = _RUNTIME_CONFIGURATION_MESSAGE
                self._runtime_client = _UnavailableRuntimeClient(
                    _RUNTIME_CONFIGURATION_MESSAGE
                )
                self._capability = None
            self._cache_ready = True

    def _probe_capability(self) -> SeparationCapability:
        capability = self._capability_probe(
            self._runtime_client,
            enabled=self.settings.stem_separation_enabled,
            device=self.settings.stem_separation_device,
        )
        assert capability is not None
        return capability

    def _cache_root(self) -> Path:
        value = (
            self.settings.stem_separation_cache_dir
            or self.settings.data_dir / "runtime-cache" / "demucs"
        )
        if not isinstance(value, Path):
            raise RuntimeConfigurationError(
                WorkerErrorDetail(
                    code="RUNTIME_CONFIGURATION_INVALID",
                    message=_RUNTIME_CONFIGURATION_MESSAGE,
                    retryable=False,
                )
            )
        return value

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

    def _record_completion(
        self,
        job_id: str,
        result: StemSeparationResult,
    ) -> None:
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
            separation_error=_sanitize_public_text(
                safe_error,
                settings=self.settings,
                fallback="Stem separation failed.",
                limit=800,
            ),
        )

    def _best_effort_record_failure(
        self,
        job_id: str,
        message: str,
        *,
        safe_error: str,
    ) -> bool:
        for attempt in range(2):
            try:
                self._record_failure(
                    job_id,
                    message,
                    safe_error=safe_error,
                )
                return True
            except Exception:
                logging.exception(
                    "Could not persist retryable separation failure for job %s "
                    "(attempt %s)",
                    job_id,
                    attempt + 1,
                )
        logging.error(
            "Stem-separation state for job %s could not be recovered because local "
            "database writes remained unavailable.",
            job_id,
        )
        return False

    def _safe_refresh_capability(self) -> None:
        try:
            self.refresh_capability()
        except Exception:
            logging.exception("Could not refresh stem-separation capability")

    def _capture_published_manifest(self, job_id: str) -> _ManifestSnapshot:
        try:
            record = db.get_job(self.settings.database_path, job_id) or {}
        except Exception:
            logging.exception(
                "Could not inspect prior stem manifest pointer for job %s", job_id
            )
            return _ManifestSnapshot(had_pointer=True, payload=None)
        if record.get("stem_manifest_file_name") != STEM_MANIFEST_RELATIVE_PATH:
            return _ManifestSnapshot(had_pointer=False, payload=None)
        try:
            path, _ = _safe_manifest_path(self.settings, job_id, required=True)
            assert path is not None
            return _ManifestSnapshot(had_pointer=True, payload=path.read_bytes())
        except Exception:
            logging.exception(
                "Could not snapshot prior published stem manifest for job %s", job_id
            )
            return _ManifestSnapshot(had_pointer=True, payload=None)

    def _restore_published_manifest(
        self,
        job_id: str,
        snapshot: _ManifestSnapshot,
    ) -> None:
        try:
            path, job_dir = _safe_manifest_path(
                self.settings,
                job_id,
                required=False,
            )
            if job_dir is None:
                return
            target = job_dir / STEM_MANIFEST_RELATIVE_PATH
            if snapshot.had_pointer:
                if snapshot.payload is None:
                    logging.error(
                        "Prior stem manifest for job %s could not be restored because "
                        "its snapshot was unavailable.",
                        job_id,
                    )
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                parent_info = target.parent.lstat()
                if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(
                    parent_info.st_mode
                ):
                    raise OSError("unsafe stem manifest directory")
                temporary = target.parent / f".{target.name}.{uuid4().hex}.restoring"
                with temporary.open("xb") as handle:
                    handle.write(snapshot.payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                return
            if path is not None:
                path.unlink()
        except Exception:
            logging.exception(
                "Could not restore prior published stem state for job %s", job_id
            )


def _runtime_settings_error(settings: Settings) -> str | None:
    if not settings.stem_separation_enabled:
        return None
    marker = getattr(settings, "stem_separation_configuration_error", None)
    if isinstance(marker, str) and marker:
        return _RUNTIME_CONFIGURATION_MESSAGE
    if (
        not isinstance(settings.stem_separation_version, str)
        or _VERSION_RE.fullmatch(settings.stem_separation_version) is None
    ):
        return _RUNTIME_CONFIGURATION_MESSAGE
    if settings.stem_separation_device not in _ALLOWED_DEVICES:
        return _RUNTIME_CONFIGURATION_MESSAGE
    timeout = settings.stem_separation_timeout_seconds
    if type(timeout) is not int or timeout <= 0:
        return _RUNTIME_CONFIGURATION_MESSAGE
    profile = settings.stem_separation_runtime_profile
    if profile is not None and (
        not isinstance(profile, str) or _PROFILE_RE.fullmatch(profile) is None
    ):
        return _RUNTIME_CONFIGURATION_MESSAGE
    for value in (
        settings.stem_separation_worker_executable,
        settings.stem_separation_runtime_lock,
        settings.stem_separation_cache_dir,
    ):
        if value is None:
            continue
        if not isinstance(value, Path):
            return _RUNTIME_CONFIGURATION_MESSAGE
        text = os.fspath(value)
        if not text or "\x00" in text or not value.is_absolute():
            return _RUNTIME_CONFIGURATION_MESSAGE
        if Path(os.path.normpath(text)) != value:
            return _RUNTIME_CONFIGURATION_MESSAGE
    return None


def _create_safe_cache_root(path: Path) -> Path:
    text = os.fspath(path)
    if not text or "\x00" in text or not path.is_absolute():
        raise OSError("invalid cache root")
    if Path(os.path.normpath(text)) != path:
        raise OSError("cache root must be normalized")

    parts = path.parts
    if not parts:
        raise OSError("invalid cache root")
    current = Path(parts[0])
    for component in parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError("cache path contains an unsafe component")

    resolved = path.resolve(strict=True)
    if os.path.normcase(str(resolved)) != os.path.normcase(str(path)):
        raise OSError("cache root contains a symbolic-link component")
    return resolved


def _safe_manifest_path(
    settings: Settings,
    job_id: str,
    *,
    required: bool,
) -> tuple[Path | None, Path | None]:
    job_dir = secure_job_dir(settings, job_id)
    target = job_dir / STEM_MANIFEST_RELATIVE_PATH
    if not target.exists():
        if required:
            raise FileNotFoundError("published stem manifest is missing")
        return None, job_dir
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OSError("published stem manifest path is unsafe")
    resolved = target.resolve(strict=True)
    root = job_dir.resolve(strict=True)
    if root not in resolved.parents:
        raise OSError("published stem manifest escapes the job directory")
    return target, job_dir


def _known_status(value: object) -> str | None:
    return str(value) if value in VALID_SEPARATION_STATUSES else None


def _safe_stage(value: object, status: str, status_known: bool) -> str:
    if not status_known:
        return "failed"
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
    *,
    status_known: bool,
    settings: Settings,
) -> str:
    if not status_known:
        return "Stem separation state is unavailable. The job cannot be started."
    value = job.get("separation_message")
    if isinstance(value, str) and value.strip():
        return _sanitize_public_text(
            value,
            settings=settings,
            fallback="Stem separation is unavailable.",
            limit=240,
        )
    if status == "completed":
        return "Stem separation complete."
    if status == "processing":
        return "Stem separation is in progress."
    if status == "failed":
        return "Stem separation could not be completed. Try again."
    return _sanitize_public_text(
        capability.message,
        settings=settings,
        fallback="Stem separation is unavailable.",
        limit=240,
    )


def _safe_message(value: object) -> str:
    if not isinstance(value, str):
        return "Stem separation is unavailable."
    message = _SPACE_RE.sub(" ", _CONTROL_RE.sub(" ", value)).strip()
    if not message:
        return "Stem separation is unavailable."
    return message[:239] + "…" if len(message) > 240 else message


def _safe_error_value(value: object, settings: Settings) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _sanitize_public_text(
        value,
        settings=settings,
        fallback="Stem separation failed.",
        limit=800,
    )


def _safe_exception(exc: BaseException, settings: Settings) -> str:
    return _sanitize_public_text(
        str(exc),
        settings=settings,
        fallback="Stem separation failed.",
        limit=800,
    )


def _sanitize_public_text(
    value: object,
    *,
    settings: Settings,
    fallback: str,
    limit: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    if "traceback (most recent call last)" in value.lower():
        return fallback
    text = _CONTROL_RE.sub(" ", value)
    text = _URL_RE.sub("[redacted]", text)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    known_paths = (
        settings.data_dir,
        settings.stem_separation_worker_executable,
        settings.stem_separation_runtime_lock,
        settings.stem_separation_cache_dir,
    )
    for path in sorted(
        {str(path) for path in known_paths if isinstance(path, Path)},
        key=len,
        reverse=True,
    ):
        if path:
            text = text.replace(path, "[redacted]")
            text = text.replace(path.replace("\\", "/"), "[redacted]")
    text = _WINDOWS_PATH_RE.sub("[redacted]", text)
    text = _POSIX_PATH_RE.sub("[redacted]", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return fallback
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


__all__ = [
    "CANONICAL_SEPARATION_STAGES",
    "SeparationJobNotFound",
    "SeparationService",
    "SeparationStartConflict",
]
