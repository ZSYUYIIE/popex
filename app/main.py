from __future__ import annotations

import logging
import math
import mimetypes
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, HttpUrl, StrictBool

from app import db
from app.analysis import (
    ANALYSIS_JSON_RELATIVE_PATH,
    AudioAnalysisError,
    AudioAnalysisResult,
    analysis_json_path,
    analyze_audio,
    load_analysis,
)
from app.config import SUPPORTED_MEDIA_EXTENSIONS, Settings
from app.media import (
    MediaProcessingError,
    MediaResult,
    cleanup_job_dir,
    dependency_report,
    friendly_error,
    generated_source_name,
    process_upload,
    process_url,
    secure_job_dir,
)
from app.separation import StemSeparationResult, separate_stems
from app.separation_artifacts import (
    StemArtifactError,
    StemKindNotFoundError,
    StemManifestUnavailableError,
    load_stem_details,
    resolve_stem_artifact,
)
from app.separation_service import (
    SeparationJobNotFound,
    SeparationService,
    SeparationStartConflict,
)
from app.interpretation_artifacts import (
    InterpretationArtifactError,
    InterpretationArtifactUnavailableError,
    interpretation_json_path,
    load_interpretation_details,
)
from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    InterpretationPipelineError,
    InterpretationPipelineResult,
    interpret_transcription_job,
)
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HarmonyArtifactError,
    HarmonyArtifactUnavailableError,
    harmony_artifact_path,
    harmony_artifact_scope,
    harmony_attempt_artifact_file_name,
    load_harmony_artifact,
    reconcile_harmony_attempt_artifacts,
    remove_harmony_artifact,
)
from app.harmony_pipeline import (
    HARMONY_PIPELINE_VERSION,
    HarmonyPipelineError,
    HarmonyPipelineResult,
    infer_harmony_job,
)
from app.transcription_draft import INTERPRETATION_DRAFT_RELATIVE_PATH
from app.transcription_artifacts import (
    TranscriptionArtifactError,
    TranscriptionArtifactUnavailableError,
    load_transcription_details,
    transcription_json_path,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    load_raw_transcription,
)
from app.transcription_pipeline import (
    TRANSCRIPTION_VERSION,
    TranscriptionPipelineError,
    TranscriptionPipelineResult,
    transcribe_job,
)


UrlProcessor = Callable[
    [str, str, Settings, Callable[[str, str, float], None], Callable[[float], None]],
    MediaResult,
]
UploadProcessor = Callable[
    [str, str, str, Settings, Callable[[str, str, float], None], Callable[[float], None]],
    MediaResult,
]
AnalysisProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    AudioAnalysisResult,
]
SeparationProcessor = Callable[..., StemSeparationResult]
TranscriptionProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    TranscriptionPipelineResult,
]
InterpretationProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    InterpretationPipelineResult,
]
HarmonyProcessor = Callable[..., HarmonyPipelineResult]

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_MIME_PREFIXES = ("audio/", "video/")
ALLOWED_GENERIC_MIME_TYPES = {
    "",
    "application/octet-stream",
    "application/x-m4a",
    "application/ogg",
}
ANALYSIS_STAGES = {
    "analyzing_audio",
    "detecting_beats",
    "estimating_key",
    "saving_analysis",
}
PREPARATION_PROGRESS_LIMIT = 64.0
ANALYSIS_FAILURE_PROGRESS = 95.0
_HARMONY_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
_HARMONY_ARTIFACT_FILE_RE = re.compile(
    r"harmony/harmonic-context(?:\.[a-f0-9]{32})?\.json"
)
_INTERNAL_SEPARATION_FIELDS = frozenset(
    {
        "separation_status",
        "separation_stage",
        "separation_progress",
        "separation_message",
        "separation_version",
        "separation_model",
        "stem_manifest_file_name",
        "separated_at",
        "separation_error",
    }
)
_INTERNAL_TRANSCRIPTION_FIELDS = frozenset(
    {
        "transcription_status",
        "transcription_stage",
        "transcription_progress",
        "transcription_message",
        "transcription_version",
        "transcription_artifact_file_name",
        "transcribed_at",
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
        "transcription_error",
    }
)

_INTERNAL_INTERPRETATION_FIELDS = frozenset(
    {
        "interpretation_status",
        "interpretation_stage",
        "interpretation_progress",
        "interpretation_message",
        "interpretation_version",
        "interpretation_artifact_file_name",
        "interpreted_at",
        "interpretation_part_count",
        "interpretation_phrase_count",
        "interpretation_pitched_item_count",
        "interpretation_percussion_item_count",
        "interpretation_warning_count",
        "interpretation_error",
    }
)


_INTERNAL_HARMONY_FIELDS = frozenset(
    {
        "harmony_status",
        "harmony_stage",
        "harmony_progress",
        "harmony_message",
        "harmony_attempt_id",
        "harmony_attempt_version",
        "harmony_version",
        "harmony_artifact_file_name",
        "harmonized_at",
        "harmony_source_transcription_version",
        "harmony_source_transcription_artifact_file_name",
        "harmony_source_transcribed_at",
        "harmony_event_count",
        "harmony_segment_count",
        "harmony_resolved_segment_count",
        "harmony_unresolved_segment_count",
        "harmony_unresolved_event_count",
        "harmony_warning_count",
        "harmony_used_interpretation_context",
        "harmony_error",
    }
)


class JobCreate(BaseModel):
    url: HttpUrl


class SeparationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowModelDownload: StrictBool = False


def create_app(
    settings: Settings | None = None,
    url_processor: UrlProcessor = process_url,
    upload_processor: UploadProcessor = process_upload,
    analysis_processor: AnalysisProcessor = analyze_audio,
    separation_runtime_client: Any | None = None,
    separation_processor: SeparationProcessor = separate_stems,
    transcription_processor: TranscriptionProcessor = transcribe_job,
    interpretation_processor: InterpretationProcessor = interpret_transcription_job,
    harmony_processor: HarmonyProcessor = infer_harmony_job,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    separation_service = SeparationService(
        app_settings,
        runtime_client=separation_runtime_client,
        processor=separation_processor,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_settings.ensure_directories()
        db.init_database(app_settings.database_path)
        db.fail_incomplete_jobs(app_settings.database_path)
        app.state.dependencies = dependency_report(app_settings)
        app.state.separation_service = separation_service
        separation_service.initialize()
        if (
            not app.state.dependencies["ffmpeg"]
            or not app.state.dependencies["ffprobe"]
        ):
            logging.warning(
                "PopEx started without FFmpeg/ffprobe. Media jobs will fail until "
                "the missing executable is installed."
            )
        yield

    app = FastAPI(
        title="PopEx",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.separation_service = separation_service
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    def serialize_job(record: dict) -> dict:
        return _serialize_job(record, app_settings, separation_service)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(BASE_DIR / "templates" / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        dependencies = getattr(
            app.state,
            "dependencies",
            dependency_report(app_settings),
        )
        return {
            "status": "ok" if all(dependencies.values()) else "degraded",
            "dependencies": dependencies,
        }

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_url_job(
        payload: JobCreate,
        background_tasks: BackgroundTasks,
    ) -> dict:
        source_url = str(payload.url)
        _validate_source_url(source_url, app_settings.allowed_hosts)
        job_id = uuid4().hex
        job = db.create_job(
            app_settings.database_path,
            job_id,
            source_type="url",
            source_url=source_url,
        )
        background_tasks.add_task(
            _run_url_job,
            job_id,
            source_url,
            app_settings,
            url_processor,
            analysis_processor,
        )
        return serialize_job(job)

    @app.post("/api/uploads", status_code=status.HTTP_202_ACCEPTED)
    async def submit_upload_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict:
        original_filename = file.filename or ""
        extension = Path(original_filename).suffix.lower()
        if extension not in SUPPORTED_MEDIA_EXTENSIONS:
            await file.close()
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported file extension. Supported types: "
                    + ", ".join(SUPPORTED_MEDIA_EXTENSIONS)
                ),
            )

        content_type = (file.content_type or "").lower()
        if not (
            content_type.startswith(ALLOWED_MIME_PREFIXES)
            or content_type in ALLOWED_GENERIC_MIME_TYPES
        ):
            await file.close()
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported media MIME type '{content_type}'.",
            )

        job_id = uuid4().hex
        job = db.create_job(
            app_settings.database_path,
            job_id,
            source_type="upload",
            original_filename=original_filename,
        )
        db.update_job(
            app_settings.database_path,
            job_id,
            status="processing",
            stage="validating",
            progress=2,
            message="Validating local upload.",
            preparation_status="processing",
        )

        try:
            job_dir = secure_job_dir(app_settings, job_id, create=True)
            safe_name = generated_source_name(extension)
            temporary_path = job_dir / f"{safe_name}.uploading"
            source_path = job_dir / safe_name
            total = 0
            with temporary_path.open("xb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > app_settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Upload exceeds the "
                                f"{settings.max_upload_mb} MB limit."
                            ),
                        )
                    output.write(chunk)
            if total == 0:
                raise HTTPException(
                    status_code=422,
                    detail="The uploaded file is empty.",
                )
            temporary_path.replace(source_path)
            db.update_job(
                app_settings.database_path,
                job_id,
                stage="importing",
                progress=15,
                message="Local source saved safely.",
                source_file_name=safe_name,
                source_format=extension.lstrip("."),
            )
        except HTTPException as exc:
            cleanup_job_dir(app_settings.exports_dir / job_id)
            db.update_job(
                app_settings.database_path,
                job_id,
                status="failed",
                stage="failed",
                message="Upload rejected.",
                preparation_status="failed",
                error=str(exc.detail),
            )
            raise
        except OSError:
            logging.exception("Could not store upload for job %s", job_id)
            cleanup_job_dir(app_settings.exports_dir / job_id)
            db.update_job(
                app_settings.database_path,
                job_id,
                status="failed",
                stage="failed",
                message="Upload storage failed.",
                preparation_status="failed",
                error="The uploaded file could not be saved.",
            )
            raise HTTPException(
                status_code=500,
                detail="The uploaded file could not be saved.",
            )
        finally:
            await file.close()

        background_tasks.add_task(
            _run_upload_job,
            job_id,
            safe_name,
            original_filename,
            app_settings,
            upload_processor,
            analysis_processor,
        )
        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or job)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [
            serialize_job(job)
            for job in db.list_jobs(app_settings.database_path)
        ]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return serialize_job(record)

    @app.post(
        "/api/jobs/{job_id}/analyze",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def analyze_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        force: bool = Query(False),
    ) -> dict:
        if not app_settings.audio_analysis_enabled:
            raise HTTPException(
                status_code=409,
                detail="Audio analysis is disabled by configuration.",
            )

        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.get("analysis_status") == "processing" or (
            record.get("status") == "processing"
            and record.get("stage") in ANALYSIS_STAGES
        ):
            raise HTTPException(
                status_code=409,
                detail="Audio analysis is already running.",
            )

        wav_name = record.get("normalized_file_name") or "analysis.wav"
        wav_path = _resolve_job_file(app_settings, job_id, wav_name)
        if not wav_path.is_file():
            raise HTTPException(
                status_code=409,
                detail="Analysis audio is missing for this job.",
            )

        if (
            record.get("analysis_status") == "completed"
            and record.get("analysis_version")
            == app_settings.audio_analysis_version
            and not force
        ):
            return serialize_job(record)

        db.update_job(
            app_settings.database_path,
            job_id,
            status="processing",
            stage="analyzing_audio",
            progress=66,
            message="Analyzing timing.",
            preparation_status="completed",
            error=None,
            analysis_status="processing",
            analysis_version=app_settings.audio_analysis_version,
            analysis_error=None,
        )
        background_tasks.add_task(
            _run_analysis_job,
            job_id,
            app_settings,
            analysis_processor,
        )
        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)

    @app.post(
        "/api/jobs/{job_id}/separate",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def separate_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        payload: SeparationStartRequest = Body(
            default_factory=SeparationStartRequest
        ),
    ) -> dict:
        try:
            record = separation_service.request_start(
                job_id,
                allow_model_download=payload.allowModelDownload,
                schedule=background_tasks.add_task,
             )
        except SeparationJobNotFound:
            raise HTTPException(status_code=404, detail="Job not found") from None
        except SeparationStartConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return serialize_job(record)

    @app.post(
        "/api/jobs/{job_id}/transcribe",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def transcribe_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        force: str | None = Query(None),
    ) -> dict:
        force_value = _strict_query_bool(force, field="force", default=False)
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if (
            record.get("preparation_status") != "completed"
            or record.get("analysis_status") != "completed"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Source preparation and audio analysis must be complete before "
                    "raw transcription."
                ),
            )
        transcription_status = record.get("transcription_status") or "not_started"
        if transcription_status == "processing":
            raise HTTPException(
                status_code=409,
                detail="Raw transcription is already running.",
            )
        if transcription_status == "completed" and not force_value:
            raise HTTPException(
                status_code=409,
                detail="Raw transcription is already complete; use force=true to rerun.",
            )
        claimed = db.claim_transcription_attempt(
            app_settings.database_path,
            job_id,
            transcription_version=TRANSCRIPTION_VERSION,
            force=force_value,
        )
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Raw transcription could not be started in the current state.",
            )
        background_tasks.add_task(
            _run_transcription_job,
            job_id,
            app_settings,
            transcription_processor,
        )
        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)

    @app.post(
        "/api/jobs/{job_id}/interpret",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def interpret_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        force: str | None = Query(None),
    ) -> dict:
        force_value = _strict_query_bool(force, field="force", default=False)
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if (
            record.get("transcription_status") != "completed"
            or record.get("transcription_artifact_file_name")
            != RAW_TRANSCRIPTION_RELATIVE_PATH
        ):
            raise HTTPException(
                status_code=409,
                detail="Completed raw transcription is required before editable interpretation.",
            )
        try:
            transcription_json_path(job_id, app_settings, record)
        except (
            TranscriptionArtifactUnavailableError,
            TranscriptionArtifactError,
        ):
            raise HTTPException(
                status_code=409,
                detail="A valid published raw transcription is required before editable interpretation.",
            ) from None

        interpretation_status = record.get("interpretation_status") or "not_started"
        if interpretation_status == "processing":
            raise HTTPException(
                status_code=409,
                detail="Editable interpretation is already running.",
            )
        if interpretation_status == "completed" and not force_value:
            raise HTTPException(
                status_code=409,
                detail="Editable interpretation is already complete; use force=true to re-interpret.",
            )
        claimed = db.claim_interpretation_attempt(
            app_settings.database_path,
            job_id,
            interpretation_version=INTERPRETATION_PIPELINE_VERSION,
            force=force_value,
        )
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Editable interpretation could not be started in the current state.",
            )
        background_tasks.add_task(
            _run_interpretation_job,
            job_id,
            app_settings,
            interpretation_processor,
        )
        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)

    @app.post(
        "/api/jobs/{job_id}/harmonize",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def harmonize_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        force: str | None = Query(None),
    ) -> dict:
        force_value = _strict_query_bool(force, field="force", default=False)
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if (
            record.get("preparation_status") != "completed"
            or record.get("analysis_status") != "completed"
            or record.get("analysis_json_file_name")
            != ANALYSIS_JSON_RELATIVE_PATH
            or not isinstance(record.get("analysis_version"), str)
            or not record.get("analysis_version")
            or record.get("transcription_status") != "completed"
            or record.get("transcription_artifact_file_name")
            != RAW_TRANSCRIPTION_RELATIVE_PATH
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Completed canonical audio analysis and raw transcription "
                    "are required before harmonic context."
                ),
            )
        raw_transcription = _load_matching_raw_transcription_for_harmony(
            job_id,
            app_settings,
            record,
        )
        if raw_transcription is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid matching raw transcription is required before "
                    "harmonic context."
                ),
            )
        try:
            analysis = load_analysis(job_id, app_settings)
        except (AudioAnalysisError, MediaProcessingError):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid matching audio analysis is required before "
                    "harmonic context."
                ),
            ) from None
        if (
            not isinstance(analysis, dict)
            or analysis.get("analysisVersion") != record.get("analysis_version")
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid matching audio analysis is required before "
                    "harmonic context."
                ),
            )

        harmony_status = record.get("harmony_status") or "not_started"
        if harmony_status == "processing" and not force_value:
            raise HTTPException(
                status_code=409,
                detail="Harmonic-context processing is already running.",
            )
        if harmony_status == "completed" and not force_value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Harmonic context is already complete; use force=true to "
                    "re-harmonize."
                ),
            )
        attempt_id = db.claim_harmony_attempt(
            app_settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
            force=force_value,
        )
        if attempt_id is None:
            raise HTTPException(
                status_code=409,
                detail="Harmonic context could not be started in the current state.",
            )
        if not isinstance(attempt_id, str) or not attempt_id:
            logging.error("Harmony claim returned an invalid attempt ID for %s", job_id)
            raise HTTPException(
                status_code=500,
                detail="Harmonic context could not be started safely.",
            )
        current = db.get_job(app_settings.database_path, job_id)
        background_tasks.add_task(
            _run_harmony_job,
            job_id,
            app_settings,
            harmony_processor,
            attempt_id,
        )
        return serialize_job(current or record)

    @app.get("/api/jobs/{job_id}/analysis")
    def get_analysis(job_id: str) -> dict:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            result = load_analysis(job_id, app_settings)
        except (AudioAnalysisError, MediaProcessingError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "available": result is not None,
            "status": record.get("analysis_status") or "not_started",
            "analysisVersion": record.get("analysis_version"),
            "summary": _analysis_summary(record),
            "result": result,
            "warnings": result.get('warnings', []) if result else [],
            "downloadUrl": (
                f"/api/jobs/{job_id}/analysis/download" if result else None
            ),
            "error": record.get("analysis_error"),
        }

    @app.get("/api/jobs/{job_id}/analysis/download")
    def download_analysis(job_id: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.get("analysis_json_file_name") != ANALYSIS_JSON_RELATIVE_PATH:
            raise HTTPException(
                status_code=404,
                detail="Audio analysis is not available",
            )
        try:
            path = analysis_json_path(job_id, app_settings)
        except MediaProcessingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Audio analysis is not available",
            )
        return FileResponse(
            path,
            filename="audio-analysis.json",
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/transcription")
    def get_transcription(
        job_id: str,
        include_events: str | None = Query(None, alias="includeEvents"),
    ) -> dict:
        include_events_value = _strict_query_bool(
            include_events, field="includeEvents", default=False
        )
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            details = load_transcription_details(job_id, app_settings, record)
        except TranscriptionArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published raw transcription is unavailable.",
            ) from None
        except TranscriptionArtifactError:
            logging.exception(
                "Published raw transcription failed validation for job %s", job_id
            )
            raise HTTPException(
                status_code=500,
                detail="Published raw transcription could not be validated.",
            ) from None
        return details.payload(include_events=include_events_value)

    @app.get("/api/jobs/{job_id}/transcription/download")
    def download_transcription(job_id: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            path = transcription_json_path(job_id, app_settings, record)
        except TranscriptionArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published raw transcription is unavailable.",
            ) from None
        except TranscriptionArtifactError:
            logging.exception(
                "Published raw transcription failed validation for job %s", job_id
            )
            raise HTTPException(
                status_code=500,
                detail="Published raw transcription could not be validated.",
            ) from None
        return FileResponse(
            path,
            filename="raw-transcription.json",
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/interpretation")
    def get_interpretation(
        job_id: str,
        include_items: str | None = Query(None, alias="includeItems"),
    ) -> dict:
        include_items_value = _strict_query_bool(
            include_items, field="includeItems", default=False
        )
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            details = load_interpretation_details(job_id, app_settings, record)
        except InterpretationArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published editable interpretation is unavailable.",
            ) from None
        except InterpretationArtifactError:
            logging.exception(
                "Published editable interpretation failed validation for job %s", job_id
            )
            raise HTTPException(
                status_code=500,
                detail="Published editable interpretation could not be validated.",
            ) from None
        return details.payload(include_items=include_items_value)

    @app.get("/api/jobs/{job_id}/interpretation/download")
    def download_interpretation(job_id: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            path = interpretation_json_path(job_id, app_settings, record)
        except InterpretationArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published editable interpretation is unavailable.",
            ) from None
        except InterpretationArtifactError:
            logging.exception(
                "Published editable interpretation failed validation for job %s", job_id
            )
            raise HTTPException(
                status_code=500,
                detail="Published editable interpretation could not be validated.",
            ) from None
        return FileResponse(
            path,
            filename="editable-interpretation.json",
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/harmony")
    def get_harmony(
        job_id: str,
        include_segments: str | None = Query(None, alias="includeSegments"),
    ) -> dict:
        include_segments_value = _strict_query_bool(
            include_segments,
            field="includeSegments",
            default=False,
        )
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        artifact = _load_harmony_for_http(job_id, app_settings, record)
        return _harmony_details_payload(
            artifact,
            record,
            include_segments=include_segments_value,
        )

    @app.get("/api/jobs/{job_id}/harmony/download")
    def download_harmony(job_id: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        artifact_file_name = record.get("harmony_artifact_file_name")
        if not _is_harmony_artifact_file_name(artifact_file_name):
            raise HTTPException(
                status_code=404,
                detail="Published harmonic context is unavailable.",
            )
        _load_harmony_for_http(job_id, app_settings, record)
        try:
            path = harmony_artifact_path(
                job_id,
                app_settings,
                artifact_file_name=artifact_file_name,
            )
        except HarmonyArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published harmonic context is unavailable.",
            ) from None
        except HarmonyArtifactError:
            logging.exception(
                "Published harmonic context failed path validation for job %s",
                job_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Published harmonic context could not be validated.",
            ) from None
        return FileResponse(
            path,
            filename="harmonic-context.json",
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/stems")
    def get_stems(job_id: str) -> dict:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            details = load_stem_details(job_id, app_settings, record)
        except StemArtifactError:
            logging.exception("Published stem details failed validation for job %s", job_id)
            raise HTTPException(
                status_code=500,
                detail="Published stem artifacts could not be validated.",
            ) from None
        if not details.available:
            raise HTTPException(
                status_code=404,
                detail="Published stem artifacts are unavailable.",
            )
        return details.payload()

    @app.get("/api/jobs/{job_id}/stems/{kind}/preview")
    def preview_stem(job_id: str, kind: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        artifact = _stem_artifact_or_http_error(
            job_id,
            kind,
            app_settings,
            record,
        )
        return FileResponse(artifact.path, media_type=artifact.media_type)

    @app.get("/api/jobs/{job_id}/stems/{kind}/download")
    def download_stem(job_id: str, kind: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        artifact = _stem_artifact_or_http_error(
            job_id,
            kind,
            app_settings,
            record,
        )
        return FileResponse(
            artifact.path,
            filename=artifact.download_name,
            media_type=artifact.media_type,
        )

    @app.get("/api/jobs/{job_id}/files/{file_name}")
    def download_file(job_id: str, file_name: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if file_name not in _persisted_artifact_names(record):
            raise HTTPException(status_code=404, detail="File not found")
        path = _resolve_job_file(app_settings, job_id, file_name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        download_name = path.name
        if (
            record.get("source_type") == "upload"
            and path.name == record.get("source_file_name")
            and record.get("original_filename")
        ):
            download_name = _safe_download_name(record["original_filename"])
        return FileResponse(
            path,
            filename=download_name,
            media_type=_media_type(path),
        )

    return app


def _run_url_job(
    job_id: str,
    source_url: str,
    settings: Settings,
    processor: UrlProcessor,
    analysis_processor: AnalysisProcessor,
) -> None:
    db.update_job(
        settings.database_path,
        job_id,
        status="processing",
        stage="validating",
        progress=1,
        message="Validating source URL.",
        preparation_status="processing",
        error=None,
    )
    _execute_processor(
        job_id,
        settings,
        lambda stage_callback, progress_callback: processor(
            job_id,
            source_url,
            settings,
            stage_callback,
            progress_callback,
        ),
        analysis_processor,
    )


def _run_upload_job(
    job_id: str,
    source_file_name: str,
    original_filename: str,
    settings: Settings,
    processor: UploadProcessor,
    analysis_processor: AnalysisProcessor,
) -> None:
    _execute_processor(
        job_id,
        settings,
        lambda stage_callback, progress_callback: processor(
            job_id,
            source_file_name,
            original_filename,
            settings,
            stage_callback,
            progress_callback,
        ),
        analysis_processor,
    )


def _execute_processor(
    job_id: str,
    settings: Settings,
    execute: Callable[
        [Callable[[str, str, float], None], Callable[[float], None]],
        MediaResult,
    ],
    analysis_processor: AnalysisProcessor,
) -> None:
    def map_preparation_progress(progress: float) -> float:
        if not settings.audio_analysis_enabled:
            return progress
        return progress * PREPARATION_PROGRESS_LIMIT / 100.0

    def update_stage(stage: str, message: str, progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            status="processing",
            stage=stage,
            message=message,
            progress=round(
                max(0.0, min(100.0, map_preparation_progress(progress))),
                1,
            ),
            preparation_status="processing",
            error=None,
        )

    def update_progress(progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            progress=round(
                max(0.0, min(100.0, map_preparation_progress(progress))),
                1,
            ),
        )

    try:
        result = execute(update_stage, update_progress)
    except MediaProcessingError as exc:
        db.update_job(
            settings.database_path,
            job_id,
            status="failed",
            stage="failed",
            message="Media preparation failed.",
            preparation_status="failed",
            error=str(exc),
        )
        return
    except Exception:
        logging.exception("Unexpected media processing failure for job %s", job_id)
        db.update_job(
            settings.database_path,
            job_id,
            status="failed",
            stage="failed",
            message="Media preparation failed.",
            preparation_status="failed",
            error="Unexpected processing failure. Check server logs.",
        )
        return

    analysis_enabled = settings.audio_analysis_enabled
    db.update_job(
        settings.database_path,
        job_id,
        status="processing" if analysis_enabled else "completed",
        stage="analyzing_audio" if analysis_enabled else "completed",
        progress=PREPARATION_PROGRESS_LIMIT if analysis_enabled else 100,
        message=(
            "Analyzing timing."
            if analysis_enabled
            else "Source and analysis audio are ready."
        ),
        title=result.title,
        uploader=result.uploader,
        duration_seconds=result.duration_seconds,
        source_format=result.source_format,
        sample_rate=result.sample_rate,
        channel_count=result.channel_count,
        source_file_name=result.source_file_name,
        normalized_file_name=result.normalized_file_name,
        metadata_file_name="metadata.json",
        preparation_status="completed",
        error=None,
        analysis_status="processing" if analysis_enabled else "not_started",
        analysis_version=(
            settings.audio_analysis_version if analysis_enabled else None
        ),
        analysis_error=None,
    )
    if analysis_enabled:
        _run_analysis_job(job_id, settings, analysis_processor)


def _run_analysis_job(
    job_id: str,
    settings: Settings,
    processor: AnalysisProcessor,
) -> None:
    def update_stage(stage: str, message: str, progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            status="processing",
            stage=stage,
            progress=round(max(65.0, min(99.0, progress)), 1),
            message=message,
            preparation_status="completed",
            error=None,
            analysis_status="processing",
            analysis_error=None,
        )

    try:
        result = processor(job_id, settings, update_stage)
    except (AudioAnalysisError, MediaProcessingError) as exc:
        _record_analysis_failure(settings, job_id, str(exc))
    except Exception:
        logging.exception("Unexpected audio analysis failure for job %s", job_id)
        _record_analysis_failure(
            settings,
            job_id,
            "Unexpected audio analysis failure. Check server logs.",
        )
    else:
        db.update_job(
            settings.database_path,
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Audio analysis complete.",
            preparation_status="completed",
            error=None,
            analysis_status="completed",
            analysis_version=result.analysis_version,
            tempo_bpm=result.tempo_bpm,
            tempo_confidence=result.tempo_confidence,
            key_symbol=result.key_symbol,
            key_confidence=result.key_confidence,
            analysis_json_file_name=result.analysis_json_file_name,
            analyzed_at=result.analyzed_at,
            analysis_error=None,
        )


def _run_transcription_job(
    job_id: str,
    settings: Settings,
    processor: TranscriptionProcessor,
) -> None:
    last_progress = 1.0

    def update_stage(stage: str, message: str, progress: float) -> None:
        nonlocal last_progress
        try:
            numeric_progress = float(progress)
        except (TypeError, ValueError):
            numeric_progress = last_progress
        numeric_progress = max(last_progress, min(99.0, max(1.0, numeric_progress)))
        last_progress = round(numeric_progress, 1)
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="processing",
            transcription_stage=stage,
            transcription_progress=last_progress,
            transcription_message=message,
            transcription_error=None,
        )

    try:
        result = processor(job_id, settings, update_stage)
    except TranscriptionPipelineError as exc:
        _record_transcription_failure(
            settings,
            job_id,
            _safe_transcription_error(str(exc), settings),
            progress=last_progress,
        )
    except Exception:
        logging.exception("Unexpected raw transcription failure for job %s", job_id)
        _record_transcription_failure(
            settings,
            job_id,
            "Unexpected raw transcription failure. Check server logs.",
            progress=last_progress,
        )
    else:
        if (
            not isinstance(result, TranscriptionPipelineResult)
            or result.artifact_file_name != RAW_TRANSCRIPTION_RELATIVE_PATH
            or not isinstance(result.transcription_version, str)
            or not result.transcription_version
            or not isinstance(result.transcribed_at, str)
            or not result.transcribed_at
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    result.pitched_event_count,
                    result.percussion_event_count,
                    result.aligned_event_count,
                )
            )
        ):
            logging.error("Transcription processor returned an invalid result for %s", job_id)
            _record_transcription_failure(
                settings,
                job_id,
                "Raw transcription returned an invalid result.",
                progress=last_progress,
            )
            return
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_stage="completed",
            transcription_progress=100,
            transcription_message="Raw transcription complete.",
            transcription_version=result.transcription_version,
            transcription_artifact_file_name=result.artifact_file_name,
            transcribed_at=result.transcribed_at,
            pitched_event_count=result.pitched_event_count,
            percussion_event_count=result.percussion_event_count,
            aligned_event_count=result.aligned_event_count,
            transcription_error=None,
        )


def _record_transcription_failure(
    settings: Settings,
    job_id: str,
    error: str,
    *,
    progress: float,
) -> None:
    db.update_job(
        settings.database_path,
        job_id,
        transcription_status="failed",
        transcription_stage="failed",
        transcription_progress=round(max(1.0, min(99.0, progress)), 1),
        transcription_message=(
            "Raw transcription stopped; prepared audio and any previous result remain available."
        ),
        transcription_error=error,
    )


def _safe_transcription_error(value: str, settings: Settings) -> str:
    text = str(value)
    lowered = text.lower()
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered:
        return "Raw transcription failed."
    try:
        cleaned = friendly_error(text, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return "Raw transcription failed."
    cleaned = re.sub(r"(?i)\b(?:https?|file)://[^\s]+", "<external location>", cleaned)
    cleaned = re.sub(
        r"(?i)\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
        r"\s*(?:=|:)?\s*[^\s,;]+",
        "<redacted>",
        cleaned,
    )
    cleaned = re.sub(r"(?i)0x[0-9a-f]{6,}", "<address>", cleaned)
    cleaned = " ".join(cleaned.replace("\x00", "").split()).strip()
    if not cleaned:
        return "Raw transcription failed."
    return cleaned[:500]


def _run_interpretation_job(
    job_id: str,
    settings: Settings,
    processor: InterpretationProcessor,
) -> None:
    last_progress = 1.0

    def update_stage(stage: str, message: str, progress: float) -> None:
        nonlocal last_progress
        try:
            numeric_progress = float(progress)
        except (TypeError, ValueError):
            numeric_progress = last_progress
        numeric_progress = max(last_progress, min(99.0, max(1.0, numeric_progress)))
        last_progress = round(numeric_progress, 1)
        db.update_job(
            settings.database_path,
            job_id,
            interpretation_status="processing",
            interpretation_stage=stage,
            interpretation_progress=last_progress,
            interpretation_message=message,
            interpretation_error=None,
        )

    try:
        result = processor(job_id, settings, update_stage)
    except InterpretationPipelineError as exc:
        _record_interpretation_failure(
            settings,
            job_id,
            _safe_interpretation_error(str(exc), settings),
            progress=last_progress,
        )
    except Exception:
        logging.exception("Unexpected editable interpretation failure for job %s", job_id)
        _record_interpretation_failure(
            settings,
            job_id,
            "Unexpected editable interpretation failure. Check server logs.",
            progress=last_progress,
        )
    else:
        counts = (
            result.part_count if isinstance(result, InterpretationPipelineResult) else None,
            result.phrase_count if isinstance(result, InterpretationPipelineResult) else None,
            result.pitched_item_count if isinstance(result, InterpretationPipelineResult) else None,
            result.percussion_item_count if isinstance(result, InterpretationPipelineResult) else None,
            result.warning_count if isinstance(result, InterpretationPipelineResult) else None,
        )
        if (
            not isinstance(result, InterpretationPipelineResult)
            or result.draft_file_name != INTERPRETATION_DRAFT_RELATIVE_PATH
            or not isinstance(result.version, str)
            or not result.version
            or not isinstance(result.created_at, str)
            or not result.created_at
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or not isinstance(result.warnings, tuple)
            or result.warning_count != len(result.warnings)
        ):
            logging.error("Interpretation processor returned an invalid result for %s", job_id)
            _record_interpretation_failure(
                settings,
                job_id,
                "Editable interpretation returned an invalid result.",
                progress=last_progress,
            )
            return
        db.update_job(
            settings.database_path,
            job_id,
            interpretation_status="completed",
            interpretation_stage="completed",
            interpretation_progress=100,
            interpretation_message="Editable interpretation complete.",
            interpretation_version=result.version,
            interpretation_artifact_file_name=result.draft_file_name,
            interpreted_at=result.created_at,
            interpretation_part_count=result.part_count,
            interpretation_phrase_count=result.phrase_count,
            interpretation_pitched_item_count=result.pitched_item_count,
            interpretation_percussion_item_count=result.percussion_item_count,
            interpretation_warning_count=result.warning_count,
            interpretation_error=None,
        )


def _record_interpretation_failure(
    settings: Settings,
    job_id: str,
    error: str,
    *,
    progress: float,
) -> None:
    db.update_job(
        settings.database_path,
        job_id,
        interpretation_status="failed",
        interpretation_stage="failed",
        interpretation_progress=round(max(1.0, min(99.0, progress)), 1),
        interpretation_message=(
            "Editable interpretation stopped; raw transcription and any previous draft remain available."
        ),
        interpretation_error=error,
    )


def _safe_interpretation_error(value: str, settings: Settings) -> str:
    text = str(value)
    lowered = text.lower()
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered:
        return "Editable interpretation failed."
    try:
        cleaned = friendly_error(text, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return "Editable interpretation failed."
    cleaned = re.sub(r"(?i)\b(?:https?|file)://[^\s]+", "<external location>", cleaned)
    cleaned = re.sub(
        r"(?i)\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
        r"\s*(?:=|:)?\s*[^\s,;]+",
        "<redacted>",
        cleaned,
    )
    cleaned = re.sub(r"(?i)0x[0-9a-f]{6,}", "<address>", cleaned)
    cleaned = " ".join(cleaned.replace("\x00", "").split()).strip()
    if not cleaned:
        return "Editable interpretation failed."
    return cleaned[:500]


def _harmony_attempt_is_active(
    record: dict[str, Any] | None,
    attempt_id: str | None,
) -> bool:
    return (
        record is not None
        and record.get("harmony_status") == "processing"
        and record.get("harmony_attempt_id") == attempt_id
    )


def _harmony_protection_state(
    record: dict[str, Any] | None,
) -> tuple[str, str | None, str | None] | None:
    if record is None:
        return None
    status = record.get("harmony_status")
    durable = record.get("harmony_artifact_file_name")
    active = record.get("harmony_attempt_id")
    if not isinstance(status, str) or status not in {
        "not_started",
        "processing",
        "completed",
        "failed",
    }:
        return None
    if durable is not None and not _is_harmony_artifact_file_name(durable):
        return None
    if active is not None:
        try:
            harmony_attempt_artifact_file_name(active)
        except HarmonyArtifactError:
            return None
    if status != "processing" and active is not None:
        return None
    return status, durable, active


def _run_harmony_job(
    job_id: str,
    settings: Settings,
    processor: HarmonyProcessor,
    attempt_id: str | None = None,
) -> None:
    try:
        expected_artifact_file_name = harmony_attempt_artifact_file_name(attempt_id)
    except HarmonyArtifactError:
        logging.error(
            "Harmony worker received an invalid attempt identity for job %s",
            job_id,
        )
        return
    try:
        started = db.start_harmony_attempt(
            settings.database_path,
            job_id,
            attempt_id=attempt_id,
        )
    except ValueError:
        logging.error(
            "Harmony worker received an invalid attempt identity for job %s",
            job_id,
        )
        return
    if not started:
        return
    last_progress = 2.0
    current_before_processor = db.get_job(settings.database_path, job_id)
    if not _harmony_attempt_is_active(current_before_processor, attempt_id):
        return

    def read_protection_state() -> tuple[str, str | None, str | None] | None:
        return _harmony_protection_state(
            db.get_job(settings.database_path, job_id)
        )

    def cleanup_lease():
        return db.harmony_cleanup_lease(
            settings.database_path,
            job_id,
            status="processing",
            durable_artifact_file_name=current_before_processor.get(
                "harmony_artifact_file_name"
            ),
            active_attempt_id=attempt_id,
        )

    try:
        reconcile_harmony_attempt_artifacts(
            job_id,
            settings,
            durable_artifact_file_name=current_before_processor.get(
                "harmony_artifact_file_name"
            ),
            active_attempt_id=attempt_id,
            protection_state_reader=read_protection_state,
            cleanup_lease=cleanup_lease,
        )
    except HarmonyArtifactError:
        logging.exception(
            "Harmony attempt artifact reconciliation failed for job %s",
            job_id,
        )
        _record_harmony_failure(
            settings,
            job_id,
            "Harmony attempt artifacts could not be reconciled safely.",
            attempt_id=attempt_id,
        )
        return
    if not _harmony_attempt_is_active(
        db.get_job(settings.database_path, job_id),
        attempt_id,
    ):
        return

    def update_stage(stage: str, message: str, progress: float) -> None:
        nonlocal last_progress
        try:
            numeric_progress = float(progress)
        except (TypeError, ValueError):
            numeric_progress = last_progress
        numeric_progress = max(
            last_progress,
            min(99.0, max(1.0, numeric_progress)),
        )
        next_progress = round(numeric_progress, 1)
        updated = db.update_harmony_progress(
            settings.database_path,
            job_id,
            stage=stage,
            progress=next_progress,
            message=message,
            attempt_id=attempt_id,
        )
        if not updated:
            raise HarmonyPipelineError(
                "Harmonic-context attempt is no longer active."
            )
        last_progress = next_progress

    try:
        with harmony_artifact_scope(expected_artifact_file_name):
            result = processor(
                job_id,
                settings,
                update_stage,
                attempt_id=attempt_id,
            )
    except HarmonyPipelineError as exc:
        failed = _record_harmony_failure(
            settings,
            job_id,
            _safe_harmony_error(str(exc), settings),
            attempt_id=attempt_id,
        )
        if failed and attempt_id is not None:
            _cleanup_harmony_artifact(
                settings,
                job_id,
                expected_artifact_file_name,
            )
    except Exception:
        logging.exception(
            "Unexpected harmonic-context failure for job %s",
            job_id,
        )
        failed = _record_harmony_failure(
            settings,
            job_id,
            "Unexpected harmonic-context failure. Check server logs.",
            attempt_id=attempt_id,
        )
        if failed and attempt_id is not None:
            _cleanup_harmony_artifact(
                settings,
                job_id,
                expected_artifact_file_name,
            )
    else:
        if not _validate_harmony_processor_result(
            job_id,
            settings,
            result,
            attempt_id=attempt_id,
        ):
            logging.error(
                "Harmony processor returned an invalid result for %s",
                job_id,
            )
            failed = _record_harmony_failure(
                settings,
                job_id,
                "Harmonic context returned an invalid result.",
                attempt_id=attempt_id,
            )
            if failed and attempt_id is not None:
                _cleanup_harmony_artifact(
                    settings,
                    job_id,
                    expected_artifact_file_name,
                )
            return
        current = db.get_job(settings.database_path, job_id)
        raw_transcription = (
            _load_matching_raw_transcription_for_harmony(
                job_id,
                settings,
                current,
            )
            if current is not None
            else None
        )
        if (
            raw_transcription is None
            or result.payload.get("rawEvidence")
            != _harmony_raw_evidence(raw_transcription)
        ):
            logging.warning(
                "Discarded stale harmonic-context result for job %s",
                job_id,
            )
            failed = _record_harmony_failure(
                settings,
                job_id,
                "Harmonic-context source changed during processing.",
                attempt_id=attempt_id,
            )
            if failed and attempt_id is not None:
                _cleanup_harmony_artifact(
                    settings,
                    job_id,
                    expected_artifact_file_name,
                )
            return
        previous_artifact_file_name = (
            current.get("harmony_artifact_file_name")
            if current is not None
            else None
        )
        completed = db.complete_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=result.version,
            artifact_file_name=result.artifact_file_name,
            harmonized_at=result.created_at,
            event_count=result.event_count,
            segment_count=result.segment_count,
            resolved_segment_count=result.resolved_segment_count,
            unresolved_segment_count=result.unresolved_segment_count,
            unresolved_event_count=result.unresolved_event_count,
            warning_count=result.warning_count,
            used_interpretation_context=result.used_interpretation_context,
            attempt_id=attempt_id,
        )
        if not completed:
            logging.warning(
                "Discarded stale harmonic-context completion for job %s",
                job_id,
            )
            return
        if (
            isinstance(previous_artifact_file_name, str)
            and previous_artifact_file_name != result.artifact_file_name
            and _is_harmony_artifact_file_name(previous_artifact_file_name)
        ):
            _cleanup_harmony_artifact(
                settings,
                job_id,
                previous_artifact_file_name,
            )


def _validate_harmony_processor_result(
    job_id: str,
    settings: Settings,
    result: object,
    *,
    attempt_id: str | None,
) -> bool:
    if not isinstance(result, HarmonyPipelineResult):
        return False
    try:
        expected_artifact_file_name = (
            HARMONY_ARTIFACT_RELATIVE_PATH
            if attempt_id is None
            else harmony_attempt_artifact_file_name(attempt_id)
        )
    except HarmonyArtifactError:
        return False
    counts = (
        result.event_count,
        result.segment_count,
        result.resolved_segment_count,
        result.unresolved_segment_count,
        result.unresolved_event_count,
        result.warning_count,
    )
    if (
        result.artifact_file_name != expected_artifact_file_name
        or not isinstance(result.version, str)
        or not result.version
        or not isinstance(result.created_at, str)
        or not result.created_at
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        )
        or result.segment_count
        != result.resolved_segment_count + result.unresolved_segment_count
        or result.unresolved_event_count > result.event_count
        or type(result.used_interpretation_context) is not bool
        or (
            result.used_interpretation_context
            and (
                not isinstance(result.interpretation_version, str)
                or not result.interpretation_version
            )
        )
        or (
            not result.used_interpretation_context
            and result.interpretation_version is not None
        )
        or not isinstance(result.warnings, tuple)
        or result.warning_count != len(result.warnings)
        or not isinstance(result.payload, dict)
    ):
        return False
    try:
        artifact = load_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=expected_artifact_file_name,
        )
    except HarmonyArtifactError:
        return False
    if artifact is None or artifact != result.payload:
        return False
    diagnostics = artifact.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    expected_counts = {
        "eventCount": result.event_count,
        "segmentCount": result.segment_count,
        "resolvedSegmentCount": result.resolved_segment_count,
        "unresolvedSegmentCount": result.unresolved_segment_count,
        "unresolvedEventCount": result.unresolved_event_count,
    }
    if (
        artifact.get("harmonyVersion") != result.version
        or artifact.get("createdAt") != result.created_at
        or any(diagnostics.get(name) != value for name, value in expected_counts.items())
        or tuple(artifact.get("warnings", ())) != result.warnings
        or ("sourceInterpretation" in artifact)
        != result.used_interpretation_context
    ):
        return False
    if result.used_interpretation_context:
        source = artifact.get("sourceInterpretation")
        if (
            not isinstance(source, dict)
            or source.get("draftVersion") != result.interpretation_version
        ):
            return False
    return True


def _cleanup_harmony_artifact(
    settings: Settings,
    job_id: str,
    artifact_file_name: str,
) -> None:
    def authorize_cleanup() -> None:
        state = _harmony_protection_state(
            db.get_job(settings.database_path, job_id)
        )
        if state is None:
            raise HarmonyArtifactError(
                "Harmony cleanup state could not be verified."
            )
        status, durable_artifact_file_name, active_attempt_id = state
        if durable_artifact_file_name == artifact_file_name:
            raise HarmonyArtifactError(
                "The durable harmony artifact cannot be removed."
            )
        if status == "processing":
            try:
                active_artifact_file_name = (
                    HARMONY_ARTIFACT_RELATIVE_PATH
                    if active_attempt_id is None
                    else harmony_attempt_artifact_file_name(active_attempt_id)
                )
            except HarmonyArtifactError as exc:
                raise HarmonyArtifactError(
                    "Harmony cleanup state could not be verified."
                ) from exc
            if active_artifact_file_name == artifact_file_name:
                raise HarmonyArtifactError(
                    "The active harmony artifact cannot be removed."
                )

    try:
        authorize_cleanup()
        remove_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=artifact_file_name,
            cleanup_authorizer=authorize_cleanup,
        )
    except HarmonyArtifactError:
        logging.warning(
            "Could not clean a non-durable harmony artifact for job %s",
            job_id,
        )


def _record_harmony_failure(
    settings: Settings,
    job_id: str,
    error: str,
    *,
    attempt_id: str | None = None,
) -> bool:
    return db.fail_harmony_attempt(
        settings.database_path,
        job_id,
        error=error,
        attempt_id=attempt_id,
    )


def _safe_harmony_error(value: str, settings: Settings) -> str:
    text = str(value)
    lowered = text.lower()
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered:
        return "Harmony processing failed."
    try:
        cleaned = friendly_error(text, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return "Harmony processing failed."
    cleaned = re.sub(
        r"(?i)\b(?:https?|file)://[^\s]+",
        "<external location>",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
        r"\s*(?:=|:)?\s*[^\s,;]+",
        "<redacted>",
        cleaned,
    )
    cleaned = re.sub(r"(?i)0x[0-9a-f]{6,}", "<address>", cleaned)
    cleaned = " ".join(cleaned.replace("\x00", "").split()).strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned:
        return "Harmony processing failed."
    return cleaned[:500]


def _record_analysis_failure(
    settings: Settings,
    job_id: str,
    error: str,
) -> None:
    db.update_job(
        settings.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=ANALYSIS_FAILURE_PROGRESS,
        message=(
            "Source preparation is complete; audio analysis could not be completed."
        ),
        preparation_status="completed",
        error=None,
        analysis_status="failed",
        analysis_error=error,
    )


def _validate_source_url(
    source_url: str,
    allowed_hosts: tuple[str, ...],
) -> None:
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=422,
            detail="Only HTTP(S) URLs are supported",
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=422,
            detail="URLs containing credentials are rejected",
        )
    allowed = any(
        hostname == allowed_host or hostname.endswith(f".{allowed_host}")
        for allowed_host in allowed_hosts
    )
    if not allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Source host '{hostname or 'unknown'}' is not allowed",
        )


def _serialize_job(
    job: dict,
    settings: Settings,
    separation_service: SeparationService | None = None,
) -> dict:
    payload = {
        key: value
        for key, value in job.items()
        if key not in _INTERNAL_SEPARATION_FIELDS
        and key not in _INTERNAL_TRANSCRIPTION_FIELDS
        and key not in _INTERNAL_INTERPRETATION_FIELDS
        and key not in _INTERNAL_HARMONY_FIELDS
    }
    payload["files"] = []
    job_dir = settings.exports_dir / job["id"]
    allowed_names = _persisted_artifact_names(job)
    if job_dir.is_dir():
        payload["files"] = [
            _serialize_file(job, path)
            for path in sorted(job_dir.iterdir())
            if path.is_file() and path.name in allowed_names
        ]
    payload["preparation"] = {
        "status": job.get("preparation_status") or "pending",
        "sourceAvailable": bool(
            job.get("source_file_name") in allowed_names
            and (job_dir / str(job.get("source_file_name"))).is_file()
        ),
        "analysisAudioAvailable": bool(
            job.get("normalized_file_name") in allowed_names
            and (job_dir / str(job.get("normalized_file_name"))).is_file()
        ),
    }
    payload["analysis"] = {
        "status": job.get("analysis_status") or "not_started",
        "version": job.get("analysis_version"),
        **_analysis_summary(job),
        "endpoint": f"/api/jobs/{job['id']}/analysis",
        "download_url": (
            f"/api/jobs/{job['id']}/analysis/download"
            if job.get("analysis_json_file_name")
            == ANALYSIS_JSON_RELATIVE_PATH
            else None
        ),
        "error": job.get("analysis_error"),
    }
    if separation_service is not None:
        separation = separation_service.serialize_job(job)
        if separation is not None:
            payload["separation"] = separation
    transcription = _serialize_transcription(job)
    if transcription is not None:
        payload["transcription"] = transcription
    interpretation = _serialize_interpretation(job)
    if interpretation is not None:
        payload["interpretation"] = interpretation
    harmony = _serialize_harmony(job)
    if harmony is not None:
        payload["harmony"] = harmony
    return payload


def _serialize_transcription(job: dict) -> dict[str, Any] | None:
    status_value = job.get("transcription_status")
    status = status_value if isinstance(status_value, str) and status_value else "not_started"
    should_expose = (
        job.get("analysis_status") == "completed"
        or status != "not_started"
        or job.get("transcription_artifact_file_name") is not None
    )
    if not should_expose:
        return None
    job_id = job["id"]
    available = (
        job.get("transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
    )
    ready = (
        job.get("preparation_status") == "completed"
        and job.get("analysis_status") == "completed"
    )
    return {
        "enabled": True,
        "status": status,
        "stage": job.get("transcription_stage") or "not_started",
        "progress": _safe_progress(job.get("transcription_progress")),
        "message": job.get("transcription_message"),
        "version": job.get("transcription_version"),
        "available": available,
        "counts": {
            "pitched": _safe_count(job.get("pitched_event_count")),
            "percussion": _safe_count(job.get("percussion_event_count")),
            "aligned": _safe_count(job.get("aligned_event_count")),
        },
        "canStart": ready and status in {"not_started", "failed"},
        "startUrl": f"/api/jobs/{job_id}/transcribe",
        "detailsUrl": f"/api/jobs/{job_id}/transcription?includeEvents=false",
        "downloadUrl": (
            f"/api/jobs/{job_id}/transcription/download" if available else None
        ),
        "error": job.get("transcription_error"),
    }


def _serialize_interpretation(job: dict) -> dict[str, Any] | None:
    status_value = job.get("interpretation_status")
    status = status_value if isinstance(status_value, str) and status_value else "not_started"
    should_expose = (
        job.get("transcription_status") == "completed"
        or status != "not_started"
        or job.get("interpretation_artifact_file_name") is not None
    )
    if not should_expose:
        return None
    job_id = job["id"]
    available = (
        job.get("interpretation_artifact_file_name")
        == INTERPRETATION_DRAFT_RELATIVE_PATH
    )
    ready = (
        job.get("transcription_status") == "completed"
        and job.get("transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
    )
    return {
        "enabled": True,
        "status": status,
        "stage": job.get("interpretation_stage") or "not_started",
        "progress": _safe_progress(job.get("interpretation_progress")),
        "message": job.get("interpretation_message"),
        "version": job.get("interpretation_version"),
        "available": available,
        "counts": {
            "parts": _safe_count(job.get("interpretation_part_count")),
            "phrases": _safe_count(job.get("interpretation_phrase_count")),
            "pitched": _safe_count(job.get("interpretation_pitched_item_count")),
            "percussion": _safe_count(job.get("interpretation_percussion_item_count")),
            "warnings": _safe_count(job.get("interpretation_warning_count")),
        },
        "canStart": ready and status in {"not_started", "failed"},
        "canReinterpret": ready and status == "completed",
        "startUrl": f"/api/jobs/{job_id}/interpret",
        "detailsUrl": f"/api/jobs/{job_id}/interpretation?includeItems=false",
        "fullDetailsUrl": f"/api/jobs/{job_id}/interpretation?includeItems=true",
        "downloadUrl": (
            f"/api/jobs/{job_id}/interpretation/download" if available else None
        ),
        "error": job.get("interpretation_error"),
    }


def _serialize_harmony(job: dict) -> dict[str, Any] | None:
    status_value = job.get("harmony_status")
    status = (
        status_value
        if isinstance(status_value, str) and status_value
        else "not_started"
    )
    should_expose = (
        job.get("transcription_status") == "completed"
        or status != "not_started"
        or job.get("harmony_artifact_file_name") is not None
    )
    if not should_expose:
        return None
    job_id = job["id"]
    available = _is_harmony_artifact_file_name(
        job.get("harmony_artifact_file_name")
    )
    ready = (
        job.get("analysis_status") == "completed"
        and job.get("analysis_json_file_name")
        == ANALYSIS_JSON_RELATIVE_PATH
        and job.get("transcription_status") == "completed"
        and job.get("transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
    )
    context_value = job.get("harmony_used_interpretation_context")
    used_context = (
        bool(context_value)
        if type(context_value) is int and context_value in {0, 1}
        else None
    )
    return {
        "enabled": True,
        "status": status,
        "stage": job.get("harmony_stage") or "not_started",
        "progress": _safe_progress(job.get("harmony_progress")),
        "message": job.get("harmony_message"),
        "attemptVersion": job.get("harmony_attempt_version"),
        "version": job.get("harmony_version"),
        "createdAt": job.get("harmonized_at"),
        "available": available,
        "counts": {
            "events": _safe_count(job.get("harmony_event_count")),
            "segments": _safe_count(job.get("harmony_segment_count")),
            "resolved": _safe_count(
                job.get("harmony_resolved_segment_count")
            ),
            "unresolved": _safe_count(
                job.get("harmony_unresolved_segment_count")
            ),
            "unresolvedEvents": _safe_count(
                job.get("harmony_unresolved_event_count")
            ),
            "warnings": _safe_count(job.get("harmony_warning_count")),
        },
        "usedInterpretationContext": used_context,
        "canStart": ready and status in {"not_started", "failed"},
        "canReharmonize": ready and status == "completed",
        "startUrl": f"/api/jobs/{job_id}/harmonize",
        "detailsUrl": f"/api/jobs/{job_id}/harmony?includeSegments=false",
        "fullDetailsUrl": f"/api/jobs/{job_id}/harmony?includeSegments=true",
        "downloadUrl": (
            f"/api/jobs/{job_id}/harmony/download" if available else None
        ),
        "error": job.get("harmony_error"),
    }


def _strict_query_bool(value: str | None, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if value == "true":
        return True
    if value == "false":
        return False
    raise HTTPException(
        status_code=422,
        detail=f"{field} must be true or false.",
    )


def _safe_progress(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(100.0, number)), 1)


def _safe_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _analysis_summary(job: dict) -> dict:
    return {
        "tempoBpm": job.get("tempo_bpm"),
        "tempoConfidence": job.get("tempo_confidence"),
        "keySymbol": job.get("key_symbol"),
        "keyConfidence": job.get("key_confidence"),
        "analyzedAt": job.get("analyzed_at"),
    }


def _persisted_artifact_names(job: dict) -> set[str]:
    return {
        name
        for name in (
            job.get("source_file_name"),
            job.get("normalized_file_name"),
            job.get("metadata_file_name"),
        )
        if isinstance(name, str) and name
    }


def _serialize_file(job: dict, path: Path) -> dict:
    if path.name == job.get("normalized_file_name"):
        label = "Analysis audio"
        kind = "analysis"
    elif path.name == job.get("source_file_name"):
        label = "Source file"
        kind = "source"
    else:
        label = "Metadata"
        kind = "metadata"
    url = f"/api/jobs/{job['id']}/files/{quote(path.name, safe='')}"
    return {
        "name": path.name,
        "label": label,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "download_url": url,
        "preview_url": (
            url
            if path.suffix.lower()
            in {".mp3", ".wav", ".ogg", ".m4a", ".aac"}
            else None
        ),
    }


def _load_matching_raw_transcription_for_harmony(
    job_id: str,
    settings: Settings,
    record: dict,
) -> dict[str, Any] | None:
    try:
        raw = load_raw_transcription(job_id, settings)
    except (RawTranscriptionError, MediaProcessingError, OSError, RuntimeError):
        return None
    if raw is None:
        return None
    source_analysis = raw.get("sourceAnalysis")
    if not isinstance(source_analysis, dict):
        return None
    aligned_count = sum(
        1
        for candidate in raw.get("alignmentCandidates", ())
        if isinstance(candidate, dict) and "alignedTimeSeconds" in candidate
    )
    expected_counts = (
        ("pitched_event_count", len(raw.get("pitchedNoteEvents", ()))),
        ("percussion_event_count", len(raw.get("percussionEvents", ()))),
        ("aligned_event_count", aligned_count),
    )
    return raw if (
        raw.get("transcriptionVersion") == record.get("transcription_version")
        and raw.get("createdAt") == record.get("transcribed_at")
        and source_analysis.get("fileName") == ANALYSIS_JSON_RELATIVE_PATH
        and source_analysis.get("analysisVersion") == record.get("analysis_version")
        and all(
            record.get(field) == value
            and isinstance(record.get(field), int)
            and not isinstance(record.get(field), bool)
            and value >= 0
            for field, value in expected_counts
        )
    ) else None


def _harmony_raw_evidence(
    raw_transcription: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence = [
        {
            "id": event["id"],
            "sourceKind": event["sourceKind"],
            "rawStartSeconds": event["startSeconds"],
            "rawEndSeconds": event["endSeconds"],
            "midiNote": event["midiNote"],
            "midiPitch": event["midiPitch"],
            "pitchClass": event["midiNote"] % 12,
            "pitchName": _HARMONY_PITCH_CLASS_NAMES[event["midiNote"] % 12],
            "confidence": event["confidence"],
            "warnings": list(event.get("warnings", [])),
        }
        for event in raw_transcription.get("pitchedNoteEvents", ())
    ]
    evidence.sort(
        key=lambda item: (
            item["rawStartSeconds"],
            item["rawEndSeconds"],
            item["id"],
        )
    )
    return evidence


def _harmony_raw_evidence_matches_current(
    artifact_evidence: object,
    current_evidence: list[dict[str, Any]],
    *,
    allow_legacy_missing_warnings: bool,
) -> bool:
    if not isinstance(artifact_evidence, list) or len(artifact_evidence) != len(
        current_evidence
    ):
        return False
    for artifact_item, current_item in zip(artifact_evidence, current_evidence):
        if not isinstance(artifact_item, dict):
            return False
        if artifact_item == current_item:
            continue
        if allow_legacy_missing_warnings and "warnings" not in artifact_item:
            expected_without_warnings = dict(current_item)
            expected_without_warnings.pop("warnings", None)
            if artifact_item == expected_without_warnings:
                continue
        return False
    return True


def _is_harmony_artifact_file_name(value: object) -> bool:
    return isinstance(value, str) and _HARMONY_ARTIFACT_FILE_RE.fullmatch(value) is not None


def _load_harmony_for_http(
    job_id: str,
    settings: Settings,
    record: dict,
) -> dict[str, Any]:
    artifact_file_name = record.get("harmony_artifact_file_name")
    if not _is_harmony_artifact_file_name(artifact_file_name):
        raise HTTPException(
            status_code=404,
            detail="Published harmonic context is unavailable.",
        )
    try:
        artifact = load_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=artifact_file_name,
        )
    except HarmonyArtifactError:
        logging.exception(
            "Published harmonic context failed validation for job %s",
            job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Published harmonic context could not be validated.",
        ) from None
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Published harmonic context is unavailable.",
        )
    raw_transcription = _load_matching_raw_transcription_for_harmony(
        job_id,
        settings,
        record,
    )
    if (
        raw_transcription is None
        or not _harmony_artifact_matches_record(
            artifact,
            record,
            raw_transcription,
        )
    ):
        logging.error(
            "Published harmonic context metadata mismatch for job %s",
            job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Published harmonic context could not be validated.",
        )
    return artifact


def _harmony_artifact_matches_record(
    artifact: dict[str, Any],
    record: dict,
    raw_transcription: dict[str, Any] | None = None,
) -> bool:
    diagnostics = artifact.get("diagnostics")
    transcription = artifact.get("sourceTranscription")
    analysis = artifact.get("sourceAnalysis")
    if not all(
        isinstance(value, dict)
        for value in (diagnostics, transcription, analysis)
    ):
        return False
    context_value = record.get("harmony_used_interpretation_context")
    if type(context_value) is not int or context_value not in {0, 1}:
        return False
    expected_counts = {
        "eventCount": record.get("harmony_event_count"),
        "segmentCount": record.get("harmony_segment_count"),
        "resolvedSegmentCount": record.get(
            "harmony_resolved_segment_count"
        ),
        "unresolvedSegmentCount": record.get(
            "harmony_unresolved_segment_count"
        ),
        "unresolvedEventCount": record.get(
            "harmony_unresolved_event_count"
        ),
    }
    raw_evidence_matches = True
    if raw_transcription is not None:
        raw_evidence_matches = _harmony_raw_evidence_matches_current(
            artifact.get("rawEvidence"),
            _harmony_raw_evidence(raw_transcription),
            allow_legacy_missing_warnings=(
                record.get("harmony_artifact_file_name")
                == HARMONY_ARTIFACT_RELATIVE_PATH
            ),
        )
    return (
        artifact.get("harmonyVersion") == record.get("harmony_version")
        and artifact.get("createdAt") == record.get("harmonized_at")
        and transcription.get("fileName")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
        and transcription.get("transcriptionVersion")
        == record.get("harmony_source_transcription_version")
        and record.get("transcription_version")
        == record.get("harmony_source_transcription_version")
        and record.get("transcription_artifact_file_name")
        == record.get("harmony_source_transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
        and record.get("transcribed_at")
        == record.get("harmony_source_transcribed_at")
        and analysis.get("fileName") == ANALYSIS_JSON_RELATIVE_PATH
        and analysis.get("analysisVersion")
        == record.get("analysis_version")
        and all(
            diagnostics.get(name) == value
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for name, value in expected_counts.items()
        )
        and len(artifact.get("warnings", ()))
        == record.get("harmony_warning_count")
        and ("sourceInterpretation" in artifact) == bool(context_value)
        and raw_evidence_matches
    )


def _harmony_details_payload(
    artifact: dict[str, Any],
    record: dict,
    *,
    include_segments: bool,
) -> dict[str, Any]:
    diagnostics = artifact["diagnostics"]
    payload: dict[str, Any] = {
        "available": True,
        "status": record.get("harmony_status") or "not_started",
        "schemaVersion": artifact["schemaVersion"],
        "harmonyVersion": artifact["harmonyVersion"],
        "createdAt": artifact["createdAt"],
        "sourceTranscription": artifact["sourceTranscription"],
        "sourceAnalysis": artifact["sourceAnalysis"],
        "algorithms": artifact["algorithms"],
        "tonalContext": artifact["tonalContext"],
        "counts": {
            "events": diagnostics["eventCount"],
            "segments": diagnostics["segmentCount"],
            "resolved": diagnostics["resolvedSegmentCount"],
            "unresolved": diagnostics["unresolvedSegmentCount"],
            "unresolvedEvents": diagnostics["unresolvedEventCount"],
            "warnings": len(artifact["warnings"]),
        },
        "usedInterpretationContext": "sourceInterpretation" in artifact,
        "warnings": artifact["warnings"],
        "diagnostics": diagnostics,
        "segmentsIncluded": include_segments,
        "downloadUrl": f"/api/jobs/{record['id']}/harmony/download",
    }
    if "sourceInterpretation" in artifact:
        payload["sourceInterpretation"] = artifact[
            "sourceInterpretation"
        ]
    if include_segments:
        payload["rawEvidence"] = artifact["rawEvidence"]
        payload["segments"] = artifact["segments"]
        payload["unresolvedEventIds"] = artifact[
            "unresolvedEventIds"
        ]
    return payload


def _stem_artifact_or_http_error(
    job_id: str,
    kind: str,
    settings: Settings,
    record: dict,
):
    try:
        return resolve_stem_artifact(job_id, kind, settings, record)
    except (StemManifestUnavailableError, StemKindNotFoundError):
        raise HTTPException(
            status_code=404,
            detail="The requested stem artifact is unavailable.",
        ) from None
    except StemArtifactError:
        logging.exception("Published stem artifact failed validation for job %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Published stem artifacts could not be validated.",
        ) from None


def _resolve_job_file(
    settings: Settings,
    job_id: str,
    file_name: str,
) -> Path:
    if Path(file_name).name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    job_dir = (settings.exports_dir / job_id).resolve()
    candidate = (job_dir / file_name).resolve()
    if settings.exports_dir.resolve() not in job_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid job path")
    if job_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return candidate


def _media_type(path: Path) -> str:
    explicit = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".json": "application/json",
    }
    return (
        explicit.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _safe_download_name(value: str) -> str:
    name = (
        Path(value.replace("\\", "/"))
        .name.replace("\r", "")
        .replace("\n", "")
    )
    return name or "source-media"


app = create_app()
