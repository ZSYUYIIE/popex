from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from app import db
from app.config import SUPPORTED_MEDIA_EXTENSIONS, Settings
from app.media import (
    MediaProcessingError,
    MediaResult,
    cleanup_job_dir,
    dependency_report,
    generated_source_name,
    process_upload,
    process_url,
    secure_job_dir,
)


UrlProcessor = Callable[
    [str, str, Settings, Callable[[str, str, float], None], Callable[[float], None]],
    MediaResult,
]
UploadProcessor = Callable[
    [str, str, str, Settings, Callable[[str, str, float], None], Callable[[float], None]],
    MediaResult,
]
BASE_DIR = Path(__file__).resolve().parent
ALLOWED_MIME_PREFIXES = ("audio/", "video/")
ALLOWED_GENERIC_MIME_TYPES = {
    "",
    "application/octet-stream",
    "application/x-m4a",
    "application/ogg",
}


class JobCreate(BaseModel):
    url: HttpUrl


def create_app(
    settings: Settings | None = None,
    url_processor: UrlProcessor = process_url,
    upload_processor: UploadProcessor = process_upload,
) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_settings.ensure_directories()
        db.init_database(app_settings.database_path)
        db.fail_incomplete_jobs(app_settings.database_path)
        app.state.dependencies = dependency_report(app_settings)
        if not app.state.dependencies["ffmpeg"] or not app.state.dependencies["ffprobe"]:
            logging.warning(
                "PopEx started without FFmpeg/ffprobe. Media jobs will fail until "
                "the missing executable is installed."
            )
        yield

    app = FastAPI(
        title="PopEx",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(BASE_DIR / "templates" / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        dependencies = getattr(app.state, "dependencies", dependency_report(app_settings))
        return {
            "status": "ok" if all(dependencies.values()) else "degraded",
            "dependencies": dependencies,
        }

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_url_job(
        payload: JobCreate, background_tasks: BackgroundTasks
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
        )
        return _serialize_job(job, app_settings)

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
                                f"{app_settings.max_upload_mb} MB limit."
                            ),
                        )
                    output.write(chunk)
            if total == 0:
                raise HTTPException(status_code=422, detail="The uploaded file is empty.")
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
                error="The uploaded file could not be saved.",
            )
            raise HTTPException(
                status_code=500, detail="The uploaded file could not be saved."
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
        )
        current = db.get_job(app_settings.database_path, job_id)
        return _serialize_job(current or job, app_settings)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [
            _serialize_job(job, app_settings)
            for job in db.list_jobs(app_settings.database_path)
        ]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _serialize_job(record, app_settings)

    @app.get("/api/jobs/{job_id}/files/{file_name}")
    def download_file(job_id: str, file_name: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record["status"] != "completed":
            raise HTTPException(status_code=409, detail="Job is not complete")
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
) -> None:
    db.update_job(
        settings.database_path,
        job_id,
        status="processing",
        stage="validating",
        progress=1,
        message="Validating source URL.",
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
    )


def _run_upload_job(
    job_id: str,
    source_file_name: str,
    original_filename: str,
    settings: Settings,
    processor: UploadProcessor,
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
    )


def _execute_processor(
    job_id: str,
    settings: Settings,
    execute: Callable[
        [Callable[[str, str, float], None], Callable[[float], None]], MediaResult
    ],
) -> None:
    def update_stage(stage: str, message: str, progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            status="processing",
            stage=stage,
            message=message,
            progress=round(max(0.0, min(100.0, progress)), 1),
            error=None,
        )

    def update_progress(progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            progress=round(max(0.0, min(100.0, progress)), 1),
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
            error=str(exc),
        )
    except Exception:
        logging.exception("Unexpected media processing failure for job %s", job_id)
        db.update_job(
            settings.database_path,
            job_id,
            status="failed",
            stage="failed",
            message="Media preparation failed.",
            error="Unexpected processing failure. Check server logs.",
        )
    else:
        db.update_job(
            settings.database_path,
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Source and analysis audio are ready.",
            title=result.title,
            uploader=result.uploader,
            duration_seconds=result.duration_seconds,
            source_format=result.source_format,
            sample_rate=result.sample_rate,
            channel_count=result.channel_count,
            source_file_name=result.source_file_name,
            normalized_file_name=result.normalized_file_name,
            error=None,
        )


def _validate_source_url(source_url: str, allowed_hosts: tuple[str, ...]) -> None:
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="Only HTTP(S) URLs are supported")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="URLs containing credentials are rejected")
    allowed = any(
        hostname == allowed_host or hostname.endswith(f".{allowed_host}")
        for allowed_host in allowed_hosts
    )
    if not allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Source host '{hostname or 'unknown'}' is not allowed",
        )


def _serialize_job(job: dict, settings: Settings) -> dict:
    payload = dict(job)
    payload["files"] = []
    if job["status"] == "completed":
        job_dir = settings.exports_dir / job["id"]
        if job_dir.is_dir():
            allowed_names = {
                job.get("source_file_name"),
                job.get("normalized_file_name"),
                "metadata.json",
            }
            payload["files"] = [
                _serialize_file(job, path)
                for path in sorted(job_dir.iterdir())
                if path.is_file() and path.name in allowed_names
            ]
    return payload


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
        "preview_url": url if path.suffix.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac"} else None,
    }


def _resolve_job_file(settings: Settings, job_id: str, file_name: str) -> Path:
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
    return explicit.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _safe_download_name(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.replace("\r", "").replace("\n", "")
    return name or "source-media"


app = create_app()
