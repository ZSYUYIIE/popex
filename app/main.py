from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from app import db
from app.config import Settings
from app.downloader import ExtractionError, ExtractionResult, extract_audio


Extractor = Callable[[str, str, Settings, Callable[[float], None]], ExtractionResult]
BASE_DIR = Path(__file__).resolve().parent


class JobCreate(BaseModel):
    url: HttpUrl


def create_app(
    settings: Settings | None = None,
    extractor: Extractor = extract_audio,
) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_settings.ensure_directories()
        db.init_database(app_settings.database_path)
        db.fail_incomplete_jobs(app_settings.database_path)
        yield

    app = FastAPI(
        title="PopEx",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(BASE_DIR / "templates" / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(payload: JobCreate, background_tasks: BackgroundTasks) -> dict:
        source_url = str(payload.url)
        _validate_source_url(source_url, app_settings.allowed_hosts)
        job_id = uuid4().hex
        job = db.create_job(app_settings.database_path, job_id, source_url)
        background_tasks.add_task(
            _run_job,
            job_id,
            source_url,
            app_settings,
            extractor,
        )
        return _serialize_job(job, app_settings)

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
        media_type = "audio/mpeg" if path.suffix.lower() == ".mp3" else "application/json"
        return FileResponse(path, filename=path.name, media_type=media_type)

    return app


def _run_job(
    job_id: str,
    source_url: str,
    settings: Settings,
    extractor: Extractor,
) -> None:
    db.update_job(settings.database_path, job_id, status="processing", progress=1)

    def update_progress(progress: float) -> None:
        db.update_job(
            settings.database_path,
            job_id,
            progress=round(max(0.0, min(100.0, progress)), 1),
        )

    try:
        result = extractor(job_id, source_url, settings, update_progress)
    except ExtractionError as exc:
        db.update_job(
            settings.database_path,
            job_id,
            status="failed",
            error=str(exc),
        )
    except Exception:
        logging.exception("Unexpected extraction failure for job %s", job_id)
        db.update_job(
            settings.database_path,
            job_id,
            status="failed",
            error="Unexpected extraction failure. Check server logs.",
        )
    else:
        db.update_job(
            settings.database_path,
            job_id,
            status="completed",
            progress=100,
            title=result.title,
            uploader=result.uploader,
            duration_seconds=result.duration_seconds,
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
            payload["files"] = [
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "download_url": f"/api/jobs/{job['id']}/files/{quote(path.name, safe='')}",
                }
                for path in sorted(job_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".mp3", ".json"}
            ]
    return payload


def _resolve_job_file(settings: Settings, job_id: str, file_name: str) -> Path:
    if Path(file_name).name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    job_dir = (settings.exports_dir / job_id).resolve()
    candidate = (job_dir / file_name).resolve()
    if job_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return candidate


app = create_app()
