import re
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.media import MediaProcessingError, MediaResult
from app.main import create_app


def make_settings(tmp_path: Path, *, max_upload_mb: int = 2) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com", "youtu.be"),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=max_upload_mb,
        audio_quality="192",
        ffmpeg_binary="missing-test-ffmpeg",
        ffprobe_binary="missing-test-ffprobe",
    )


def fake_url_processor(
    job_id, source_url, settings, stage_callback, progress_callback
):
    stage_callback("extracting_audio", "Extracting URL source.", 30)
    job_dir = settings.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.mp3").write_bytes(b"fake url mp3")
    (job_dir / "analysis.wav").write_bytes(b"RIFFfake wav")
    (job_dir / "metadata.json").write_text('{"title":"URL Song"}', encoding="utf-8")
    progress_callback(100)
    return MediaResult(
        title="URL Song",
        uploader="URL Artist",
        duration_seconds=123,
        source_format="mp3",
        sample_rate=44100,
        channel_count=2,
        source_file_name="source.mp3",
        normalized_file_name="analysis.wav",
        files=("source.mp3", "analysis.wav", "metadata.json"),
    )


def fake_upload_processor(
    job_id,
    source_file_name,
    original_filename,
    settings,
    stage_callback,
    progress_callback,
):
    stage_callback("validating", "Inspecting source.", 40)
    job_dir = settings.exports_dir / job_id
    assert (job_dir / source_file_name).is_file()
    stage_callback("normalizing", "Creating WAV.", 75)
    (job_dir / "analysis.wav").write_bytes(b"RIFFfake analysis")
    (job_dir / "metadata.json").write_text(
        '{"source_type":"upload"}', encoding="utf-8"
    )
    progress_callback(100)
    return MediaResult(
        title=Path(original_filename).stem,
        uploader=None,
        duration_seconds=2.5,
        source_format=Path(source_file_name).suffix.lstrip("."),
        sample_rate=48000,
        channel_count=2,
        source_file_name=source_file_name,
        normalized_file_name="analysis.wav",
        files=(source_file_name, "analysis.wav", "metadata.json"),
    )


def failing_upload_processor(*args, **kwargs):
    raise MediaProcessingError(
        "FFmpeg normalization failed for <local media>: invalid audio data."
    )


def make_app(tmp_path: Path, **kwargs):
    return create_app(
        settings=make_settings(tmp_path, **kwargs),
        url_processor=fake_url_processor,
        upload_processor=fake_upload_processor,
    )


def test_valid_local_mp3_upload_and_downloads(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("My Song.mp3", b"synthetic mp3 bytes", "audio/mpeg")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        payload = client.get(f"/api/jobs/{job_id}").json()
        assert payload["status"] == "completed"
        assert payload["stage"] == "completed"
        assert payload["source_type"] == "upload"
        assert payload["original_filename"] == "My Song.mp3"
        assert payload["normalized_file_name"] == "analysis.wav"
        assert re.fullmatch(
            r"source-[a-f0-9]{32}\.mp3", payload["source_file_name"]
        )
        assert {item["label"] for item in payload["files"]} == {
            "Source file",
            "Analysis audio",
            "Metadata",
        }

        source = next(item for item in payload["files"] if item["kind"] == "source")
        analysis = next(
            item for item in payload["files"] if item["kind"] == "analysis"
        )
        assert client.get(source["download_url"]).content == b"synthetic mp3 bytes"
        assert client.get(analysis["download_url"]).content == b"RIFFfake analysis"


def test_valid_wav_upload(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("tone.wav", b"RIFFsynthetic", "audio/wav")},
        )

    assert response.status_code == 202
    with TestClient(app) as client:
        payload = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert payload["status"] == "completed"
    assert payload["source_file_name"].endswith(".wav")


def test_rejects_unsupported_extension(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )

    assert response.status_code == 422
    assert "Unsupported file extension" in response.json()["detail"]


def test_rejects_empty_file_and_cleans_storage(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("empty.mp3", b"", "audio/mpeg")},
        )
        jobs = client.get("/api/jobs").json()

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()
    assert jobs[0]["status"] == "failed"
    assert not (tmp_path / "exports" / jobs[0]["id"]).exists()


def test_rejects_oversized_file(tmp_path: Path):
    app = make_app(tmp_path, max_upload_mb=1)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={
                "file": (
                    "large.mp3",
                    b"x" * (1024 * 1024 + 1),
                    "audio/mpeg",
                )
            },
        )

    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]


def test_traversal_filename_is_metadata_only(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={
                "file": (
                    "../../outside.mp3",
                    b"synthetic",
                    "audio/mpeg",
                )
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert ".." not in payload["source_file_name"]
    assert "/" not in payload["source_file_name"]
    source_path = (
        tmp_path / "exports" / payload["id"] / payload["source_file_name"]
    ).resolve()
    assert (tmp_path / "exports" / payload["id"]).resolve() in source_path.parents
    assert source_path.is_file()


def test_url_workflow_still_creates_source_and_analysis_audio(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs", json={"url": "https://www.youtube.com/watch?v=abc123"}
        )

    assert response.status_code == 202
    with TestClient(app) as client:
        payload = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert payload["source_type"] == "url"
    assert payload["status"] == "completed"
    assert {item["kind"] for item in payload["files"]} == {
        "source",
        "analysis",
        "metadata",
    }


def test_persistence_after_restart(tmp_path: Path):
    settings = make_settings(tmp_path)
    app = create_app(
        settings=settings,
        url_processor=fake_url_processor,
        upload_processor=fake_upload_processor,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/uploads",
            files={"file": ("persist.mp3", b"persistent", "audio/mpeg")},
        ).json()
        job_id = created["id"]

    restarted = create_app(
        settings=settings,
        url_processor=fake_url_processor,
        upload_processor=fake_upload_processor,
    )
    with TestClient(restarted) as client:
        persisted = client.get(f"/api/jobs/{job_id}")

    assert persisted.status_code == 200
    assert persisted.json()["status"] == "completed"
    assert any(item["kind"] == "analysis" for item in persisted.json()["files"])


def test_database_migration_preserves_existing_url_jobs(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                title TEXT,
                uploader TEXT,
                duration_seconds REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs VALUES (
                'legacy', 'https://youtube.com/watch?v=old', 'completed', 100,
                'Old URL Job', 'Artist', 10, NULL, '2026-01-01', '2026-01-01'
            )
            """
        )

    app = create_app(
        settings=make_settings(tmp_path),
        url_processor=fake_url_processor,
        upload_processor=fake_upload_processor,
    )
    with TestClient(app) as client:
        legacy = client.get("/api/jobs/legacy")

    assert legacy.status_code == 200
    assert legacy.json()["title"] == "Old URL Job"
    assert legacy.json()["source_type"] == "url"
    assert legacy.json()["stage"] == "completed"


def test_failed_ffmpeg_normalization_is_persisted(tmp_path: Path):
    app = create_app(
        settings=make_settings(tmp_path),
        url_processor=fake_url_processor,
        upload_processor=failing_upload_processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("broken.mp3", b"bad", "audio/mpeg")},
        )

    assert response.status_code == 202
    with TestClient(app) as client:
        payload = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert payload["status"] == "failed"
    assert payload["stage"] == "failed"
    assert "FFmpeg normalization failed" in payload["error"]
    assert str(tmp_path) not in payload["error"]


def test_rejects_path_traversal_download(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("song.mp3", b"media", "audio/mpeg")},
        )
        job_id = response.json()["id"]
        traversal = client.get(
            f"/api/jobs/{job_id}/files/%2E%2E%2Fpopex.sqlite3"
        )

    assert traversal.status_code in {400, 404}


def test_index_and_degraded_health_are_clear(tmp_path: Path):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/api/health").json()

    assert health["status"] == "degraded"
    assert health["dependencies"]["data_directory_writable"] is True
