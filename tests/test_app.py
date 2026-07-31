from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.downloader import ExtractionResult
from app.main import create_app


def fake_extractor(job_id, source_url, settings, progress_callback):
    progress_callback(50)
    job_dir = settings.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "test-song.mp3").write_bytes(b"fake mp3 content")
    (job_dir / "metadata.json").write_text('{"title":"Test Song"}', encoding="utf-8")
    progress_callback(100)
    return ExtractionResult(
        title="Test Song",
        uploader="Test Artist",
        duration_seconds=123,
        files=("test-song.mp3", "metadata.json"),
    )


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com", "youtu.be"),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        audio_quality="192",
    )


def test_create_extract_and_download_job(tmp_path: Path):
    app = create_app(settings=make_settings(tmp_path), extractor=fake_extractor)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs", json={"url": "https://www.youtube.com/watch?v=abc123"}
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        job = client.get(f"/api/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["status"] == "completed"
        assert job.json()["title"] == "Test Song"
        assert {item["name"] for item in job.json()["files"]} == {
            "test-song.mp3",
            "metadata.json",
        }

        download = client.get(f"/api/jobs/{job_id}/files/test-song.mp3")
        assert download.status_code == 200
        assert download.content == b"fake mp3 content"


def test_rejects_non_allowlisted_source(tmp_path: Path):
    app = create_app(settings=make_settings(tmp_path), extractor=fake_extractor)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs", json={"url": "https://example.com/video.mp4"}
        )

    assert response.status_code == 422
    assert "not allowed" in response.json()["detail"]


def test_rejects_path_traversal(tmp_path: Path):
    app = create_app(settings=make_settings(tmp_path), extractor=fake_extractor)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs", json={"url": "https://youtube.com/watch?v=abc123"}
        )
        job_id = response.json()["id"]
        traversal = client.get(f"/api/jobs/{job_id}/files/%2E%2E%2Fpopex.sqlite3")

    assert traversal.status_code in {400, 404}


def test_index_and_health(tmp_path: Path):
    app = create_app(settings=make_settings(tmp_path), extractor=fake_extractor)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json() == {"status": "ok"}
