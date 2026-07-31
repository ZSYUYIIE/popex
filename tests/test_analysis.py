import json
import sqlite3
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import db
from app.analysis import AudioAnalysisError, AudioAnalysisResult, analyze_audio
from app.config import Settings
from app.main import create_app
from app.media import MediaResult

SR = 44100


def settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com", "youtu.be"),
        max_duration_seconds=60,
        max_filesize_mb=10,
        max_upload_mb=10,
        audio_quality="192",
        ffmpeg_binary="missing-test-ffmpeg",
        ffprobe_binary="missing-test-ffprobe",
        audio_analysis_enabled=enabled,
        audio_analysis_version="baseline-librosa-v1",
        audio_analysis_timeout_seconds=60,
        audio_silence_rms_threshold=0.0001,
    )


def create_job_audio(tmp_path: Path, audio: np.ndarray, job_id: str = "a" * 32) -> tuple[Settings, str]:
    config = settings(tmp_path)
    config.ensure_directories()
    db.init_database(config.database_path)
    db.create_job(config.database_path, job_id, source_type="upload", original_filename="synthetic.wav")
    job_dir = config.exports_dir / job_id
    job_dir.mkdir(parents=True)
    sf.write(job_dir / "analysis.wav", audio, SR, subtype="PCM_16")
    (job_dir / "source-test.wav").write_bytes(b"synthetic")
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    db.update_job(config.database_path, job_id, status="completed", stage="completed", normalized_file_name="analysis.wav", source_file_name="source-test.wav", title="Synthetic")
    return config, job_id


def test_click_track_tempo_and_beats_are_persisted(tmp_path):
    duration = 10.0
    times = np.arange(0.5, duration, 0.5)
    audio = librosa.clicks(times=times, sr=SR, length=int(duration * SR), click_freq=1000.0)
    config, job_id = create_job_audio(tmp_path, audio)
    stages = []
    result = analyze_audio(job_id, config, lambda stage, message, progress: stages.append(stage))
    assert result.tempo_bpm == pytest.approx(120, abs=8)
    assert len(result.payload["timing"]["beatsSeconds"]) >= 12
    assert result.payload["timing"]["tempoConfidence"] is not None
    assert stages == ["analyzing_audio", "detecting_beats", "estimating_key", "saving_analysis"]
    saved = json.loads((config.exports_dir / job_id / "analysis" / "audio-analysis.json").read_text())
    assert saved["schemaVersion"] == 1
    assert saved["analysisVersion"] == "baseline-librosa-v1"
    assert saved["audio"]["sampleRate"] == SR


def test_synthetic_a_minor_tonal_signal_estimates_key(tmp_path):
    t = np.arange(SR * 8) / SR
    audio = (0.45 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 261.6256 * t) + 0.25 * np.sin(2 * np.pi * 329.6276 * t)).astype(np.float32)
    config, job_id = create_job_audio(tmp_path, audio)
    result = analyze_audio(job_id, config, lambda *args: None)
    assert result.payload["tonality"]["key"] == "A"
    assert result.payload["tonality"]["mode"] == "minor"
    assert len(result.payload["tonality"]["chromaMean"]) == 12
    assert result.payload["tonality"]["confidence"] is not None


def test_silent_audio_rejected(tmp_path):
    config, job_id = create_job_audio(tmp_path, np.zeros(SR, dtype=np.float32))
    with pytest.raises(AudioAnalysisError, match="silent|quiet"):
        analyze_audio(job_id, config, lambda *args: None)


def test_malformed_audio_failure(tmp_path):
    config = settings(tmp_path)
    config.ensure_directories()
    db.init_database(config.database_path)
    job_id = "b" * 32
    db.create_job(config.database_path, job_id, source_type="upload")
    job_dir = config.exports_dir / job_id
    job_dir.mkdir()
    (job_dir / "analysis.wav").write_bytes(b"not a wav")
    with pytest.raises(AudioAnalysisError, match="unreadable|corrupted"):
        analyze_audio(job_id, config, lambda *args: None)


def test_missing_analysis_wav_failure(tmp_path):
    config = settings(tmp_path)
    config.ensure_directories()
    db.init_database(config.database_path)
    job_id = "c" * 32
    db.create_job(config.database_path, job_id, source_type="upload")
    (config.exports_dir / job_id).mkdir()
    with pytest.raises(AudioAnalysisError, match="missing"):
        analyze_audio(job_id, config, lambda *args: None)


def fake_ingestion(job_id, source_file_name, original_filename, config, stage_callback, progress_callback):
    job_dir = config.exports_dir / job_id
    t = np.arange(SR * 4) / SR
    sf.write(job_dir / "analysis.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    return MediaResult(title="Synthetic", uploader=None, duration_seconds=4, source_format="wav", sample_rate=SR, channel_count=1, source_file_name=source_file_name, normalized_file_name="analysis.wav", files=(source_file_name, "analysis.wav", "metadata.json"))


def fake_url(job_id, source_url, config, stage_callback, progress_callback):
    job_dir = config.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.mp3").write_bytes(b"url")
    t = np.arange(SR * 4) / SR
    sf.write(job_dir / "analysis.wav", 0.2 * np.sin(2 * np.pi * 220 * t), SR)
    (job_dir / "metadata.json").write_text("{}")
    return MediaResult(title="URL", uploader="Artist", duration_seconds=4, source_format="mp3", sample_rate=SR, channel_count=1, source_file_name="source.mp3", normalized_file_name="analysis.wav", files=("source.mp3", "analysis.wav", "metadata.json"))


def test_upload_analysis_api_summary_and_restart_persistence(tmp_path):
    config = settings(tmp_path)
    app = create_app(settings=config, url_processor=fake_url, upload_processor=fake_ingestion)
    with TestClient(app) as client:
        response = client.post("/api/uploads", files={"file": ("tone.wav", b"source", "audio/wav")})
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed" and job["analysis"]["status"] == "completed"
        assert job["analysis"]["keySymbol"]
        analysis = client.get(f"/api/jobs/{job_id}/analysis").json()
        assert analysis["available"] is True
        assert analysis["result"]["schemaVersion"] == 1
        assert client.get(f"/api/jobs/{job_id}/analysis/download").status_code == 200
    restarted = create_app(settings=config, url_processor=fake_url, upload_processor=fake_ingestion)
    with TestClient(restarted) as client:
        persisted = client.get(f"/api/jobs/{job_id}/analysis").json()
        assert persisted["available"] is True and persisted["status"] == "completed"


def test_url_flow_uses_shared_post_normalization_analysis(tmp_path):
    config = settings(tmp_path)
    app = create_app(settings=config, url_processor=fake_url, upload_processor=fake_ingestion)
    with TestClient(app) as client:
        created = client.post("/api/jobs", json={"url": "https://youtube.com/watch?v=test"})
        job = client.get(f"/api/jobs/{created.json()['id']}").json()
    assert job["source_type"] == "url" and job["analysis"]["status"] == "completed"


def test_existing_completed_job_can_be_analyzed_and_duplicate_is_not_repeated(tmp_path):
    config, job_id = create_job_audio(tmp_path, 0.2 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR), job_id="d" * 32)
    calls = []

    def fake_analyzer(job_id, settings, stage_callback):
        calls.append(job_id)
        stage_callback("saving_analysis", "Saving", 94)
        path = settings.exports_dir / job_id / "analysis"
        path.mkdir(exist_ok=True)
        payload = {"schemaVersion": 1, "analysisVersion": settings.audio_analysis_version, "warnings": [], "timing": {}, "tonality": {}, "audio": {}}
        (path / "audio-analysis.json").write_text(json.dumps(payload))
        return AudioAnalysisResult(settings.audio_analysis_version, 120.0, 0.8, "A minor", 0.7, "analysis/audio-analysis.json", "2026-07-31T00:00:00+00:00", payload)

    app = create_app(settings=config, analysis_processor=fake_analyzer)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        job = client.get(f"/api/jobs/{job_id}").json()
    assert calls == [job_id]
    assert job["analysis"]["tempoBpm"] == 120.0


def test_analysis_failure_preserves_source_and_wav_then_retry_succeeds(tmp_path):
    config, job_id = create_job_audio(tmp_path, 0.2 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR), job_id="f" * 32)
    attempts = {"count": 0}

    def flaky(job_id, settings, stage_callback):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise AudioAnalysisError("Synthetic analysis failure.")
        path = settings.exports_dir / job_id / "analysis"
        path.mkdir(exist_ok=True)
        payload = {"schemaVersion": 1, "analysisVersion": settings.audio_analysis_version, "warnings": [], "timing": {}, "tonality": {}, "audio": {}}
        (path / "audio-analysis.json").write_text(json.dumps(payload))
        return AudioAnalysisResult(settings.audio_analysis_version, 100.0, 0.5, "C major", 0.5, "analysis/audio-analysis.json", "2026-07-31T00:00:00+00:00", payload)

    app = create_app(settings=config, analysis_processor=flaky)
    with TestClient(app) as client:
        client.post(f"/api/jobs/{job_id}/analyze")
        failed = client.get(f"/api/jobs/{job_id}").json()
        assert failed["status"] == "failed" and failed["analysis"]["status"] == "failed"
        assert (config.exports_dir / job_id / "source-test.wav").is_file()
        assert (config.exports_dir / job_id / "analysis.wav").is_file()
        client.post(f"/api/jobs/{job_id}/analyze")
        completed = client.get(f"/api/jobs/{job_id}").json()
    assert completed["status"] == "completed" and attempts["count"] == 2


def test_migration_adds_analysis_columns_without_losing_legacy_job(tmp_path):
    database = tmp_path / "popex.sqlite3"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, source_url TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, title TEXT, uploader TEXT, duration_seconds REAL, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO jobs VALUES ('legacy','https://youtube.com/watch?v=old','completed',100,'Old','Artist',10,NULL,'2026-01-01','2026-01-01')")
    db.init_database(database)
    legacy = db.get_job(database, "legacy")
    assert legacy["title"] == "Old" and legacy["analysis_status"] == "not_started"
