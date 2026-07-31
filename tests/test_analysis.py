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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def create_job_audio(
    tmp_path: Path,
    audio: np.ndarray,
    job_id: str = "a" * 32,
) -> tuple[Settings, str]:
    config = settings(tmp_path)
    config.ensure_directories()
    db.init_database(config.database_path)
    db.create_job(
        config.database_path,
        job_id,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = config.exports_dir / job_id
    job_dir.mkdir(parents=True)
    sf.write(job_dir / "analysis.wav", audio, SR, subtype="PCM_16")
    (job_dir / "source-test.wav").write_bytes(b"synthetic")
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    db.update_job(
        config.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        normalized_file_name="analysis.wav",
        source_file_name="source-test.wav",
        metadata_file_name="metadata.json",
        title="Synthetic",
    )
    return config, job_id


def test_click_track_tempo_and_beats_are_persisted(tmp_path: Path):
    duration = 10.0
    times = np.arange(0.5, duration, 0.5)
    audio = librosa.clicks(
        times=times,
        sr=SR,
        length=int(duration * SR),
        click_freq=1000.0,
    )
    config, job_id = create_job_audio(tmp_path, audio)
    stages = []

    result = analyze_audio(
        job_id,
        config,
        lambda stage, message, progress: stages.append(stage),
    )

    assert result.tempo_bpm == pytest.approx(120, abs=8)
    assert len(result.payload["timing"]["beatsSeconds"]) >= 12
    assert result.payload["timing"]["tempoConfidence"] is not None
    assert stages == [
        "analyzing_audio",
        "detecting_beats",
        "estimating_key",
        "saving_analysis",
    ]

    saved = json.loads(
        (
            config.exports_dir
            / job_id
            / "analysis"
            / "audio-analysis.json"
        ).read_text()
    )
    assert saved["schemaVersion"] == 1
    assert saved["analysisVersion"] == "baseline-librosa-v1"
    assert saved["audio"]["sampleRate"] == SR


def test_synthetic_a_minor_tonal_signal_uses_extensible_schema(tmp_path: Path):
    t = np.arange(SR * 8) / SR
    audio = (
        0.45 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 261.6256 * t)
        + 0.25 * np.sin(2 * np.pi * 329.6276 * t)
    ).astype(np.float32)
    config, job_id = create_job_audio(tmp_path, audio)

    result = analyze_audio(job_id, config, lambda *args: None)
    tonality = result.payload["tonality"]

    assert tonality["tonalCenter"] == "A"
    assert tonality["primaryCandidate"]["collection"] == "aeolian"
    assert tonality["primaryCandidate"]["displayName"] == "A minor"
    assert tonality["primaryCandidate"]["supportedByBaseline"] is True
    assert tonality["localRegions"] == []
    assert tonality["chromaticismScore"] is None
    assert tonality["baselineCollections"] == ["ionian", "aeolian"]
    assert tonality["candidates"]
    assert all(
        isinstance(candidate["collection"], str)
        for candidate in tonality["candidates"]
    )

    # Temporary compatibility fields remain available to existing clients.
    assert tonality["key"] == "A"
    assert tonality["mode"] == "minor"
    assert tonality["symbol"] == "A minor"
    assert len(tonality["chromaMean"]) == 12
    assert tonality["confidence"] is not None


def test_silent_audio_rejected(tmp_path: Path):
    config, job_id = create_job_audio(
        tmp_path,
        np.zeros(SR, dtype=np.float32),
    )
    with pytest.raises(AudioAnalysisError, match="silent|quiet"):
        analyze_audio(job_id, config, lambda *args: None)


def test_malformed_audio_failure(tmp_path: Path):
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


def test_missing_analysis_wav_failure(tmp_path: Path):
    config = settings(tmp_path)
    config.ensure_directories()
    db.init_database(config.database_path)
    job_id = "c" * 32
    db.create_job(config.database_path, job_id, source_type="upload")
    (config.exports_dir / job_id).mkdir()

    with pytest.raises(AudioAnalysisError, match="missing"):
        analyze_audio(job_id, config, lambda *args: None)


def fake_ingestion(
    job_id,
    source_file_name,
    original_filename,
    config,
    stage_callback,
    progress_callback,
):
    job_dir = config.exports_dir / job_id
    t = np.arange(SR * 4) / SR
    sf.write(
        job_dir / "analysis.wav",
        (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
        SR,
    )
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    return MediaResult(
        title="Synthetic",
        uploader=None,
        duration_seconds=4,
        source_format="wav",
        sample_rate=SR,
        channel_count=1,
        source_file_name=source_file_name,
        normalized_file_name="analysis.wav",
        files=(source_file_name, "analysis.wav", "metadata.json"),
    )


def fake_url(
    job_id,
    source_url,
    config,
    stage_callback,
    progress_callback,
):
    job_dir = config.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.mp3").write_bytes(b"url")
    t = np.arange(SR * 4) / SR
    sf.write(
        job_dir / "analysis.wav",
        0.2 * np.sin(2 * np.pi * 220 * t),
        SR,
    )
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    return MediaResult(
        title="URL",
        uploader="Artist",
        duration_seconds=4,
        source_format="mp3",
        sample_rate=SR,
        channel_count=1,
        source_file_name="source.mp3",
        normalized_file_name="analysis.wav",
        files=("source.mp3", "analysis.wav", "metadata.json"),
    )


def successful_analysis_result(
    job_id: str,
    config: Settings,
    *,
    tempo: float = 120.0,
    key: str = "A minor",
) -> AudioAnalysisResult:
    analysis_dir = config.exports_dir / job_id / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "analysisVersion": config.audio_analysis_version,
        "warnings": [],
        "timing": {"tempoBpm": tempo},
        "tonality": {
            "tonalCenter": key.split()[0],
            "primaryCandidate": {
                "collection": "aeolian" if key.endswith("minor") else "ionian",
                "displayName": key,
                "confidence": 0.7,
            },
            "key": key.split()[0],
            "mode": key.split()[1],
            "symbol": key,
        },
        "audio": {},
    }
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return AudioAnalysisResult(
        config.audio_analysis_version,
        tempo,
        0.8,
        key,
        0.7,
        "analysis/audio-analysis.json",
        "2026-07-31T00:00:00+00:00",
        payload,
    )


def test_progress_stays_below_100_while_analysis_runs(tmp_path: Path):
    config = settings(tmp_path)
    observed_progress = []

    def inspecting_analyzer(job_id, config, stage_callback):
        before = db.get_job(config.database_path, job_id)
        observed_progress.append(before["progress"])
        assert before["status"] == "processing"
        assert before["preparation_status"] == "completed"
        assert before["analysis_status"] == "processing"
        stage_callback("saving_analysis", "Saving", 94)
        during = db.get_job(config.database_path, job_id)
        observed_progress.append(during["progress"])
        return successful_analysis_result(job_id, config)

    app = create_app(
        settings=config,
        url_processor=fake_url,
        upload_processor=fake_ingestion,
        analysis_processor=inspecting_analyzer,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("tone.wav", b"source", "audio/wav")},
        )
        job = client.get(f"/api/jobs/{response.json()['id']}").json()

    assert observed_progress
    assert all(progress < 100 for progress in observed_progress)
    assert job["progress"] == 100
    assert job["analysis"]["status"] == "completed"


def test_upload_analysis_api_summary_and_restart_persistence(tmp_path: Path):
    config = settings(tmp_path)
    app = create_app(
        settings=config,
        url_processor=fake_url,
        upload_processor=fake_ingestion,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("tone.wav", b"source", "audio/wav")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed"
        assert job["preparation"]["status"] == "completed"
        assert job["analysis"]["status"] == "completed"
        assert job["analysis"]["keySymbol"]

        analysis = client.get(f"/api/jobs/{job_id}/analysis").json()
        assert analysis["available"] is True
        assert analysis["result"]["schemaVersion"] == 1
        assert analysis["summary"]["keySymbol"] == job["analysis"]["keySymbol"]
        assert client.get(
            f"/api/jobs/{job_id}/analysis/download"
        ).status_code == 200

    restarted = create_app(
        settings=config,
        url_processor=fake_url,
        upload_processor=fake_ingestion,
    )
    with TestClient(restarted) as client:
        persisted = client.get(f"/api/jobs/{job_id}/analysis").json()
        assert persisted["available"] is True
        assert persisted["status"] == "completed"


def test_url_and_upload_flows_share_post_normalization_analysis(tmp_path: Path):
    config = settings(tmp_path)
    app = create_app(
        settings=config,
        url_processor=fake_url,
        upload_processor=fake_ingestion,
    )
    with TestClient(app) as client:
        url_created = client.post(
            "/api/jobs",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        url_job = client.get(
            f"/api/jobs/{url_created.json()['id']}"
        ).json()

        upload_created = client.post(
            "/api/uploads",
            files={"file": ("tone.wav", b"source", "audio/wav")},
        )
        upload_job = client.get(
            f"/api/jobs/{upload_created.json()['id']}"
        ).json()

    assert url_job["source_type"] == "url"
    assert upload_job["source_type"] == "upload"
    assert url_job["preparation"]["status"] == "completed"
    assert upload_job["preparation"]["status"] == "completed"
    assert url_job["analysis"]["status"] == "completed"
    assert upload_job["analysis"]["status"] == "completed"


def test_existing_job_analysis_is_reused_and_active_duplicate_is_rejected(
    tmp_path: Path,
):
    audio = 0.2 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)
    config, job_id = create_job_audio(
        tmp_path,
        audio,
        job_id="d" * 32,
    )
    calls = []

    def fake_analyzer(job_id, config, stage_callback):
        calls.append(job_id)
        stage_callback("saving_analysis", "Saving", 94)
        return successful_analysis_result(job_id, config)

    app = create_app(settings=config, analysis_processor=fake_analyzer)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        job = client.get(f"/api/jobs/{job_id}").json()

        db.update_job(
            config.database_path,
            job_id,
            status="processing",
            stage="analyzing_audio",
            progress=66,
            analysis_status="processing",
        )
        duplicate = client.post(f"/api/jobs/{job_id}/analyze")

    assert calls == [job_id]
    assert job["analysis"]["tempoBpm"] == 120.0
    assert duplicate.status_code == 409


def test_analysis_failure_preserves_artifacts_and_ingestion_success(tmp_path: Path):
    audio = 0.2 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)
    config, job_id = create_job_audio(
        tmp_path,
        audio,
        job_id="f" * 32,
    )
    attempts = {"count": 0}

    def flaky(job_id, config, stage_callback):
        attempts["count"] += 1
        stage_callback("estimating_key", "Estimating", 84)
        if attempts["count"] == 1:
            raise AudioAnalysisError("Synthetic analysis failure.")
        return successful_analysis_result(
            job_id,
            config,
            tempo=100.0,
            key="C major",
        )

    app = create_app(settings=config, analysis_processor=flaky)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        failed = client.get(f"/api/jobs/{job_id}").json()

        assert failed["status"] == "completed"
        assert failed["stage"] == "completed"
        assert failed["progress"] < 100
        assert failed["preparation"]["status"] == "completed"
        assert failed["preparation"]["sourceAvailable"] is True
        assert failed["preparation"]["analysisAudioAvailable"] is True
        assert failed["analysis"]["status"] == "failed"
        assert failed["error"] is None
        assert "Synthetic analysis failure" in failed["analysis"]["error"]

        files = {item["kind"]: item for item in failed["files"]}
        assert client.get(files["source"]["download_url"]).status_code == 200
        assert client.get(files["analysis"]["download_url"]).status_code == 200
        assert client.get(files["metadata"]["download_url"]).status_code == 200

        arbitrary = client.get(
            f"/api/jobs/{job_id}/files/not-persisted.wav"
        )
        traversal = client.get(
            f"/api/jobs/{job_id}/files/%2E%2E%2Fpopex.sqlite3"
        )
        assert arbitrary.status_code == 404
        assert traversal.status_code in {400, 404}

        assert client.post(f"/api/jobs/{job_id}/analyze").status_code == 202
        completed = client.get(f"/api/jobs/{job_id}").json()

    assert completed["status"] == "completed"
    assert completed["progress"] == 100
    assert completed["analysis"]["status"] == "completed"
    assert attempts["count"] == 2


def test_migration_adds_analysis_fields_without_losing_legacy_job(tmp_path: Path):
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
                'legacy',
                'https://youtube.com/watch?v=old',
                'completed',
                100,
                'Old',
                'Artist',
                10,
                NULL,
                '2026-01-01',
                '2026-01-01'
            )
            """
        )

    db.init_database(database)
    legacy = db.get_job(database, "legacy")

    assert legacy["title"] == "Old"
    assert legacy["preparation_status"] == "completed"
    assert legacy["analysis_status"] == "not_started"


def test_stylesheet_remains_readable_and_not_minified():
    stylesheet = (PROJECT_ROOT / "app" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )
    lines = stylesheet.splitlines()

    assert "\n.analysis-panel {\n" in stylesheet
    assert "*{box-sizing" not in stylesheet
    assert len(lines) > 250
    assert max(len(line) for line in lines) < 180
