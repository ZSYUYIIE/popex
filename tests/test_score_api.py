from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.main import create_app
from app.score_construction import SCORE_BUILDER_VERSION
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    write_raw_transcription,
)

ANALYSIS_VERSION = "baseline-librosa-v1"
RAW_CREATED_AT = "2026-08-14T04:00:00+00:00"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("example.invalid",),
        max_duration_seconds=60,
        max_filesize_mb=16,
        max_upload_mb=16,
        audio_quality="192",
        ffmpeg_binary="missing-test-ffmpeg",
        ffprobe_binary="missing-test-ffprobe",
        audio_analysis_enabled=True,
    )


def analysis_payload(*, tempo=120.0, meter=4) -> dict:
    return {
        "schemaVersion": 1,
        "analysisVersion": ANALYSIS_VERSION,
        "createdAt": "2026-08-14T03:55:00+00:00",
        "sourceAsset": "analysis.wav",
        "libraries": {},
        "audio": {
            "durationSeconds": 2.0,
            "sampleRate": 44100,
            "channels": 1,
            "peakAmplitude": 0.8,
            "rms": 0.2,
            "rmsDbfs": -13.9,
            "silent": False,
        },
        "timing": {
            "tempoBpm": tempo,
            "tempoConfidence": 0.9,
            "tempoStable": True,
            "beatsSeconds": [0.0, 0.5, 1.0],
            "beatConfidence": 0.9,
            "downbeatsSeconds": [0.0],
            "meter": meter,
            "meterConfidence": 0.8,
        },
        "tonality": {
            "tonalCenter": "C",
            "primaryCandidate": {
                "tonalCenter": "C",
                "collection": "ionian",
                "displayName": "C major",
                "confidence": 0.8,
                "supportedByBaseline": True,
            },
            "candidates": [],
            "localRegions": [],
            "chromaticismScore": None,
            "baselineCollections": ["ionian", "aeolian"],
            "key": "C",
            "mode": "major",
            "symbol": "C major",
            "confidence": 0.8,
            "scoreMargin": 0.1,
            "tuningOffsetCents": 0.0,
            "chromaMean": [0.0] * 12,
            "alternatives": [],
        },
        "warnings": [],
    }


def raw_payload() -> dict:
    events = [
        {
            "id": "p_c",
            "sourceKind": "full_mix",
            "startSeconds": 0.0,
            "endSeconds": 0.4,
            "midiNote": 60,
            "midiPitch": 60.05,
            "frequencyHz": 261.6,
            "noteName": "C4",
            "confidence": 0.9,
            "warnings": [],
        },
        {
            "id": "p_e",
            "sourceKind": "full_mix",
            "startSeconds": 0.5,
            "endSeconds": 0.9,
            "midiNote": 64,
            "midiPitch": 64.02,
            "frequencyHz": 329.6,
            "noteName": "E4",
            "confidence": 0.9,
            "warnings": [],
        },
    ]
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "raw-transcription-v1",
        "createdAt": RAW_CREATED_AT,
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": ANALYSIS_VERSION,
        },
        "algorithms": {"testRaw": {"version": "raw-transcription-v1"}},
        "pitchedNoteEvents": events,
        "percussionEvents": [],
        "alignmentCandidates": [],
        "warnings": [],
    }


def create_job(settings: Settings, *, transcribed=True, tempo=120.0, meter=4) -> str:
    settings.ensure_directories()
    db.init_database(settings.database_path)
    job_id = uuid4().hex
    db.create_job(
        settings.database_path,
        job_id,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = settings.exports_dir / job_id
    (job_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (job_dir / "analysis" / "audio-analysis.json").write_text(
        json.dumps(analysis_payload(tempo=tempo, meter=meter), allow_nan=False),
        encoding="utf-8",
    )
    db.update_job(
        settings.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        normalized_file_name="analysis.wav",
        analysis_status="completed",
        analysis_version=ANALYSIS_VERSION,
        analysis_json_file_name=ANALYSIS_JSON_RELATIVE_PATH,
        analyzed_at="2026-08-14T03:55:00+00:00",
        transcription_status="completed" if transcribed else "not_started",
        transcription_stage="completed" if transcribed else "not_started",
        transcription_progress=100 if transcribed else 0,
        transcription_version="raw-transcription-v1" if transcribed else None,
        transcription_artifact_file_name=(
            RAW_TRANSCRIPTION_RELATIVE_PATH if transcribed else None
        ),
        transcribed_at=RAW_CREATED_AT if transcribed else None,
        pitched_event_count=2 if transcribed else None,
        percussion_event_count=0 if transcribed else None,
        aligned_event_count=0 if transcribed else None,
    )
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    return job_id


def test_unknown_job_returns_404(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    db.init_database(settings.database_path)
    client = TestClient(create_app(settings))
    assert client.get(f"/api/jobs/{'0' * 32}/score").status_code == 404


def test_untranscribed_job_has_no_score_preview(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings, transcribed=False)
    client = TestClient(create_app(settings))
    response = client.get(f"/api/jobs/{job_id}/score")
    assert response.status_code == 404


def test_score_preview_reports_provenance_and_counts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    client = TestClient(create_app(settings))
    response = client.get(f"/api/jobs/{job_id}/score")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["builderVersion"] == SCORE_BUILDER_VERSION
    assert payload["tempoBpm"] == 120.0
    assert payload["beatsPerMeasure"] == 4
    assert payload["meterSource"] == "analysis"
    assert payload["noteCount"] == 2
    assert "measures" not in payload
    assert payload["provenance"]["transcriptionVersion"] == "raw-transcription-v1"
    assert payload["provenance"]["analysisVersion"] == ANALYSIS_VERSION


def test_score_preview_with_measures_and_meter_fallback(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings, meter=None)
    client = TestClient(create_app(settings))
    response = client.get(
        f"/api/jobs/{job_id}/score", params={"includeMeasures": "true"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["meterSource"] == "fallback-4/4"
    assert any("4/4" in warning for warning in payload["warnings"])
    assert len(payload["measures"]) == payload["measureCount"]
    assert payload["measures"][0]["notes"][0]["midiNote"] == 60


def test_score_preview_rejects_bad_flag(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    client = TestClient(create_app(settings))
    response = client.get(
        f"/api/jobs/{job_id}/score", params={"includeMeasures": "yes"}
    )
    assert response.status_code == 422


def test_score_midi_download_has_smf_magic(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    client = TestClient(create_app(settings))
    response = client.get(
        f"/api/jobs/{job_id}/score/download", params={"format": "midi"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"
    assert response.content[:4] == b"MThd"
    assert bytes((0x90, 60)) in response.content


def test_score_musicxml_download_parses(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    client = TestClient(create_app(settings))
    response = client.get(
        f"/api/jobs/{job_id}/score/download", params={"format": "musicxml"}
    )
    assert response.status_code == 200
    assert "musicxml" in response.headers["content-type"]
    root = ET.fromstring(response.text)
    assert root.tag == "score-partwise"
    assert SCORE_BUILDER_VERSION in response.text


def test_score_download_rejects_unknown_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    client = TestClient(create_app(settings))
    assert (
        client.get(
            f"/api/jobs/{job_id}/score/download", params={"format": "pdf"}
        ).status_code
        == 422
    )
    assert (
        client.get(f"/api/jobs/{job_id}/score/download").status_code == 422
    )
