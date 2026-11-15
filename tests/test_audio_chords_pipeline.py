from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import audio_chords_pipeline, db
from app.audio_chords import audio_chords_file_name, load_audio_chords_artifact
from app.audio_chords_pipeline import publish_audio_chords
from app.harmony_pipeline import infer_harmony_job
from app.main import create_app
from test_audio_chords import SR, drums, progression
from test_harmony_api import (
    ANALYSIS_VERSION,
    HARMONY_CREATED_AT,
    analysis_payload,
    create_job,
    make_settings,
    publish_harmony,
    raw_payload,
)

BARS = ["C", "C", "G", "G", "Am", "Am", "F", "F"]


def write_analysis_audio(settings, job_id: str, bars: list[str | None] = BARS) -> None:
    """Synthetic 120 BPM recording: one chord per bar, bass and drum noise."""
    accompaniment, bass, _ = progression(bars)
    mix = accompaniment + bass + drums(len(accompaniment) / SR)
    sf.write(settings.exports_dir / job_id / "analysis.wav", mix, SR, subtype="PCM_16")


def publish(settings, job_id: str, **overrides):
    arguments = {
        "raw_transcription": raw_payload(),
        "timing": analysis_payload()["timing"],
        "analysis_version": ANALYSIS_VERSION,
        "harmony_created_at": HARMONY_CREATED_AT,
    }
    arguments.update(overrides)
    return publish_audio_chords(job_id, settings, **arguments)


def test_chords_are_saved_beside_the_harmony_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    write_analysis_audio(settings, job_id)

    relative = publish(settings, job_id)

    assert relative == audio_chords_file_name(HARMONY_CREATED_AT)
    artifact = load_audio_chords_artifact(settings, job_id, HARMONY_CREATED_AT)
    assert artifact["source"] == "mix"
    assert artifact["analysisVersion"] == ANALYSIS_VERSION
    # The fixture analysis tracked too few beats for a beat grid, so beats
    # come from its constant tempo.
    assert artifact["gridVersion"] is None
    per_beat = [segment["symbol"] for segment in artifact["segments"] for _ in range(segment["beats"])]
    assert per_beat[:32] == [symbol for symbol in BARS for _ in range(4)]


def test_nothing_is_saved_without_audio_tempo_or_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)

    assert publish(settings, job_id) is None  # no analysis.wav
    write_analysis_audio(settings, job_id)
    assert publish(settings, job_id, timing=None) is None
    assert publish(settings, job_id, timing={"tempoBpm": None, "beatsSeconds": []}) is None
    assert publish(settings, "f" * 32) is None
    assert not (settings.exports_dir / job_id / "chords").exists()


def test_very_short_audio_is_skipped(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    sf.write(settings.exports_dir / job_id / "analysis.wav", np.zeros(SR, dtype="float32"), SR)
    assert publish(settings, job_id) is None


def test_a_failure_while_listening_never_escapes(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    write_analysis_audio(settings, job_id)

    def explode(*args, **kwargs):
        raise RuntimeError("boom at C:/Users/someone/secret")

    monkeypatch.setattr(audio_chords_pipeline, "recognize_chords", explode)
    assert publish(settings, job_id) is None
    assert not list((settings.exports_dir / job_id).glob("chords/*.json"))


def test_harmony_stage_listens_and_survives_a_listening_failure(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    write_analysis_audio(settings, job_id)
    stages: list[str] = []

    result = infer_harmony_job(job_id, settings, lambda stage, message, progress: stages.append(stage))

    assert "listening_for_chords" in stages and stages[-1] == "completed"
    assert load_audio_chords_artifact(settings, job_id, result.created_at) is not None

    monkeypatch.setattr(audio_chords_pipeline, "recognize_chords", lambda *a, **k: 1 / 0)
    second = infer_harmony_job(job_id, settings, created_at="2026-11-16T09:00:00+00:00")
    assert second.created_at == "2026-11-16T09:00:00+00:00"
    assert load_audio_chords_artifact(settings, job_id, second.created_at) is None
    # The earlier result keeps its chords.
    assert load_audio_chords_artifact(settings, job_id, result.created_at) is not None


def test_harmony_stage_without_audio_is_unchanged(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    result = infer_harmony_job(job_id, settings)
    assert result.segment_count >= 0
    assert not (settings.exports_dir / job_id / "chords").exists()


def build_score(settings, job_id: str) -> dict:
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        return client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()


def test_score_prefers_chords_heard_in_the_audio(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    write_analysis_audio(settings, job_id)
    publish(settings, job_id)

    details = build_score(settings, job_id)

    assert details["measures"][0]["chordSymbol"] == "C"
    assert details["layers"]["chordSymbols"]["status"] == "included"
    assert "heard in the audio (the full mix)" in details["layers"]["chordSymbols"]["note"]
    assert "Sevenths, suspensions and inversions are not claimed" in details["layers"]["chordSymbols"]["note"]
    windows = details["measures"][0]["harmony"]
    assert windows and all(entry["segmentId"].startswith("c") for entry in windows)
    assert details["counts"]["chordSymbols"] >= 1


def test_score_falls_back_to_note_based_harmony_without_audio_chords(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)

    details = build_score(settings, job_id)

    assert "heard in the audio" not in details["layers"]["chordSymbols"]["note"]
    assert details["layers"]["chordSymbols"]["status"] == "included"


def test_audio_chords_from_another_harmony_run_are_ignored(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    write_analysis_audio(settings, job_id)
    publish(settings, job_id, harmony_created_at="2026-01-01T00:00:00+00:00")

    details = build_score(settings, job_id)
    assert "heard in the audio" not in details["layers"]["chordSymbols"]["note"]


def test_corrupt_audio_chords_do_not_break_the_score(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    write_analysis_audio(settings, job_id)
    relative = publish(settings, job_id)
    (settings.exports_dir / job_id / relative).write_text("{broken", encoding="utf-8")

    details = build_score(settings, job_id)
    assert details["status"] == "completed"
    assert "heard in the audio" not in details["layers"]["chordSymbols"]["note"]


def test_audio_chords_for_other_analysis_are_ignored(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    write_analysis_audio(settings, job_id)
    relative = publish(settings, job_id)
    path = settings.exports_dir / job_id / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    document["analysisVersion"] = "another-analysis-v9"
    path.write_text(json.dumps(document), encoding="utf-8")

    details = build_score(settings, job_id)
    assert "heard in the audio" not in details["layers"]["chordSymbols"]["note"]


@pytest.mark.parametrize("meter", [3, 4])
def test_meter_from_the_analysis_is_used(tmp_path: Path, meter: int) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    accompaniment, bass, _ = progression(["C", "C", "G", "G", "Am", "Am", "F", "F"], meter=meter)
    sf.write(settings.exports_dir / job_id / "analysis.wav", accompaniment + bass, SR)
    timing = {**analysis_payload()["timing"], "meter": meter}
    assert publish(settings, job_id, timing=timing) is not None
    artifact = load_audio_chords_artifact(settings, job_id, HARMONY_CREATED_AT)
    per_beat = [segment["symbol"] for segment in artifact["segments"] for _ in range(segment["beats"])]
    assert per_beat[: 4 * meter] == ["C"] * (2 * meter) + ["G"] * (2 * meter)
