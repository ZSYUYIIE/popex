"""Meter from where the drums and bass repeat (cycle 46)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import db
from app.audio_chords import audio_chords_file_name
from app.beat_grid import build_beat_grid, infer_meter
from app.main import create_app
from app.score_api import score_meter
from app.score_pipeline import score_outdated_reason
from test_audio_chords_pipeline import write_analysis_audio
from test_harmony_api import create_job, make_settings

BEAT = 0.5


def beats(count: int) -> list[float]:
    return [round(index * BEAT, 6) for index in range(count)]


def hits(pattern: dict[str, list[int]], meter: int, bars: int) -> list[tuple[float, str]]:
    """Drum hits at the given beat positions (0-based) of every bar."""
    return [
        (round((bar * meter + position) * BEAT, 6), voice)
        for bar in range(bars)
        for voice, positions in pattern.items()
        for position in positions
    ]


def test_a_waltz_kick_on_every_third_beat_is_three_four():
    drum_hits = hits({"low": [0], "mid": [1, 2]}, 3, 16)
    meter, margin = infer_meter(beats(48), drum_hits=drum_hits)
    assert meter == 3 and margin > 0.5


def test_a_rock_beat_is_four_four():
    drum_hits = hits({"low": [0, 2], "mid": [1, 3]}, 4, 12)
    assert infer_meter(beats(48), drum_hits=drum_hits)[0] == 4


def test_the_bass_on_beat_one_settles_the_meter_without_drums():
    bass = [bar * 3 * BEAT for bar in range(16)]
    assert infer_meter(beats(48), bass_onsets=bass)[0] == 3
    bass = [bar * 4 * BEAT for bar in range(12)]
    assert infer_meter(beats(48), bass_onsets=bass)[0] == 4


def test_no_decision_without_a_repeating_accent():
    # A kick on every beat repeats with any period, so it says nothing about the bar.
    assert infer_meter(beats(48), drum_hits=[(time, "low") for time in beats(48)]) is None
    # Too few hits to count.
    assert infer_meter(beats(48), drum_hits=hits({"low": [0]}, 3, 3)) is None
    assert infer_meter([0.0, 0.5], drum_hits=hits({"low": [0]}, 3, 16)) is None


def test_a_waltz_bar_starts_on_the_kick():
    drum_hits = hits({"low": [0], "mid": [1, 2]}, 3, 16)
    # Shift everything by one beat: the first tracked beat is a beat 3, so
    # the bar phase must come from the kick, not from the first beat.
    shifted = [(time + BEAT, voice) for time, voice in drum_hits]
    grid = build_beat_grid(beats(50), meter=3, end_seconds=24.5, drum_hits=shifted)
    assert grid.bar_phase_source == "drums"
    downbeats = grid.times[::3]
    assert all(round((time - BEAT) / BEAT) % 3 == 0 for time in downbeats if time >= 0)


def _raw(drum_hits, bass_onsets=()):
    events = [
        {"timeSeconds": time, "hits": [{"kind": "low_impulse" if voice == "low" else "mid_impulse", "confidence": 0.9}]}
        for time, voice in drum_hits
    ]
    notes = [{"sourceKind": "bass", "startSeconds": time} for time in bass_onsets]
    return {"percussionEvents": events, "pitchedNoteEvents": notes}


def test_score_meter_prefers_the_drums_then_the_analysis_then_four_four():
    analysis = {"timing": {"beatsSeconds": beats(48), "meter": 4, "meterConfidence": 0.3}}
    bass = [bar * 3 * BEAT for bar in range(16)]
    assert score_meter(analysis, _raw([], bass))[:2] == (3, "drums-and-bass")
    assert score_meter(analysis, _raw([])) == (4, "analysis", 0.3)
    assert score_meter({"timing": {"beatsSeconds": beats(48)}}, _raw([])) == (4, "fallback-4/4", None)


def test_older_scores_are_offered_a_redraft_for_the_meter():
    assert score_outdated_reason({"score_version": "score-pipeline-v9", "score_measure_count": 12}) == "meter"
    assert score_outdated_reason({"score_version": "score-pipeline-v10", "score_measure_count": 12}) is None
    assert score_outdated_reason({"score_version": "score-pipeline-v9", "score_measure_count": 1}) is None


def test_drafting_again_reruns_harmony_heard_on_an_older_beat_grid(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    write_analysis_audio(settings, job_id)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/draft").status_code == 202
        record = db.get_job(settings.database_path, job_id)
        path = settings.exports_dir / job_id / audio_chords_file_name(record["harmonized_at"])
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["gridVersion"] = "beat-grid-v1"  # heard before meter came from the drums
        path.write_text(json.dumps(saved), encoding="utf-8")
        assert client.post(f"/api/jobs/{job_id}/draft").status_code == 202
        progress = client.get(f"/api/jobs/{job_id}").json()["draft"]["progress"]
    outcomes = {step["key"]: step["outcome"] for step in progress["steps"]}
    assert outcomes["harmonize"] == "ran"
