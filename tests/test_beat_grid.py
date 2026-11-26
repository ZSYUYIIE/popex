from __future__ import annotations

import math
import random

import pytest

from app.beat_grid import BEAT_GRID_VERSION, BeatGrid, build_beat_grid
from app.score_construction import ScoreConstructionError, build_score_document



def _distance_to_bar_start(beat: float, meter: int = 4) -> float:
    """Distance to the nearest bar start, so 3.9999999 and 0.0000001 both count as 0."""
    phase = beat % meter
    return min(phase, meter - phase)

def steady_beats(tempo: float = 120.0, count: int = 64, first: float = 0.0) -> list[float]:
    beat = 60.0 / tempo
    return [first + index * beat for index in range(count)]


def backbeat(beats: list[float], *, downbeat_index: int = 0) -> list[tuple[float, str]]:
    """Kick on beat 1 (and softer use of beat 3), snare on beats 2 and 4."""
    hits = []
    for index, seconds in enumerate(beats):
        position = (index - downbeat_index) % 4
        if position == 0:
            hits.append((seconds, "low"))
        elif position in (1, 3):
            hits.append((seconds, "mid"))
        elif index % 8 == 2:
            hits.append((seconds, "low"))
    return hits


def note(note_id: str, start: float, end: float, midi: int = 60) -> dict:
    return {
        "id": note_id,
        "sourceKind": "full_mix",
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": midi,
        "midiPitch": float(midi),
        "confidence": 0.9,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Building the grid


def test_steady_beats_become_one_straight_line_with_the_true_tempo():
    # The tracker reports each beat on a coarse frame grid (about 23 ms), the
    # way a hop of 512 samples at 22.05 kHz does.
    hop = 512 / 22050
    exact = steady_beats(160.0, 400, first=0.13)
    tracked = [round(value / hop) * hop for value in exact]
    grid = build_beat_grid(tracked, meter=4, end_seconds=exact[-1])

    assert grid is not None
    assert grid.mode == "steady"
    assert grid.tempo_bpm == pytest.approx(160.0, abs=0.02)
    assert grid.tracked_beats == 400
    # No drift: the last tracked beat still sits on its own grid beat.
    assert grid.beat_at(exact[-1]) == pytest.approx(399.0, abs=0.06)
    intervals = [b - a for a, b in zip(grid.times, grid.times[1:])]
    assert max(intervals) - min(intervals) < 1e-9


def test_a_slightly_wrong_global_tempo_no_longer_matters():
    # The beats are right even when the analysis reports 161.5 BPM for a
    # 160 BPM recording; the grid is built from the beats, not that number.
    grid = build_beat_grid(steady_beats(160.0, 640, first=0.1), meter=4, end_seconds=240.0)
    document = build_score_document(
        [note("p1", 0.1 + 600 * 0.375, 0.1 + 601 * 0.375)],
        tempo_bpm=161.499,
        beats_per_measure=4,
        beat_grid=grid,
    )
    placed = document["measures"][150]["notes"][0]

    assert document["tempoBpm"] == pytest.approx(160.0, abs=0.001)
    assert placed["quantizedBeat"] == 600.0
    assert placed["quantizationShiftSeconds"] < 0.001


def test_moving_tempo_is_followed_instead_of_averaged():
    # The tempo falls from 120 to 90 BPM across the recording.
    beats = [0.0]
    for index in range(1, 160):
        tempo = 120.0 - 30.0 * index / 159
        beats.append(beats[-1] + 60.0 / tempo)
    grid = build_beat_grid(beats, meter=4, end_seconds=beats[-1])

    assert grid is not None
    assert grid.mode == "following"
    for index in (0, 40, 80, 120, 159):
        assert grid.beat_at(beats[index]) == pytest.approx(index, abs=0.05)


def test_grid_is_deterministic():
    beats = steady_beats(96.0, 120, first=0.2)
    first = build_beat_grid(beats, meter=4, end_seconds=80.0, drum_hits=backbeat(beats))
    second = build_beat_grid(beats, meter=4, end_seconds=80.0, drum_hits=backbeat(beats))
    assert first.times == second.times
    assert first.summary() == second.summary()


def test_skipped_beats_are_filled_before_counting():
    beats = steady_beats(120.0, 80)
    tracked = [value for index, value in enumerate(beats) if index not in (20, 21, 50)]
    grid = build_beat_grid(tracked, meter=4, end_seconds=beats[-1])

    assert grid is not None
    assert grid.tracked_beats == 77
    assert grid.beat_at(beats[79]) == pytest.approx(79.0, abs=0.01)


def test_beat_and_seconds_are_inverse_and_extrapolate():
    grid = build_beat_grid(steady_beats(100.0, 40, first=0.2), meter=4, end_seconds=30.0)
    for beat in (-0.5, 0.0, 3.25, 17.5, 39.0, 60.0):
        assert grid.beat_at(grid.seconds_at(beat)) == pytest.approx(beat, abs=1e-9)
    assert grid.seconds_at(0.0) == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize(
    "tracked",
    [
        None,
        "0 1 2 3 4 5 6 7 8",
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, True],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float("nan")],
        [0.0, 0.5, 1.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0],
        [-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        [index * 0.05 for index in range(40)],
        [index * 6.0 for index in range(12)],
    ],
)
def test_unusable_beat_tracks_give_no_grid(tracked):
    assert build_beat_grid(tracked, meter=4, end_seconds=20.0) is None


@pytest.mark.parametrize("meter", [0, 13, 4.0, True, None])
def test_invalid_meter_gives_no_grid(meter):
    assert build_beat_grid(steady_beats(), meter=meter, end_seconds=30.0) is None


def test_grid_rejects_times_that_do_not_increase():
    with pytest.raises(ValueError):
        BeatGrid(
            [0.0, 0.5, 0.5],
            meter=4,
            mode="steady",
            tempo_bpm=120.0,
            bar_phase_source="none",
            bar_phase_confidence=None,
            onset_shift_seconds=0.0,
            tracked_beats=3,
        )


# ---------------------------------------------------------------------------
# Settling on the played beat


def test_onsets_pull_a_late_beat_tracker_onto_the_played_beat():
    played = steady_beats(120.0, 120, first=0.50)
    tracked = [value + 0.03 for value in played]  # tracker reports 30 ms late
    grid = build_beat_grid(tracked, meter=4, end_seconds=played[-1], onsets=played)

    assert grid.onset_shift_seconds == pytest.approx(-0.03, abs=0.002)
    assert grid.beat_at(played[40]) % 1.0 == pytest.approx(0.0, abs=0.005) or grid.beat_at(played[40]) % 1.0 == pytest.approx(1.0, abs=0.005)


def test_too_few_onsets_leave_the_grid_alone():
    tracked = steady_beats(120.0, 64, first=0.5)
    grid = build_beat_grid(tracked, meter=4, end_seconds=32.0, onsets=[0.48, 0.98, 1.48])
    assert grid.onset_shift_seconds == 0.0


def test_swung_offbeats_do_not_shift_the_grid():
    tracked = steady_beats(120.0, 96, first=0.5)
    onsets = tracked + [value + 0.5 * (2 / 3) for value in tracked]  # swung off-beats
    grid = build_beat_grid(tracked, meter=4, end_seconds=48.0, onsets=onsets)
    assert abs(grid.onset_shift_seconds) < 0.002


# ---------------------------------------------------------------------------
# Finding beat 1


@pytest.mark.parametrize("downbeat_index", [0, 1, 2, 3])
def test_backbeat_finds_the_first_beat_of_the_bar(downbeat_index):
    beats = steady_beats(120.0, 96, first=0.1)
    grid = build_beat_grid(
        beats, meter=4, end_seconds=48.0, drum_hits=backbeat(beats, downbeat_index=downbeat_index)
    )

    assert grid.bar_phase_source == "drums"
    assert grid.bar_phase_confidence is not None and grid.bar_phase_confidence >= 0.15
    # Every kick on beat 1 falls on a bar line.
    for index in range(downbeat_index, 96, 4):
        assert _distance_to_bar_start(grid.beat_at(beats[index])) == pytest.approx(0.0, abs=1e-6)


def test_a_pickup_gets_its_own_first_bar():
    # Three beats of pickup before the first downbeat at 1.6 s.
    beats = steady_beats(120.0, 96, first=0.1)
    grid = build_beat_grid(beats, meter=4, end_seconds=48.0, drum_hits=backbeat(beats, downbeat_index=3))

    assert grid.times[0] < 0.0  # bar 1 begins before the recording
    assert grid.seconds_at(4.0) == pytest.approx(1.6, abs=1e-6)
    document = build_score_document(
        [note("pickup", 1.1, 1.6), note("downbeat", 1.6, 2.1)],
        tempo_bpm=120.0,
        beats_per_measure=4,
        beat_grid=grid,
    )
    assert document["measures"][0]["startSeconds"] == 0.0
    assert document["measures"][0]["endSeconds"] == pytest.approx(1.6, abs=1e-6)
    assert document["measures"][0]["notes"][0]["quantizedBeat"] == 3.0
    assert document["measures"][1]["notes"][0]["quantizedBeat"] == 4.0


def test_four_on_the_floor_without_a_backbeat_does_not_guess():
    beats = steady_beats(124.0, 96, first=0.1)
    grid = build_beat_grid(beats, meter=4, end_seconds=46.0, drum_hits=[(value, "low") for value in beats])
    assert grid.bar_phase_source == "none"
    assert grid.bar_phase_confidence is None


def test_drum_pattern_is_only_trusted_in_four_four():
    beats = steady_beats(120.0, 96, first=0.1)
    grid = build_beat_grid(beats, meter=3, end_seconds=48.0, drum_hits=backbeat(beats))
    assert grid.bar_phase_source == "none"


def test_confident_analysis_downbeats_are_used_when_the_drums_are_silent():
    beats = steady_beats(120.0, 64, first=0.1)
    downbeats = beats[2::4]
    confident = build_beat_grid(
        beats, meter=4, end_seconds=32.0, analysis_downbeats=downbeats, analysis_meter_confidence=0.7
    )
    unsure = build_beat_grid(
        beats, meter=4, end_seconds=32.0, analysis_downbeats=downbeats, analysis_meter_confidence=0.27
    )

    assert confident.bar_phase_source == "analysis"
    assert _distance_to_bar_start(confident.beat_at(beats[2])) == pytest.approx(0.0, abs=1e-6)
    assert unsure.bar_phase_source == "none"


def test_drums_outrank_analysis_downbeats():
    beats = steady_beats(120.0, 64, first=0.1)
    grid = build_beat_grid(
        beats,
        meter=4,
        end_seconds=32.0,
        drum_hits=backbeat(beats, downbeat_index=0),
        analysis_downbeats=beats[3::4],
        analysis_meter_confidence=0.9,
    )
    assert grid.bar_phase_source == "drums"
    assert _distance_to_bar_start(grid.beat_at(beats[0])) == pytest.approx(0.0, abs=1e-6)


def test_grid_covers_the_whole_recording_in_whole_bars():
    beats = steady_beats(120.0, 40, first=0.1)
    grid = build_beat_grid(beats, meter=4, end_seconds=60.0)
    assert grid.times[-1] > 60.0
    assert (len(grid.times) - 1) % 4 == 0


def test_summary_is_serializable_and_versioned():
    beats = steady_beats(120.0, 64, first=0.1)
    summary = build_beat_grid(beats, meter=4, end_seconds=32.0, drum_hits=backbeat(beats)).summary()
    assert summary["version"] == BEAT_GRID_VERSION
    assert set(summary) == {
        "version",
        "mode",
        "barPhaseSource",
        "barPhaseConfidence",
        "onsetShiftSeconds",
        "trackedBeats",
        "firstBeatSeconds",
    }
    assert summary["mode"] == "steady" and summary["barPhaseSource"] == "drums"


# ---------------------------------------------------------------------------
# Score construction on a grid


def test_bars_follow_real_beat_times_and_raw_times_are_untouched():
    beats = [0.0]
    for index in range(1, 80):
        beats.append(beats[-1] + (0.5 if index < 40 else 0.6))
    grid = build_beat_grid(beats, meter=4, end_seconds=beats[-1])
    events = [note("a", beats[8], beats[9]), note("b", beats[60], beats[61])]
    document = build_score_document(events, tempo_bpm=120.0, beats_per_measure=4, beat_grid=grid)
    measures = document["measures"]

    assert measures[2]["notes"][0]["startSeconds"] == beats[8]
    assert measures[2]["notes"][0]["quantizedBeat"] == 8.0
    assert measures[15]["notes"][0]["quantizedBeat"] == 60.0
    # Early bars are shorter than late ones because the tempo slows down.
    early = measures[2]["endSeconds"] - measures[2]["startSeconds"]
    late = measures[15]["endSeconds"] - measures[15]["startSeconds"]
    assert late > early * 1.1
    for previous, current in zip(measures, measures[1:]):
        assert current["startSeconds"] == previous["endSeconds"]
    assert document["beatGrid"]["mode"] == "following"


def test_sound_just_before_bar_one_lands_on_its_first_beat():
    grid = build_beat_grid(steady_beats(120.0, 64, first=0.12), meter=4, end_seconds=32.0)
    document = build_score_document(
        [note("early", 0.02, 0.4)], tempo_bpm=120.0, beats_per_measure=4, beat_grid=grid
    )
    placed = document["measures"][0]["notes"][0]
    assert placed["quantizedBeat"] == 0.0
    assert placed["startBeat"] < 0.0
    assert placed["startSeconds"] == 0.02


def test_percussion_uses_the_same_grid_as_pitched_notes():
    beats = steady_beats(160.0, 200, first=0.11)
    grid = build_beat_grid(beats, meter=4, end_seconds=beats[-1])
    hit = {
        "eventId": "r000001",
        "hitIndex": 0,
        "sourceKind": "drums",
        "rawKind": "snare",
        "broadVoice": "mid_drum",
        "resolved": True,
        "timeSeconds": beats[161],
        "strength": 0.8,
        "confidence": 0.9,
        "interpretationPlacement": None,
    }
    document = build_score_document(
        [note("n", beats[161], beats[162])],
        tempo_bpm=161.499,
        beats_per_measure=4,
        percussion_hits=[hit],
        beat_grid=grid,
    )
    measure = document["measures"][40]
    assert measure["notes"][0]["quantizedBeat"] == 161.0
    assert measure["percussionHits"][0]["quantizedBeat"] == 161.0
    assert measure["percussionHits"][0]["quantizationShiftSeconds"] < 0.001


def test_without_a_grid_the_builder_is_unchanged():
    events = [note("a", 0.0, 0.5), note("b", 2.0, 2.5)]
    document = build_score_document(events, tempo_bpm=120.0, beats_per_measure=4)
    assert "beatGrid" not in document
    assert [m["startSeconds"] for m in document["measures"]] == [0.0, 2.0]
    assert document["measures"][1]["notes"][0]["quantizedBeat"] == 4.0


def test_random_jitter_never_breaks_bar_order():
    randomizer = random.Random(7)
    beats = [0.2]
    for _ in range(300):
        beats.append(beats[-1] + 0.5 + randomizer.uniform(-0.06, 0.06))
    grid = build_beat_grid(beats, meter=4, end_seconds=beats[-1])
    assert grid is not None
    assert all(b > a for a, b in zip(grid.times, grid.times[1:]))
    events = [note(f"n{index}", beats[index], beats[index] + 0.2) for index in range(0, 300, 7)]
    document = build_score_document(events, tempo_bpm=120.0, beats_per_measure=4, beat_grid=grid)
    for measure in document["measures"]:
        assert measure["endSeconds"] > measure["startSeconds"] >= 0.0
        assert math.isfinite(measure["endSeconds"])


def test_a_grid_that_cannot_give_bars_a_duration_is_refused():
    class Collapsed:
        tempo_bpm = 120.0

        def beat_at(self, seconds: float) -> float:
            return seconds * 2.0

        def seconds_at(self, beat: float) -> float:
            return 0.0

        def summary(self) -> dict:
            return {}

    with pytest.raises(ScoreConstructionError):
        build_score_document(
            [note("a", 0.0, 0.5)], tempo_bpm=120.0, beats_per_measure=4, beat_grid=Collapsed()
        )


# ---------------------------------------------------------------------------
# Saved scores built through the pipeline


def _set_tracked_beats(settings, job_id: str, beats: list[float], **timing) -> None:
    import json

    path = settings.exports_dir / job_id / "analysis" / "audio-analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    analysis["timing"]["beatsSeconds"] = beats
    analysis["timing"].update(timing)
    path.write_text(json.dumps(analysis), encoding="utf-8")


def _build_saved_score(tmp_path, beats: list[float] | None, **timing) -> dict:
    from fastapi.testclient import TestClient

    from app.main import create_app
    from test_harmony_api import create_job, make_settings

    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    if beats is not None:
        _set_tracked_beats(settings, job_id, beats, **timing)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        return client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()


def test_saved_score_records_the_grid_and_follows_tracked_beats(tmp_path):
    # The analysis reports 120 BPM but the tracked beats run at 100 BPM.
    details = _build_saved_score(tmp_path, steady_beats(100.0, 32, first=0.25))

    assert details["version"] == "score-pipeline-v9"
    grid = details["timing"]["beatGrid"]
    assert grid["version"] == BEAT_GRID_VERSION
    assert grid["mode"] == "steady"
    assert grid["barPhaseSource"] == "none" and grid["barPhaseConfidence"] is None
    assert grid["trackedBeats"] == 32
    assert details["timing"]["tempoBpm"] == pytest.approx(100.0, abs=0.01)
    first = details["measures"][0]
    assert first["startSeconds"] == pytest.approx(0.25, abs=1e-6)
    assert first["endSeconds"] - first["startSeconds"] == pytest.approx(2.4, abs=1e-6)
    assert any("first beat of the bar could not be confirmed" in text for text in details["warnings"])
    assert not any("one constant tempo" in text for text in details["warnings"])


def test_saved_score_without_enough_beats_says_it_uses_one_tempo(tmp_path):
    details = _build_saved_score(tmp_path, None)

    assert details["timing"]["beatGrid"] is None
    assert details["timing"]["tempoBpm"] == 120.0
    assert details["measures"][0]["startSeconds"] == 0.0
    assert any("one constant tempo" in text for text in details["warnings"])


def test_scores_built_on_one_tempo_are_flagged_for_rebuild():
    from app.score_pipeline import score_outdated_reason

    old = {"score_version": "score-pipeline-v5", "score_note_count": 3, "score_measure_count": 12}
    assert score_outdated_reason(old) == "beat-grid"
    # A one-bar score has no beat-grid drift; it is only flagged for the
    # per-line rhythm grids its notes predate.
    assert score_outdated_reason({**old, "score_measure_count": 1}) == "rhythm-grid"
    # Scores from before song sections are rebuilt once they are long enough to have them.
    assert score_outdated_reason({**old, "score_version": "score-pipeline-v6"}) == "rhythm-grid"
    assert score_outdated_reason({**old, "score_version": "score-pipeline-v6", "score_measure_count": 40}) == "sections"
    assert score_outdated_reason({**old, "score_version": "score-pipeline-v7", "score_measure_count": 40}) == "rhythm-grid"
    assert score_outdated_reason({**old, "score_version": "score-pipeline-v9", "score_measure_count": 40}) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda grid: grid.update(mode="wobbly"),
        lambda grid: grid.update(barPhaseSource="guess"),
        lambda grid: grid.update(barPhaseConfidence=0.5),
        lambda grid: grid.update(barPhaseConfidence=2.0, barPhaseSource="drums"),
        lambda grid: grid.update(trackedBeats=0),
        lambda grid: grid.update(onsetShiftSeconds="0"),
        lambda grid: grid.update(version="beat grid"),
        lambda grid: grid.pop("firstBeatSeconds"),
        lambda grid: grid.update(extra=1),
    ],
)
def test_tampered_grid_summaries_are_rejected(tmp_path, mutate):
    from app.score_artifacts import ScoreArtifactValidationError, validate_score_artifact
    from app.score_pipeline import construct_score
    from app.score_sources import current_score_fingerprint
    from app import db, score_pipeline
    from test_harmony_api import create_job, make_settings

    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    _set_tracked_beats(settings, job_id, steady_beats(100.0, 32, first=0.25))
    record = db.get_job(settings.database_path, job_id)
    result = construct_score(
        job_id,
        settings,
        record,
        lambda *args: None,
        attempt_id="a" * 32,
        expected_fingerprint=current_score_fingerprint(record),
    )
    document = result.payload
    assert validate_score_artifact(document)["schemaVersion"] == 9
    mutate(document["timing"]["beatGrid"])
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(document)
    assert score_pipeline.SCORE_PIPELINE_VERSION == "score-pipeline-v9"


def test_schema_five_scores_cannot_carry_a_grid_and_schema_six_must(tmp_path):
    from app.score_artifacts import ScoreArtifactValidationError, validate_score_artifact
    from app.score_pipeline import construct_score
    from app.score_sources import current_score_fingerprint
    from app import db
    from test_harmony_api import create_job, make_settings

    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    record = db.get_job(settings.database_path, job_id)
    document = construct_score(
        job_id,
        settings,
        record,
        lambda *args: None,
        attempt_id="b" * 32,
        expected_fingerprint=current_score_fingerprint(record),
    ).payload
    older = {key: value for key, value in document.items() if key not in {"sections", "rhythmGrids"}}
    older["schemaVersion"] = 5
    older["measures"] = [{k: v for k, v in m.items() if k != "chordChanges"} for m in document["measures"]]
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(older)
    older["timing"] = {key: value for key, value in document["timing"].items() if key != "beatGrid"}
    assert validate_score_artifact(older)["schemaVersion"] == 5
    missing = {**document, "timing": older["timing"]}
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(missing)


def test_browser_playhead_follows_saved_bar_lines():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "function rollBeatAt(frame,time,secondsPerBeat)" in script
    assert 'data-bar-lines="' in script and 'data-beats-per-bar="' in script
    assert '"beat-grid"' in script
    assert "Rebuild it so bars follow the tracked beats." in script