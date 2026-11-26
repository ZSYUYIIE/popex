"""Evidence-gated sixteenth-note grids per source line (Cycle 37)."""

from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app import db
from app.beat_grid import BeatGrid
from app.main import create_app
from app.score_artifacts import (
    ScoreArtifactValidationError,
    load_score_artifact,
    validate_score_artifact,
)
from app.score_construction import (
    build_score_document,
    choose_rhythm_grids,
    score_to_musicxml_text,
)
from app.score_pipeline import (
    ScorePipelineError,
    _check_notes_on_their_grids,
    rhythm_grid_note,
    score_outdated_reason,
)
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings

SPB = 0.6  # 100 BPM: a sixteenth lasts 150 ms


def line(kind: str, onsets: list[float], *, length: float = 0.12, confidence: float = 0.9) -> list[dict]:
    return [
        {"id": f"{kind}{index}", "sourceKind": kind, "startSeconds": start,
         "endSeconds": start + length, "midiNote": 60 + index % 5, "confidence": confidence}
        for index, start in enumerate(onsets)
    ]


def sixteenth_figure(bars: int = 2, spb: float = SPB) -> list[float]:
    """Four sixteenths on every beat: half of the onsets fall on the "e" and "a"."""
    return [beat * spb + step * spb / 4 + 0.008 for beat in range(4 * bars) for step in range(4)]


def straight_eighths(bars: int = 2) -> list[float]:
    return [half * SPB / 2 + 0.01 for half in range(8 * bars)]


def grid_for(grids: list[dict], kind: str) -> dict:
    return next(item for item in grids if item["sourceKind"] == kind)


def drifting_grid(beats: int = 33) -> BeatGrid:
    """Tracked beats that speed up from 0.66 s to 0.54 s per beat."""
    times, now = [], 0.0
    for index in range(beats):
        times.append(now)
        now += 0.66 - 0.12 * index / (beats - 1)
    return BeatGrid(
        times, meter=4, mode="following", tempo_bpm=100.0, bar_phase_source="drums",
        bar_phase_confidence=0.9, onset_shift_seconds=0.0, tracked_beats=beats,
    )


# ---------------------------------------------------------------------------
# The evidence rule


def test_sixteenth_figures_earn_a_sixteenth_grid_with_smaller_shifts() -> None:
    notes = line("vocals", sixteenth_figure())
    vocals = grid_for(choose_rhythm_grids(notes, seconds_per_beat=SPB), "vocals")
    assert vocals == {"sourceKind": "vocals", "unit": "sixteenth", "unitBeats": 0.25,
                      "confidentOnsets": 32, "offEighthOnsets": 16, "reason": "off-eighth-onsets"}
    document = build_score_document(notes, tempo_bpm=100.0)
    placed = [note for measure in document["measures"] for note in measure["notes"]]
    assert {note["quantizedBeat"] % 0.5 for note in placed} == {0.0, 0.25}
    assert min(note["quantizedDurationBeats"] for note in placed) == 0.25
    assert max(note["quantizationShiftSeconds"] for note in placed) < 0.05
    assert not any("shifted by more than 50ms" in warning for warning in document["warnings"])
    assert document["rhythmGrids"] == [vocals]


def test_straight_eighths_keep_the_eighth_grid() -> None:
    vocals = grid_for(choose_rhythm_grids(line("vocals", straight_eighths()), seconds_per_beat=SPB), "vocals")
    assert (vocals["unit"], vocals["offEighthOnsets"], vocals["reason"]) == ("eighth", 0, "insufficient-evidence")


def test_sparse_low_confidence_or_diluted_evidence_keeps_eighths() -> None:
    sparse = straight_eighths() + [0.15 + 0.005, 1.35 + 0.005]  # only two off-eighth onsets
    assert grid_for(choose_rhythm_grids(line("bass", sparse), seconds_per_beat=SPB), "bass")["unit"] == "eighth"
    unsure = line("bass", sixteenth_figure(), confidence=0.3)
    result = grid_for(choose_rhythm_grids(unsure, seconds_per_beat=SPB), "bass")
    assert (result["unit"], result["confidentOnsets"]) == ("eighth", 0)
    # Off-eighth onsets must also be a meaningful share of the line (15%).
    diluted = straight_eighths(bars=8) + [beat * SPB + SPB / 4 + 0.005 for beat in range(3)]
    result = grid_for(choose_rhythm_grids(line("bass", diluted), seconds_per_beat=SPB), "bass")
    assert (result["unit"], result["offEighthOnsets"]) == ("eighth", 3)


def test_onsets_between_grid_positions_are_not_evidence() -> None:
    # 50 ms after the "e": closer to it than to the beat, but outside the 30 ms tolerance.
    loose = [beat * SPB + SPB / 4 + 0.05 for beat in range(8)]
    result = grid_for(choose_rhythm_grids(line("vocals", loose), seconds_per_beat=SPB), "vocals")
    assert (result["unit"], result["offEighthOnsets"]) == ("eighth", 0)


def test_fast_tempos_keep_eighths_and_say_why() -> None:
    spb = 60.0 / 180.0  # a sixteenth lasts 83 ms
    notes = line("vocals", sixteenth_figure(spb=spb), length=0.05)
    result = grid_for(choose_rhythm_grids(notes, seconds_per_beat=spb), "vocals")
    assert (result["unit"], result["reason"]) == ("eighth", "tempo-too-fast")


def test_each_line_gets_its_own_grid_and_a_plain_description() -> None:
    notes = line("vocals", sixteenth_figure()) + line("bass", straight_eighths())
    document = build_score_document(notes, tempo_bpm=100.0)
    grids = document["rhythmGrids"]
    assert [(item["sourceKind"], item["unit"]) for item in grids] == [("bass", "eighth"), ("vocals", "sixteenth")]
    bass = [note for measure in document["measures"] for note in measure["notes"] if note["sourceKind"] == "bass"]
    assert all(note["quantizedBeat"] % 0.5 == 0 for note in bass)
    assert rhythm_grid_note(grids) == (
        "Quantized from raw pitched events to a sixteenth-note grid for the vocal-stem line, "
        "where the note onsets support it, and an eighth-note grid elsewhere."
    )
    assert rhythm_grid_note(grids[:1]) == "Quantized from raw pitched events to an eighth-note grid."
    assert "which the note onsets support" in rhythm_grid_note(grids[1:])
    assert rhythm_grid_note([]).endswith("no notes were available.")


def test_mixed_grid_warning_names_both_grids() -> None:
    late = [onset + 0.07 for onset in straight_eighths()]  # 70 ms late on the eighth grid
    notes = line("vocals", sixteenth_figure()) + line("bass", late)
    warnings = build_score_document(notes, tempo_bpm=100.0)["warnings"]
    assert any("during quantization to each line's 8th- or 16th-note grid." in item for item in warnings)


# ---------------------------------------------------------------------------
# Measured on the beat-following bar grid


def test_off_eighth_onsets_are_judged_on_the_tracked_beats() -> None:
    grid = drifting_grid()
    # Every "e" and "a" of the tracked beats, played 8 ms late.
    onsets = [grid.seconds_at(step / 4) + 0.008 for step in range(4 * 32) if step % 2 == 1]
    notes = line("vocals", onsets)
    on_grid = grid_for(choose_rhythm_grids(notes, seconds_per_beat=SPB, beat_grid=grid), "vocals")
    assert (on_grid["unit"], on_grid["offEighthOnsets"]) == ("sixteenth", len(onsets))
    # One constant tempo would lose most of these onsets as the beats drift.
    constant = grid_for(choose_rhythm_grids(notes, seconds_per_beat=SPB), "vocals")
    assert constant["offEighthOnsets"] < len(onsets) // 2

    document = build_score_document(notes, tempo_bpm=100.0, beat_grid=grid)
    placed = [note for measure in document["measures"] for note in measure["notes"]]
    # Every onset lands on exactly the "e" or "a" it was played on.
    assert sorted(note["quantizedBeat"] for note in placed) == [
        step / 4 for step in range(4 * 32) if step % 2 == 1
    ]


def test_beat_grid_tempo_decides_whether_sixteenths_are_readable() -> None:
    spb = 60.0 / 180.0
    times = [index * spb for index in range(33)]
    fast = BeatGrid(times, meter=4, mode="steady", tempo_bpm=180.0, bar_phase_source="drums",
                    bar_phase_confidence=0.9, onset_shift_seconds=0.0, tracked_beats=33)
    notes = line("vocals", sixteenth_figure(spb=spb), length=0.05)
    result = grid_for(choose_rhythm_grids(notes, seconds_per_beat=SPB, beat_grid=fast), "vocals")
    assert result["reason"] == "tempo-too-fast"


def test_a_fast_passage_on_a_moving_grid_cannot_earn_sixteenths() -> None:
    # Eight beats at 100 BPM, then twenty-four at 180 BPM; the average stays under 166 BPM.
    times, now = [], 0.0
    for index in range(33):
        times.append(now)
        now += 0.6 if index < 8 else 60.0 / 180.0
    grid = BeatGrid(times, meter=4, mode="following", tempo_bpm=150.0, bar_phase_source="drums",
                    bar_phase_confidence=0.9, onset_shift_seconds=0.0, tracked_beats=33)
    fast_offbeats = [grid.seconds_at(step / 4) for step in range(32, 128) if step % 2 == 1]
    result = grid_for(
        choose_rhythm_grids(line("vocals", fast_offbeats, length=0.05), seconds_per_beat=0.4,
                            beat_grid=grid),
        "vocals",
    )
    assert (result["unit"], result["offEighthOnsets"], result["reason"]) == ("eighth", 0, "tempo-too-fast")
    slow_offbeats = [grid.seconds_at(step / 4) for step in range(0, 32) if step % 2 == 1]
    slow = grid_for(
        choose_rhythm_grids(line("vocals", slow_offbeats), seconds_per_beat=0.4, beat_grid=grid),
        "vocals",
    )
    assert (slow["unit"], slow["offEighthOnsets"]) == ("sixteenth", 16)


# ---------------------------------------------------------------------------
# Exports


def test_sixteenth_musicxml_writes_sixteenth_durations() -> None:
    document = build_score_document(line("vocals", sixteenth_figure(bars=1)), tempo_bpm=100.0)
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    sounding = [note for note in root.iter("note") if note.find("rest") is None]
    assert sounding and {note.findtext("duration") for note in sounding} == {"120"}
    xsd = os.environ.get("POPEX_MUSICXML_XSD")
    if xsd:
        from lxml import etree

        schema = etree.XMLSchema(etree.parse(xsd))
        assert schema.validate(etree.fromstring(text.encode("utf-8"))), schema.error_log


# ---------------------------------------------------------------------------
# Saved scores


def _saved_document(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
    record = db.get_job(settings.database_path, job_id)
    return settings, job_id, load_score_artifact(
        job_id, settings, artifact_file_name=record["score_artifact_file_name"]
    )


def test_saved_scores_record_one_grid_per_line(tmp_path) -> None:
    _settings, _job_id, document = _saved_document(tmp_path)
    assert document["schemaVersion"] == 10
    assert document["pipelineVersion"] == "score-pipeline-v11"
    kinds = {note["sourceKind"] for measure in document["measures"] for note in measure["notes"]}
    assert [item["sourceKind"] for item in document["rhythmGrids"]] == sorted(kinds)
    assert document["layers"]["pitchedNotes"]["note"] == rhythm_grid_note(document["rhythmGrids"])


@pytest.mark.parametrize(
    "tamper",
    [
        # A note between sixteenth positions.
        lambda doc: next(n for m in doc["measures"] for n in m["notes"]).update(
            quantizedEndBeat=next(n for m in doc["measures"] for n in m["notes"])["quantizedEndBeat"] + 0.125
        ),
        lambda doc: doc["rhythmGrids"][0].update(unitBeats=0.25),
        lambda doc: doc["rhythmGrids"][0].update(unit="sixteenth", unitBeats=0.25),
        lambda doc: doc["rhythmGrids"][0].update(
            unit="sixteenth", unitBeats=0.25, reason="off-eighth-onsets", offEighthOnsets=0
        ),
        lambda doc: doc["rhythmGrids"][0].update(offEighthOnsets=doc["rhythmGrids"][0]["confidentOnsets"] + 1),
        # Sixteenths claimed with fewer onsets than the rule needs (3 and 15%).
        lambda doc: doc["rhythmGrids"][0].update(
            unit="sixteenth", unitBeats=0.25, reason="off-eighth-onsets",
            confidentOnsets=100, offEighthOnsets=2,
        ),
        lambda doc: doc["rhythmGrids"][0].update(
            unit="sixteenth", unitBeats=0.25, reason="off-eighth-onsets",
            confidentOnsets=100, offEighthOnsets=10,
        ),
        # Malformed note timing must fail validation, not crash it.
        lambda doc: next(n for m in doc["measures"] for n in m["notes"]).pop("quantizedEndBeat"),
        lambda doc: next(n for m in doc["measures"] for n in m["notes"]).update(quantizedBeat="1.5"),
        lambda doc: doc["rhythmGrids"].append(copy.deepcopy(doc["rhythmGrids"][0])),
        lambda doc: doc.update(rhythmGrids=[]),
        lambda doc: doc.pop("rhythmGrids"),
    ],
)
def test_tampered_rhythm_grids_are_rejected(tmp_path, tamper) -> None:
    _settings, _job_id, document = _saved_document(tmp_path)
    tampered = copy.deepcopy(document)
    tamper(tampered)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(tampered)


def test_corrections_may_move_notes_off_their_predicted_grid(tmp_path) -> None:
    _settings, _job_id, document = _saved_document(tmp_path)
    corrected = copy.deepcopy(document)
    note = next(n for m in corrected["measures"] for n in m["notes"])
    assert grid_for(corrected["rhythmGrids"], note["sourceKind"])["unit"] == "eighth"
    # A musician moves an eighth-grid note onto a sixteenth position.
    note["quantizedBeat"] += 0.25
    note["quantizedEndBeat"] += 0.25
    validate_score_artifact(corrected)


def test_deleting_every_note_of_a_line_keeps_a_valid_corrected_score(tmp_path) -> None:
    settings, job_id, document = _saved_document(tmp_path)
    ids = [note["id"] for measure in document["measures"] for note in measure["notes"]]
    with TestClient(create_app(settings)) as client:
        for revision, note_id in enumerate(ids):
            response = client.post(
                f"/api/jobs/{job_id}/score/corrections",
                json={"expectedRevision": revision,
                      "operation": {"op": "delete_note", "target": {"noteId": note_id}}},
            )
            assert response.status_code == 200, response.text
        corrected = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert not [n for m in corrected["measures"] for n in m["notes"]]
    assert musicxml.status_code == 200


def test_pipeline_refuses_a_prediction_off_its_own_grid() -> None:
    grids = [{"sourceKind": "vocals", "unit": "eighth", "unitBeats": 0.5}]
    good = [{"notes": [{"sourceKind": "vocals", "quantizedBeat": 1.5, "quantizedEndBeat": 2.0}]}]
    _check_notes_on_their_grids(good, grids)
    off = [{"notes": [{"sourceKind": "vocals", "quantizedBeat": 1.25, "quantizedEndBeat": 2.0}]}]
    with pytest.raises(ScorePipelineError):
        _check_notes_on_their_grids(off, grids)
    with pytest.raises(ScorePipelineError):
        _check_notes_on_their_grids([{"notes": [{"sourceKind": "bass", "quantizedBeat": 0.0,
                                                  "quantizedEndBeat": 0.5}]}], grids)


def test_schema_seven_scores_without_grids_stay_readable(tmp_path) -> None:
    _settings, _job_id, document = _saved_document(tmp_path)
    older = {key: value for key, value in document.items() if key != "rhythmGrids"}
    older["schemaVersion"] = 7
    older["measures"] = [{k: v for k, v in m.items() if k != "chordChanges"} for m in document["measures"]]
    if isinstance(older.get("tonality"), dict):
        older["tonality"] = {k: v for k, v in older["tonality"].items() if k != "signatureConfidence"}
    validate_score_artifact(older)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact({**older, "rhythmGrids": document["rhythmGrids"]})


def test_scores_from_before_rhythm_grids_are_out_of_date(tmp_path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        record = db.get_job(settings.database_path, job_id)
        document = load_score_artifact(
            job_id, settings, artifact_file_name=record["score_artifact_file_name"]
        )
        db.update_job(settings.database_path, job_id, score_version="score-pipeline-v7")
        summary = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)["score"]
    assert summary["stale"] is True and summary["staleReason"] == "rhythm-grid"
    from app.main import _saved_score_payload

    old_record = {**record, "score_version": "score-pipeline-v7"}
    details = _saved_score_payload(job_id, document, old_record, include_measures=False)
    assert details["stale"] is True
    assert "sixteenth-note rhythms" in details["warnings"][0]
    assert score_outdated_reason({"score_version": "score-pipeline-v7", "score_note_count": 0}) is None
    assert score_outdated_reason({"score_version": "score-pipeline-v11", "score_note_count": 9}) is None


def test_panel_explains_rhythm_grid_and_section_staleness() -> None:
    result = _run_node(
        f"""
const grid=t.renderScore({{id:"g",score:{{...{json.dumps(SUMMARY)}, stale:true, staleReason:"rhythm-grid"}}}});
const sections=t.renderScore({{id:"s",score:{{...{json.dumps(SUMMARY)}, stale:true, staleReason:"sections"}}}});
console.log(JSON.stringify({{grid,sections}}));
"""
    )
    assert "built before sixteenth-note rhythms were supported" in result["grid"]
    assert "every line uses an eighth-note grid" in result["grid"]
    assert "built before song sections were available" in result["sections"]
    assert "has no section marks" in result["sections"]
