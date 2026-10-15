"""Evidence-gated sixteenth-note grids per source line (Cycle 17)."""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.score_artifacts import (
    ScoreArtifactValidationError,
    load_score_artifact,
    validate_score_artifact,
)
from app.score_construction import build_score_document, choose_rhythm_grids, score_to_musicxml_text
from app.score_pipeline import rhythm_grid_note, score_outdated_reason
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings

SPB = 0.6  # 100 BPM: a sixteenth lasts 150 ms


def line(kind: str, onsets: list[float], *, length: float = 0.12, confidence: float = 0.9) -> list[dict]:
    return [
        {"id": f"{kind}{index}", "sourceKind": kind, "startSeconds": start, "endSeconds": start + length,
         "midiNote": 60 + index % 5, "confidence": confidence}
        for index, start in enumerate(onsets)
    ]


def sixteenth_figure(bars: int = 2) -> list[float]:
    """Four sixteenths on every beat: half of the onsets fall on the "e" and "a"."""
    return [beat * SPB + step * SPB / 4 + 0.008 for beat in range(4 * bars) for step in range(4)]


def straight_eighths(bars: int = 2) -> list[float]:
    return [half * SPB / 2 + 0.01 for half in range(8 * bars)]


def grid(grids: list[dict], kind: str) -> dict:
    return next(item for item in grids if item["sourceKind"] == kind)


def test_sixteenth_figures_earn_a_sixteenth_grid_with_smaller_shifts() -> None:
    notes = line("vocals", sixteenth_figure())
    vocals = grid(choose_rhythm_grids(notes, seconds_per_beat=SPB), "vocals")
    assert vocals == {"sourceKind": "vocals", "unit": "sixteenth", "unitBeats": 0.25,
                      "confidentOnsets": 32, "offEighthOnsets": 16, "reason": "off-eighth-onsets"}
    document = build_score_document(notes, tempo_bpm=100.0)
    placed = [note for measure in document["measures"] for note in measure["notes"]]
    assert {note["quantizedBeat"] % 0.5 for note in placed} == {0.0, 0.25}
    assert max(note["quantizationShiftSeconds"] for note in placed) < 0.05
    assert not any("shifted by more than 50ms" in warning for warning in document["warnings"])
    assert document["rhythmGrids"] == [vocals]


def test_straight_eighths_keep_the_eighth_grid() -> None:
    vocals = grid(choose_rhythm_grids(line("vocals", straight_eighths()), seconds_per_beat=SPB), "vocals")
    assert (vocals["unit"], vocals["offEighthOnsets"], vocals["reason"]) == ("eighth", 0, "insufficient-evidence")


def test_sparse_or_low_confidence_evidence_keeps_eighths() -> None:
    sparse = straight_eighths() + [0.15 + 0.005, 1.35 + 0.005]  # only two off-eighth onsets
    assert grid(choose_rhythm_grids(line("bass", sparse), seconds_per_beat=SPB), "bass")["unit"] == "eighth"
    unsure = line("bass", sixteenth_figure(), confidence=0.3)
    result = grid(choose_rhythm_grids(unsure, seconds_per_beat=SPB), "bass")
    assert (result["unit"], result["confidentOnsets"]) == ("eighth", 0)
    # Off-eighth onsets must also be a meaningful share of the line.
    diluted = straight_eighths(bars=8) + [beat * SPB + SPB / 4 + 0.005 for beat in range(3)]
    result = grid(choose_rhythm_grids(line("bass", diluted), seconds_per_beat=SPB), "bass")
    assert (result["unit"], result["offEighthOnsets"]) == ("eighth", 3)


def test_fast_tempos_keep_eighths_and_say_why() -> None:
    spb = 60.0 / 180.0  # a sixteenth lasts 83 ms
    onsets = [beat * spb + step * spb / 4 for beat in range(8) for step in range(4)]
    notes = line("vocals", onsets, length=0.05)
    result = grid(choose_rhythm_grids(notes, seconds_per_beat=spb), "vocals")
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
        "where onsets support it, and an eighth-note grid elsewhere."
    )
    assert rhythm_grid_note(grids[:1]) == "Quantized from raw pitched events to an eighth-note grid."
    assert "which the onsets support" in rhythm_grid_note(grids[1:])


def test_mixed_grid_warning_names_both_grids() -> None:
    late = [onset + 0.07 for onset in straight_eighths()]  # 70 ms late on the eighth grid
    notes = line("vocals", sixteenth_figure()) + line("bass", late)
    warnings = build_score_document(notes, tempo_bpm=100.0)["warnings"]
    assert any("during quantization to each line's 8th- or 16th-note grid." in item for item in warnings)


def test_sixteenth_musicxml_has_quarter_divisions_and_validates() -> None:
    document = build_score_document(line("vocals", sixteenth_figure(bars=1)), tempo_bpm=100.0)
    root = ET.fromstring(score_to_musicxml_text(document))
    assert {note.findtext("duration") for note in root.iter("note") if note.find("rest") is None} == {"120"}
    xsd = os.environ.get("POPEX_MUSICXML_XSD")
    if xsd:
        from lxml import etree

        schema = etree.XMLSchema(etree.parse(xsd))
        assert schema.validate(etree.fromstring(score_to_musicxml_text(document).encode("utf-8")))


def test_saved_scores_record_and_enforce_rhythm_grids(tmp_path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
    record = db.get_job(settings.database_path, job_id)
    document = load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])
    assert document["schemaVersion"] == 6
    kinds = {note["sourceKind"] for measure in document["measures"] for note in measure["notes"]}
    assert [item["sourceKind"] for item in document["rhythmGrids"]] == sorted(kinds)
    assert document["layers"]["pitchedNotes"]["note"].startswith("Quantized from raw pitched events")

    off_grid = copy.deepcopy(document)
    note = next(note for measure in off_grid["measures"] for note in measure["notes"])
    note["quantizedEndBeat"] += 0.125
    note["quantizedDurationBeats"] += 0.125
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(off_grid)
    wrong_unit = copy.deepcopy(document)
    wrong_unit["rhythmGrids"][0]["unitBeats"] = 0.25
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(wrong_unit)
    unexplained = copy.deepcopy(document)
    unexplained["rhythmGrids"][0].update(unit="sixteenth", unitBeats=0.25)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(unexplained)
    missing = copy.deepcopy(document)
    missing["rhythmGrids"] = []
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(missing)


def test_scores_from_before_rhythm_grids_are_out_of_date(tmp_path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        client.post(f"/api/jobs/{job_id}/score/construct")
        db.update_job(settings.database_path, job_id, score_version="score-pipeline-v5")
        summary = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)["score"]
    assert summary["stale"] is True and summary["staleReason"] == "rhythm-grid"
    assert score_outdated_reason({"score_version": "score-pipeline-v5", "score_note_count": 0}) is None
    result = _run_node(
        f"""
const stale=t.renderScore({{id:"s",score:{{...{__import__("json").dumps(SUMMARY)}, stale:true, staleReason:"rhythm-grid"}}}});
console.log(JSON.stringify({{stale}}));
"""
    )
    assert "built before sixteenth-note rhythms were supported" in result["stale"]
