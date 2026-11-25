from __future__ import annotations

import json

import numpy as np
import pytest

from app.song_sections import (
    SECTIONS_VERSION,
    SectionError,
    bar_features,
    describe_heard,
    find_sections,
    public_sections,
    validate_sections,
)

FULL = [-18.0, -16.0, -17.0, -24.0]
NO_BASS = [-19.0, -18.0, -80.0, -28.0]
DRUMS_ONLY = [-90.0, -17.0, -90.0, -70.0]


def _song(plan, seed=7):
    """Synthetic bar features: each letter is one sound, each bar a little noisy."""
    rng = np.random.default_rng(seed)
    looks = {letter: rng.normal(0.0, 1.0, 25) for letter in "ABCDE"}
    features, levels = [], []
    for letter, bars, stems in plan:
        for _ in range(bars):
            features.append(looks[letter] + rng.normal(0.0, 0.05, 25))
            levels.append(np.array(stems) + rng.normal(0.0, 0.3, 4))
    return np.array(features), np.array(levels)


def test_sections_follow_changes_and_letter_repeats() -> None:
    features, levels = _song(
        [("A", 4, DRUMS_ONLY), ("B", 8, NO_BASS), ("C", 8, FULL), ("B", 8, NO_BASS), ("C", 8, FULL)]
    )
    result = find_sections(features, levels)
    assert result["version"] == SECTIONS_VERSION and result["note"] is None
    spans = [(s["startMeasure"], s["endMeasure"], s["label"]) for s in result["sections"]]
    assert spans == [(0, 4, "A"), (4, 12, "B"), (12, 20, "C"), (20, 28, "B"), (28, 36, "C")]
    heard = [describe_heard(s["heard"]) for s in result["sections"]]
    assert heard == ["drums only", "vocals, drums, accompaniment", "full band",
                     "vocals, drums, accompaniment", "full band"]
    assert result["sections"][0]["boundaryStrength"] == 1.0
    assert validate_sections(result, 36) == result


def test_sections_are_at_least_four_bars_and_cover_every_bar() -> None:
    features, levels = _song([("A", 10, FULL), ("B", 2, NO_BASS), ("A", 10, FULL)])
    sections = find_sections(features, levels)["sections"]
    assert all(s["endMeasure"] - s["startMeasure"] >= 4 for s in sections)
    assert sections[0]["startMeasure"] == 0 and sections[-1]["endMeasure"] == 22


def test_without_stems_sections_say_nothing_about_instruments() -> None:
    features, _ = _song([("A", 8, FULL), ("B", 8, FULL), ("A", 8, FULL)])
    result = find_sections(features, None)
    assert [s["label"] for s in result["sections"]] == ["A", "B", "A"]
    assert all(s["heard"] is None for s in result["sections"])
    assert describe_heard(None) is None and describe_heard([]) == "quiet"


def test_short_or_unchanging_recordings_get_a_note_not_guesses() -> None:
    features, levels = _song([("A", 12, FULL)])
    short = find_sections(features, levels)
    assert short["sections"] == [] and "16 bars" in short["note"]
    flat = find_sections(np.ones((32, 25)), None)
    assert flat["sections"] == [] and "does not change" in flat["note"]
    validate_sections(short, 12)
    validate_sections(flat, 32)


def test_bar_features_measure_stems_on_the_bar_grid() -> None:
    rate = 22050
    seconds = np.arange(rate * 8) / rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * seconds)
    silent = np.zeros_like(tone)
    drums = np.where(seconds < 4, 0.3 * np.sin(2 * np.pi * 60 * seconds), 0.0)
    spans = [(-0.5, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    features, levels = bar_features(
        spans, mix=tone + drums, stems={"vocals": tone, "drums": drums, "bass": silent}
    )
    assert features.shape == (4, 25) and levels.shape == (4, 4)
    assert (levels[:2, 1] > -20).all() and (levels[2:, 1] < -100).all()
    assert (levels[:, 2] < -100).all() and (levels[:, 3] == -120.0).all()
    assert bar_features(spans, mix=tone)[1] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(version="other"),
        lambda v: v["sections"][1].update(startMeasure=5),
        lambda v: v["sections"][0].update(label="a"),
        lambda v: v["sections"][0].update(heard=["guitar"]),
        lambda v: v["sections"][0].update(boundaryStrength=2),
        lambda v: v["sections"][-1].update(endMeasure=30),
        lambda v: v.update(note="also a note"),
        lambda v: v["sections"][0].update(extra=1),
    ],
)
def test_saved_sections_fail_closed(change) -> None:
    features, levels = _song([("A", 8, FULL), ("B", 8, NO_BASS), ("A", 8, FULL)])
    value = find_sections(features, levels)
    change(value)
    with pytest.raises(SectionError):
        validate_sections(value, 24)


def _built_document(tmp_path):
    from app import db
    from app.score_pipeline import construct_score
    from app.score_sources import current_score_fingerprint
    from test_harmony_api import create_job, make_settings

    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    record = db.get_job(settings.database_path, job_id)
    return construct_score(
        job_id,
        settings,
        record,
        lambda *args: None,
        attempt_id="c" * 32,
        expected_fingerprint=current_score_fingerprint(record),
    ).payload


def test_built_scores_carry_sections_or_say_why_not(tmp_path) -> None:
    from app.score_artifacts import validate_score_artifact

    document = _built_document(tmp_path)
    assert document["schemaVersion"] == 8
    sections = document["sections"]
    assert sections["version"] == SECTIONS_VERSION
    # The fixture is a few bars long, so it carries a note instead of guesses.
    assert sections["sections"] == [] and "16 bars" in sections["note"]
    validate_score_artifact(document)


def test_schema_six_scores_without_sections_stay_readable(tmp_path) -> None:
    from app.score_artifacts import ScoreArtifactValidationError, validate_score_artifact
    from app.score_construction import score_to_musicxml_text
    from app.score_pipeline import score_export_document

    document = _built_document(tmp_path)
    older = {key: value for key, value in document.items() if key not in {"sections", "rhythmGrids"}}
    older["schemaVersion"] = 6
    assert validate_score_artifact(older)["schemaVersion"] == 6
    assert "<rehearsal" not in score_to_musicxml_text(score_export_document(older))
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact({**older, "schemaVersion": 7})
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact({**document, "sections": {**document["sections"], "version": "x"}})


def test_musicxml_marks_each_section_start_once_with_its_letter() -> None:
    import xml.etree.ElementTree as ET

    from app.score_construction import build_score_document, score_to_musicxml_text

    events = [
        {"id": f"n{bar}", "startSeconds": bar * 2.0, "endSeconds": bar * 2.0 + 0.5, "midiNote": 60 + bar % 5,
         "confidence": 0.9, "sourceKind": "vocals" if bar % 2 else "other"}
        for bar in range(20)
    ]
    document = build_score_document(events, tempo_bpm=120.0)
    document["sections"] = {
        "version": SECTIONS_VERSION,
        "note": None,
        "sections": [
            {"startMeasure": 0, "endMeasure": 8, "label": "A", "boundaryStrength": 1.0, "heard": None},
            {"startMeasure": 8, "endMeasure": 16, "label": "B", "boundaryStrength": 0.6, "heard": None},
            {"startMeasure": 16, "endMeasure": 20, "label": "A", "boundaryStrength": 0.4, "heard": None},
        ],
    }
    root = ET.fromstring(score_to_musicxml_text(document))
    first, *others = root.findall("part")
    marks = [
        (int(measure.get("number")), mark.text, mark.get("enclosure"))
        for measure in first.findall("measure")
        for mark in measure.iter("rehearsal")
    ]
    assert marks == [(1, "A", "square"), (9, "B", "square"), (17, "A", "square")]
    assert all(part.find(".//rehearsal") is None for part in others)
    broken = {**document, "sections": {**document["sections"], "sections": document["sections"]["sections"][:2]}}
    from app.score_construction import ScoreConstructionError

    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(broken)


def test_sections_are_measured_from_the_recording_on_the_bar_grid(tmp_path) -> None:
    import soundfile as sf

    from app.score_pipeline import find_score_sections
    from test_harmony_api import make_settings

    settings = make_settings(tmp_path)
    job_id = "e" * 32
    rate = 22050
    seconds = np.arange(int(rate * 64)) / rate
    # 16 bars of a soft high tone, 16 bars of a loud low buzz, then the tone again.
    soft = 0.05 * np.sin(2 * np.pi * 880 * seconds)
    loud = 0.5 * np.sign(np.sin(2 * np.pi * 110 * seconds))
    audio = np.where((seconds >= 21.33) & (seconds < 42.67), loud, soft).astype("float32")
    folder = settings.exports_dir / job_id
    folder.mkdir(parents=True)
    sf.write(folder / "analysis.wav", audio, rate)
    bar = 64 / 48
    measures = [{"startSeconds": i * bar, "endSeconds": (i + 1) * bar} for i in range(48)]
    result = find_score_sections(job_id, settings, {"separation_status": "failed"}, measures)
    validate_sections(result, 48)
    spans = [(s["startMeasure"], s["endMeasure"], s["label"]) for s in result["sections"]]
    assert spans == [(0, 16, "A"), (16, 32, "B"), (32, 48, "A")]
    missing = find_score_sections("f" * 32, settings, {}, measures)
    assert missing["sections"] == [] and "unavailable" in missing["note"]
    short = find_score_sections(job_id, settings, {}, measures[:8])
    assert short["sections"] == [] and "16 bars" in short["note"]


def _document_with_sections(tmp_path):
    document = _built_document(tmp_path)
    # The fixture score is one bar long: one section is enough to name.
    document["sections"] = {
        "version": SECTIONS_VERSION,
        "note": None,
        "sections": [
            {"startMeasure": 0, "endMeasure": len(document["measures"]), "label": "A",
             "boundaryStrength": 1.0, "heard": None},
        ],
    }
    return document


@pytest.mark.parametrize(
    ("name", "error"),
    [("x" * 41, "at most 40"), ("Verse<1>", "at most 40"), ("  Chorus", "extra spaces"), (7, "must be text")],
)
def test_section_names_are_short_single_lines(name, error) -> None:
    from app.score_corrections import CorrectionError, validate_operation

    with pytest.raises(CorrectionError, match=error):
        validate_operation({"op": "set_section_name", "target": {"startMeasure": 0}, "name": name})
    assert validate_operation({"op": "set_section_name", "target": {"startMeasure": 3}, "name": "Pre-chorus 2"}) == {
        "op": "set_section_name", "target": {"startMeasure": 3}, "name": "Pre-chorus 2",
    }
    assert validate_operation({"op": "set_section_name", "target": {"startMeasure": 3}, "name": None})["name"] is None


def test_section_names_are_corrections_beside_the_detected_letters(tmp_path) -> None:
    from app.score_artifacts import validate_score_artifact
    from app.score_construction import score_to_musicxml_text
    from app.score_corrections import apply_corrections, new_operation
    from app.score_pipeline import score_export_document

    document = _document_with_sections(tmp_path)
    original = json.loads(json.dumps(document))
    ops = [
        new_operation({"op": "set_section_name", "target": {"startMeasure": 0}, "name": "Chorus"}),
        new_operation({"op": "set_section_name", "target": {"startMeasure": 5}, "name": "Nowhere"}),
    ]
    corrected, report = apply_corrections(document, ops)
    assert document == original  # the prediction is untouched
    items = corrected["sections"]["sections"]
    assert items[0]["name"] == "Chorus" and items[0]["label"] == "A"
    assert report["sections"] == {"0": None}
    assert report["notApplicable"][0]["reason"] == "No section starts at that bar in this score."
    validate_score_artifact(corrected)
    public = public_sections(corrected)
    assert [item["name"] for item in public["sections"]] == ["Chorus"]
    import xml.etree.ElementTree as ET

    marks = [mark.text for mark in ET.fromstring(score_to_musicxml_text(score_export_document(corrected))).iter("rehearsal")]
    assert marks == ["Chorus"]

    # Clearing the name again returns to the prediction and leaves nothing to review.
    cleared, report = apply_corrections(
        document, [*ops, new_operation({"op": "set_section_name", "target": {"startMeasure": 0}, "name": None})]
    )
    assert "name" not in cleared["sections"]["sections"][0] and report["sections"] == {}


def test_saved_section_names_fail_closed() -> None:
    features, levels = _song([("A", 8, FULL), ("B", 8, NO_BASS), ("A", 8, FULL)])
    value = find_sections(features, levels)
    value["sections"][0]["name"] = "Intro"
    assert validate_sections(value, 24)["sections"][0]["name"] == "Intro"
    for bad in (None, "", " Intro", "x" * 41, "<b>"):
        value["sections"][0]["name"] = bad
        with pytest.raises(SectionError):
            validate_sections(value, 24)
