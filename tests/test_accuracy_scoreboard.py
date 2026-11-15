from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "accuracy_scoreboard.py"
_SPEC = importlib.util.spec_from_file_location("popex_accuracy_scoreboard", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
scoreboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scoreboard)

ScoreboardError = scoreboard.ScoreboardError
EXAMPLE_REFERENCE_PATH = _REPO_ROOT / "docs" / "accuracy-reference.example.json"

# The example reference: 120 BPM, 4/4, C major, one chord per bar
# (C, Am, F, G) twice, bass on each chord root at the start of each bar.
EXAMPLE_CHORDS = ["C", "Am", "F", "G", "C", "Am", "F", "G"]
EXAMPLE_BASS = [36, 33, 29, 31, 36, 33, 29, 31]


def example_reference() -> dict:
    return json.loads(EXAMPLE_REFERENCE_PATH.read_text(encoding="utf-8"))


def make_score(
    *,
    tempo: float = 120.0,
    bars: int = 8,
    beats: int = 4,
    chords: list | None = None,
    bass: list | None = None,
    offset: float = 0.0,
    tonality: dict | None = None,
) -> dict:
    """Build a saved-score-shaped document in code. No audio is involved."""
    bar_seconds = beats * 60.0 / tempo
    measures = []
    for index in range(bars):
        start = offset + index * bar_seconds
        notes = []
        if bass is not None and index < len(bass) and bass[index] is not None:
            notes.append(
                {
                    "id": f"p{index:06d}",
                    "midiNote": bass[index],
                    "sourceKind": "bass",
                    "startSeconds": start,
                    "quantizedBeat": float(index * beats),
                    "confidence": 0.9,
                }
            )
        measures.append(
            {
                "measureIndex": index,
                "startSeconds": start,
                "endSeconds": start + bar_seconds,
                "chordSymbol": chords[index] if chords is not None and index < len(chords) else None,
                "harmony": [],
                "notes": notes,
                "percussionHits": [],
            }
        )
    return {
        "artifactType": "popex-score-document",
        "schemaVersion": 5,
        "builderVersion": "score-construction-v5",
        "jobId": "0" * 32,
        "timing": {"tempoBpm": tempo, "beatsPerMeasure": beats},
        "tonality": tonality,
        "measures": measures,
    }


def c_major(confidence: float = 0.8, *, signature: bool = True) -> dict:
    return {
        "primaryCandidate": {
            "tonalCenter": "C",
            "rootPitchClass": 0,
            "collection": "ionian",
            "displayName": "C Ionian (major)",
            "confidence": confidence,
        },
        "keySignature": {"fifths": 0} if signature else None,
    }


def perfect_score() -> dict:
    return make_score(chords=EXAMPLE_CHORDS, bass=EXAMPLE_BASS, tonality=c_major())


# ---------------------------------------------------------------------------
# Whole evaluation


def test_perfect_draft_scores_full_marks():
    result = scoreboard.evaluate(perfect_score(), example_reference(), label="perfect")

    assert result["artifactType"] == "popex-accuracy-result"
    assert result["schemaVersion"] == 1
    assert result["reference"]["id"] == "synthetic-demo-phrase"
    timing = result["timing"]
    assert timing["tempoErrorPercent"] == 0.0
    assert timing["tempoOctaveError"] is None
    assert timing["meterMatches"] is True
    assert timing["bars"] == timing["expectedBars"] == 8
    assert timing["finalGridDriftBeats"] == 0.0
    assert timing["maxGridDriftBeats"] == 0.0
    assert timing["alignedBarsPercent"] == 100.0
    assert result["key"]["score"]["tonalCenterMatches"] is True
    assert result["key"]["score"]["collectionMatches"] is True
    assert result["key"]["score"]["keySignatureWritten"] is True
    assert result["chords"]["shown"]["coveragePercent"] == 100.0
    assert result["chords"]["shown"]["rootAccuracyPercent"] == 100.0
    assert result["chords"]["shown"]["majMinAccuracyPercent"] == 100.0
    assert result["notes"]["bass"]["detected"]["f1Percent"] == 100.0
    assert result["notes"]["bass"]["notated"]["f1Percent"] == 100.0
    assert result["indicators"]["bass"]["onChordRootPercent"] == 100.0


def test_result_is_deterministic_and_strict_json():
    first = scoreboard.evaluate(perfect_score(), example_reference())
    second = scoreboard.evaluate(perfect_score(), example_reference())
    assert first == second
    json.dumps(first, allow_nan=False)


def test_evaluation_does_not_modify_its_inputs():
    score, reference = perfect_score(), example_reference()
    score_before, reference_before = copy.deepcopy(score), copy.deepcopy(reference)
    scoreboard.evaluate(score, reference)
    assert score == score_before
    assert reference == reference_before


def test_accepts_the_saved_score_api_payload_shape():
    payload = perfect_score()
    payload.update({"available": True, "status": "completed", "view": "corrected", "counts": {"measures": 8}})
    result = scoreboard.evaluate(payload, example_reference())
    assert result["score"]["view"] == "corrected"
    assert result["score"]["measures"] == 8


# ---------------------------------------------------------------------------
# Timing


def test_constant_tempo_error_is_reported_as_grid_drift():
    # A draft that uses 121.2 BPM for a 120 BPM recording notates 400 beats in
    # the time 396.04 real beats pass: the grid ends almost four beats ahead.
    reference = example_reference()
    result = scoreboard.evaluate(make_score(tempo=121.2, bars=100), reference)
    timing = result["timing"]

    assert timing["tempoErrorPercent"] == pytest.approx(1.0, abs=0.01)
    assert timing["finalGridDriftBeats"] == pytest.approx(-3.96, abs=0.01)
    assert timing["maxGridDriftBeats"] == pytest.approx(3.96, abs=0.01)
    assert timing["alignedBarsPercent"] < 30.0
    assert timing["bars"] == 100
    assert timing["expectedBars"] == 99


def test_grid_drift_does_not_depend_on_where_the_first_bar_starts():
    reference = example_reference()
    at_zero = scoreboard.evaluate(make_score(tempo=121.2, bars=100), reference)["timing"]
    shifted = scoreboard.evaluate(make_score(tempo=121.2, bars=100, offset=0.37), reference)["timing"]

    assert shifted["finalGridDriftBeats"] == at_zero["finalGridDriftBeats"]
    assert shifted["maxGridDriftBeats"] == at_zero["maxGridDriftBeats"]


def test_right_tempo_with_wrong_bar_phase_keeps_zero_drift_but_no_aligned_bars():
    # Bars start one beat late: the tempo is right, the bar lines are not.
    result = scoreboard.evaluate(make_score(offset=0.5), example_reference())["timing"]
    assert result["maxGridDriftBeats"] == 0.0
    assert result["alignedBarsPercent"] == 0.0
    assert result["medianBarOffsetBeats"] == 1.0


def test_aligned_bars_are_not_reported_without_a_first_downbeat():
    reference = example_reference()
    del reference["timing"]["firstDownbeatSeconds"]
    del reference["chords"]
    result = scoreboard.evaluate(perfect_score(), reference)["timing"]
    assert result["alignedBarsPercent"] is None
    assert result["medianBarOffsetBeats"] is None
    assert result["finalGridDriftBeats"] == 0.0


@pytest.mark.parametrize(("tempo", "expected"), [(60.0, "half"), (240.0, "double"), (120.0, None), (126.0, None)])
def test_half_and_double_tempo_are_flagged(tempo, expected):
    result = scoreboard.evaluate(make_score(tempo=tempo), example_reference())["timing"]
    assert result["tempoOctaveError"] == expected


def test_meter_mismatch_is_reported():
    result = scoreboard.evaluate(make_score(beats=3), example_reference())["timing"]
    assert result["meterMatches"] is False
    assert result["beatsPerMeasure"] == 3
    assert result["referenceBeatsPerMeasure"] == 4


# ---------------------------------------------------------------------------
# Key


def tonality(center: str, collection: str, confidence: float = 0.6, signature: bool = False) -> dict:
    return {
        "primaryCandidate": {"tonalCenter": center, "collection": collection, "displayName": f"{center} {collection}", "confidence": confidence},
        "keySignature": {"fifths": 1} if signature else None,
    }


def g_major_reference() -> dict:
    reference = example_reference()
    reference["key"] = {"tonalCenter": "G", "collections": ["ionian", "mixolydian"]}
    return reference


def test_any_accepted_collection_counts_as_the_right_key():
    result = scoreboard.evaluate(make_score(tonality=tonality("G", "mixolydian", 0.156)), g_major_reference())
    entry = result["key"]["score"]
    assert entry["tonalCenterMatches"] is True
    assert entry["collectionMatches"] is True
    assert entry["confidence"] == 0.156
    assert entry["keySignatureWritten"] is False


def test_relative_minor_has_the_same_notes_but_the_wrong_centre():
    entry = scoreboard.evaluate(make_score(tonality=tonality("E", "aeolian")), g_major_reference())["key"]["score"]
    assert entry["tonalCenterMatches"] is False
    assert entry["collectionMatches"] is False
    assert entry["sameNotes"] is True


def test_unrelated_key_is_wrong_on_every_count():
    entry = scoreboard.evaluate(make_score(tonality=tonality("B", "aeolian")), g_major_reference())["key"]["score"]
    assert entry == {
        "displayName": "B aeolian",
        "confidence": 0.6,
        "keySignatureWritten": False,
        "tonalCenterMatches": False,
        "collectionMatches": False,
        "sameNotes": False,
    }


def test_right_centre_with_an_unaccepted_collection():
    entry = scoreboard.evaluate(make_score(tonality=tonality("G", "aeolian")), g_major_reference())["key"]["score"]
    assert entry["tonalCenterMatches"] is True
    assert entry["collectionMatches"] is False


def test_analysis_key_is_judged_separately_when_supplied():
    analysis = {"tonality": {"primaryCandidate": {"tonalCenter": "B", "collection": "aeolian", "displayName": "B minor", "confidence": 0.04}}}
    result = scoreboard.evaluate(make_score(tonality=tonality("G", "ionian")), g_major_reference(), analysis=analysis)
    assert result["key"]["score"]["tonalCenterMatches"] is True
    assert result["key"]["analysis"]["displayName"] == "B minor"
    assert result["key"]["analysis"]["tonalCenterMatches"] is False
    assert "keySignatureWritten" not in result["key"]["analysis"]


def test_score_without_tonal_context_is_reported_as_unavailable_not_wrong():
    result = scoreboard.evaluate(make_score(tonality=None), g_major_reference())
    assert result["key"]["score"] is None
    assert result["key"]["analysis"] is None


def test_reference_without_a_key_skips_key_scoring():
    reference = example_reference()
    del reference["key"]
    result = scoreboard.evaluate(perfect_score(), reference)
    assert result["key"] is None
    assert result["indicators"]["sources"]["bass"]["inReferenceKeyPercent"] is None


# ---------------------------------------------------------------------------
# Chords


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("Am", (9, "min")),
        ("Am7", (9, "min")),
        ("A:min", (9, "min")),
        ("C-7", (0, "min")),
        ("C", (0, "maj")),
        ("G7", (7, "maj")),
        ("Cmaj7", (0, "maj")),
        ("CM7", (0, "maj")),
        ("C/E", (0, "maj")),
        ("F#m", (6, "min")),
        ("Bb", (10, "maj")),
        ("Dsus4", (2, "other")),
        ("Bdim", (11, "other")),
        ("E5", (4, "other")),
        ("N", None),
        ("", None),
        (None, None),
    ],
)
def test_chord_symbols_reduce_to_root_and_quality(symbol, expected):
    assert scoreboard.parse_chord(symbol) == expected


@pytest.mark.parametrize("symbol", ["H", "x", "7", 5])
def test_unreadable_chord_symbols_are_rejected(symbol):
    with pytest.raises(ScoreboardError):
        scoreboard.parse_chord(symbol)


def test_bars_without_a_symbol_count_as_wrong():
    chords = ["C", None, "F", None, "C", None, "F", None]
    result = scoreboard.evaluate(make_score(chords=chords), example_reference())["chords"]
    assert result["barsWithSymbol"] == 4
    assert result["shown"]["coveragePercent"] == 50.0
    assert result["shown"]["rootAccuracyPercent"] == 50.0
    assert result["shown"]["rootAccuracyWhereShownPercent"] == 100.0


def test_right_root_with_wrong_quality_is_separated_from_a_wrong_root():
    # Bar 2 is A major instead of A minor; bar 3 is G instead of F.
    chords = ["C", "A", "G", "G", "C", "Am", "F", "G"]
    shown = scoreboard.evaluate(make_score(chords=chords), example_reference())["chords"]["shown"]
    assert shown["coveragePercent"] == 100.0
    assert shown["rootAccuracyPercent"] == 87.5
    assert shown["majMinAccuracyPercent"] == 75.0


def test_sevenths_and_slash_basses_are_not_penalised():
    chords = ["Cmaj7", "Am7", "F/A", "G7", "C", "Am", "F", "G"]
    shown = scoreboard.evaluate(make_score(chords=chords), example_reference())["chords"]["shown"]
    assert shown["majMinAccuracyPercent"] == 100.0


def test_no_symbols_at_all_scores_zero_not_unavailable():
    result = scoreboard.evaluate(make_score(), example_reference())["chords"]
    assert result["barsWithSymbol"] == 0
    assert result["shown"]["coveragePercent"] == 0.0
    assert result["shown"]["rootAccuracyPercent"] == 0.0
    assert result["shown"]["rootAccuracyWhereShownPercent"] is None


def test_harmony_windows_are_scored_separately_from_shown_symbols():
    score = make_score()
    # Bar 1 has a resolved C window over its first half and an unresolved one
    # over the second; bar 2 has a wrong window. Nothing is shown on a bar.
    score["measures"][0]["harmony"] = [
        {"startBeat": 0.0, "endBeat": 2.0, "symbol": "C", "unresolved": False},
        {"startBeat": 2.0, "endBeat": 4.0, "symbol": None, "unresolved": True},
    ]
    score["measures"][1]["harmony"] = [{"startBeat": 0.0, "endBeat": 4.0, "symbol": "E", "unresolved": False}]
    result = scoreboard.evaluate(score, example_reference())["chords"]

    assert result["shown"]["coveragePercent"] == 0.0
    # 1 s right + 2 s wrong out of 16 s of reference chords.
    assert result["candidates"]["coveragePercent"] == pytest.approx(18.8, abs=0.1)
    assert result["candidates"]["rootAccuracyPercent"] == pytest.approx(6.2, abs=0.1)


def test_overlapping_windows_count_once_and_are_right_if_any_is_right():
    score = make_score()
    score["measures"][0]["harmony"] = [
        {"startBeat": 0.0, "endBeat": 4.0, "symbol": "C", "unresolved": False},
        {"startBeat": 0.0, "endBeat": 4.0, "symbol": "E", "unresolved": False},
        {"startBeat": 1.0, "endBeat": 3.0, "symbol": "C", "unresolved": False},
    ]
    candidates = scoreboard.evaluate(score, example_reference())["chords"]["candidates"]
    assert candidates["coveragePercent"] == 12.5
    assert candidates["rootAccuracyPercent"] == 12.5


def test_unreadable_score_symbols_are_counted_not_fatal():
    chords = ["C", "???", "F", "G", "C", "Am", "F", "G"]
    result = scoreboard.evaluate(make_score(chords=chords), example_reference())["chords"]
    assert result["unreadableSymbols"] == 1
    assert result["shown"]["coveragePercent"] == 87.5


def test_half_bar_reference_chords_split_the_bar():
    reference = example_reference()
    reference["chords"] = {"bars": [{"startBar": 1, "endBar": 2, "pattern": [["C", "F"]]}]}
    loaded = scoreboard.load_reference(reference)
    assert [(s["start"], s["end"], s["symbol"]) for s in loaded["chords"]] == [
        (0.0, 1.0, "C"),
        (1.0, 2.0, "F"),
        (2.0, 3.0, "C"),
        (3.0, 4.0, "F"),
    ]
    shown = scoreboard.evaluate(make_score(bars=2, chords=["C", "C"]), reference)["chords"]["shown"]
    assert shown["rootAccuracyPercent"] == 50.0


def test_no_chord_bars_are_left_out_of_the_reference_time():
    reference = example_reference()
    reference["chords"] = {"bars": [{"startBar": 1, "endBar": 4, "pattern": ["N", "C"]}]}
    shown = scoreboard.evaluate(make_score(bars=4, chords=["C", "C", "C", "C"]), reference)["chords"]["shown"]
    assert shown["referenceSeconds"] == 4.0
    assert shown["rootAccuracyPercent"] == 100.0


def test_timed_segments_are_an_alternative_to_bar_patterns():
    reference = example_reference()
    reference["chords"] = {
        "segments": [
            {"startSeconds": 0.0, "endSeconds": 3.0, "symbol": "C"},
            {"startSeconds": 3.0, "endSeconds": 4.0, "symbol": "G"},
        ]
    }
    shown = scoreboard.evaluate(make_score(bars=2, chords=["C", "C"]), reference)["chords"]["shown"]
    assert shown["referenceSeconds"] == 4.0
    assert shown["rootAccuracyPercent"] == 75.0


def test_chords_follow_real_time_so_a_drifting_grid_loses_accuracy():
    # Every bar carries the chord a correct grid would show, but the bars are
    # 5% too short, so later symbols sit over the wrong stretch of the recording.
    reference = example_reference()
    reference["chords"] = {"bars": [{"startBar": 1, "endBar": 40, "pattern": ["C", "Am", "F", "G"]}]}
    drifting = make_score(tempo=126.0, bars=40, chords=["C", "Am", "F", "G"] * 10)
    steady = make_score(tempo=120.0, bars=40, chords=["C", "Am", "F", "G"] * 10)

    assert scoreboard.evaluate(steady, reference)["chords"]["shown"]["rootAccuracyPercent"] == 100.0
    assert scoreboard.evaluate(drifting, reference)["chords"]["shown"]["rootAccuracyPercent"] < 60.0


# ---------------------------------------------------------------------------
# Notes


def test_note_f1_counts_matches_misses_and_extras():
    reference = [(0.0, 60), (1.0, 62), (2.0, 64), (3.0, 65)]
    estimated = [(0.01, 60), (1.04, 62), (2.06, 64), (5.0, 70), (6.0, 71)]
    result = scoreboard.note_f1(estimated, reference)
    assert result["matched"] == 2
    assert result["precisionPercent"] == 40.0
    assert result["recallPercent"] == 50.0
    assert result["f1Percent"] == pytest.approx(44.4, abs=0.1)


def test_octave_errors_only_count_in_the_octave_tolerant_figure():
    reference = [(0.0, 60), (1.0, 62)]
    estimated = [(0.0, 72), (1.0, 50)]
    assert scoreboard.note_f1(estimated, reference)["f1Percent"] == 0.0
    assert scoreboard.note_f1(estimated, reference, octave_tolerant=True)["f1Percent"] == 100.0


def test_one_reference_note_cannot_be_matched_twice():
    result = scoreboard.note_f1([(0.0, 60), (0.02, 60), (0.04, 60)], [(0.01, 60)])
    assert result["matched"] == 1
    assert result["precisionPercent"] == pytest.approx(33.3, abs=0.1)
    assert result["recallPercent"] == 100.0


def test_empty_estimate_scores_zero_without_dividing_by_zero():
    result = scoreboard.note_f1([], [(0.0, 60)])
    assert (result["precisionPercent"], result["recallPercent"], result["f1Percent"]) == (0.0, 0.0, 0.0)


def test_detected_and_notated_time_are_scored_separately():
    # The bass is detected 30 ms late (inside tolerance) but the draft uses a
    # 5%-fast grid, so the notated starts fall away from the recording.
    score = make_score(tempo=126.0, chords=EXAMPLE_CHORDS, bass=EXAMPLE_BASS)
    for index, measure in enumerate(score["measures"]):
        measure["notes"][0]["startSeconds"] = index * 2.0 + 0.03
    result = scoreboard.evaluate(score, example_reference())["notes"]["bass"]

    assert result["toleranceSeconds"] == 0.05
    assert result["detected"]["f1Percent"] == 100.0
    assert result["notated"]["matched"] == 1
    assert result["notated"]["f1Percent"] == 12.5


def test_reference_notes_for_all_lines_ignore_the_source():
    reference = example_reference()
    reference["notes"] = {"all": reference["notes"]["bass"]}
    result = scoreboard.evaluate(perfect_score(), reference)["notes"]
    assert result["all"]["detected"]["f1Percent"] == 100.0


def test_missing_source_line_scores_zero():
    reference = example_reference()
    reference["notes"] = {"vocals": [{"startSeconds": 0.0, "midiNote": 60}]}
    result = scoreboard.evaluate(perfect_score(), reference)["notes"]
    assert result["vocals"]["detected"]["estimatedNotes"] == 0
    assert result["vocals"]["detected"]["f1Percent"] == 0.0


def test_reference_without_notes_does_not_invent_note_accuracy():
    reference = example_reference()
    del reference["notes"]
    result = scoreboard.evaluate(perfect_score(), reference)
    assert result["notes"] is None
    assert "not measured" in scoreboard.format_report(result)


# ---------------------------------------------------------------------------
# Indicators


def test_indicators_describe_the_draft_and_say_they_are_not_accuracy():
    # Bars 1-4 on the chord root, bars 5-8 a tritone away and low confidence.
    bass = [36, 33, 29, 31, 42, 39, 35, 37]
    score = make_score(chords=EXAMPLE_CHORDS, bass=bass)
    for measure in score["measures"][4:]:
        measure["notes"][0]["confidence"] = 0.2
    score["measures"][0]["percussionHits"] = [{"broadVoice": "low_drum"}, {"broadVoice": "low_drum"}, {"broadVoice": "mid_drum"}]
    indicators = scoreboard.evaluate(score, example_reference())["indicators"]

    assert "not accuracy" in indicators["note"]
    assert indicators["sources"]["bass"]["notes"] == 8
    assert indicators["sources"]["bass"]["notesPerBar"] == 1.0
    assert indicators["sources"]["bass"]["lowConfidencePercent"] == 50.0
    assert indicators["sources"]["bass"]["inReferenceKeyPercent"] == 62.5
    assert indicators["bass"]["onChordRootPercent"] == 50.0
    assert indicators["bass"]["barsWithBassPercent"] == 100.0
    assert indicators["percussion"] == {"hits": 3, "hitsPerBar": 0.38, "byVoice": {"low_drum": 2, "mid_drum": 1}}


def test_bass_indicator_is_absent_without_bass_notes_or_reference_chords():
    assert scoreboard.evaluate(make_score(chords=EXAMPLE_CHORDS), example_reference())["indicators"]["bass"] is None
    reference = example_reference()
    del reference["chords"]
    assert scoreboard.evaluate(perfect_score(), reference)["indicators"]["bass"] is None


def test_malformed_notes_in_a_score_are_skipped_not_fatal():
    score = perfect_score()
    score["measures"][0]["notes"] += ["text", {"midiNote": "60", "startSeconds": 0.0}, {"midiNote": 60, "startSeconds": None}]
    result = scoreboard.evaluate(score, example_reference())
    assert result["score"]["notes"] == 8


# ---------------------------------------------------------------------------
# Reference validation


def _mutated(mutate) -> dict:
    reference = example_reference()
    mutate(reference)
    return reference


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(artifactType="something-else"),
        lambda r: r.update(schemaVersion=2),
        lambda r: r.update(id="Has Spaces"),
        lambda r: r.update(id=""),
        lambda r: r.update(title="x" * 201),
        lambda r: r.pop("timing"),
        lambda r: r["timing"].update(tempoBpm=0),
        lambda r: r["timing"].update(tempoBpm=float("nan")),
        lambda r: r["timing"].update(tempoBpm=True),
        lambda r: r["timing"].update(tempoBpm="120"),
        lambda r: r["timing"].update(beatsPerMeasure=4.0),
        lambda r: r["timing"].update(beatsPerMeasure=0),
        lambda r: r["timing"].update(firstDownbeatSeconds=-1),
        lambda r: r["key"].update(tonalCenter="H"),
        lambda r: r["key"].update(tonalCenter="C##"),
        lambda r: r["key"].update(collections=[]),
        lambda r: r["key"].update(collections=["happy"]),
        lambda r: r.update(key="C major"),
        lambda r: r.update(chords={"bars": [], "segments": []}),
        lambda r: r.update(chords={}),
        lambda r: r.update(chords={"bars": []}),
        lambda r: r.update(chords={"bars": [{"startBar": 0, "endBar": 4, "pattern": ["C"]}]}),
        lambda r: r.update(chords={"bars": [{"startBar": 5, "endBar": 4, "pattern": ["C"]}]}),
        lambda r: r.update(chords={"bars": [{"startBar": 1, "endBar": 4, "pattern": []}]}),
        lambda r: r.update(chords={"bars": [{"startBar": 1, "endBar": 4, "pattern": ["Q"]}]}),
        lambda r: r.update(chords={"bars": [{"startBar": 1, "endBar": 4, "pattern": [[]]}]}),
        lambda r: r.update(chords={"bars": [{"startBar": 1, "endBar": 4, "pattern": ["C"]}, {"startBar": 4, "endBar": 6, "pattern": ["G"]}]}),
        lambda r: r.update(chords={"segments": [{"startSeconds": 2, "endSeconds": 1, "symbol": "C"}]}),
        lambda r: r.update(chords={"segments": [{"startSeconds": 0, "endSeconds": 2, "symbol": "C"}, {"startSeconds": 1, "endSeconds": 3, "symbol": "G"}]}),
        lambda r: r.update(notes={}),
        lambda r: r.update(notes={"Bass Line": [{"startSeconds": 0, "midiNote": 40}]}),
        lambda r: r.update(notes={"bass": []}),
        lambda r: r.update(notes={"bass": [{"startSeconds": 0, "midiNote": 128}]}),
        lambda r: r.update(notes={"bass": [{"startSeconds": 0, "midiNote": 40.5}]}),
        lambda r: r.update(notes={"bass": [{"startSeconds": -1, "midiNote": 40}]}),
    ],
)
def test_invalid_references_are_rejected(mutate):
    with pytest.raises(ScoreboardError):
        scoreboard.load_reference(_mutated(mutate))


def test_bar_patterns_need_a_first_downbeat():
    reference = example_reference()
    del reference["timing"]["firstDownbeatSeconds"]
    with pytest.raises(ScoreboardError, match="firstDownbeatSeconds"):
        scoreboard.load_reference(reference)


def test_collection_names_accept_common_aliases():
    reference = example_reference()
    reference["key"]["collections"] = ["Major", "natural minor", "harmonic-minor", "ionian"]
    assert scoreboard.load_reference(reference)["key"]["collections"] == ["ionian", "aeolian", "harmonic_minor"]


def test_known_collections_match_the_score_tonal_context():
    from app.tonal_context import COLLECTIONS

    assert set(scoreboard.COLLECTION_INTERVALS) == set(COLLECTIONS)
    for name, definition in COLLECTIONS.items():
        assert scoreboard.COLLECTION_INTERVALS[name] == tuple(definition[1])


# ---------------------------------------------------------------------------
# Score validation


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("measures"),
        lambda s: s.update(measures=[]),
        lambda s: s.pop("timing"),
        lambda s: s["timing"].update(tempoBpm=None),
        lambda s: s["timing"].update(beatsPerMeasure="4"),
        lambda s: s["measures"][0].update(endSeconds=s["measures"][0]["startSeconds"]),
        lambda s: s["measures"][1].update(startSeconds=-5.0, endSeconds=-4.0),
        lambda s: s["measures"][2].update(startSeconds=float("inf")),
        lambda s: s["measures"].__setitem__(3, "bar"),
    ],
)
def test_invalid_scores_are_rejected(mutate):
    score = perfect_score()
    mutate(score)
    with pytest.raises(ScoreboardError):
        scoreboard.evaluate(score, example_reference())


def test_score_summary_without_measures_explains_what_to_use():
    with pytest.raises(ScoreboardError, match="includes its measures"):
        scoreboard.evaluate({"timing": {"tempoBpm": 120, "beatsPerMeasure": 4}}, example_reference())


@pytest.mark.parametrize("label", ["", "x" * 81, "bad\nlabel", "/etc/passwd"])
def test_invalid_labels_are_rejected(label):
    with pytest.raises(ScoreboardError):
        scoreboard.evaluate(perfect_score(), example_reference(), label=label)


# ---------------------------------------------------------------------------
# Command line and saved runs


def write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_command_prints_a_report_and_separates_accuracy_from_indicators(tmp_path, capsys):
    score_path = write_json(tmp_path / "score.json", perfect_score())
    code = scoreboard.main(["--score", str(score_path), "--reference", str(EXAMPLE_REFERENCE_PATH), "--label", "main"])
    output = capsys.readouterr().out

    assert code == 0
    assert "ACCURACY AGAINST THE REFERENCE" in output
    assert "INDICATORS (describe the draft; not accuracy)" in output
    assert output.index("ACCURACY AGAINST THE REFERENCE") < output.index("INDICATORS")
    assert "Bar-grid drift: 0.0 beats by the end" in output
    assert str(tmp_path) not in output


def test_command_can_print_json(tmp_path, capsys):
    score_path = write_json(tmp_path / "score.json", perfect_score())
    code = scoreboard.main(["--score", str(score_path), "--reference", str(EXAMPLE_REFERENCE_PATH), "--json"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["artifactType"] == "popex-accuracy-result"
    assert result["scoreboardVersion"] == scoreboard.SCOREBOARD_VERSION


def test_saved_runs_accumulate_and_a_repeated_label_replaces_its_run(tmp_path, capsys):
    results_path = tmp_path / "runs" / "results.json"
    good = write_json(tmp_path / "good.json", perfect_score())
    fast = write_json(tmp_path / "fast.json", make_score(tempo=121.2, bars=100))
    reference = str(EXAMPLE_REFERENCE_PATH)

    assert scoreboard.main(["--score", str(fast), "--reference", reference, "--label", "before", "--save", str(results_path)]) == 0
    assert scoreboard.main(["--score", str(good), "--reference", reference, "--label", "after", "--save", str(results_path)]) == 0
    assert scoreboard.main(["--score", str(good), "--reference", reference, "--label", "before", "--save", str(results_path)]) == 0
    capsys.readouterr()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["artifactType"] == "popex-accuracy-results"
    assert [run["label"] for run in saved["runs"]] == ["after", "before"]
    assert saved["runs"][1]["timing"]["maxGridDriftBeats"] == 0.0
    assert not list(results_path.parent.glob(".scoreboard-*"))

    assert scoreboard.main(["--table", str(results_path)]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith("reference")
    assert "max drift (beats)" in table[0]
    assert len(table) == 4
    assert "after" in table[2] and "before" in table[3]


def test_command_errors_are_plain_and_write_nothing(tmp_path, capsys):
    results_path = tmp_path / "results.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    not_finite = tmp_path / "nan.json"
    not_finite.write_text('{"measures": [], "timing": {"tempoBpm": NaN}}', encoding="utf-8")
    array = write_json(tmp_path / "array.json", [1, 2, 3])
    good = write_json(tmp_path / "good.json", perfect_score())
    reference = str(EXAMPLE_REFERENCE_PATH)

    cases = [
        ["--score", str(broken), "--reference", reference, "--save", str(results_path)],
        ["--score", str(not_finite), "--reference", reference, "--save", str(results_path)],
        ["--score", str(array), "--reference", reference, "--save", str(results_path)],
        ["--score", str(tmp_path / "missing.json"), "--reference", reference, "--save", str(results_path)],
        ["--score", str(tmp_path), "--reference", reference, "--save", str(results_path)],
        ["--score", str(good), "--reference", str(good), "--save", str(results_path)],
        ["--score", str(good)],
        ["--table", str(good)],
        ["--table", str(good), "--score", str(good)],
    ]
    for arguments in cases:
        assert scoreboard.main(arguments) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("Scoreboard error: ")
        assert "Traceback" not in captured.err
        assert str(tmp_path) not in captured.err
    assert not results_path.exists()


def test_oversized_input_is_refused_before_parsing(tmp_path, monkeypatch):
    path = write_json(tmp_path / "score.json", perfect_score())
    monkeypatch.setattr(scoreboard, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ScoreboardError, match="larger than"):
        scoreboard.read_json_file(path, "score")


def test_save_refuses_a_file_it_did_not_write(tmp_path):
    other = write_json(tmp_path / "results.json", {"artifactType": "something-else", "runs": []})
    result = scoreboard.evaluate(perfect_score(), example_reference(), label="main")
    with pytest.raises(ScoreboardError):
        scoreboard.save_run(other, result)
    assert json.loads(other.read_text(encoding="utf-8")) == {"artifactType": "something-else", "runs": []}


# ---------------------------------------------------------------------------
# Repository hygiene


def test_example_reference_is_valid_and_declares_itself_synthetic():
    reference = example_reference()
    loaded = scoreboard.load_reference(reference)
    assert loaded["id"].startswith("synthetic-")
    assert "no real recording" in reference["source"]


def test_no_reference_for_a_real_recording_is_committed():
    """References for real recordings stay outside the repository."""
    git = shutil.which("git")
    if git is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    listed = subprocess.run([git, "ls-files", "-z", "--", "*.json"], cwd=_REPO_ROOT, capture_output=True, check=False)
    if listed.returncode != 0:
        pytest.skip("git ls-files not available")
    for name in filter(None, listed.stdout.decode("utf-8", "replace").split("\0")):
        path = _REPO_ROOT / name
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("artifactType") == scoreboard.REFERENCE_ARTIFACT:
            assert str(payload.get("id", "")).startswith("synthetic-"), f"{name} is not a synthetic reference"
