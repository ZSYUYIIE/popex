from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

import pytest

from app.score_construction import (
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    score_to_midi_bytes,
    score_to_musicxml_text,
)


def _note(
    note_id: str,
    start: float,
    end: float,
    midi: int = 69,
    confidence: float = 0.9,
    *,
    midi_pitch: float | None = None,
    source_kind: str = "full_mix",
    warnings: list[str] | None = None,
) -> dict:
    return {
        "id": note_id,
        "sourceKind": source_kind,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": midi,
        "midiPitch": midi if midi_pitch is None else midi_pitch,
        "confidence": confidence,
        "warnings": [] if warnings is None else warnings,
    }


def _midi_note_events(payload: bytes) -> list[tuple[int, int, int]]:
    track_length = struct.unpack_from(">I", payload, 18)[0]
    data = payload[22 : 22 + track_length]
    cursor = 0
    tick = 0
    events: list[tuple[int, int, int]] = []

    def read_varlen(offset: int) -> tuple[int, int]:
        value = 0
        while True:
            byte = data[offset]
            offset += 1
            value = (value << 7) | (byte & 0x7F)
            if byte < 0x80:
                return value, offset

    while cursor < len(data):
        delta, cursor = read_varlen(cursor)
        tick += delta
        status = data[cursor]
        cursor += 1
        if status == 0xFF:
            meta_type = data[cursor]
            cursor += 1
            size, cursor = read_varlen(cursor)
            cursor += size
            if meta_type == 0x2F:
                break
            continue
        first = data[cursor]
        second = data[cursor + 1]
        cursor += 2
        if (status & 0xF0) in (0x80, 0x90):
            events.append((tick, status & 0xF0, first))
    return events


def test_empty_evidence_builds_single_empty_measure_with_warning() -> None:
    document = build_score_document([], tempo_bpm=120.0)
    assert document["schemaVersion"] == SCORE_SCHEMA_VERSION
    assert document["builderVersion"] == SCORE_BUILDER_VERSION
    assert document["measureCount"] == 1
    assert document["noteCount"] == 0
    assert any("empty" in warning.lower() for warning in document["warnings"])


def test_notes_split_across_measures_with_quantization() -> None:
    document = build_score_document(
        [_note("a" * 1 + "1", 0.10, 0.40, 69), _note("b2", 2.10, 2.40, 71)],
        tempo_bpm=120.0,
        beats_per_measure=4,
    )
    assert document["measureCount"] >= 2
    assert document["noteCount"] == 2
    first = document["measures"][0]["notes"][0]
    assert first["quantizedBeat"] == pytest.approx(0.0, abs=0.25)
    assert first["noteName"].startswith("A")


def test_low_confidence_and_large_shift_warn_honestly() -> None:
    document = build_score_document(
        [_note("n1", 0.13, 0.50, 60, confidence=0.10)],
        tempo_bpm=120.0,
    )
    assert any("0.50" in warning for warning in document["warnings"])


def test_score_document_preserves_raw_pitch_source_and_event_warnings() -> None:
    document = build_score_document(
        [
            _note(
                "evidence1",
                0.0,
                0.5,
                60,
                midi_pitch=60.27,
                source_kind="vocals",
                warnings=["weak onset evidence"],
            )
        ],
        tempo_bpm=120.0,
    )
    note = document["measures"][0]["notes"][0]
    assert note["id"] == "evidence1"
    assert note["sourceKind"] == "vocals"
    assert note["rawMidiPitch"] == pytest.approx(60.27)
    assert note["sourceWarnings"] == ["weak onset evidence"]
    assert note["startSeconds"] == 0.0
    assert note["endSeconds"] == 0.5


def test_score_document_preserves_all_bounded_event_warnings() -> None:
    warnings = [f"event warning {index}" for index in range(8)]
    warnings.append("line one\nline two")
    document = build_score_document(
        [_note("warnings1", 0.0, 0.5, warnings=warnings)], tempo_bpm=120.0
    )
    assert document["measures"][0]["notes"][0]["sourceWarnings"] == warnings
    with pytest.raises(ScoreConstructionError, match="bounded sequence"):
        build_score_document(
            [_note("warnings2", 0.0, 0.5, warnings=["warning"] * 129)],
            tempo_bpm=120.0,
        )


def test_musicxml_preserves_quantized_onset_and_duration() -> None:
    document = build_score_document([_note("timed1", 0.5, 1.5, 60)], tempo_bpm=120.0)
    root = ET.fromstring(score_to_musicxml_text(document))
    measure = root.find(".//measure")
    assert measure is not None
    forwards = measure.findall("forward")
    notes = [node for node in measure.findall("note") if node.find("pitch") is not None]
    assert [node.findtext("duration") for node in forwards] == ["480", "480"]
    assert [node.findtext("duration") for node in notes] == ["960"]
    assert notes[0].find("pitch/step").text == "C"
    assert notes[0].find("pitch/octave").text == "4"
    assert notes[0].findtext("voice") == "1"
    assert not measure.findall("note/rest")


def test_musicxml_emits_notated_sharps() -> None:
    document = build_score_document([_note("sharp1", 0.0, 0.5, 61)], tempo_bpm=120.0)
    root = ET.fromstring(score_to_musicxml_text(document))
    note = next(node for node in root.findall(".//measure/note") if node.find("pitch") is not None)
    assert note.findtext("pitch/step") == "C"
    assert note.findtext("pitch/alter") == "1"
    assert note.findtext("accidental") == "sharp"


def test_musicxml_overlapping_notes_use_separate_timed_voices() -> None:
    document = build_score_document(
        [_note("long1", 0.0, 1.5, 60), _note("overlap1", 0.5, 1.0, 64)],
        tempo_bpm=120.0,
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    measure = root.find(".//measure")
    pitched = [note for note in measure.findall("note") if note.find("pitch") is not None]
    assert [note.findtext("voice") for note in pitched] == ["1", "2"]
    assert [note.findtext("duration") for note in pitched] == ["1440", "480"]
    backup = measure.find("backup")
    assert backup is not None
    assert backup.findtext("duration") == "1920"


def test_musicxml_splits_a_long_note_with_ties_at_barline() -> None:
    document = build_score_document([_note("held1", 1.5, 3.0, 67)], tempo_bpm=120.0)
    root = ET.fromstring(score_to_musicxml_text(document))
    measures = root.findall(".//measure")
    assert len(measures) == 2
    first_note = next(note for note in measures[0].findall("note") if note.find("pitch") is not None)
    second_note = next(note for note in measures[1].findall("note") if note.find("pitch") is not None)
    assert first_note.findtext("duration") == "480"
    assert second_note.findtext("duration") == "960"
    assert first_note.find("tie").get("type") == "start"
    assert second_note.find("tie").get("type") == "stop"
    assert first_note.find("notations/tied").get("type") == "start"
    assert second_note.find("notations/tied").get("type") == "stop"


def test_midi_and_musicxml_keep_duration_beyond_sixteen_beats() -> None:
    document = build_score_document([_note("long-midi", 0.0, 12.0, 55)], tempo_bpm=120.0)
    events = _midi_note_events(score_to_midi_bytes(document))
    assert (0, 0x90, 55) in events
    assert (24 * 480, 0x80, 55) in events


def test_musicxml_bounds_note_fragment_expansion() -> None:
    document = build_score_document(
        [_note(f"long-{index}", 0.0, 6144.0, 48 + index) for index in range(5)],
        tempo_bpm=20.0,
        beats_per_measure=1,
    )
    assert document["measureCount"] == 2048
    with pytest.raises(ScoreConstructionError, match="fragment limit"):
        score_to_musicxml_text(document)


def test_score_fails_closed_instead_of_collapsing_excess_measures() -> None:
    with pytest.raises(ScoreConstructionError, match="measure limit"):
        build_score_document(
            [_note("late", 35999.0, 36000.0, 60)],
            tempo_bpm=20.0,
            beats_per_measure=1,
        )


def test_exporters_reject_inconsistent_divisions_and_measure_indices() -> None:
    document = build_score_document([_note("second-bar", 2.0, 2.5)], tempo_bpm=120.0)
    wrong_divisions = {**document, "divisions": 960}
    with pytest.raises(ScoreConstructionError, match="divisions"):
        score_to_midi_bytes(wrong_divisions)
    with pytest.raises(ScoreConstructionError, match="divisions"):
        score_to_musicxml_text(wrong_divisions)

    wrong_index = {
        **document,
        "measures": [document["measures"][0], {**document["measures"][1], "measureIndex": 1.0}],
    }
    with pytest.raises(ScoreConstructionError, match="measureIndex"):
        score_to_midi_bytes(wrong_index)
    with pytest.raises(ScoreConstructionError, match="measureIndex"):
        score_to_musicxml_text(wrong_index)


def test_duplicate_ids_and_bad_ranges_fail_closed() -> None:
    with pytest.raises(ScoreConstructionError):
        build_score_document(
            [_note("same", 0.0, 0.2), _note("same", 0.3, 0.5)], tempo_bpm=120.0
        )
    with pytest.raises(ScoreConstructionError):
        build_score_document([_note("x1", 0.5, 0.2)], tempo_bpm=120.0)
    with pytest.raises(ScoreConstructionError):
        build_score_document([_note("x2", 0.0, 0.2)], tempo_bpm=5.0)


def test_midi_bytes_have_smf_header_and_note_events() -> None:
    document = build_score_document([_note("m1", 0.0, 0.5, 69)], tempo_bpm=120.0)
    payload = score_to_midi_bytes(document)
    assert payload[:4] == b"MThd"
    assert b"MTrk" in payload
    assert bytes((0x90, 69)) in payload


def test_musicxml_parses_and_marks_draft() -> None:
    document = build_score_document(
        [_note("w1", 0.0, 0.5, 60)],
        tempo_bpm=100.0,
        chord_symbols=["Cmaj"],
    )
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    assert root.tag == "score-partwise"
    assert SCORE_BUILDER_VERSION in text
    assert root.find(".//pitch") is not None
    assert root.find(".//metronome/beat-unit") is not None
    assert root.find(".//metronome/per-minute") is not None
    encoding = root.find("identification/encoding")
    assert encoding is not None
    assert not (encoding.text or "").strip()
    assert encoding.findtext("software") == f"PopEx {SCORE_BUILDER_VERSION}"
    assert "review required" in encoding.findtext("encoding-description")


@pytest.mark.parametrize("midi_note", [0, 11])
def test_musicxml_rejects_pitch_below_its_octave_range_but_midi_preserves_it(
    midi_note: int,
) -> None:
    document = build_score_document(
        [_note("low-pitch", 0.0, 0.5, midi_note)], tempo_bpm=120.0
    )
    assert (0, 0x90, midi_note) in _midi_note_events(score_to_midi_bytes(document))
    with pytest.raises(ScoreConstructionError, match="MusicXML pitch range"):
        score_to_musicxml_text(document)


@pytest.mark.parametrize("midi_note", [12, 127])
def test_musicxml_preserves_pitches_at_its_octave_boundaries(midi_note: int) -> None:
    document = build_score_document(
        [_note("boundary-pitch", 0.0, 0.5, midi_note)], tempo_bpm=120.0
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    assert root.findtext(".//pitch/octave") == str(midi_note // 12 - 1)


def test_musicxml_preserves_chord_text_without_fabricated_harmony() -> None:
    document = build_score_document(
        [_note("w2", 0.0, 0.5, 60)],
        tempo_bpm=100.0,
        chord_symbols=["Am7"],
    )
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    assert root.find(".//harmony") is None
    words = [node.text for node in root.iter("words")]
    assert "Am7" in words


def test_musicxml_rejects_tampered_measure_index_or_chord() -> None:
    document = build_score_document([_note("w3", 0.0, 0.5, 60)], tempo_bpm=100.0)
    tampered = dict(document)
    tampered_measures = [dict(measure) for measure in document["measures"]]
    tampered_measures[0]["measureIndex"] = 7
    tampered["measures"] = tampered_measures
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(tampered)
    tampered2 = dict(document)
    tampered_measures2 = [dict(measure) for measure in document["measures"]]
    tampered_measures2[0]["chordSymbol"] = "Bad <tag>"
    tampered2["measures"] = tampered_measures2
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(tampered2)
