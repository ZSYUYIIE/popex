from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

import pytest
from midi_reader import note_events

from app.score_construction import (
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    parse_chord_symbol,
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
    return [(tick, status, key) for tick, status, key, _velocity in note_events(payload)]


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


def test_midi_keeps_overlapping_unisons_on_independent_channels() -> None:
    document = build_score_document(
        [_note("held", 0.0, 2.0, 60), _note("short", 0.5, 1.0, 60)],
        tempo_bpm=120.0,
    )
    assert _midi_note_events(score_to_midi_bytes(document)) == [
        (0, 0x90, 60), (480, 0x91, 60), (960, 0x81, 60), (1920, 0x80, 60),
    ]


def test_midi_unison_channel_limit_fails_without_using_percussion_channel() -> None:
    events = [_note(f"unison-{index}", 0, 0.5, 60) for index in range(16)]
    document = build_score_document(events[:15], tempo_bpm=120.0)
    on_channels = {
        status & 0x0F for _, status, _ in _midi_note_events(score_to_midi_bytes(document))
        if status & 0xF0 == 0x90
    }
    assert on_channels == set(range(16)) - {9}
    with pytest.raises(ScoreConstructionError, match="overlapping unison"):
        score_to_midi_bytes(build_score_document(events, tempo_bpm=120.0))


def test_midi_preserves_meter_and_trailing_measure_time() -> None:
    document = build_score_document(
        [_note("short", 0, 0.5, 60)], tempo_bpm=120.0, beats_per_measure=3
    )
    payload = score_to_midi_bytes(document)
    assert b"\x00\xff\x58\x04\x03\x02\x18\x08" in payload
    # The last note ends at tick 480; the 3/4 measure ends at tick 1440.
    assert payload.endswith(b"\x87\x40\xff\x2f\x00")


def test_midi_reuses_finished_channels_with_note_off_before_note_on() -> None:
    events = [_note("first", 0, 0.5, 60), _note("next", 0.5, 1, 60)]
    payload = score_to_midi_bytes(build_score_document(events, tempo_bpm=120.0))
    assert _midi_note_events(payload) == [
        (0, 0x90, 60), (480, 0x80, 60), (480, 0x90, 60), (960, 0x80, 60),
    ]
    assert payload == score_to_midi_bytes(
        build_score_document(list(reversed(events)), tempo_bpm=120.0)
    )


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


def test_the_printed_tempo_mark_is_a_whole_number_and_playback_stays_exact():
    document = build_score_document(
        [{"id": "n1", "startSeconds": 0.0, "endSeconds": 0.4, "midiNote": 60, "confidence": 0.9}],
        tempo_bpm=100.055,
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    assert root.findtext(".//metronome/per-minute") == "100"
    assert root.find(".//direction/sound").get("tempo") == "100.055"
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


def test_musicxml_exports_chord_symbols_as_structured_harmony() -> None:
    document = build_score_document(
        [_note("w2", 0.0, 0.5, 60), _note("w2b", 2.4, 2.9, 62), _note("w2c", 4.8, 5.3, 64)],
        tempo_bpm=100.0,
        chord_symbols=["Am7", "Bbmaj7/D", "F#m"],
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    harmonies = root.findall(".//harmony")
    assert [h.findtext("root/root-step") for h in harmonies] == ["A", "B", "F"]
    assert [h.findtext("root/root-alter") for h in harmonies] == [None, "-1", "1"]
    assert [h.findtext("kind") for h in harmonies] == ["minor-seventh", "major-seventh", "minor"]
    # The printed suffix is the one the musician sees in PopEx.
    assert [h.find("kind").get("text") for h in harmonies] == ["m7", "maj7", "m"]
    assert harmonies[1].findtext("bass/bass-step") == "D" and harmonies[1].find("bass/bass-alter") is None
    assert harmonies[0].find("bass") is None
    assert not [node.text for node in root.iter("words")]
    # The chord sits before the bar's first note, at the downbeat.
    first_measure = list(root.find(".//measure"))
    tags = [child.tag for child in first_measure]
    assert tags.index("harmony") < tags.index("note")


def test_musicxml_keeps_unparsed_chord_text_as_words() -> None:
    document = build_score_document(
        [_note("w4", 0.0, 0.5, 60), _note("w4b", 2.4, 2.9, 62)],
        tempo_bpm=100.0,
        chord_symbols=["Am(add9)", "C"],
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    assert [node.text for node in root.iter("words")] == ["Am(add9)"]
    assert [h.findtext("kind") for h in root.iter("harmony")] == ["major"]
    assert root.find(".//harmony/kind").get("text") == ""


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("C", ("C", 0, "major", "", None, 0)),
        ("Ebm", ("E", -1, "minor", "m", None, 0)),
        ("Bdim", ("B", 0, "diminished", "dim", None, 0)),
        ("Dsus2", ("D", 0, "suspended-second", "sus2", None, 0)),
        ("Gsus4", ("G", 0, "suspended-fourth", "sus4", None, 0)),
        ("E5", ("E", 0, "power", "5", None, 0)),
        ("G7", ("G", 0, "dominant", "7", None, 0)),
        ("C#m7b5", ("C", 1, "half-diminished", "m7b5", None, 0)),
        ("Cm/Eb", ("C", 0, "minor", "m", "E", -1)),
        ("F/C", ("F", 0, "major", "", "C", 0)),
        ("A♭maj7", ("A", -1, "major-seventh", "maj7", None, 0)),
        (" D9 ", ("D", 0, "dominant-ninth", "9", None, 0)),
    ],
)
def test_parse_chord_symbol_reads_root_kind_and_bass(symbol, expected) -> None:
    parsed = parse_chord_symbol(symbol)
    assert parsed is not None
    assert (
        parsed["rootStep"], parsed["rootAlter"], parsed["kind"],
        parsed["kindText"], parsed["bassStep"], parsed["bassAlter"],
    ) == expected


@pytest.mark.parametrize("symbol", ["", "H", "Cmaj7#11", "Am(add9)", "N.C.", "C/", "C/H", "cm", "C##", 7, None])
def test_parse_chord_symbol_refuses_text_it_does_not_fully_understand(symbol) -> None:
    assert parse_chord_symbol(symbol) is None


def test_every_harmony_stage_quality_exports_as_structured_harmony() -> None:
    from app.harmony_inference import _QUALITY_SUFFIX

    for quality, suffix in _QUALITY_SUFFIX.items():
        parsed = parse_chord_symbol(f"A{suffix}")
        assert parsed is not None, quality


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


_XSD = __import__("os").environ.get("POPEX_MUSICXML_XSD")


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_structured_chord_symbols_validate_against_musicxml_schema() -> None:
    from lxml import etree  # required whenever the schema path is configured

    symbols = ["C", "Ebm", "Bdim", "Dsus2", "Gsus4", "E5", "G7", "Fmaj7", "Am7",
               "C#m7b5", "Cm/Eb", "Bbmaj7/D", "Am(add9)"]
    notes = [_note(f"x{i}", i * 2.4, i * 2.4 + 0.5, 60) for i in range(len(symbols))]
    document = build_score_document(notes, tempo_bpm=100.0, chord_symbols=symbols)
    schema = etree.XMLSchema(etree.parse(_XSD))
    tree = etree.fromstring(score_to_musicxml_text(document).encode("utf-8"))
    assert schema.validate(tree), schema.error_log
    assert len(tree.findall(".//harmony")) == len(symbols) - 1


def _part_document(*, with_drums: bool = True) -> dict:
    from app.score_construction import plan_score_parts
    from test_score_percussion import hit_input

    events = [
        _note("v1", 0.0, 0.5, 67, source_kind="vocals"),
        _note("a1", 0.0, 0.5, 60, source_kind="other"),
        _note("a2", 0.0, 0.5, 64, source_kind="other"),
        _note("a3", 0.25, 0.5, 60, source_kind="other"),
        _note("b1", 0.0, 1.0, 40, source_kind="bass"),
    ]
    hits = [hit_input("k1", "low_drum", 0.0), hit_input("s1", "mid_drum", 0.5)] if with_drums else None
    document = build_score_document(events, tempo_bpm=120.0, percussion_hits=hits)
    for measure in document["measures"]:
        for note in measure["notes"]:
            note["tab"] = (
                {"instrument": "bass", "status": "unplayable", "string": None, "fret": None}
                if note["sourceKind"] == "bass" else None
            )
    document["scoreParts"] = plan_score_parts(
        [note for measure in document["measures"] for note in measure["notes"] if note["tab"] is None]
    )
    return document


def test_midi_writes_one_named_track_per_part_like_musicxml() -> None:
    from midi_reader import read_midi

    document = _part_document()
    midi = read_midi(score_to_midi_bytes(document))
    assert midi.format == 1 and midi.division == 480
    names = [track.name for track in midi.tracks]
    assert names == [
        None,
        "Lead vocal (draft)",
        "Accompaniment reduction",
        "Bass (draft, with TAB)",
        "Drum Kit (draft, broad voices)",
    ]
    xml = ET.fromstring(score_to_musicxml_text(document))
    assert [name.text for name in xml.iter("part-name")] == names[1:]
    # The conductor track holds tempo and meter, and no notes.
    conductor = midi.tracks[0]
    assert not conductor.channel
    assert {meta_type for _tick, meta_type, _data in conductor.meta} >= {0x51, 0x58, 0x2F}
    # Every track ends with the score.
    assert {track.end_tick for track in midi.tracks} == {4 * 480}

    def programs(track):
        return {status & 0x0F: data[0] + 1 for _t, status, data in track.channel if status & 0xF0 == 0xC0}

    def note_channels(track):
        return {status & 0x0F for _t, status, _d in track.channel if status & 0xF0 == 0x90}

    vocal, accompaniment, bass, drums = midi.tracks[1:]
    assert programs(vocal) == {0: 54}
    assert note_channels(vocal) == {0}
    # The overlapping unison C4 spills onto a spare channel with the same program.
    assert programs(accompaniment) == {1: 1, 2: 1}
    assert note_channels(accompaniment) == {1, 2}
    assert programs(bass) == {3: 34} and note_channels(bass) == {3}
    assert not programs(drums) and note_channels(drums) == {9}
    pitched = [note for track in midi.tracks[1:4] for note in track.channel if note[1] & 0xF0 == 0x90]
    assert sorted(data[0] for _t, _s, data in pitched) == [40, 60, 60, 64, 67]


def test_midi_without_score_parts_keeps_one_combined_track() -> None:
    from midi_reader import read_midi

    document = build_score_document(
        [_note("v1", 0.0, 0.5, 67, source_kind="vocals"), _note("a1", 0.0, 0.5, 60, source_kind="other")],
        tempo_bpm=120.0,
    )
    midi = read_midi(score_to_midi_bytes(document))
    assert [track.name for track in midi.tracks] == [None, "Draft pitched events"]
    assert not [event for event in midi.tracks[1].channel if event[1] & 0xF0 == 0xC0]


def test_midi_of_an_empty_score_is_a_valid_conductor_track() -> None:
    from midi_reader import read_midi

    midi = read_midi(score_to_midi_bytes(build_score_document([], tempo_bpm=120.0)))
    assert len(midi.tracks) == 1 and midi.tracks[0].end_tick == 4 * 480


def test_musicxml_writes_notes_that_start_and_end_together_as_one_chord() -> None:
    document = build_score_document(
        [
            _note("c", 0.0, 1.0, 64),
            _note("e", 0.0, 1.0, 60),
            _note("g", 0.0, 1.0, 67),
            _note("held", 0.0, 2.0, 72),  # same start, longer: needs its own voice
            _note("next", 1.0, 1.5, 62),
        ],
        tempo_bpm=120.0,
    )
    measure = ET.fromstring(score_to_musicxml_text(document)).find("part/measure")
    notes = measure.findall("note")
    written = [
        (n.findtext("voice"), n.find("chord") is not None, n.findtext("pitch/step") + n.findtext("pitch/octave"),
         n.findtext("duration"))
        for n in notes
    ]
    # The triad is one chord in voice 1, written lowest note first; the
    # following note continues the same voice; the held C5 gets voice 2.
    assert written == [
        ("1", False, "C4", "960"),
        ("1", True, "E4", "960"),
        ("1", True, "G4", "960"),
        ("1", False, "D4", "480"),
        ("2", False, "C5", "1920"),
    ]
    # <chord/> is the first child, as MusicXML requires.
    assert [child.tag for child in notes[1]][:2] == ["chord", "pitch"]


def test_chords_tie_across_bar_lines_note_by_note() -> None:
    document = build_score_document(
        [_note("low", 1.5, 2.5, 60), _note("high", 1.5, 2.5, 64)], tempo_bpm=120.0
    )
    measures = ET.fromstring(score_to_musicxml_text(document)).findall("part/measure")
    for measure, tie in zip(measures, ("start", "stop")):
        notes = [n for n in measure.findall("note") if n.find("pitch") is not None]
        assert [n.find("chord") is not None for n in notes] == [False, True]
        assert {n.findtext("voice") for n in notes} == {"1"}
        assert all(n.find(f"tie[@type='{tie}']") is not None for n in notes)


def test_different_sources_never_share_a_chord() -> None:
    document = build_score_document(
        [_note("v", 0.0, 1.0, 64, source_kind="vocals"), _note("o", 0.0, 1.0, 60, source_kind="other")],
        tempo_bpm=120.0,
    )
    notes = ET.fromstring(score_to_musicxml_text(document)).findall("part/measure/note")
    assert all(n.find("chord") is None for n in notes)
    assert len({n.findtext("voice") for n in notes}) == 2


def _reading_note(note_id, start, end, midi, confidence=0.9):
    return {"id": note_id, "midiNote": midi, "quantizedBeat": start, "quantizedEndBeat": end,
            "confidence": confidence, "sourceKind": "other", "tab": None}


def test_reading_reduction_holds_each_chord_until_the_next_one() -> None:
    from app.score_construction import reading_reduction

    notes = [
        _reading_note("a", 0.0, 3.0, 64), _reading_note("b", 0.0, 1.5, 67), _reading_note("c", 0.0, 2.0, 72),
        _reading_note("d", 1.0, 2.0, 65),            # next chord on the upper staff
        _reading_note("low", 0.0, 4.0, 48),           # lower staff keeps its own timing
    ]
    reduced, left_out = reading_reduction(notes)
    by_id = {note["id"]: note for note in reduced}
    assert left_out == 0
    assert {by_id[key]["quantizedEndBeat"] for key in ("a", "b", "c")} == {1.0}
    assert by_id["d"]["quantizedEndBeat"] == 2.0 and by_id["low"]["quantizedEndBeat"] == 4.0
    assert notes[0]["quantizedEndBeat"] == 3.0  # the input is not changed


def test_reading_reduction_keeps_the_four_most_confident_notes_per_staff() -> None:
    from app.score_construction import reading_reduction

    notes = [_reading_note(f"n{i}", 0.0, 1.0, 60 + i, confidence=0.1 * (i + 1)) for i in range(6)]
    notes.append(_reading_note("again", 0.0, 1.0, 65, confidence=0.95))  # same pitch as n5
    reduced, left_out = reading_reduction(notes)
    assert sorted(note["id"] for note in reduced) == ["again", "n2", "n3", "n4"]
    assert left_out == 3
    everything, none_left = reading_reduction(notes, split_staves=False, max_per_chord=None)
    assert len(everything) == len(notes) and none_left == 0


def test_accompaniment_notation_reads_in_one_voice_per_staff() -> None:
    from app.score_construction import plan_score_parts

    events = [
        _note(f"o{i}", start, start + length, midi, source_kind="other")
        for i, (start, length, midi) in enumerate(
            [(0.0, 1.5, 64), (0.0, 0.75, 67), (0.0, 1.0, 72), (0.5, 1.0, 69), (1.0, 1.5, 48), (1.0, 0.5, 52)]
        )
    ]
    document = build_score_document(events, tempo_bpm=120.0)
    for measure in document["measures"]:
        for note in measure["notes"]:
            note["tab"] = None
    document["scoreParts"] = plan_score_parts([n for m in document["measures"] for n in m["notes"]])
    part = ET.fromstring(score_to_musicxml_text(document)).find("part")
    voices = {(note.findtext("staff"), note.findtext("voice")) for note in part.iter("note") if note.find("pitch") is not None}
    assert voices == {("1", "1"), ("2", "5")}
    # MIDI keeps every note with its own length.
    from midi_reader import read_midi

    midi = read_midi(score_to_midi_bytes(document))
    assert sum(1 for _t, status, _d in midi.tracks[1].channel if status & 0xF0 == 0x90) == 6
