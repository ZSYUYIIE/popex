"""Unit tests for string/fret suggestions (separate from pitch transcription)."""

from __future__ import annotations

import pytest

from app.tablature import (
    TAB_INSTRUMENTS,
    TablatureError,
    assign_tablature,
    positions_for,
    tab_position_is_consistent,
)


def note(note_id: str, midi: int, beat: float, confidence: float = 0.9) -> dict:
    return {"id": note_id, "midiNote": midi, "quantizedBeat": beat, "confidence": confidence}


def placed(result: dict, notes: list[dict]) -> list[tuple]:
    return [(result[item["id"]]["string"], result[item["id"]]["fret"]) for item in notes]


def test_standard_tunings_and_ranges() -> None:
    guitar, bass = TAB_INSTRUMENTS["guitar"], TAB_INSTRUMENTS["bass"]
    assert guitar.strings == (64, 59, 55, 50, 45, 40) and guitar.frets == 21
    assert bass.strings == (43, 38, 33, 28) and bass.frets == 20
    assert (guitar.lowest, guitar.highest) == (40, 85)
    assert (bass.lowest, bass.highest) == (28, 63)


def test_positions_cover_every_string_that_sounds_the_pitch() -> None:
    assert positions_for(64, TAB_INSTRUMENTS["guitar"]) == [(1, 0), (2, 5), (3, 9), (4, 14), (5, 19)]
    assert positions_for(39, TAB_INSTRUMENTS["guitar"]) == []


def test_scale_uses_a_compact_open_position() -> None:
    notes = [note(f"n{i}", midi, i * 0.5) for i, midi in enumerate([48, 50, 52, 53, 55, 57, 59, 60])]
    result = assign_tablature(notes, "guitar")
    assert placed(result, notes) == [(5, 3), (4, 0), (4, 2), (4, 3), (3, 0), (3, 2), (2, 0), (2, 1)]
    assert all(value["status"] == "assigned" for value in result.values())


def test_a_high_phrase_stays_in_one_hand_position() -> None:
    # C5 D5 E5 F5 G5 and back fit one five-fret window around the 12th fret.
    melody = [72, 74, 76, 77, 79, 77, 76, 74, 72]
    notes = [note(f"n{i}", midi, i * 0.5) for i, midi in enumerate(melody)]
    result = assign_tablature(notes, "guitar")
    frets = [result[item["id"]]["fret"] for item in notes]
    assert 0 not in frets
    assert max(frets) - min(frets) <= 4


def test_a_phrase_that_fits_the_open_position_is_not_shifted() -> None:
    notes = [note(f"n{i}", midi, i * 0.5) for i, midi in enumerate([62, 64, 66, 67, 69, 67, 66, 64])]
    result = assign_tablature(notes, "guitar")
    fretted = [result[item["id"]]["fret"] for item in notes if result[item["id"]]["fret"] > 0]
    assert max(fretted) - min(fretted) <= 4


def test_chords_use_distinct_strings_within_the_span() -> None:
    chord = [note(str(i), midi, 0.0) for i, midi in enumerate([40, 47, 52, 56, 59, 64])]
    result = assign_tablature(chord, "guitar")
    strings = [result[item["id"]]["string"] for item in chord]
    assert sorted(strings) == [1, 2, 3, 4, 5, 6]
    fretted = [result[item["id"]]["fret"] for item in chord if result[item["id"]]["fret"] > 0]
    assert max(fretted) - min(fretted) <= 4
    for item in chord:
        position = result[item["id"]]
        assert tab_position_is_consistent(item["midiNote"], "guitar", position["string"], position["fret"])


def test_least_confident_notes_give_way_in_an_unplayable_chord() -> None:
    chord = [note(str(i), midi, 0.0, 0.9 - i * 0.01) for i, midi in enumerate([40, 47, 52, 56, 59, 64, 67])]
    result = assign_tablature(chord, "guitar")
    assert result["6"] == {"status": "unplayable", "string": None, "fret": None}
    assert sum(value["status"] == "assigned" for value in result.values()) == 6


def test_out_of_range_notes_are_kept_without_a_position() -> None:
    notes = [note("low", 24, 0.0), note("ok", 28, 1.0), note("high", 70, 2.0)]
    result = assign_tablature(notes, "bass")
    assert result["low"]["status"] == "out_of_range"
    assert result["high"]["status"] == "out_of_range"
    assert result["ok"] == {"status": "assigned", "string": 4, "fret": 0}


def test_unison_pitches_in_one_chord_use_different_strings() -> None:
    chord = [note("a", 64, 0.0), note("b", 64, 0.0)]
    result = assign_tablature(chord, "guitar")
    assert {result["a"]["string"], result["b"]["string"]} == {1, 2}


def test_assignment_is_deterministic() -> None:
    notes = [note(f"n{i}", 45 + (i * 5) % 19, i * 0.5) for i in range(40)]
    assert assign_tablature(notes, "guitar") == assign_tablature(list(reversed(notes)), "guitar")


@pytest.mark.parametrize(
    "bad",
    [
        [note("a", 60, 0.0), note("a", 62, 1.0)],
        [{"id": "x", "midiNote": True, "quantizedBeat": 0.0, "confidence": 0.5}],
        [{"id": "x", "midiNote": 60, "quantizedBeat": float("nan"), "confidence": 0.5}],
    ],
)
def test_malformed_notes_are_rejected(bad: list[dict]) -> None:
    with pytest.raises(TablatureError):
        assign_tablature(bad, "guitar")


def test_unknown_instrument_is_rejected() -> None:
    with pytest.raises(TablatureError):
        assign_tablature([], "banjo")


def test_position_consistency_check() -> None:
    assert tab_position_is_consistent(45, "bass", 1, 2)
    assert not tab_position_is_consistent(45, "bass", 1, 3)
    assert not tab_position_is_consistent(45, "bass", 5, 0)
    assert not tab_position_is_consistent(45, "ukulele", 1, 2)
