"""Full-mix note lines when no separated stems exist (cycle 38)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app import note_model, transcription_pipeline
from app.note_model import (
    MIX_BASS_MAX_MIDI,
    MIX_LINE_KINDS,
    MIX_MELODY_MIN_MIDI,
    NOTE_MODEL_VERSION,
    NoteModelError,
    split_mix_lines,
)
from app.score_construction import SCORE_PARTS, plan_score_parts

SR = note_model.SAMPLE_RATE


def _tone(midi: int, seconds: float, level: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(level / k * np.sin(2 * np.pi * freq * k * t) for k in (1, 2, 3))
    fade = min(len(t) // 10, int(0.01 * SR))
    envelope = np.ones_like(t)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (wave * envelope).astype(np.float32)


def _mix(path: Path) -> Path:
    """Four beats: a bass note, a held chord and a melody above both."""
    bass = [40, 45, 43, 38]  # E2 A2 G2 D2
    chords = [[59, 64], [57, 64], [59, 62], [57, 62]]  # B3-E4 ... inside the middle
    melody = [76, 72, 74, 71]  # E5 C5 D5 B4
    beat = 0.6
    pieces = []
    for low, chord, high in zip(bass, chords, melody):
        block = _tone(low, beat, 0.25) + _tone(high, beat, 0.2)
        for pitch in chord:
            block = block + _tone(pitch, beat, 0.12)
        pieces.append(block)
    audio = np.concatenate(pieces)
    audio = audio / max(1.0, float(np.max(np.abs(audio))) / 0.9)
    target = path / "mix.wav"
    sf.write(target, audio, SR)
    return target


# ---------------------------------------------------------------------------
# The split rule


def test_split_keeps_the_lowest_and_highest_lines_and_every_other_note():
    notes = [
        (0.0, 0.5, 40, 0.6),  # bass
        (0.0, 0.5, 52, 0.9),  # louder octave overtone: lowest still wins
        (0.0, 0.5, 60, 0.5),  # chord tone
        (0.0, 0.5, 64, 0.5),  # chord tone
        (0.0, 0.5, 72, 0.4),  # melody: highest wins even when quieter
        (0.5, 1.0, 43, 0.6),
        (0.5, 1.0, 74, 0.7),
    ]
    lines = split_mix_lines(notes)
    assert [note[2] for note in lines["mix_bass"]] == [40, 43]
    assert [note[2] for note in lines["mix_melody"]] == [72, 74]
    assert sorted(note[2] for note in lines["mix_chords"]) == [52, 60, 64]
    assert sorted(note for line in lines.values() for note in line) == sorted(notes)


def test_split_leaves_short_and_middle_register_notes_in_the_chords_line():
    notes = [
        (0.0, 0.05, 36, 0.9),  # too short for the bass line
        (0.0, 0.05, 84, 0.9),  # too short for the top line
        (0.0, 0.5, MIX_BASS_MAX_MIDI + 1, 0.9),  # F3: between the two ranges
        (0.0, 0.5, MIX_MELODY_MIN_MIDI - 1, 0.9),  # F#3
    ]
    lines = split_mix_lines(notes)
    assert lines["mix_bass"] == [] and lines["mix_melody"] == []
    assert sorted(lines["mix_chords"]) == sorted(notes)


def test_a_small_legato_overlap_keeps_both_notes_and_trims_the_first():
    lines = split_mix_lines([(0.0, 0.63, 40, 0.6), (0.58, 1.2, 45, 0.7), (0.3, 0.9, 38, 0.6)])
    assert [note[2] for note in lines["mix_bass"]] == [38]  # 38 clashes with both
    lines = split_mix_lines([(0.0, 0.63, 40, 0.6), (0.58, 1.2, 45, 0.7)])
    assert lines["mix_bass"] == [(0.0, 0.58, 40, 0.6), (0.58, 1.2, 45, 0.7)]
    assert lines["mix_chords"] == []


def test_a_weak_note_inside_a_strong_one_is_not_a_line_note():
    strong = (0.0, 0.6, 43, 0.76)
    undertone = (0.2, 0.35, 31, 0.36)  # starts inside, ends inside, under 60%
    overtone = (0.2, 0.35, 93, 0.3)
    melody = (0.0, 0.6, 74, 0.67)
    lines = split_mix_lines([strong, undertone, overtone, melody])
    assert lines["mix_bass"] == [strong]
    assert lines["mix_melody"] == [melody]
    assert sorted(lines["mix_chords"]) == sorted([undertone, overtone])
    # A note of similar strength inside it still wins the lowest line.
    rival = (0.2, 0.35, 31, 0.6)
    assert split_mix_lines([strong, rival])["mix_bass"] == [rival]


def test_split_never_drops_identical_notes():
    note = (0.0, 0.5, 40, 0.5)
    lines = split_mix_lines([note, note])
    assert lines["mix_bass"] == [note]
    assert lines["mix_chords"] == [note]


def test_split_lines_have_no_overlapping_notes():
    rng = np.random.default_rng(7)
    notes = []
    for _ in range(300):
        start = float(rng.uniform(0, 20))
        notes.append((start, start + float(rng.uniform(0.05, 1.5)), int(rng.integers(28, 96)), float(rng.uniform(0.3, 1))))
    lines = split_mix_lines(notes)
    for kind in ("mix_bass", "mix_melody"):
        line = lines[kind]
        for earlier, later in zip(line, line[1:]):
            assert later[0] >= earlier[1] - 1e-9 or earlier[1] - earlier[0] <= 0.02 + 1e-9
    assert all(note[2] <= MIX_BASS_MAX_MIDI for note in lines["mix_bass"])
    assert all(note[2] >= MIX_MELODY_MIN_MIDI for note in lines["mix_melody"])
    assert sum(len(line) for line in lines.values()) == len(notes)


# ---------------------------------------------------------------------------
# The model on a synthetic full mix


def test_the_full_mix_gives_three_honest_lines(tmp_path):
    pytest.importorskip("onnxruntime")
    results = note_model.transcribe_mix_lines(_mix(tmp_path))
    assert [item["sourceKind"] for item in results] == list(MIX_LINE_KINDS)
    by_kind = {item["sourceKind"]: item for item in results}
    bass = [event["midiNote"] for event in by_kind["mix_bass"]["events"]]
    melody = [event["midiNote"] for event in by_kind["mix_melody"]["events"]]
    # The model can hear a short ghost note an octave or a fifth below a bass
    # note; the lowest line may keep it, so the bass check counts recall.
    assert len({40, 45, 43, 38} & set(bass)) >= 3
    assert all(pitch <= MIX_BASS_MAX_MIDI for pitch in bass)
    assert melody == [76, 72, 74, 71]
    chords = {event["midiNote"] for event in by_kind["mix_chords"]["events"]}
    assert {59, 64, 57, 62} <= chords
    for item in results:
        assert item["algorithmVersion"] == NOTE_MODEL_VERSION
        assert item["warnings"][0].startswith("Without separated stems")
        assert all(event["sourceKind"] == item["sourceKind"] for event in item["events"])


def test_the_model_runs_once_for_all_three_lines(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    path = _mix(tmp_path)
    calls = []
    real = note_model.model_activations

    def counting(audio):
        calls.append(len(audio))
        return real(audio)

    monkeypatch.setattr(note_model, "_mix_cache", None)
    monkeypatch.setattr(note_model, "model_activations", counting)
    first = note_model.transcribe_mix_lines(path)
    second = note_model.transcribe_mix_lines(path)
    assert len(calls) == 1
    assert first == second


def test_silence_gives_three_empty_lines_that_say_why(tmp_path):
    pytest.importorskip("onnxruntime")
    target = tmp_path / "quiet.wav"
    sf.write(target, np.zeros(SR, dtype=np.float32), SR)
    results = note_model.transcribe_mix_lines(target)
    assert [len(item["events"]) for item in results] == [0, 0, 0]
    assert all("No notes passed" in item["warnings"][-1] for item in results)


# ---------------------------------------------------------------------------
# The transcription pipeline


def test_the_pipeline_expands_mix_lines_into_validated_sources(tmp_path):
    pytest.importorskip("onnxruntime")
    path = _mix(tmp_path)
    result = transcription_pipeline.transcribe_pitched_source(path, source_kind="full_mix")
    assert result["events"] == [] and len(result["lines"]) == 3
    outputs = transcription_pipeline._pitch_output(
        transcription_pipeline.transcribe_pitched_source,
        path,
        "full_mix",
        "baseline-librosa-v1",
        input_origin="full_mix",
    )
    assert [output.source_kind for output in outputs] == list(MIX_LINE_KINDS)
    assert all(output.algorithm_version == NOTE_MODEL_VERSION for output in outputs)
    assert sum(len(output.events) for output in outputs) >= 12


def test_malformed_mix_lines_fail_the_source(tmp_path):
    def wrong(path, *, source_kind):
        return {"sourceKind": "full_mix", "events": [], "lines": [{"sourceKind": "vocals"}]}

    with pytest.raises(transcription_pipeline._SourceFailure):
        transcription_pipeline._pitch_output(
            wrong, tmp_path / "mix.wav", "full_mix", "baseline-librosa-v1", input_origin="full_mix"
        )


def test_the_full_mix_falls_back_to_pyin_and_says_so(tmp_path, monkeypatch):
    path = _mix(tmp_path)

    def broken(*args, **kwargs):
        raise NoteModelError("boom")

    monkeypatch.setattr(transcription_pipeline, "note_model_available", lambda: True)
    monkeypatch.setattr(transcription_pipeline, "transcribe_mix_lines", broken)
    result = transcription_pipeline.transcribe_pitched_source(path, source_kind="full_mix")
    assert "lines" not in result
    assert result["algorithmVersion"] == "baseline-pyin-v1"
    assert result["warnings"][0].startswith("The note model could not process the full mix")

    monkeypatch.setattr(transcription_pipeline, "note_model_available", lambda: False)
    result = transcription_pipeline.transcribe_pitched_source(path, source_kind="full_mix")
    assert result["warnings"][0].startswith("The note model is unavailable")


def test_transcription_version_changed_so_old_drafts_are_redone():
    assert transcription_pipeline.TRANSCRIPTION_VERSION == "stem-notes-onset-v3"


# ---------------------------------------------------------------------------
# Score parts


def _note(kind: str, midi: int) -> dict:
    return {"sourceKind": kind, "midiNote": midi}


def test_mix_lines_get_their_own_honest_parts():
    notes = [_note("mix_melody", 72), _note("mix_melody", 76)]
    notes += [_note("mix_chords", 48), _note("mix_chords", 64)]
    notes += [_note("mix_bass", 40), _note("mix_bass", 45)]
    plan = plan_score_parts(notes)
    assert [item["id"] for item in plan] == ["melody", "accompaniment", "bass-line"]
    by_id = {item["id"]: item for item in plan}
    assert by_id["melody"]["name"] == "Melody (top line of the full mix, draft)"
    assert by_id["melody"]["clef"] == "treble"
    assert by_id["accompaniment"]["clef"] == "grand"
    assert by_id["bass-line"]["sourceKinds"] == ["mix_bass"]
    assert SCORE_PARTS["bass-line"][2] == 34


def test_stem_parts_are_unchanged():
    plan = plan_score_parts([_note("other", 60), _note("bass", 40), _note("full_mix", 70)])
    assert [(item["id"], item["sourceKinds"]) for item in plan] == [
        ("pitched-lines", ["full_mix"]),
        ("accompaniment", ["other"]),
        ("bass-line", ["bass"]),
    ]


def test_the_full_mix_lowest_line_is_the_tonal_bass_evidence():
    from app.tonal_context import _bass_notes

    def note(kind, midi, beat):
        return {"sourceKind": kind, "midiNote": midi, "quantizedBeat": beat}

    measures = [{"notes": [note("mix_chords", 31, 0.0), note("mix_bass", 36, 0.0), note("mix_melody", 72, 0.0)]}]
    line, source = _bass_notes(measures)
    assert source == "lowest line of the full mix" and [item["midiNote"] for item in line] == [36]
    measures[0]["notes"].append(note("bass", 38, 1.0))
    assert _bass_notes(measures)[1] == "bass-stem line"
