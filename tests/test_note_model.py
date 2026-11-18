from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app import note_model, transcription_pipeline
from app.note_model import (
    MODEL_PATH,
    MODEL_SHA256,
    NOTE_MODEL_VERSION,
    NoteModelError,
    activations_to_notes,
    one_line,
    transcribe_notes,
)

pytest.importorskip("onnxruntime")

SR = note_model.SAMPLE_RATE


def tone(midi: int, seconds: float, level: float = 0.3) -> np.ndarray:
    """A plucked synthetic tone with harmonics. No recording is involved."""
    time = np.arange(int(seconds * SR)) / SR
    frequency = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(np.sin(2 * np.pi * frequency * k * time) / k for k in (1, 2, 3))
    envelope = np.exp(-2.0 * time) * np.minimum(1.0, time / 0.005)
    return (level * wave * envelope).astype("float32")


def sequence(notes: list[list[int]], seconds: float = 0.5) -> np.ndarray:
    """Each entry is a list of MIDI notes sounding together for ``seconds``."""
    parts = []
    for chord in notes:
        part = np.zeros(int(seconds * SR), dtype="float32")
        for midi in chord:
            part += tone(midi, seconds) / max(1, len(chord)) ** 0.5
        parts.append(part)
    return np.concatenate(parts)


def write(tmp_path: Path, audio: np.ndarray, name: str = "stem.wav", rate: int = SR) -> Path:
    path = tmp_path / name
    sf.write(path, audio, rate)
    return path


def test_bundled_model_is_intact_and_licensed():
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    folder = MODEL_PATH.parent
    assert "Apache License" in (folder / "LICENSE").read_text(encoding="utf-8")
    assert "Spotify AB" in (folder / "NOTICE").read_text(encoding="utf-8")
    assert note_model.note_model_available()


def test_a_bass_line_is_found_as_one_line(tmp_path):
    line = [43, 45, 47, 48, 50, 48, 47, 45]  # G2 A2 B2 C3 D3 C3 B2 A2
    result = transcribe_notes(write(tmp_path, sequence([[m] for m in line])), source_kind="bass")

    assert result["algorithmVersion"] == NOTE_MODEL_VERSION
    assert result["sourceKind"] == "bass"
    pitches = [event["midiNote"] for event in result["events"]]
    assert pitches == line
    starts = [event["startSeconds"] for event in result["events"]]
    assert starts == pytest.approx([index * 0.5 for index in range(8)], abs=0.04)
    for event in result["events"]:
        assert 0.0 <= event["confidence"] <= 1.0
        assert 1 <= event["velocity"] <= 127
        assert event["endSeconds"] > event["startSeconds"]
        assert event["id"].startswith("p")


def test_the_accompaniment_keeps_its_chords(tmp_path):
    chords = [[60, 64, 67], [57, 60, 64], [55, 59, 62], [53, 57, 60]]  # C, Am, G, F
    result = transcribe_notes(write(tmp_path, sequence(chords, 0.75)), source_kind="other")
    found = {}
    for event in result["events"]:
        found.setdefault(round(event["startSeconds"] / 0.75), set()).add(event["midiNote"])
    for index, chord in enumerate(chords):
        assert set(chord) <= found.get(index, set()), (index, found.get(index))
    assert any("several notes at once" in warning for warning in result["warnings"])


def test_vocals_read_as_one_line_even_with_harmony(tmp_path):
    # A doubled harmony a third below each melody note.
    melody = [67, 69, 71, 72]
    audio = sequence([[m, m - 4] for m in melody])
    result = transcribe_notes(write(tmp_path, audio), source_kind="vocals")
    starts = [round(event["startSeconds"], 1) for event in result["events"]]
    assert len(starts) == len(set(starts))  # never two notes at one moment


def test_resampled_input_gives_the_same_pitches(tmp_path):
    line = [43, 47, 50, 55]
    audio = sequence([[m] for m in line])
    import librosa

    high_rate = librosa.resample(audio, orig_sr=SR, target_sr=44100)
    result = transcribe_notes(write(tmp_path, high_rate, rate=44100), source_kind="bass")
    assert [event["midiNote"] for event in result["events"]] == line


def test_silence_gives_no_notes_and_says_why(tmp_path):
    result = transcribe_notes(write(tmp_path, np.zeros(SR * 2, dtype="float32")), source_kind="bass")
    assert result["events"] == []
    assert any("silent" in warning for warning in result["warnings"])


def test_only_separated_stems_are_accepted(tmp_path):
    path = write(tmp_path, sequence([[60]]))
    with pytest.raises(NoteModelError):
        transcribe_notes(path, source_kind="full_mix")
    with pytest.raises(NoteModelError):
        transcribe_notes(tmp_path / "missing.wav", source_kind="bass")


def test_a_tampered_model_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "nmp.onnx"
    fake.write_bytes(b"not a model")
    monkeypatch.setattr(note_model, "MODEL_PATH", fake)
    monkeypatch.setattr(note_model, "_session", None)
    assert not note_model.note_model_available()
    with pytest.raises(NoteModelError, match="integrity"):
        note_model.model_activations(np.zeros(SR, dtype="float32"))
    monkeypatch.setattr(note_model, "_session", None)


def test_one_line_keeps_the_strongest_overlapping_note():
    notes = [(0.0, 1.0, 60, 0.4), (0.1, 0.9, 64, 0.9), (1.0, 2.0, 62, 0.5)]
    assert one_line(notes) == [(0.1, 0.9, 64, 0.9), (1.0, 2.0, 62, 0.5)]


def test_activations_decode_to_notes_and_respect_the_frequency_range():
    frames = np.zeros((200, 88))
    onsets = np.zeros((200, 88))
    frames[20:80, 40] = 0.9  # MIDI 61
    onsets[20, 40] = 0.9
    frames[100:160, 5] = 0.8  # MIDI 26, below 40 Hz
    onsets[100, 5] = 0.8
    notes = activations_to_notes(
        {"note": frames, "onset": onsets},
        onset_threshold=0.5,
        frame_threshold=0.3,
        min_note_frames=5,
        min_hz=40.0,
        max_hz=None,
    )
    assert [note[2] for note in notes] == [61]
    assert notes[0][3] == pytest.approx(0.9)
    assert notes[0][0] == pytest.approx(20 * 256 / SR, abs=0.02)


def test_dense_stems_keep_only_the_strongest_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(note_model, "_MAX_NOTES_PER_SOURCE", 3)
    chords = [[60, 64, 67], [57, 60, 64]]
    result = transcribe_notes(write(tmp_path, sequence(chords, 0.75)), source_kind="other")
    assert len(result["events"]) == 3
    assert any("strongest are kept" in warning for warning in result["warnings"])


# ---------------------------------------------------------------------------
# Choice of detector in the transcription pipeline


def test_stems_use_the_note_model_and_the_full_mix_uses_pyin(tmp_path):
    path = write(tmp_path, sequence([[43], [47], [50], [55]]))
    stem = transcription_pipeline.transcribe_pitched_source(path, source_kind="bass")
    mix = transcription_pipeline.transcribe_pitched_source(path, source_kind="full_mix")
    assert stem["algorithmVersion"] == NOTE_MODEL_VERSION
    assert mix["algorithmVersion"] == "baseline-pyin-v1"


def test_the_pipeline_falls_back_to_pyin_and_says_so(tmp_path, monkeypatch):
    path = write(tmp_path, sequence([[43], [47], [50], [55]]))

    def broken(*args, **kwargs):
        raise NoteModelError("boom")

    monkeypatch.setattr(transcription_pipeline, "transcribe_notes", broken)
    result = transcription_pipeline.transcribe_pitched_source(path, source_kind="bass")
    assert result["algorithmVersion"] == "baseline-pyin-v1"
    assert result["warnings"][0].startswith("The note model could not process this stem")

    monkeypatch.setattr(transcription_pipeline, "note_model_available", lambda: False)
    result = transcription_pipeline.transcribe_pitched_source(path, source_kind="vocals")
    assert result["warnings"][0].startswith("The note model is unavailable")


def test_note_model_results_pass_raw_transcription_validation(tmp_path):
    path = write(tmp_path, sequence([[60, 64, 67], [57, 60, 64]], 0.75))
    result = transcribe_notes(path, source_kind="other")
    output = transcription_pipeline._detector_output(
        result, "other", "pitched", "baseline-librosa-v1", input_origin="stem"
    )
    assert len(output.events) == len(result["events"])
    assert output.algorithm_version == NOTE_MODEL_VERSION
