"""Note detection with a small neural network (Basic Pitch, ICASSP 2022).

The first note detector, a pYIN pitch tracker, follows one pitch at a time. On
a bass stem it missed most notes, and on the accompaniment stem it turned each
chord into one note. This module runs Spotify's Basic Pitch model, which hears
several notes at once, on the separated stems:

- bass and vocals are reduced to one line (the strongest note wherever notes
  overlap), because each is one voice;
- the accompaniment keeps its chords;
- every note keeps the model's own activation as its confidence.

The model (``app/models/basic_pitch/nmp.onnx``, 230 KB) and the inference and
note-creation steps below come from https://github.com/spotify/basic-pitch,
Copyright 2022 Spotify AB, licensed under the Apache License 2.0; see
``app/models/basic_pitch/LICENSE`` and ``NOTICE``. The code here is a reduced
port of ``basic_pitch/inference.py`` and ``basic_pitch/note_creation.py``
(version 0.4.0): pitch bends and MIDI writing are left out. It runs with
ONNX Runtime on the CPU and downloads nothing.
"""

from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

NOTE_MODEL_VERSION = "basic-pitch-icassp2022-v1"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "basic_pitch" / "nmp.onnx"
MODEL_SHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"

# Constants from basic_pitch/constants.py.
SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_WINDOW_LENGTH = 2
ANNOTATIONS_FPS = SAMPLE_RATE // FFT_HOP
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
AUDIO_N_SAMPLES = SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
_N_OVERLAPPING_FRAMES = 30
_OVERLAP_LEN = _N_OVERLAPPING_FRAMES * FFT_HOP
_HOP_SIZE = AUDIO_N_SAMPLES - _OVERLAP_LEN
_ENERGY_TOLERANCE = 11

# Per-stem settings: frequency range, thresholds, shortest note, one line or chords.
_SOURCE_SETTINGS: dict[str, dict[str, Any]] = {
    "bass": {"min_hz": 30.0, "max_hz": 400.0, "onset": 0.5, "frame": 0.3, "min_ms": 80.0, "one_line": True},
    "vocals": {"min_hz": 80.0, "max_hz": 1200.0, "onset": 0.5, "frame": 0.3, "min_ms": 100.0, "one_line": True},
    "other": {"min_hz": 60.0, "max_hz": 2100.0, "onset": 0.5, "frame": 0.3, "min_ms": 80.0, "one_line": False},
}
MODEL_SOURCE_KINDS = frozenset(_SOURCE_SETTINGS)

# Without stems, the full mix is heard once and split by register into three
# estimated lines. They are labelled as lines of the mix, never as stems.
MIX_LINE_KINDS = ("mix_melody", "mix_bass", "mix_chords")
_MIX_SETTINGS = {"min_hz": 30.0, "max_hz": 2100.0, "onset": 0.5, "frame": 0.3, "min_ms": 80.0}
MIX_BASS_MAX_MIDI = 52  # E3: the lowest line is taken from notes at or below it
MIX_MELODY_MIN_MIDI = 55  # G3: the top line is taken from notes at or above it
MIX_LINE_MIN_SECONDS = 0.100
# Two notes clash in a line when they overlap by more than a quarter of the shorter one.
MIX_LINE_OVERLAP_SHARE = 0.25
# A note inside a stronger one (starting 20 ms or more later, under 60% of its
# strength) is not a line candidate; it stays in the chords line.
MIX_GHOST_LEAD_SECONDS = 0.020
MIX_GHOST_STRENGTH_SHARE = 0.6
_mix_cache_lock = threading.Lock()
_mix_cache: tuple[tuple[str, int, int], list[tuple[float, float, int, float]], float] | None = None
# The score holds at most 5000 notes; a dense accompaniment keeps its strongest.
_MAX_NOTES_PER_SOURCE = 2400
# All three full-mix lines together stay within the score's note limit.
_MAX_MIX_NOTES = 4800
_MAX_DURATION_SECONDS = 1800.0
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_session_lock = threading.Lock()
_session: Any = None


class NoteModelError(RuntimeError):
    """The note model is unavailable or could not process the audio."""


def note_model_available() -> bool:
    """True when ONNX Runtime imports and the bundled model is intact."""
    try:
        _get_session()
    except NoteModelError:
        return False
    return True


def _get_session() -> Any:
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise NoteModelError("ONNX Runtime is not installed.") from exc
        try:
            data = MODEL_PATH.read_bytes()
        except OSError as exc:
            raise NoteModelError("The bundled note model is missing.") from exc
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise NoteModelError("The bundled note model failed its integrity check.")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        try:
            _session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise NoteModelError("The note model could not be loaded.") from exc
        return _session


# ---------------------------------------------------------------------------
# Model inference (port of basic_pitch.inference)


def model_activations(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Frame-level ``note``, ``onset`` and ``contour`` activations for mono 22050 Hz audio."""
    session = _get_session()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    original_length = audio.shape[0]
    padded = np.concatenate([np.zeros(_OVERLAP_LEN // 2, dtype=np.float32), audio])
    outputs: dict[str, list[np.ndarray]] = {"note": [], "onset": [], "contour": []}
    for start in range(0, padded.shape[0], _HOP_SIZE):
        window = padded[start : start + AUDIO_N_SAMPLES]
        if window.shape[0] < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - window.shape[0]))
        batch = window.reshape(1, AUDIO_N_SAMPLES, 1)
        contour, note, onset = session.run(
            ["StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
            {"serving_default_input_2:0": batch},
        )
        outputs["note"].append(note)
        outputs["onset"].append(onset)
        outputs["contour"].append(contour)
    trim = _N_OVERLAPPING_FRAMES // 2
    frames = int(math.floor(original_length * (ANNOTATIONS_FPS / SAMPLE_RATE)))
    result = {}
    for key, chunks in outputs.items():
        stacked = np.concatenate(chunks)[:, trim:-trim, :]
        result[key] = stacked.reshape(-1, stacked.shape[-1])[:frames, :]
    return result


# ---------------------------------------------------------------------------
# Note creation (port of basic_pitch.note_creation, without pitch bends)


def _hz_to_midi(hz: float) -> float:
    return 12.0 * math.log2(hz / 440.0) + 69.0


def _infer_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(appended[n:, :] - appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    peak = np.max(frame_diff)
    if peak > 0:
        frame_diff = np.max(onsets) * frame_diff / peak
    return np.max([onsets, frame_diff], axis=0)


def _frame_times(n_frames: int) -> np.ndarray:
    original = np.arange(n_frames) * FFT_HOP / SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / SAMPLE_RATE) * (ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)) + 0.0018
    return original - window_offset * window_numbers


def activations_to_notes(
    activations: dict[str, np.ndarray],
    *,
    onset_threshold: float,
    frame_threshold: float,
    min_note_frames: int,
    min_hz: float | None,
    max_hz: float | None,
) -> list[tuple[float, float, int, float]]:
    """Decode activations into ``(start_s, end_s, midi, amplitude)`` notes."""
    import scipy.signal

    frames = np.array(activations["note"], dtype=np.float64)
    onsets = np.array(activations["onset"], dtype=np.float64)
    n_frames = frames.shape[0]
    if n_frames < 3:
        return []
    if max_hz is not None:
        index = int(round(_hz_to_midi(max_hz) - MIDI_OFFSET))
        onsets[:, max(0, index):] = 0
        frames[:, max(0, index):] = 0
    if min_hz is not None:
        index = int(round(_hz_to_midi(min_hz) - MIDI_OFFSET))
        onsets[:, : max(0, index)] = 0
        frames[:, : max(0, index)] = 0
    onsets = _infer_onsets(onsets, frames)

    peak_matrix = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_matrix[peaks] = onsets[peaks]
    onset_time, onset_freq = np.where(peak_matrix >= onset_threshold)
    remaining = frames.copy()
    found: list[tuple[int, int, int, float]] = []
    for start, freq in zip(onset_time[::-1], onset_freq[::-1]):
        if start >= n_frames - 1:
            continue
        i, below = start + 1, 0
        while i < n_frames - 1 and below < _ENERGY_TOLERANCE:
            below = below + 1 if remaining[i, freq] < frame_threshold else 0
            i += 1
        i -= below
        if i - start <= min_note_frames:
            continue
        remaining[start:i, freq] = 0
        if freq < MAX_FREQ_IDX:
            remaining[start:i, freq + 1] = 0
        if freq > 0:
            remaining[start:i, freq - 1] = 0
        found.append((int(start), int(i), int(freq) + MIDI_OFFSET, float(np.mean(frames[start:i, freq]))))

    # "Melodia trick": notes that sustain without a clear onset.
    while np.max(remaining) > frame_threshold:
        middle, freq = np.unravel_index(np.argmax(remaining), remaining.shape)
        remaining[middle, freq] = 0
        i, below = middle + 1, 0
        while i < n_frames - 1 and below < _ENERGY_TOLERANCE:
            below = below + 1 if remaining[i, freq] < frame_threshold else 0
            remaining[i, freq] = 0
            if freq < MAX_FREQ_IDX:
                remaining[i, freq + 1] = 0
            if freq > 0:
                remaining[i, freq - 1] = 0
            i += 1
        end = i - 1 - below
        i, below = middle - 1, 0
        while i > 0 and below < _ENERGY_TOLERANCE:
            below = below + 1 if remaining[i, freq] < frame_threshold else 0
            remaining[i, freq] = 0
            if freq < MAX_FREQ_IDX:
                remaining[i, freq + 1] = 0
            if freq > 0:
                remaining[i, freq - 1] = 0
            i -= 1
        start = i + 1 + below
        if end - start <= min_note_frames:
            continue
        found.append((int(start), int(end), int(freq) + MIDI_OFFSET, float(np.mean(frames[start:end, freq]))))

    times = _frame_times(n_frames)
    notes = []
    for start, end, midi, amplitude in found:
        start_s, end_s = float(times[start]), float(times[min(end, n_frames - 1)])
        if end_s > start_s and math.isfinite(amplitude):
            notes.append((max(0.0, start_s), end_s, midi, float(min(1.0, max(0.0, amplitude)))))
    notes.sort(key=lambda note: (note[0], note[2]))
    return notes


def one_line(notes: list[tuple[float, float, int, float]], *, tolerance: float = 0.03) -> list[tuple[float, float, int, float]]:
    """Keep the strongest note wherever notes overlap, so one voice reads as one line."""
    kept: list[tuple[float, float, int, float]] = []
    for note in sorted(notes, key=lambda item: (-item[3], item[0], item[2])):
        start, end = note[0], note[1]
        if all(end - tolerance <= other[0] or start + tolerance >= other[1] for other in kept):
            kept.append(note)
    return sorted(kept, key=lambda item: (item[0], item[2]))


def split_mix_lines(
    notes: list[tuple[float, float, int, float]], *, tolerance: float = 0.03
) -> dict[str, list[tuple[float, float, int, float]]]:
    """Split full-mix notes into a lowest line, a top line and the chords between.

    The lowest line keeps the lowest note wherever notes at or below E3
    overlap (the loudest low note is often the bass's octave overtone). The
    top line keeps the highest note wherever the remaining notes at or above
    G3 overlap. Notes overlap when they share more than a quarter of the
    shorter note; a smaller legato overlap keeps both, and the earlier note is
    shortened to end where the next begins. Both lines ignore notes shorter
    than 100 ms and weak notes that start inside a much stronger note (often
    its overtone or undertone). Every other note stays in the chords line, so
    no detected note is dropped.
    """

    def clash(first: tuple[float, float, int, float], second: tuple[float, float, int, float]) -> bool:
        overlap = min(first[1], second[1]) - max(first[0], second[0])
        shorter = min(first[1] - first[0], second[1] - second[0])
        return overlap > max(tolerance, MIX_LINE_OVERLAP_SHARE * shorter)

    def ghost(index: int, candidates: list[int]) -> bool:
        # A weak note that starts inside a much stronger note and ends with it
        # is usually that note's overtone or undertone, not a line of its own.
        note = notes[index]
        return any(
            other != index
            and notes[other][0] <= note[0] - MIX_GHOST_LEAD_SECONDS
            and notes[other][1] >= note[1] - tolerance
            and note[3] < MIX_GHOST_STRENGTH_SHARE * notes[other][3]
            for other in candidates
        )

    def single_line(candidates: list[int], key) -> list[int]:
        candidates = [index for index in candidates if not ghost(index, candidates)]
        kept: list[int] = []
        for index in sorted(candidates, key=key):
            if not any(clash(notes[index], notes[other]) for other in kept):
                kept.append(index)
        return kept

    def trimmed(indices: list[int]) -> list[tuple[float, float, int, float]]:
        # A legato line keeps both notes; the earlier one ends where the next begins.
        line = sorted((notes[index] for index in indices), key=lambda note: (note[0], note[2]))
        for position in range(len(line) - 1):
            start, end, midi, amplitude = line[position]
            following = line[position + 1][0]
            if end > following:
                line[position] = (start, max(start + 0.02, following), midi, amplitude)
        return line

    long_enough = [index for index, note in enumerate(notes) if note[1] - note[0] >= MIX_LINE_MIN_SECONDS]
    bass = single_line(
        [index for index in long_enough if notes[index][2] <= MIX_BASS_MAX_MIDI],
        key=lambda index: (notes[index][2], -notes[index][3], notes[index][0], index),
    )
    taken = set(bass)
    melody = single_line(
        [index for index in long_enough if index not in taken and notes[index][2] >= MIX_MELODY_MIN_MIDI],
        key=lambda index: (-notes[index][2], -notes[index][3], notes[index][0], index),
    )
    taken.update(melody)
    chords = [index for index in range(len(notes)) if index not in taken]
    order = lambda note: (note[0], note[2])  # noqa: E731
    return {
        "mix_melody": trimmed(melody),
        "mix_bass": trimmed(bass),
        "mix_chords": sorted((notes[index] for index in chords), key=order),
    }


def _full_mix_notes(path: Path) -> tuple[list[tuple[float, float, int, float]], float]:
    """Run the model once per full-mix file and reuse it for all three lines."""
    global _mix_cache
    try:
        info = path.stat()
    except OSError as exc:
        raise NoteModelError("The audio could not be read for note detection.") from exc
    key = (str(path.resolve()), info.st_mtime_ns, info.st_size)
    with _mix_cache_lock:
        if _mix_cache is not None and _mix_cache[0] == key:
            return list(_mix_cache[1]), _mix_cache[2]
    audio = _read_audio(path)
    duration = audio.shape[0] / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if audio.size < SAMPLE_RATE // 4 or rms <= 1e-5:
        notes: list[tuple[float, float, int, float]] = []
    else:
        notes = activations_to_notes(
            model_activations(audio),
            onset_threshold=_MIX_SETTINGS["onset"],
            frame_threshold=_MIX_SETTINGS["frame"],
            min_note_frames=int(round(_MIX_SETTINGS["min_ms"] / 1000.0 * (SAMPLE_RATE / FFT_HOP))),
            min_hz=_MIX_SETTINGS["min_hz"],
            max_hz=_MIX_SETTINGS["max_hz"],
        )
    with _mix_cache_lock:
        _mix_cache = (key, list(notes), duration)
    return notes, duration


# ---------------------------------------------------------------------------
# Detector with the same result contract as app.pitch_transcription


def _read_audio(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise NoteModelError("The audio could not be read for note detection.") from exc
    mono = samples.mean(axis=1)
    if not np.all(np.isfinite(mono)):
        raise NoteModelError("The audio contains invalid samples.")
    if mono.shape[0] / rate > _MAX_DURATION_SECONDS:
        raise NoteModelError("The audio is longer than note detection supports.")
    if rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE)
    return mono.astype(np.float32, copy=False)


def transcribe_notes(audio_path: Path | str, *, source_kind: str, algorithm_version: str = NOTE_MODEL_VERSION) -> dict[str, Any]:
    """Detect notes on one separated stem and return raw pitched events."""
    if source_kind not in _SOURCE_SETTINGS:
        raise NoteModelError("The note model only runs on the vocal, bass and accompaniment stems.")
    settings = _SOURCE_SETTINGS[source_kind]
    audio = _read_audio(Path(audio_path))
    warnings: list[str] = []
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    diagnostics: dict[str, Any] = {
        "model": {
            "version": algorithm_version,
            "sha256": MODEL_SHA256,
            "sampleRate": SAMPLE_RATE,
            "minHz": settings["min_hz"],
            "maxHz": settings["max_hz"],
            "onsetThreshold": settings["onset"],
            "frameThreshold": settings["frame"],
            "minNoteMs": settings["min_ms"],
            "oneLine": settings["one_line"],
        },
        "eventing": {"eventEvidence": []},
    }
    if audio.size < SAMPLE_RATE // 4 or rms <= 1e-5:
        warnings.append("The stem is silent or too quiet for note detection.")
        return {"algorithmVersion": algorithm_version, "sourceKind": source_kind, "events": [], "warnings": warnings, "diagnostics": diagnostics}

    activations = model_activations(audio)
    notes = activations_to_notes(
        activations,
        onset_threshold=settings["onset"],
        frame_threshold=settings["frame"],
        min_note_frames=int(round(settings["min_ms"] / 1000.0 * (SAMPLE_RATE / FFT_HOP))),
        min_hz=settings["min_hz"],
        max_hz=settings["max_hz"],
    )
    detected = len(notes)
    if settings["one_line"]:
        notes = one_line(notes)
        if detected > len(notes):
            warnings.append(
                f"{detected - len(notes)} overlapping note(s) were left out so the {source_kind} reads as one line."
            )
    if len(notes) > _MAX_NOTES_PER_SOURCE:
        strongest = sorted(notes, key=lambda note: -note[3])[:_MAX_NOTES_PER_SOURCE]
        warnings.append(
            f"The {source_kind} stem gave {len(notes)} notes; the {_MAX_NOTES_PER_SOURCE} strongest are kept."
        )
        notes = sorted(strongest, key=lambda note: (note[0], note[2]))
    if not settings["one_line"]:
        warnings.append("Chords in the accompaniment are kept as several notes at once; the voicing needs review.")

    duration = audio.shape[0] / SAMPLE_RATE
    events = []
    evidence = []
    for index, (start, end, midi, amplitude) in enumerate(notes):
        if not 0 <= midi <= 127:
            continue
        event_id = f"p{index + 1:06d}"
        event_warnings = ["This note has a weak model activation."] if amplitude < 0.45 else []
        events.append(
            {
                "id": event_id,
                "sourceKind": source_kind,
                "startSeconds": round(start, 4),
                "endSeconds": round(min(duration, max(end, start + 0.02)), 4),
                "midiNote": int(midi),
                "midiPitch": float(midi),
                "frequencyHz": float(440.0 * 2 ** ((midi - 69) / 12)),
                "noteName": f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}",
                "confidence": round(amplitude, 4),
                "velocity": int(min(127, max(1, round(1 + 126 * math.sqrt(amplitude))))),
                "warnings": event_warnings,
            }
        )
        evidence.append({"eventId": event_id, "modelActivation": round(amplitude, 4)})
    diagnostics["eventing"]["eventEvidence"] = evidence
    diagnostics["model"].update(detectedNotes=detected, keptNotes=len(events))
    if not events:
        warnings.append("No notes passed the note model's thresholds on this stem.")
    return {"algorithmVersion": algorithm_version, "sourceKind": source_kind, "events": events, "warnings": warnings, "diagnostics": diagnostics}


def _events_from_notes(
    notes: list[tuple[float, float, int, float]], source_kind: str, duration: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = []
    evidence = []
    for index, (start, end, midi, amplitude) in enumerate(notes):
        if not 0 <= midi <= 127:
            continue
        event_id = f"p{index + 1:06d}"
        event_warnings = ["This note has a weak model activation."] if amplitude < 0.45 else []
        events.append(
            {
                "id": event_id,
                "sourceKind": source_kind,
                "startSeconds": round(start, 4),
                "endSeconds": round(min(duration, max(end, start + 0.02)), 4),
                "midiNote": int(midi),
                "midiPitch": float(midi),
                "frequencyHz": float(440.0 * 2 ** ((midi - 69) / 12)),
                "noteName": f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}",
                "confidence": round(amplitude, 4),
                "velocity": int(min(127, max(1, round(1 + 126 * math.sqrt(amplitude))))),
                "warnings": event_warnings,
            }
        )
        evidence.append({"eventId": event_id, "modelActivation": round(amplitude, 4)})
    return events, evidence


_MIX_LINE_WARNINGS = {
    "mix_melody": (
        "Without separated stems, the melody is the top line of the full mix: the highest "
        "note at or above G3 wherever notes overlap. It may follow a harmony or a chord's top note."
    ),
    "mix_bass": (
        "Without separated stems, the bass line is the lowest line of the full mix: the lowest "
        "note at or below E3 wherever notes overlap. Low chord notes can appear in it."
    ),
    "mix_chords": (
        "Without separated stems, the chords line holds every other note heard in the full mix; "
        "voicings mix several instruments and need review."
    ),
}


def transcribe_mix_lines(audio_path: Path | str, *, algorithm_version: str = NOTE_MODEL_VERSION) -> list[dict[str, Any]]:
    """Detect notes in the full mix once and return the three estimated lines."""
    notes, duration = _full_mix_notes(Path(audio_path))
    lines = split_mix_lines(notes)
    results = []
    kept_total = 0
    for source_kind in MIX_LINE_KINDS:
        line_notes = lines[source_kind]
        warnings = [_MIX_LINE_WARNINGS[source_kind]]
        # The chords line comes last and gets what the score has room for.
        limit = min(_MAX_NOTES_PER_SOURCE, max(0, _MAX_MIX_NOTES - kept_total))
        if len(line_notes) > limit:
            warnings.append(
                f"The {source_kind.replace('mix_', '')} line gave {len(line_notes)} notes; "
                f"the {limit} strongest are kept."
            )
            line_notes = sorted(
                sorted(line_notes, key=lambda note: -note[3])[:limit],
                key=lambda note: (note[0], note[2]),
            )
        kept_total += len(line_notes)
        events, evidence = _events_from_notes(line_notes, source_kind, duration)
        if not events:
            warnings.append("No notes passed the note model's thresholds for this line of the mix.")
        results.append(
            {
                "algorithmVersion": algorithm_version,
                "sourceKind": source_kind,
                "events": events,
                "warnings": warnings,
                "diagnostics": {
                    "model": {
                        "version": algorithm_version,
                        "sha256": MODEL_SHA256,
                        "sampleRate": SAMPLE_RATE,
                        "minHz": _MIX_SETTINGS["min_hz"],
                        "maxHz": _MIX_SETTINGS["max_hz"],
                        "onsetThreshold": _MIX_SETTINGS["onset"],
                        "frameThreshold": _MIX_SETTINGS["frame"],
                        "minNoteMs": _MIX_SETTINGS["min_ms"],
                        "oneLine": source_kind != "mix_chords",
                        "detectedNotes": len(notes),
                        "keptNotes": len(events),
                    },
                    "eventing": {"eventEvidence": evidence},
                },
            }
        )
    return results


__all__ = [
    "MIX_BASS_MAX_MIDI",
    "MIX_LINE_KINDS",
    "MIX_MELODY_MIN_MIDI",
    "MODEL_SOURCE_KINDS",
    "NOTE_MODEL_VERSION",
    "NoteModelError",
    "activations_to_notes",
    "model_activations",
    "note_model_available",
    "one_line",
    "split_mix_lines",
    "transcribe_mix_lines",
    "transcribe_notes",
]
