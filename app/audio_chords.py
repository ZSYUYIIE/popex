"""Chord recognition straight from the audio.

The first harmony pass names chords from the notes a single-line pitch tracker
found, which leaves most of a real recording without a chord. This module
listens to the sound instead:

- a pitch-class profile (chroma) is taken on every beat of the score's beat
  grid, from the separated accompaniment and bass stems when they exist and
  from the harmonic part of the full mix otherwise;
- where the accompaniment stems fall quiet, the full mix takes over, so a verse
  carried by voice and drums still gets its implied harmony, at lower weight;
- each beat is compared with major and minor triads on all twelve roots and
  with "no chord"; the bass supports the root, and triads inside the
  recording's overall scale get a small preference;
- one path through the beats is chosen that changes chord mainly on bar lines
  and half bars.

Results are broad on purpose: major, minor, or no chord. Sevenths, suspensions
and inversions are not claimed. Every segment carries a confidence, and weak
segments are left unnamed rather than guessed.

Recognition needs numpy and librosa, which the application already depends
on. Artifact reading and validation use the standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AUDIO_CHORDS_VERSION = "audio-chords-v1"
AUDIO_CHORDS_SCHEMA_VERSION = 1
AUDIO_CHORDS_ARTIFACT_TYPE = "popex-audio-chords"
AUDIO_CHORDS_DIRECTORY = "chords"
ANALYSIS_SAMPLE_RATE = 22050

EVIDENCE_SOURCES = frozenset({"stems+mix", "stems", "mix"})
_HOP_LENGTH = 2048
_BINS_PER_OCTAVE = 36
_BASS_WEIGHT = 0.5
_SCALE_BONUS = 0.15
_STEM_FULL_WEIGHT_LEVEL = 0.25  # stem energy, relative to its median, that earns full trust
_SILENCE_RMS = 1e-4
# Real chords never give a clean triad profile, so "no chord" must clearly beat
# every triad before it is chosen from the pitch-class profile alone.
_NO_CHORD_PENALTY = 0.7
# A beat whose sound is mostly percussive (a drum-only bar) carries no chord.
_MIN_HARMONIC_SHARE = 0.2
_BETA = 6.0
# Cost of changing chord on a bar line, on the half bar, and anywhere else.
_CHANGE_COST = (1.5, 2.5, 4.0)
MIN_NAMED_CONFIDENCE = 0.4
_MIN_BEATS = 4
_MAX_BEATS = 20000
_MAX_SEGMENTS = 20000
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_FLAT_MAJOR_TONICS = frozenset({5, 10, 3, 8, 1})  # F, Bb, Eb, Ab, Db
_MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
_TRIADS = (("maj", (0, 4, 7)), ("min", (0, 3, 7)))
_SYMBOL_RE = re.compile(r"[A-G](?:#|b)?m?\Z")
_SEGMENT_ID_RE = re.compile(r"c[0-9]{6}\Z")
_FILE_RE = re.compile(r"audio-chords\.[0-9a-f]{16}\.json\Z")


class AudioChordsError(RuntimeError):
    """Chord recognition or its saved result could not be used."""


# ---------------------------------------------------------------------------
# Recognition


def load_audio(path: str | os.PathLike[str]) -> Any:
    """Read one audio file as mono float32 at the analysis sample rate."""
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several unrelated types
        raise AudioChordsError("An audio file could not be read for chord recognition.") from exc
    mono = samples.mean(axis=1)
    if not np.all(np.isfinite(mono)):
        raise AudioChordsError("An audio file contains non-finite samples.")
    if sample_rate != ANALYSIS_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=ANALYSIS_SAMPLE_RATE)
    return mono.astype("float32", copy=False)


def _features(signal: Any, bass_signal: Any) -> tuple[Any, Any, Any]:
    """Frame-level treble chroma, bass chroma and loudness."""
    import librosa

    treble = librosa.feature.chroma_cqt(
        y=signal,
        sr=ANALYSIS_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        fmin=librosa.note_to_hz("C2"),
        n_octaves=5,
        bins_per_octave=_BINS_PER_OCTAVE,
    )
    bass = librosa.feature.chroma_cqt(
        y=bass_signal,
        sr=ANALYSIS_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
        n_octaves=2,
        bins_per_octave=_BINS_PER_OCTAVE,
    )
    loudness = librosa.feature.rms(y=signal, frame_length=2 * _HOP_LENGTH, hop_length=_HOP_LENGTH)[0]
    return treble, bass, loudness


def _per_beat(matrix: Any, frame_times: Any, edges: Sequence[tuple[float, float]], reducer: Any) -> Any:
    import numpy as np

    rows = []
    for start, end in edges:
        low, high = np.searchsorted(frame_times, [start, end])
        low = min(int(low), matrix.shape[-1] - 1)
        high = max(int(high), low + 1)
        rows.append(reducer(matrix[..., low:high], axis=-1))
    return np.asarray(rows, dtype=float)


def _unit_rows(matrix: Any) -> Any:
    import numpy as np

    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def recognize_chords(
    beat_times: Sequence[float],
    *,
    meter: int,
    mix: Any = None,
    accompaniment: Any = None,
    bass: Any = None,
) -> dict[str, Any]:
    """Name one chord, or none, for every beat and merge beats into segments.

    ``beat_times`` are the beat grid's times in seconds, starting on beat 1 of
    bar 1. Audio arrays are mono float at :data:`ANALYSIS_SAMPLE_RATE`.
    ``accompaniment`` is the separated accompaniment stem (pass ``bass`` with
    it); ``mix`` is the full recording. At least one of them is required.
    """
    import librosa
    import numpy as np

    if isinstance(meter, bool) or not isinstance(meter, int) or not 1 <= meter <= 12:
        raise AudioChordsError("The meter for chord recognition is invalid.")
    times = [float(value) for value in beat_times]
    if not _MIN_BEATS + 1 <= len(times) <= _MAX_BEATS or any(
        not math.isfinite(value) for value in times
    ) or any(b <= a for a, b in zip(times, times[1:])):
        raise AudioChordsError("The beat grid for chord recognition is invalid.")
    if mix is None and accompaniment is None:
        raise AudioChordsError("Chord recognition needs the full mix or the accompaniment stem.")
    if accompaniment is not None and bass is None:
        raise AudioChordsError("The accompaniment stem must come with the bass stem.")

    edges = [(max(0.0, a), max(0.0, b)) for a, b in zip(times, times[1:])]
    edges = [(a, b) if b > a else (a, a + 1e-3) for a, b in edges]
    beat_count = len(edges)

    def evidence(signal: Any, bass_signal: Any) -> tuple[Any, Any, Any]:
        treble, low, loudness = _features(signal, bass_signal)
        frame_times = librosa.frames_to_time(
            np.arange(treble.shape[1]), sr=ANALYSIS_SAMPLE_RATE, hop_length=_HOP_LENGTH
        )
        treble_beats = np.log1p(10.0 * _per_beat(treble, frame_times, edges, np.median))
        bass_beats = np.log1p(10.0 * _per_beat(low, frame_times, edges, np.median))
        loud_beats = _per_beat(loudness[: treble.shape[1]], frame_times, edges, np.mean)
        return _unit_rows(treble_beats), _unit_rows(bass_beats), loud_beats

    stem = None
    if accompaniment is not None:
        length = min(len(accompaniment), len(bass))
        stem = evidence(accompaniment[:length] + bass[:length], bass[:length])
    full = None
    harmonic_share = None
    if mix is not None:
        harmonic, percussive = librosa.effects.hpss(mix, margin=3.0)
        full = evidence(harmonic, harmonic)
        percussive_level = librosa.feature.rms(
            y=percussive, frame_length=2 * _HOP_LENGTH, hop_length=_HOP_LENGTH
        )[0]
        frame_times = librosa.frames_to_time(
            np.arange(len(percussive_level)), sr=ANALYSIS_SAMPLE_RATE, hop_length=_HOP_LENGTH
        )
        percussive_beats = _per_beat(percussive_level, frame_times, edges, np.mean)
        harmonic_share = full[2] / (full[2] + percussive_beats + 1e-12)

    states: list[tuple[int, str]] = [(root, quality) for root in range(12) for quality, _ in _TRIADS]
    templates = np.zeros((len(states), 12))
    for index, (root, quality) in enumerate(states):
        for tone in dict(_TRIADS)[quality]:
            templates[index, (root + tone) % 12] = 1.0
    templates = _unit_rows(templates)
    flat = np.full(12, 1.0 / math.sqrt(12.0))
    none_state = len(states)

    def emission(source: tuple[Any, Any, Any]) -> Any:
        treble, low, _ = source
        scores = np.zeros((beat_count, len(states) + 1))
        scores[:, : len(states)] = treble @ templates.T
        for index, (root, _) in enumerate(states):
            scores[:, index] += _BASS_WEIGHT * low[:, root]
        scores[:, none_state] = treble @ flat + _BASS_WEIGHT * (low @ flat) - _NO_CHORD_PENALTY
        return scores

    if stem is not None and full is not None:
        positive = stem[2][stem[2] > 0]
        reference_level = float(np.median(positive)) if positive.size else 0.0
        weight = (
            np.clip(stem[2] / (reference_level * _STEM_FULL_WEIGHT_LEVEL), 0.0, 1.0)
            if reference_level > 0
            else np.zeros(beat_count)
        )
        scores = weight[:, None] * emission(stem) + (1.0 - weight[:, None]) * emission(full)
        profile = (weight[:, None] * stem[0] + (1.0 - weight[:, None]) * full[0]).sum(axis=0)
        loudness = np.maximum(stem[2], full[2])
        source_name = "stems+mix"
    else:
        chosen = stem if stem is not None else full
        weight = np.ones(beat_count) if stem is not None else np.zeros(beat_count)
        scores = emission(chosen)
        profile = chosen[0].sum(axis=0)
        loudness = chosen[2]
        source_name = "stems" if stem is not None else "mix"

    # The recording's overall scale: the major-scale pitch set that holds the
    # most energy. A triad inside it gets a small preference, which settles
    # major against minor where the third is weak.
    major = np.zeros(12)
    major[list(_MAJOR_STEPS)] = 1.0
    scale_tonic = int(np.argmax([float(profile @ np.roll(major, shift)) for shift in range(12)]))
    scale = {(scale_tonic + step) % 12 for step in _MAJOR_STEPS}
    for index, (root, quality) in enumerate(states):
        if all((root + tone) % 12 in scale for tone in dict(_TRIADS)[quality]):
            scores[:, index] += _SCALE_BONUS
    silent = loudness < _SILENCE_RMS
    if harmonic_share is not None:
        # Judge over about two bars, so one loud drum hit does not blank a beat.
        half = meter
        smoothed = np.asarray(
            [
                float(np.median(harmonic_share[max(0, beat - half) : beat + half + 1]))
                for beat in range(beat_count)
            ]
        )
        silent = silent | (smoothed < _MIN_HARMONIC_SHARE)
    scores[silent, none_state] = scores[silent].max(axis=1) + 1.0

    position = np.arange(beat_count) % meter
    change = np.full(beat_count, _CHANGE_COST[2])
    if meter % 2 == 0:
        change[position == meter // 2] = _CHANGE_COST[1]
    change[position == 0] = _CHANGE_COST[0]
    scaled = _BETA * scores
    best = scaled[0].copy()
    back = np.zeros((beat_count, scaled.shape[1]), dtype=int)
    identity = np.arange(scaled.shape[1])
    for beat in range(1, beat_count):
        jump = float(best.max()) - change[beat]
        origin = int(best.argmax())
        take = jump > best
        back[beat] = np.where(take, origin, identity)
        best = np.where(take, jump, best) + scaled[beat]
    path = [int(best.argmax())]
    for beat in range(beat_count - 1, 0, -1):
        path.append(int(back[beat][path[-1]]))
    path.reverse()

    names = _FLAT_NAMES if scale_tonic in _FLAT_MAJOR_TONICS else _SHARP_NAMES
    confidences = []
    for beat, state in enumerate(path):
        row = scores[beat]
        if state == none_state:
            rivals = row[:none_state]
        else:
            root = states[state][0]
            rivals = np.asarray(
                [row[index] for index, (other, _) in enumerate(states) if other != root] + [row[none_state]]
            )
        confidences.append(float(np.clip(0.5 + 1.25 * (row[state] - rivals.max()), 0.02, 0.98)))

    segments: list[dict[str, Any]] = []
    start_beat = 0
    for beat in range(1, beat_count + 1):
        if beat < beat_count and path[beat] == path[start_beat]:
            continue
        state = path[start_beat]
        confidence = float(np.mean(confidences[start_beat:beat]))
        stem_share = float(np.mean(weight[start_beat:beat]))
        named = state != none_state and confidence >= MIN_NAMED_CONFIDENCE
        symbol = None
        if named:
            root, quality = states[state]
            symbol = names[root] + ("m" if quality == "min" else "")
        segments.append(
            {
                "id": f"c{len(segments) + 1:06d}",
                "startSeconds": round(edges[start_beat][0], 4),
                "endSeconds": round(edges[beat - 1][1], 4),
                "beats": beat - start_beat,
                "symbol": symbol,
                "confidence": round(confidence, 3) if named else None,
                "noChord": state == none_state,
                "stemShare": round(stem_share, 3),
            }
        )
        start_beat = beat
    # Rounding must not make neighbours overlap or collapse.
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        if cleaned:
            segment["startSeconds"] = cleaned[-1]["endSeconds"]
        if segment["endSeconds"] > segment["startSeconds"]:
            cleaned.append(segment)
    for index, segment in enumerate(cleaned):
        segment["id"] = f"c{index + 1:06d}"
    return {
        "source": source_name,
        "scaleTonic": names[scale_tonic],
        "beats": beat_count,
        "segments": cleaned,
    }


# ---------------------------------------------------------------------------
# Saved result


def audio_chords_file_name(harmony_created_at: str) -> str:
    """Job-relative file name of the chords heard for one harmony result."""
    if not isinstance(harmony_created_at, str) or not 1 <= len(harmony_created_at) <= 128:
        raise AudioChordsError("The harmony timestamp for audio chords is invalid.")
    digest = hashlib.sha256(harmony_created_at.encode("utf-8")).hexdigest()[:16]
    return f"{AUDIO_CHORDS_DIRECTORY}/audio-chords.{digest}.json"


def build_audio_chords_artifact(
    result: Mapping[str, Any],
    *,
    job_id: str,
    created_at: str,
    harmony_created_at: str,
    analysis_version: str,
    grid_version: str | None,
) -> dict[str, Any]:
    payload = {
        "schemaVersion": AUDIO_CHORDS_SCHEMA_VERSION,
        "artifactType": AUDIO_CHORDS_ARTIFACT_TYPE,
        "chordsVersion": AUDIO_CHORDS_VERSION,
        "jobId": job_id,
        "createdAt": created_at,
        "harmonyCreatedAt": harmony_created_at,
        "analysisVersion": analysis_version,
        "gridVersion": grid_version,
        "source": result["source"],
        "scaleTonic": result["scaleTonic"],
        "beats": result["beats"],
        "segments": [dict(segment) for segment in result["segments"]],
    }
    return validate_audio_chords_artifact(payload)


def _fail(message: str) -> AudioChordsError:
    return AudioChordsError(f"Saved audio chords are invalid: {message}")


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{label} is not a number.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise _fail(f"{label} is out of range.")
    return number


def validate_audio_chords_artifact(payload: Any) -> dict[str, Any]:
    """Validate a saved result and return a detached copy."""
    if not isinstance(payload, Mapping):
        raise _fail("the document is not an object.")
    expected = {
        "schemaVersion", "artifactType", "chordsVersion", "jobId", "createdAt",
        "harmonyCreatedAt", "analysisVersion", "gridVersion", "source", "scaleTonic",
        "beats", "segments",
    }
    if set(payload) != expected:
        raise _fail("the document has missing or unsupported fields.")
    if payload["schemaVersion"] != AUDIO_CHORDS_SCHEMA_VERSION or type(payload["schemaVersion"]) is not int:
        raise _fail("the schema version is unsupported.")
    if payload["artifactType"] != AUDIO_CHORDS_ARTIFACT_TYPE:
        raise _fail("the document type is unsupported.")
    for key in ("chordsVersion", "analysisVersion"):
        if not isinstance(payload[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", payload[key]):
            raise _fail(f"{key} is invalid.")
    if payload["gridVersion"] is not None and (
        not isinstance(payload["gridVersion"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", payload["gridVersion"])
    ):
        raise _fail("gridVersion is invalid.")
    if not isinstance(payload["jobId"], str) or not re.fullmatch(r"[a-f0-9]{32}", payload["jobId"]):
        raise _fail("jobId is invalid.")
    for key in ("createdAt", "harmonyCreatedAt"):
        if not isinstance(payload[key], str) or not 1 <= len(payload[key]) <= 128:
            raise _fail(f"{key} is invalid.")
    if payload["source"] not in EVIDENCE_SOURCES:
        raise _fail("source is invalid.")
    if payload["scaleTonic"] not in _SHARP_NAMES and payload["scaleTonic"] not in _FLAT_NAMES:
        raise _fail("scaleTonic is invalid.")
    if type(payload["beats"]) is not int or not 1 <= payload["beats"] <= _MAX_BEATS:
        raise _fail("beats is invalid.")
    segments = payload["segments"]
    if not isinstance(segments, list) or len(segments) > _MAX_SEGMENTS:
        raise _fail("segments is not a bounded list.")
    checked = []
    previous_end = -1.0
    total_beats = 0
    for index, raw in enumerate(segments):
        if not isinstance(raw, Mapping) or set(raw) != {
            "id", "startSeconds", "endSeconds", "beats", "symbol", "confidence", "noChord", "stemShare"
        }:
            raise _fail(f"segments[{index}] has missing or unsupported fields.")
        if not isinstance(raw["id"], str) or not _SEGMENT_ID_RE.fullmatch(raw["id"]) or raw["id"] != f"c{index + 1:06d}":
            raise _fail(f"segments[{index}].id is invalid.")
        start = _finite(raw["startSeconds"], f"segments[{index}].startSeconds", 0.0, 1e6)
        end = _finite(raw["endSeconds"], f"segments[{index}].endSeconds", 0.0, 1e6)
        if end <= start or start < previous_end:
            raise _fail(f"segments[{index}] has an invalid time range.")
        previous_end = end
        if type(raw["beats"]) is not int or raw["beats"] < 1:
            raise _fail(f"segments[{index}].beats is invalid.")
        total_beats += raw["beats"]
        if type(raw["noChord"]) is not bool:
            raise _fail(f"segments[{index}].noChord is invalid.")
        _finite(raw["stemShare"], f"segments[{index}].stemShare", 0.0, 1.0)
        if raw["symbol"] is None:
            if raw["confidence"] is not None:
                raise _fail(f"segments[{index}] has a confidence without a chord.")
        else:
            if raw["noChord"] or not isinstance(raw["symbol"], str) or not _SYMBOL_RE.fullmatch(raw["symbol"]):
                raise _fail(f"segments[{index}].symbol is invalid.")
            if _finite(raw["confidence"], f"segments[{index}].confidence", 0.0, 1.0) < MIN_NAMED_CONFIDENCE:
                raise _fail(f"segments[{index}] names a chord below the confidence floor.")
        checked.append(dict(raw))
    if total_beats > payload["beats"]:
        raise _fail("segments cover more beats than the grid has.")
    return {**{key: payload[key] for key in expected if key != "segments"}, "segments": checked}


def _chords_directory(settings: Any, job_id: str) -> Path:
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise AudioChordsError("The job identity for audio chords is invalid.")
    return Path(settings.exports_dir) / job_id / AUDIO_CHORDS_DIRECTORY


def write_audio_chords_artifact(settings: Any, artifact: Mapping[str, Any], *, keep: int = 2) -> str:
    """Atomically save a validated result; returns the job-relative file name.

    Results for earlier harmony runs are pruned to the ``keep`` newest files so
    the previous harmony result keeps its chords while a new run is unconfirmed.
    """
    checked = validate_audio_chords_artifact(artifact)
    relative = audio_chords_file_name(checked["harmonyCreatedAt"])
    directory = _chords_directory(settings, checked["jobId"])
    directory.mkdir(parents=False, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise AudioChordsError("The audio-chords directory is unsafe.")
    target = directory / Path(relative).name
    temporary = directory / f".{Path(relative).name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(checked, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise AudioChordsError("The audio chords could not be saved.") from exc
    others = sorted(
        (path for path in directory.iterdir() if _FILE_RE.fullmatch(path.name) and path != target),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in others[max(0, keep - 1):]:
        try:
            stale.unlink()
        except OSError:
            pass
    return relative


def load_audio_chords_artifact(settings: Any, job_id: str, harmony_created_at: str) -> dict[str, Any] | None:
    """Load the chords heard for one harmony result, or ``None`` if there are none."""
    path = _chords_directory(settings, job_id) / Path(audio_chords_file_name(harmony_created_at)).name
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AudioChordsError("Saved audio chords are unavailable.") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_ARTIFACT_BYTES:
        raise AudioChordsError("Saved audio chords are unsafe or too large.")

    def reject(token: str) -> Any:
        raise ValueError(token)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise AudioChordsError("Saved audio chords are unreadable.") from exc
    checked = validate_audio_chords_artifact(payload)
    if checked["jobId"] != job_id or checked["harmonyCreatedAt"] != harmony_created_at:
        raise AudioChordsError("Saved audio chords belong to another result.")
    return checked


__all__ = [
    "ANALYSIS_SAMPLE_RATE",
    "AUDIO_CHORDS_VERSION",
    "AudioChordsError",
    "MIN_NAMED_CONFIDENCE",
    "audio_chords_file_name",
    "build_audio_chords_artifact",
    "load_audio",
    "load_audio_chords_artifact",
    "recognize_chords",
    "validate_audio_chords_artifact",
    "write_audio_chords_artifact",
]
