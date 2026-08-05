"""Deterministic, unquantized pitched-event transcription with librosa pYIN."""

from __future__ import annotations

import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf


class PitchedTranscriptionError(RuntimeError):
    """Raised when pitched transcription cannot safely produce evidence."""


_SOURCE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_RANGES = {
    "bass": (float(librosa.note_to_hz("E1")), float(librosa.note_to_hz("C5"))),
    "vocal": (float(librosa.note_to_hz("C2")), float(librosa.note_to_hz("C7"))),
    "vocals": (float(librosa.note_to_hz("C2")), float(librosa.note_to_hz("C7"))),
}
_GENERAL_RANGE = (float(librosa.note_to_hz("C2")), float(librosa.note_to_hz("C6")))
_POLYPHONIC_KINDS = frozenset({"other", "full_mix", "full-mix", "mix"})
_MIN_SR, _MAX_SR = 8_000, 192_000
_MAX_DURATION = 360.0
_MAX_DURATION_SECONDS = _MAX_DURATION
_MAX_SAMPLE_VALUES = 40_000_000
_MAX_FRAMES = 100_000
_MIN_PROBABILITY = 0.10
_MAX_TRANSITION_RATE = 12.0
_SPLIT_SEMITONES = 0.75
_MIN_EVENT_SECONDS = 0.045
_MAX_WARNINGS, _MAX_WARNING_LENGTH = 8, 200


def transcribe_pitched_audio(
    audio_path: Path,
    *,
    source_kind: str,
    algorithm_version: str = "baseline-pyin-v1",
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
) -> dict[str, Any]:
    """Return raw pYIN events without beat quantization or score construction.

    Bass defaults to approximately E1-C5, vocals to C2-C7, and other safe
    open source kinds to a conservative C2-C6 range. Raw frame times,
    frequencies, fractional MIDI pitches, voiced probabilities, and RMS values
    remain available in ``diagnostics.pyin.frameEvidence``.
    """

    kind = _token(source_kind, _SOURCE_RE, "source kind")
    version = _token(algorithm_version, _VERSION_RE, "algorithm version")
    path = _wav_path(audio_path)
    info = _wav_info(path)
    fmin, fmax, default_range = _range(kind, int(info.samplerate), fmin_hz, fmax_hz)
    audio, sample_rate = _read(path, info)
    mono = np.mean(audio, axis=1, dtype=np.float64)
    duration = float(mono.size / sample_rate)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    if not all(math.isfinite(value) for value in (duration, peak, rms)):
        raise PitchedTranscriptionError(
            "Pitched transcription input contains invalid numerical samples."
        )

    frame_length, hop_length = _window(sample_rate, fmin)
    warnings: list[str] = []
    if kind in _POLYPHONIC_KINDS:
        warnings.append(
            "This monophonic pYIN baseline extracts only a dominant pitch line; "
            "it is not polyphonic transcription."
        )
    diagnostics = _diagnostics(
        duration,
        sample_rate,
        int(audio.shape[1]),
        int(mono.size),
        peak,
        rms,
        fmin,
        fmax,
        default_range,
        frame_length,
        hop_length,
    )
    if rms <= 1e-5:
        warnings.append(
            "The input is silent or too quiet for reliable pitched-event detection."
        )
        return _result(version, kind, [], warnings, diagnostics)
    if rms <= 5e-4:
        warnings.append(
            "The input is very quiet; voiced probabilities and event boundaries may be weak."
        )

    try:
        f0, voiced, probability = librosa.pyin(
            mono,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
            fill_na=np.nan,
            max_transition_rate=_MAX_TRANSITION_RATE,
        )
        frame_rms = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
        ).reshape(-1)
    except Exception as exc:
        raise PitchedTranscriptionError(
            "Pitched transcription could not analyze this WAV reliably."
        ) from exc

    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    voiced = np.asarray(voiced, dtype=bool).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    frame_rms = np.asarray(frame_rms, dtype=np.float64).reshape(-1)
    if len({f0.size, voiced.size, probability.size, frame_rms.size}) != 1:
        raise PitchedTranscriptionError(
            "Pitched transcription produced misaligned frame evidence."
        )
    if f0.size <= 0 or f0.size > _MAX_FRAMES:
        raise PitchedTranscriptionError(
            "Pitched transcription produced an unsupported amount of frame evidence."
        )

    probability = np.clip(
        np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
    )
    frame_rms = np.maximum(
        np.nan_to_num(frame_rms, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    times = np.arange(f0.size, dtype=np.float64) * hop_length / sample_rate
    midi = np.full(f0.size, np.nan, dtype=np.float64)
    finite_f0 = np.isfinite(f0) & (f0 > 0)
    midi[finite_f0] = librosa.hz_to_midi(f0[finite_f0])
    accepted = (
        voiced
        & finite_f0
        & np.isfinite(midi)
        & (midi >= 0)
        & (midi <= 127)
        & (probability >= _MIN_PROBABILITY)
    )
    smoothed, used = _smooth(midi, accepted)
    evidence = [
        {
            "index": int(index),
            "timeSeconds": float(times[index]),
            "voiced": bool(voiced[index]),
            "accepted": bool(used[index]),
            "frequencyHz": float(f0[index]) if finite_f0[index] else None,
            "midiPitch": float(midi[index]) if np.isfinite(midi[index]) else None,
            "smoothedMidiPitch": (
                float(smoothed[index]) if np.isfinite(smoothed[index]) else None
            ),
            "voicedProbability": float(probability[index]),
            "rms": float(frame_rms[index]),
        }
        for index in range(f0.size)
    ]
    voiced_count = int(np.count_nonzero(used))
    diagnostics["pyin"].update(
        frameCount=int(f0.size),
        voicedFrameCount=voiced_count,
        voicedFraction=float(voiced_count / f0.size),
        frameEvidence=evidence,
    )
    if voiced_count < 3 or voiced_count / f0.size < 0.005:
        warnings.append("pYIN found too little reliable voicing to form pitched events.")
        return _result(version, kind, [], warnings, diagnostics)

    events, short_count, unstable_count, event_evidence = _events(
        kind,
        _segments(smoothed, used),
        used,
        midi,
        probability,
        frame_rms,
        times,
        sample_rate,
        hop_length,
        duration,
    )
    diagnostics["eventing"].update(
        discardedShortSegments=short_count,
        discardedUnstableSegments=unstable_count,
        eventEvidence=event_evidence,
    )
    if events:
        warnings.append(
            "Velocity values are bounded local-RMS estimates, not precise performance dynamics."
        )
    else:
        warnings.append(
            "No stable pitched events survived the minimum-duration and confidence checks."
        )
    if short_count:
        warnings.append("Extremely short voiced glitches were excluded from pitched events.")
    if unstable_count:
        warnings.append(
            "Ambiguous or unstable pitch segments were excluded rather than quantized."
        )
    if events and voiced_count / f0.size < 0.05:
        warnings.append("Only a small portion of the input contained reliable voicing.")
    return _result(version, kind, events, warnings, diagnostics)


def _diagnostics(
    duration: float,
    sample_rate: int,
    channels: int,
    sample_frames: int,
    peak: float,
    rms: float,
    fmin: float,
    fmax: float,
    default_range: bool,
    frame_length: int,
    hop_length: int,
) -> dict[str, Any]:
    return {
        "libraries": {
            "librosa": str(librosa.__version__),
            "numpy": str(np.__version__),
            "soundfile": str(sf.__version__),
        },
        "audio": {
            "durationSeconds": duration,
            "sampleRate": sample_rate,
            "channels": channels,
            "sampleFrames": sample_frames,
            "peakAmplitude": peak,
            "rms": rms,
        },
        "range": {
            "fminHz": fmin,
            "fmaxHz": fmax,
            "usedSourceDefault": default_range,
        },
        "pyin": {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "frameCount": 0,
            "voicedFrameCount": 0,
            "voicedFraction": 0.0,
            "minimumVoicedProbability": _MIN_PROBABILITY,
            "maxTransitionRateOctavesPerSecond": _MAX_TRANSITION_RATE,
            "frameEvidence": [],
        },
        "eventing": {
            "pitchSplitSemitones": _SPLIT_SEMITONES,
            "minimumEventSeconds": _MIN_EVENT_SECONDS,
            "bridgedGapFrames": 1,
            "discardedShortSegments": 0,
            "discardedUnstableSegments": 0,
            "timingQuantized": False,
            "boundaryModel": "half-hop around accepted pYIN frame centers",
            "eventEvidence": [],
        },
    }


def _events(
    kind: str,
    segments: list[tuple[int, int]],
    used: np.ndarray,
    midi: np.ndarray,
    probability: np.ndarray,
    frame_rms: np.ndarray,
    times: np.ndarray,
    sample_rate: int,
    hop_length: int,
    duration: float,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    event_evidence: list[dict[str, Any]] = []
    short_count = unstable_count = 0
    half_hop = 0.5 * hop_length / sample_rate
    minimum_frames = max(3, math.ceil(_MIN_EVENT_SECONDS * sample_rate / hop_length))
    voiced_rms = frame_rms[used & np.isfinite(frame_rms) & (frame_rms > 0)]
    rms_reference = (
        max(float(np.percentile(voiced_rms, 95)), np.finfo(float).tiny)
        if voiced_rms.size
        else 1.0
    )
    for start, end in segments:
        indices = np.arange(start, end + 1)
        indices = indices[used[indices]]
        start_seconds = max(0.0, float(times[indices[0]] - half_hop))
        end_seconds = min(duration, float(times[indices[-1]] + half_hop))
        if indices.size < minimum_frames or end_seconds - start_seconds < _MIN_EVENT_SECONDS:
            short_count += 1
            continue
        pitch = _weighted_median(midi[indices], probability[indices])
        mad = float(np.median(np.abs(midi[indices] - pitch)))
        mean_probability = float(np.mean(probability[indices]))
        confidence = float(
            np.clip(
                0.75 * mean_probability + 0.25 * (1.0 - mad / 0.75),
                0.0,
                1.0,
            )
        )
        if (
            not math.isfinite(pitch)
            or not 0 <= pitch <= 127
            or confidence < 0.18
            or mad > 1.25
        ):
            unstable_count += 1
            continue
        midi_note = int(np.clip(np.rint(pitch), 0, 127))
        local_rms = float(np.median(frame_rms[indices]))
        velocity = int(
            np.clip(
                round(
                    1.0
                    + 126.0
                    * math.sqrt(np.clip(local_rms / rms_reference, 0.0, 1.0))
                ),
                1,
                127,
            )
        )
        event_warnings = []
        if mad > 0.35:
            event_warnings.append(
                "Pitch evidence is unstable within this candidate event."
            )
        if confidence < 0.45:
            event_warnings.append("This candidate has low pYIN confidence.")
        event_id = f"p{len(events) + 1:06d}"
        events.append(
            {
                "id": event_id,
                "sourceKind": kind,
                "startSeconds": start_seconds,
                "endSeconds": end_seconds,
                "midiNote": midi_note,
                "midiPitch": float(pitch),
                "frequencyHz": float(librosa.midi_to_hz(pitch)),
                "noteName": str(
                    librosa.midi_to_note(
                        midi_note,
                        octave=True,
                        cents=False,
                        unicode=False,
                    )
                ),
                "confidence": confidence,
                "velocity": velocity,
                "warnings": _warnings(event_warnings),
            }
        )
        event_evidence.append(
            {
                "eventId": event_id,
                "firstFrameIndex": int(indices[0]),
                "lastFrameIndex": int(indices[-1]),
                "voicedFrameCount": int(indices.size),
                "meanVoicedProbability": mean_probability,
                "pitchMadCents": float(mad * 100.0),
            }
        )
    return events, short_count, unstable_count, event_evidence


def _wav_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise PitchedTranscriptionError(
            "Pitched transcription requires a pathlib.Path WAV input."
        )
    if "\x00" in os.fspath(value):
        raise PitchedTranscriptionError(
            "The pitched transcription WAV path is invalid."
        )
    path = Path(os.path.abspath(value))
    if path.suffix.lower() != ".wav":
        raise PitchedTranscriptionError("Pitched transcription requires a WAV input.")
    try:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise PitchedTranscriptionError(
                    "The pitched transcription WAV must not use symbolic links."
                )
    except FileNotFoundError:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV is missing."
        ) from None
    except PitchedTranscriptionError:
        raise
    except OSError:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV could not be inspected safely."
        ) from None
    if not stat.S_ISREG(path.lstat().st_mode):
        raise PitchedTranscriptionError(
            "The pitched transcription input must be a regular WAV file."
        )
    return path


def _wav_info(path: Path) -> Any:
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError, ValueError) as exc:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV is unreadable or corrupted."
        ) from exc
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    duration = frames / sample_rate if sample_rate > 0 else 0.0
    if str(info.format).upper() != "WAV":
        raise PitchedTranscriptionError(
            "The pitched transcription input is not a WAV file."
        )
    if channels not in {1, 2}:
        raise PitchedTranscriptionError(
            "Pitched transcription supports only mono or stereo WAV input."
        )
    if not _MIN_SR <= sample_rate <= _MAX_SR:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV sample rate is unsupported."
        )
    if frames <= 0 or not math.isfinite(duration) or duration <= 0:
        raise PitchedTranscriptionError("The pitched transcription WAV is empty.")
    if duration > _MAX_DURATION or frames * channels > _MAX_SAMPLE_VALUES:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV exceeds the local duration or memory limit."
        )
    return info


def _read(path: Path, info: Any) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        raise PitchedTranscriptionError(
            "The pitched transcription WAV is unreadable or corrupted."
        ) from exc
    audio = np.asarray(audio, dtype=np.float32)
    if int(sample_rate) != int(info.samplerate) or audio.shape != (
        int(info.frames),
        int(info.channels),
    ):
        raise PitchedTranscriptionError(
            "The pitched transcription WAV changed during loading."
        )
    if not np.isfinite(audio).all():
        raise PitchedTranscriptionError(
            "The pitched transcription WAV contains invalid numerical samples."
        )
    return audio, int(sample_rate)


def _range(
    kind: str,
    sample_rate: int,
    fmin_hz: object,
    fmax_hz: object,
) -> tuple[float, float, bool]:
    default_min, default_max = _RANGES.get(kind, _GENERAL_RANGE)
    fmin = default_min if fmin_hz is None else _positive(fmin_hz)
    fmax = default_max if fmax_hz is None else _positive(fmax_hz)
    if (
        fmin >= fmax
        or fmax > sample_rate * 0.49
        or fmin < float(librosa.midi_to_hz(0))
        or fmax > float(librosa.midi_to_hz(127))
    ):
        raise PitchedTranscriptionError(
            "The pitched transcription frequency range is invalid for this WAV."
        )
    return float(fmin), float(fmax), fmin_hz is None and fmax_hz is None


def _token(value: object, pattern: re.Pattern[str], label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or pattern.fullmatch(value) is None
    ):
        raise PitchedTranscriptionError(
            f"The pitched transcription {label} is invalid."
        )
    return value


def _positive(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PitchedTranscriptionError(
            "Pitched transcription frequency limits must be positive finite numbers."
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise PitchedTranscriptionError(
            "Pitched transcription frequency limits must be positive finite numbers."
        )
    return number


def _window(sample_rate: int, fmin: float) -> tuple[int, int]:
    target = max(2048.0, 4.0 * sample_rate / fmin)
    frame_length = int(
        np.clip(1 << math.ceil(math.log2(target)), 2048, 16384)
    )
    return frame_length, int(np.clip(frame_length // 8, 128, 512))


def _smooth(
    midi: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = accepted.copy()
    smoothed = midi.copy()
    for index in range(1, used.size - 1):
        if not used[index] and used[index - 1] and used[index + 1]:
            if abs(smoothed[index - 1] - smoothed[index + 1]) <= 0.5:
                used[index] = True
                smoothed[index] = (
                    smoothed[index - 1] + smoothed[index + 1]
                ) / 2
    for index in range(used.size):
        if used[index] and (index == 0 or not used[index - 1]) and (
            index == used.size - 1 or not used[index + 1]
        ):
            used[index] = False
    for index in np.flatnonzero(used):
        values = [
            smoothed[position]
            for position in range(
                max(0, index - 1),
                min(used.size, index + 2),
            )
            if used[position] and math.isfinite(float(smoothed[position]))
        ]
        smoothed[index] = float(np.median(values))
    smoothed[~used] = np.nan
    return smoothed, used


def _segments(
    smoothed: np.ndarray,
    used: np.ndarray,
) -> list[tuple[int, int]]:
    indices = np.flatnonzero(used)
    if not indices.size:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current - previous > 2:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    segments: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        start = run_start
        history: list[float] = []
        index = run_start
        while index <= run_end:
            if not used[index]:
                index += 1
                continue
            value = float(smoothed[index])
            reference = float(np.median(history[-9:])) if history else value
            lookahead = [
                float(smoothed[position])
                for position in range(
                    index,
                    min(run_end + 1, index + 3),
                )
                if used[position]
            ]
            changed = sum(
                abs(candidate - reference) >= _SPLIT_SEMITONES
                for candidate in lookahead
            )
            if (
                history
                and abs(value - reference) >= _SPLIT_SEMITONES
                and changed >= 2
            ):
                segments.append((start, index - 1))
                start = index
                history = [value]
            else:
                history.append(value)
            index += 1
        segments.append((start, run_end))
    return segments


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    if not values.size:
        return math.nan
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= np.finfo(float).eps:
        return float(np.median(values))
    index = int(np.searchsorted(np.cumsum(weights), total / 2, side="left"))
    return float(values[min(index, values.size - 1)])


def _warnings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = " ".join(value.split()).strip() if isinstance(value, str) else ""
        if not text:
            continue
        if len(text) > _MAX_WARNING_LENGTH:
            text = text[: _MAX_WARNING_LENGTH - 1].rstrip() + "…"
        if text not in result:
            result.append(text)
        if len(result) >= _MAX_WARNINGS:
            break
    return result


def _result(
    version: str,
    kind: str,
    events: list[dict[str, Any]],
    warnings: Iterable[str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "algorithmVersion": version,
        "sourceKind": kind,
        "events": events,
        "warnings": _warnings(warnings),
        "diagnostics": diagnostics,
    }
    _public(result)
    return result


def _public(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PitchedTranscriptionError(
                "Pitched transcription produced invalid numerical evidence."
            )
        return
    if isinstance(value, list):
        for item in value:
            _public(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _public(item)
        return
    raise PitchedTranscriptionError(
        "Pitched transcription produced an invalid public result."
    )
