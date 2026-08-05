from __future__ import annotations

import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


DEFAULT_ALGORITHM_VERSION = "baseline-onset-bands-v1"

_SAFE_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_MIN_SAMPLE_RATE = 8_000
_MAX_SAMPLE_RATE = 192_000
_MAX_CHANNELS = 2
_MAX_DURATION_SECONDS = 1_800.0
_MAX_EVENTS = 2_048
_MIN_EVENT_CAP = 24
_MAX_EVENTS_PER_SECOND = 8.0
_MERGE_WINDOW_SECONDS = 0.070
_RESONANCE_WINDOW_SECONDS = 0.240
_RESONANCE_STRENGTH_RATIO = 0.42
_EPSILON = float(np.finfo(np.float64).tiny)


class PercussionTranscriptionError(RuntimeError):
    """A percussion transcription request could not be completed safely."""


def transcribe_percussion_audio(
    audio_path: Path,
    *,
    source_kind: str,
    algorithm_version: str = DEFAULT_ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Return deterministic raw percussion-event candidates for one local WAV."""
    safe_source_kind = _safe_slug(source_kind, "source kind", limit=64)
    safe_algorithm_version = _safe_slug(
        algorithm_version,
        "algorithm version",
        limit=128,
        allow_period=True,
    )
    audio, sample_rate, channels, duration_seconds = _read_wav(audio_path)
    mono = np.mean(audio, axis=1, dtype=np.float64)
    warnings: list[str] = []
    if safe_source_kind == "full_mix":
        warnings.append(
            "Full-mix transcription may include non-percussion transients; "
            "a drums stem is preferred."
        )

    hop_length = 256 if sample_rate >= 16_000 else 128
    event_cap = min(
        _MAX_EVENTS,
        max(
            _MIN_EVENT_CAP,
            int(math.ceil(duration_seconds * _MAX_EVENTS_PER_SECOND)),
        ),
    )
    diagnostics: dict[str, Any] = {
        "sampleRate": sample_rate,
        "channels": channels,
        "durationSeconds": _round_seconds(duration_seconds),
        "hopLength": hop_length,
        "timeResolutionSeconds": round(hop_length / sample_rate, 6),
        "onsetCandidatesBeforeCap": 0,
        "eventCap": event_cap,
        "eventsReturned": 0,
        "strengthScale": "relative_onset_envelope",
        "labelsAreBroad": True,
    }

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = (
        float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        if mono.size
        else 0.0
    )
    if peak <= 1e-7 or rms <= 1e-8:
        warnings.append("No reliable percussion onsets were detected.")
        result = {
            "algorithmVersion": safe_algorithm_version,
            "sourceKind": safe_source_kind,
            "events": [],
            "warnings": warnings,
            "diagnostics": diagnostics,
        }
        _validate_json_result(result)
        return result

    try:
        onset_envelope, candidates = _detect_onsets(
            mono,
            sample_rate,
            hop_length,
        )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        raise PercussionTranscriptionError(
            "Percussion onset analysis could not be completed."
        ) from exc

    diagnostics["onsetCandidatesBeforeCap"] = len(candidates)
    if len(candidates) > event_cap:
        candidates = sorted(
            sorted(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )[:event_cap],
            key=lambda item: item[0],
        )
        warnings.append(
            "Event density exceeded the baseline limit; weaker onset candidates "
            "were omitted."
        )

    if not candidates:
        warnings.append("No reliable percussion onsets were detected.")
        result = {
            "algorithmVersion": safe_algorithm_version,
            "sourceKind": safe_source_kind,
            "events": [],
            "warnings": warnings,
            "diagnostics": diagnostics,
        }
        _validate_json_result(result)
        return result

    features = [
        _spectral_features(mono, frame, sample_rate, hop_length)
        for frame, _ in candidates
    ]
    band_maxima = np.max(
        np.asarray(
            [feature["bandEnergies"] for feature in features],
            dtype=np.float64,
        ),
        axis=0,
    )
    envelope_reference = max(float(np.max(onset_envelope)), _EPSILON)

    events: list[dict[str, Any]] = []
    for index, ((frame, raw_strength), feature) in enumerate(
        zip(candidates, features, strict=True),
        start=1,
    ):
        transient_strength = _unit_interval(raw_strength / envelope_reference)
        salience = tuple(
            _unit_interval(energy / max(float(reference), _EPSILON))
            for energy, reference in zip(
                feature["bandEnergies"],
                band_maxima,
                strict=True,
            )
        )
        hits, event_warnings = _classify_hits(
            feature,
            salience,
            transient_strength,
        )
        raw_time = min(
            duration_seconds,
            max(0.0, float(frame * hop_length / sample_rate)),
        )
        events.append(
            {
                "id": f"r{index:06d}",
                "sourceKind": safe_source_kind,
                "timeSeconds": _round_seconds(raw_time),
                "strength": round(transient_strength, 3),
                "hits": hits,
                "rawFeatureSummary": {
                    "lowBandRatio": round(
                        float(feature["lowBandRatio"]),
                        3,
                    ),
                    "midBandRatio": round(
                        float(feature["midBandRatio"]),
                        3,
                    ),
                    "highBandRatio": round(
                        float(feature["highBandRatio"]),
                        3,
                    ),
                    "spectralCentroidHz": _round_hz(
                        feature["spectralCentroidHz"]
                    ),
                    "spectralRolloffHz": _round_hz(
                        feature["spectralRolloffHz"]
                    ),
                    "transientStrength": round(transient_strength, 3),
                },
                "warnings": event_warnings,
            }
        )

    diagnostics["eventsReturned"] = len(events)
    result = {
        "algorithmVersion": safe_algorithm_version,
        "sourceKind": safe_source_kind,
        "events": events,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
    _validate_json_result(result)
    return result


def _safe_slug(
    value: object,
    label: str,
    *,
    limit: int,
    allow_period: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or "\x00" in value
    ):
        raise PercussionTranscriptionError(f"The {label} is invalid.")
    pattern = (
        re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
        if allow_period
        else _SAFE_SLUG_RE
    )
    if pattern.fullmatch(value) is None:
        raise PercussionTranscriptionError(f"The {label} is invalid.")
    return value


def _read_wav(path: Path) -> tuple[np.ndarray, int, int, float]:
    if not isinstance(path, Path) or "\x00" in os.fspath(path):
        raise PercussionTranscriptionError(
            "The percussion audio input is invalid."
        )
    try:
        path_info = path.lstat()
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise PercussionTranscriptionError(
            "The percussion audio input is missing or unavailable."
        ) from exc
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        raise PercussionTranscriptionError(
            "The percussion audio input must be a regular non-symlink file."
        )
    try:
        descriptor = sf.info(path)
    except (RuntimeError, OSError, ValueError) as exc:
        raise PercussionTranscriptionError(
            "The percussion audio input is unreadable or corrupted."
        ) from exc
    if str(descriptor.format).upper() not in {"WAV", "WAVEX"}:
        raise PercussionTranscriptionError(
            "The percussion audio input must be a readable WAV file."
        )
    sample_rate = int(descriptor.samplerate)
    channels = int(descriptor.channels)
    frames = int(descriptor.frames)
    if sample_rate < _MIN_SAMPLE_RATE or sample_rate > _MAX_SAMPLE_RATE:
        raise PercussionTranscriptionError(
            "The percussion audio sample rate is unsupported."
        )
    if channels < 1 or channels > _MAX_CHANNELS:
        raise PercussionTranscriptionError(
            "The percussion audio channel count is unsupported."
        )
    if frames <= 0:
        raise PercussionTranscriptionError(
            "The percussion audio input is empty."
        )
    duration_seconds = float(frames / sample_rate)
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or duration_seconds > _MAX_DURATION_SECONDS
    ):
        raise PercussionTranscriptionError(
            "The percussion audio duration is unsupported."
        )
    try:
        audio, decoded_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise PercussionTranscriptionError(
            "The percussion audio input is unreadable or corrupted."
        ) from exc
    audio = np.asarray(audio, dtype=np.float32)
    if (
        int(decoded_rate) != sample_rate
        or audio.ndim != 2
        or audio.shape != (frames, channels)
        or audio.size == 0
    ):
        raise PercussionTranscriptionError(
            "The percussion audio input has unsupported sample data."
        )
    if not np.isfinite(audio).all():
        raise PercussionTranscriptionError(
            "The percussion audio input contains invalid numerical samples."
        )
    return audio, sample_rate, channels, duration_seconds


def _detect_onsets(
    mono: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    n_fft = 2_048 if sample_rate >= 16_000 else 1_024
    analysis_audio = np.asarray(mono, dtype=np.float32)
    if analysis_audio.size < n_fft:
        analysis_audio = np.pad(
            analysis_audio,
            (0, n_fft - analysis_audio.size),
        )
    onset_envelope = np.asarray(
        librosa.onset.onset_strength(
            y=analysis_audio,
            sr=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            aggregate=np.median,
            center=True,
        ),
        dtype=np.float64,
    )
    if onset_envelope.ndim != 1 or not np.isfinite(onset_envelope).all():
        raise PercussionTranscriptionError(
            "Percussion onset analysis produced invalid numerical data."
        )
    if onset_envelope.size == 0 or float(np.max(onset_envelope)) <= 1e-10:
        return onset_envelope, []
    frames = np.asarray(
        librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
            pre_max=3,
            post_max=3,
            pre_avg=8,
            post_avg=8,
            delta=0.07,
            wait=2,
        ),
        dtype=np.int64,
    )
    valid_frames = [
        int(frame)
        for frame in frames.tolist()
        if 0 <= int(frame) < onset_envelope.size
        and int(frame) * hop_length < mono.size
    ]
    return onset_envelope, _suppress_duplicate_frames(
        valid_frames,
        onset_envelope,
        sample_rate,
        hop_length,
    )


def _suppress_duplicate_frames(
    frames: list[int],
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> list[tuple[int, float]]:
    selected: list[list[float]] = []
    for frame in sorted(set(frames)):
        strength = max(0.0, float(onset_envelope[frame]))
        if not selected:
            selected.append([float(frame), strength])
            continue
        previous_frame = int(selected[-1][0])
        previous_strength = float(selected[-1][1])
        distance_seconds = (frame - previous_frame) * hop_length / sample_rate
        if distance_seconds <= _MERGE_WINDOW_SECONDS:
            selected[-1][1] = max(previous_strength, strength)
            continue
        if (
            distance_seconds <= _RESONANCE_WINDOW_SECONDS
            and strength < previous_strength * _RESONANCE_STRENGTH_RATIO
        ):
            continue
        selected.append([float(frame), strength])
    return [
        (int(frame), float(strength))
        for frame, strength in selected
    ]


def _spectral_features(
    mono: np.ndarray,
    frame: int,
    sample_rate: int,
    hop_length: int,
) -> dict[str, Any]:
    onset_sample = frame * hop_length
    start = max(0, onset_sample - int(round(0.020 * sample_rate)))
    end = min(
        mono.size,
        onset_sample + int(round(0.160 * sample_rate)),
    )
    segment = np.asarray(mono[start:end], dtype=np.float32)
    if segment.size < 64:
        segment = np.pad(segment, (0, 64 - segment.size))
    segment = np.asarray(segment, dtype=np.float64)
    segment -= float(np.mean(segment))
    n_fft = 2_048 if sample_rate >= 16_000 else 1_024
    if segment.size < n_fft:
        segment = np.pad(segment, (0, n_fft - segment.size))
    magnitude = np.abs(
        librosa.stft(
            segment.astype(np.float32),
            n_fft=n_fft,
            hop_length=max(64, n_fft // 8),
            window="hann",
            center=False,
        )
    )
    power = np.sum(
        np.square(magnitude, dtype=np.float64),
        axis=1,
    )
    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=n_fft,
    )
    valid = frequencies >= 30.0
    total_energy = max(
        float(np.sum(power[valid], dtype=np.float64)),
        _EPSILON,
    )
    masks = (
        (frequencies >= 30.0) & (frequencies < 250.0),
        (frequencies >= 250.0) & (frequencies < 4_000.0),
        frequencies >= 4_000.0,
    )
    energies = tuple(
        max(0.0, float(np.sum(power[mask], dtype=np.float64)))
        for mask in masks
    )
    ratios = tuple(
        _unit_interval(energy / total_energy)
        for energy in energies
    )
    aggregate_magnitude = np.sqrt(
        power,
        dtype=np.float64,
    )[:, np.newaxis]
    centroid = float(
        np.asarray(
            librosa.feature.spectral_centroid(
                S=aggregate_magnitude,
                freq=frequencies[:, np.newaxis],
            )
        ).reshape(-1)[0]
    )
    rolloff = float(
        np.asarray(
            librosa.feature.spectral_rolloff(
                S=aggregate_magnitude,
                freq=frequencies[:, np.newaxis],
                roll_percent=0.85,
            )
        ).reshape(-1)[0]
    )
    flatness = tuple(
        _band_flatness(aggregate_magnitude[:, 0], mask)
        for mask in masks
    )
    third = max(1, segment.size // 3)
    early_energy = float(
        np.mean(np.square(segment[:third]), dtype=np.float64)
    )
    tail_energy = float(
        np.mean(np.square(segment[-third:]), dtype=np.float64)
    )
    tail_ratio = _unit_interval(
        tail_energy / max(early_energy, _EPSILON)
    )
    values = (
        *energies,
        *ratios,
        centroid,
        rolloff,
        *flatness,
        tail_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise PercussionTranscriptionError(
            "Percussion spectral analysis produced invalid numerical data."
        )
    return {
        "bandEnergies": energies,
        "lowBandRatio": ratios[0],
        "midBandRatio": ratios[1],
        "highBandRatio": ratios[2],
        "spectralCentroidHz": max(0.0, centroid),
        "spectralRolloffHz": max(0.0, rolloff),
        "lowBandFlatness": flatness[0],
        "midBandFlatness": flatness[1],
        "highBandFlatness": flatness[2],
        "tailEnergyRatio": tail_ratio,
    }


def _band_flatness(
    magnitude: np.ndarray,
    mask: np.ndarray,
) -> float:
    band = np.asarray(magnitude[mask], dtype=np.float64)
    if band.size < 2 or float(np.max(band, initial=0.0)) <= 1e-12:
        return 0.0
    value = float(
        np.asarray(
            librosa.feature.spectral_flatness(
                S=band[:, np.newaxis],
                power=2.0,
            )
        ).reshape(-1)[0]
    )
    return _unit_interval(value)


def _classify_hits(
    feature: dict[str, Any],
    salience: tuple[float, float, float],
    transient_strength: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    low = float(feature["lowBandRatio"])
    mid = float(feature["midBandRatio"])
    high = float(feature["highBandRatio"])
    low_salience, mid_salience, high_salience = salience
    centroid = float(feature["spectralCentroidHz"])
    rolloff = float(feature["spectralRolloffHz"])
    mid_flatness = float(feature["midBandFlatness"])
    high_flatness = float(feature["highBandFlatness"])
    tail_ratio = float(feature["tailEnergyRatio"])

    kick_evidence = max(
        low,
        (
            0.45 * low_salience + min(0.45, low * 3.0)
            if low >= 0.02
            else 0.0
        ),
    )
    mid_evidence = max(
        mid,
        (
            0.40 * mid_salience + min(0.45, mid * 1.2)
            if mid >= 0.12
            else 0.0
        ),
    )
    high_evidence = max(
        high,
        (
            0.42 * high_salience + min(0.45, high * 4.0)
            if high >= 0.025
            else 0.0
        ),
    )

    candidates: list[tuple[str, float]] = []
    if low >= 0.50 or (
        low >= 0.18
        and low_salience >= 0.65
        and centroid < 2_800.0
    ):
        candidates.append(
            (
                "kick",
                _unit_interval(
                    min(
                        0.96,
                        0.42
                        + 0.45 * kick_evidence
                        + 0.08 * transient_strength,
                    )
                ),
            )
        )
    if (
        not candidates
        and 0.18 <= low < 0.58
        and mid >= 0.28
        and high < 0.22
        and centroid < 3_000.0
    ):
        candidates.append(
            (
                "tom",
                _unit_interval(
                    min(
                        0.82,
                        0.35
                        + 0.38 * max(low, mid)
                        + 0.08 * transient_strength,
                    )
                ),
            )
        )
    if (
        (
            mid >= 0.48
            or (mid >= 0.25 and mid_salience >= 0.72)
        )
        and mid_flatness >= 0.01
    ):
        candidates.append(
            (
                "snare",
                _unit_interval(
                    min(
                        0.90,
                        0.38
                        + 0.42 * mid_evidence
                        + 0.08 * transient_strength,
                    )
                ),
            )
        )
    if (
        (
            high >= 0.45
            or (
                high >= 0.035
                and high_salience >= 0.72
                and rolloff >= 5_500.0
            )
        )
        and high_flatness >= 0.004
    ):
        high_kind = (
            "cymbal"
            if tail_ratio >= 0.08 and rolloff >= 7_000.0
            else "closed_hihat"
        )
        candidates.append(
            (
                high_kind,
                _unit_interval(
                    min(
                        0.90,
                        0.36
                        + 0.44 * high_evidence
                        + 0.08 * transient_strength,
                    )
                ),
            )
        )

    event_warnings: list[str] = []
    if not candidates:
        confidence = min(
            0.49,
            0.24
            + 0.18 * max(low, mid, high)
            + 0.08 * transient_strength,
        )
        candidates = [
            (
                "unknown_percussion",
                _unit_interval(confidence),
            )
        ]
        event_warnings.append(
            "Spectral evidence was ambiguous; only a broad percussion candidate "
            "is reported."
        )
    elif len(candidates) > 1:
        event_warnings.append(
            "Independent spectral bands support simultaneous broad hit candidates."
        )

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, confidence in candidates:
        if kind in seen:
            continue
        seen.add(kind)
        hits.append(
            {
                "kind": kind,
                "confidence": round(confidence, 3),
            }
        )
        if len(hits) == 3:
            break
    return hits, event_warnings


def _unit_interval(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 3)


def _round_hz(value: float) -> int:
    return max(0, int(round(float(value) / 10.0) * 10))


def _validate_json_result(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PercussionTranscriptionError(
                    "Percussion transcription produced an invalid result."
                )
            _validate_json_result(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_result(item)
        return
    if (
        isinstance(value, bool)
        or value is None
        or isinstance(value, str)
    ):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise PercussionTranscriptionError(
        "Percussion transcription produced an invalid result."
    )
