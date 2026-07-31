from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf

from app.config import Settings
from app.media import MediaProcessingError, secure_job_dir

ANALYSIS_JSON_RELATIVE_PATH = "analysis/audio-analysis.json"
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
StageCallback = Callable[[str, str, float], None]


class AudioAnalysisError(RuntimeError):
    """Raised when normalized audio cannot be analyzed reliably."""


@dataclass(frozen=True)
class AudioAnalysisResult:
    analysis_version: str
    tempo_bpm: float | None
    tempo_confidence: float | None
    key_symbol: str | None
    key_confidence: float | None
    analysis_json_file_name: str
    analyzed_at: str
    payload: dict[str, Any]


def analyze_audio(job_id: str, settings: Settings, stage_callback: StageCallback) -> AudioAnalysisResult:
    started = time.monotonic()
    job_dir = secure_job_dir(settings, job_id)
    wav_path = job_dir / "analysis.wav"
    if not wav_path.is_file():
        raise AudioAnalysisError("Analysis audio is missing. Prepare analysis.wav before running audio analysis.")

    stage_callback("analyzing_audio", "Analyzing audio level and signal quality.", 66)
    audio, sample_rate = _read_wav(wav_path)
    _check_timeout(started, settings)
    channels = int(audio.shape[1])
    duration_seconds = float(audio.shape[0] / sample_rate)
    if duration_seconds <= 0 or duration_seconds > settings.max_duration_seconds + 1:
        raise AudioAnalysisError("Analysis audio has an invalid or unsupported duration.")

    mono = np.mean(audio, axis=1, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if mono.size else 0.0
    if not math.isfinite(peak) or not math.isfinite(rms):
        raise AudioAnalysisError("Analysis audio contains invalid numerical samples.")
    if rms <= settings.audio_silence_rms_threshold:
        raise AudioAnalysisError("Analysis audio is silent or too quiet to analyze reliably.")

    warnings: list[str] = []
    if rms <= settings.audio_silence_rms_threshold * 10:
        warnings.append("The signal is very quiet; timing and key estimates may be unreliable.")
    rms_dbfs = float(20.0 * math.log10(max(rms, np.finfo(np.float64).tiny)))

    stage_callback("detecting_beats", "Analyzing timing and estimating beats.", 74)
    timing = _analyze_timing(mono, sample_rate)
    _check_timeout(started, settings)
    if timing["tempoBpm"] is None:
        warnings.append("A reliable global tempo could not be estimated.")

    stage_callback("estimating_key", "Estimating tuning, tonal center, and mode.", 84)
    tonality = _analyze_tonality(mono, sample_rate)
    _check_timeout(started, settings)
    if tonality["symbol"] is None:
        warnings.append("A reliable global key could not be estimated.")

    created_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "analysisVersion": settings.audio_analysis_version,
        "createdAt": created_at,
        "sourceAsset": "analysis.wav",
        "libraries": {
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "soundfile": sf.__version__,
        },
        "audio": {
            "durationSeconds": duration_seconds,
            "sampleRate": int(sample_rate),
            "channels": channels,
            "peakAmplitude": peak,
            "rms": rms,
            "rmsDbfs": rms_dbfs,
            "silent": False,
        },
        "timing": timing,
        "tonality": tonality,
        "warnings": warnings,
    }
    _validate_json_numbers(payload)

    stage_callback("saving_analysis", "Saving versioned audio analysis.", 94)
    analysis_dir = job_dir / "analysis"
    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        destination = analysis_dir / "audio-analysis.json"
        temporary = analysis_dir / ".audio-analysis.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        raise AudioAnalysisError("Audio analysis completed but its JSON result could not be saved.") from exc
    _check_timeout(started, settings)

    return AudioAnalysisResult(
        analysis_version=settings.audio_analysis_version,
        tempo_bpm=timing["tempoBpm"],
        tempo_confidence=timing["tempoConfidence"],
        key_symbol=tonality["symbol"],
        key_confidence=tonality["confidence"],
        analysis_json_file_name=ANALYSIS_JSON_RELATIVE_PATH,
        analyzed_at=created_at,
        payload=payload,
    )


def load_analysis(job_id: str, settings: Settings) -> dict[str, Any] | None:
    job_dir = secure_job_dir(settings, job_id)
    path = job_dir / "analysis" / "audio-analysis.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioAnalysisError("Saved audio analysis is unreadable.") from exc
    if payload.get("schemaVersion") != 1:
        raise AudioAnalysisError("Saved audio analysis uses an unsupported schema version.")
    return payload


def analysis_json_path(job_id: str, settings: Settings) -> Path:
    return secure_job_dir(settings, job_id) / "analysis" / "audio-analysis.json"


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        raise AudioAnalysisError("Analysis audio is unreadable or corrupted.") from exc
    if sample_rate <= 0 or audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] == 0:
        raise AudioAnalysisError("Analysis audio is empty or has unsupported sample data.")
    if not np.isfinite(audio).all():
        raise AudioAnalysisError("Analysis audio contains invalid numerical samples.")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _analyze_timing(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop_length)
    if onset_envelope.size < 2 or float(np.max(onset_envelope)) <= np.finfo(float).eps:
        return {
            "tempoBpm": None,
            "tempoConfidence": None,
            "tempoStable": None,
            "beatsSeconds": [],
            "beatConfidence": None,
            "downbeatsSeconds": [],
            "meter": None,
            "meterConfidence": None,
        }
    tempo_raw, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        trim=False,
    )
    tempo = _finite_or_none(float(np.asarray(tempo_raw).reshape(-1)[0]))
    beat_frames = np.asarray(beat_frames, dtype=int)
    beats = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beats = beats[np.isfinite(beats)]
    intervals = np.diff(beats)
    interval_cv = float(np.std(intervals) / np.mean(intervals)) if intervals.size >= 2 and np.mean(intervals) > 0 else None
    tempo_stable = bool(interval_cv <= 0.08) if interval_cv is not None else None
    if beat_frames.size:
        valid_frames = beat_frames[(beat_frames >= 0) & (beat_frames < onset_envelope.size)]
        beat_strength = float(np.mean(onset_envelope[valid_frames])) if valid_frames.size else 0.0
        envelope_reference = float(np.percentile(onset_envelope, 90)) or 1.0
        strength_score = min(1.0, beat_strength / envelope_reference)
    else:
        strength_score = 0.0
    stability_score = max(0.0, 1.0 - min(1.0, (interval_cv or 0.5) / 0.25)) if intervals.size >= 2 else 0.0
    beat_confidence = float(np.clip(0.55 * strength_score + 0.45 * stability_score, 0.0, 1.0)) if beat_frames.size >= 2 else None
    tempo_confidence = beat_confidence
    meter, meter_confidence, downbeats = _estimate_meter(onset_envelope, beat_frames, beats)
    return {
        "tempoBpm": tempo if tempo and tempo > 0 else None,
        "tempoConfidence": tempo_confidence,
        "tempoStable": tempo_stable,
        "beatsSeconds": [float(value) for value in beats.tolist()],
        "beatConfidence": beat_confidence,
        "downbeatsSeconds": downbeats,
        "meter": meter,
        "meterConfidence": meter_confidence,
    }


def _estimate_meter(onset_envelope: np.ndarray, beat_frames: np.ndarray, beats: np.ndarray) -> tuple[int | None, float | None, list[float]]:
    if beat_frames.size < 12:
        return None, None, []
    strengths = np.asarray([
        onset_envelope[frame] if 0 <= frame < onset_envelope.size else 0.0 for frame in beat_frames
    ], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for meter in (3, 4):
        phase_means = np.asarray([np.mean(strengths[phase::meter]) for phase in range(meter)])
        order = np.argsort(phase_means)[::-1]
        best = float(phase_means[order[0]])
        second = float(phase_means[order[1]]) if meter > 1 else 0.0
        confidence = max(0.0, (best - second) / (abs(best) + 1e-9))
        candidates.append((confidence, meter, int(order[0])))
    confidence, meter, phase = max(candidates)
    if confidence < 0.12:
        return None, None, []
    downbeats = [float(value) for value in beats[phase::meter].tolist()]
    return meter, float(min(1.0, confidence)), downbeats


def _analyze_tonality(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    tuning_bins = librosa.estimate_tuning(y=mono, sr=sample_rate)
    tuning_cents = _finite_or_none(float(tuning_bins) * 100.0)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sample_rate, tuning=tuning_bins, n_fft=4096, hop_length=2048)
    chroma_mean = np.mean(chroma, axis=1) if chroma.size else np.zeros(12, dtype=np.float64)
    total = float(np.sum(chroma_mean))
    if total <= np.finfo(float).eps or not np.isfinite(chroma_mean).all():
        return {
            "key": None, "mode": None, "symbol": None, "confidence": None,
            "tuningOffsetCents": tuning_cents,
            "chromaMean": [0.0] * 12,
            "alternatives": [],
        }
    chroma_normalized = chroma_mean / total
    candidates: list[dict[str, Any]] = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        profile = profile / np.linalg.norm(profile)
        for root, name in enumerate(KEY_NAMES):
            rotated = np.roll(profile, root)
            score = float(np.dot(chroma_normalized, rotated) / (np.linalg.norm(chroma_normalized) + 1e-12))
            candidates.append({"key": name, "mode": mode, "symbol": f"{name} {mode}", "score": score})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    second = candidates[1]
    margin = max(0.0, float(best["score"] - second["score"]))
    confidence = float(np.clip(margin / 0.08, 0.0, 1.0))
    alternatives = [
        {"symbol": item["symbol"], "score": float(item["score"])} for item in candidates[1:4]
    ]
    return {
        "key": best["key"],
        "mode": best["mode"],
        "symbol": best["symbol"],
        "confidence": confidence,
        "scoreMargin": margin,
        "tuningOffsetCents": tuning_cents,
        "chromaMean": [float(value) for value in chroma_normalized.tolist()],
        "alternatives": alternatives,
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _check_timeout(started: float, settings: Settings) -> None:
    if time.monotonic() - started > settings.audio_analysis_timeout_seconds:
        raise AudioAnalysisError("Audio analysis timed out.")


def _validate_json_numbers(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _validate_json_numbers(child)
    elif isinstance(value, list):
        for child in value:
            _validate_json_numbers(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise AudioAnalysisError("Audio analysis produced an invalid numerical result.")
