from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.percussion_transcription import (
    DEFAULT_ALGORITHM_VERSION,
    PercussionTranscriptionError,
    transcribe_percussion_audio,
)


SR = 44_100


def _write_wav(
    path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int = SR,
    subtype: str = "FLOAT",
) -> Path:
    sf.write(
        path,
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        subtype=subtype,
    )
    return path


def _sine_pulse(
    frequency: float,
    *,
    time_seconds: float = 0.5,
    pulse_seconds: float = 0.18,
    duration_seconds: float = 2.0,
    amplitude: float = 0.9,
) -> np.ndarray:
    audio = np.zeros(int(duration_seconds * SR), dtype=np.float32)
    length = int(pulse_seconds * SR)
    time = np.arange(length, dtype=np.float64) / SR
    envelope = np.exp(-20.0 * time)
    start = int(time_seconds * SR)
    audio[start : start + length] = (
        amplitude
        * np.sin(2.0 * np.pi * frequency * time)
        * envelope
    ).astype(np.float32)
    return audio


def _band_noise_pulse(
    low_hz: float,
    high_hz: float,
    *,
    time_seconds: float = 0.5,
    pulse_seconds: float = 0.08,
    duration_seconds: float = 2.0,
    amplitude: float = 0.7,
    seed: int = 0,
) -> np.ndarray:
    audio = np.zeros(int(duration_seconds * SR), dtype=np.float32)
    length = int(pulse_seconds * SR)
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(length)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(length, d=1.0 / SR)
    spectrum[
        (frequencies < low_hz) | (frequencies > high_hz)
    ] = 0
    pulse = np.fft.irfft(spectrum, n=length)
    pulse /= max(
        float(np.max(np.abs(pulse))),
        np.finfo(float).eps,
    )
    pulse *= np.exp(-30.0 * np.arange(length) / SR)
    start = int(time_seconds * SR)
    audio[start : start + length] = (
        amplitude * pulse
    ).astype(np.float32)
    return audio


def _kinds(event: dict) -> set[str]:
    return {hit["kind"] for hit in event["hits"]}


def _first_event(result: dict) -> dict:
    assert result["events"]
    return result["events"][0]


def test_low_frequency_pulse_is_broadly_classified_as_kick(
    tmp_path: Path,
) -> None:
    path = _write_wav(
        tmp_path / "kick.wav",
        _sine_pulse(70.0),
    )

    result = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )
    event = _first_event(result)

    assert "kick" in _kinds(event)
    assert event["timeSeconds"] == pytest.approx(0.5, abs=0.04)
    assert result["algorithmVersion"] == DEFAULT_ALGORITHM_VERSION


def test_mid_band_noise_pulse_has_snare_or_unknown_evidence(
    tmp_path: Path,
) -> None:
    path = _write_wav(
        tmp_path / "snare.wav",
        _band_noise_pulse(
            500.0,
            3_500.0,
            pulse_seconds=0.12,
            seed=1,
        ),
    )

    event = _first_event(
        transcribe_percussion_audio(
            path,
            source_kind="drums_stem",
        )
    )

    assert _kinds(event) & {"snare", "unknown_percussion"}


def test_high_frequency_pulse_has_hihat_or_cymbal_evidence(
    tmp_path: Path,
) -> None:
    path = _write_wav(
        tmp_path / "hat.wav",
        _band_noise_pulse(
            7_000.0,
            16_000.0,
            pulse_seconds=0.04,
            seed=2,
        ),
    )

    event = _first_event(
        transcribe_percussion_audio(
            path,
            source_kind="drums_stem",
        )
    )

    assert _kinds(event) & {"closed_hihat", "cymbal"}


def test_simultaneous_low_and_high_pulses_share_one_raw_event(
    tmp_path: Path,
) -> None:
    audio = _sine_pulse(70.0) + _band_noise_pulse(
        7_000.0,
        16_000.0,
        pulse_seconds=0.04,
        seed=2,
    )
    path = _write_wav(
        tmp_path / "simultaneous.wav",
        audio,
    )

    event = _first_event(
        transcribe_percussion_audio(
            path,
            source_kind="drums_stem",
        )
    )
    kinds = _kinds(event)

    assert "kick" in kinds
    assert kinds & {"closed_hihat", "cymbal"}
    assert len(event["hits"]) >= 2
    assert "simultaneous" in " ".join(event["warnings"]).lower()


def test_separated_pulses_preserve_raw_ordered_timestamps(
    tmp_path: Path,
) -> None:
    audio = _sine_pulse(
        70.0,
        time_seconds=0.35,
        duration_seconds=2.0,
    )
    audio += _band_noise_pulse(
        500.0,
        3_500.0,
        time_seconds=1.2,
        pulse_seconds=0.12,
        duration_seconds=2.0,
        seed=4,
    )
    path = _write_wav(
        tmp_path / "separated.wav",
        audio,
    )

    events = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )["events"]

    assert len(events) >= 2
    times = [event["timeSeconds"] for event in events]
    assert times == sorted(times)
    assert times[0] == pytest.approx(0.35, abs=0.05)
    assert any(time == pytest.approx(1.2, abs=0.05) for time in times)
    assert all(time * 4 != round(time * 4) for time in times)


def test_ambiguous_pitched_transient_uses_unknown_lower_confidence(
    tmp_path: Path,
) -> None:
    path = _write_wav(
        tmp_path / "ambiguous.wav",
        _sine_pulse(300.0),
    )

    event = _first_event(
        transcribe_percussion_audio(
            path,
            source_kind="drums_stem",
        )
    )

    assert event["hits"] == [
        {
            "kind": "unknown_percussion",
            "confidence": pytest.approx(0.49),
        }
    ]
    assert event["warnings"]


def test_silence_produces_no_fabricated_hits(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "silence.wav",
        np.zeros(SR, dtype=np.float32),
    )

    result = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )

    assert result["events"] == []
    assert "no reliable" in " ".join(result["warnings"]).lower()


def test_dense_noise_is_capped_and_warned(tmp_path: Path) -> None:
    duration = 3.0
    audio = np.zeros(int(duration * SR), dtype=np.float32)
    pulse = _band_noise_pulse(
        500.0,
        12_000.0,
        time_seconds=0.0,
        pulse_seconds=0.012,
        duration_seconds=0.012,
        seed=8,
    )
    for index, time_seconds in enumerate(
        np.arange(0.1, duration - 0.1, 0.04)
    ):
        start = int(time_seconds * SR)
        end = min(audio.size, start + pulse.size)
        scale = 0.6 + 0.1 * (index % 5)
        audio[start:end] += pulse[: end - start] * scale
    path = _write_wav(
        tmp_path / "dense.wav",
        np.clip(audio, -1.0, 1.0),
    )

    result = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )
    diagnostics = result["diagnostics"]
    warnings = " ".join(result["warnings"]).lower()

    assert diagnostics["onsetCandidatesBeforeCap"] > diagnostics["eventCap"]
    assert len(result["events"]) == diagnostics["eventCap"]
    assert "capped" in warnings or "omitted" in warnings


def test_stereo_input_and_repeated_calls_are_deterministic(
    tmp_path: Path,
) -> None:
    mono = _sine_pulse(70.0) + _band_noise_pulse(
        7_000.0,
        16_000.0,
        pulse_seconds=0.04,
        seed=9,
    )
    stereo = np.column_stack((mono, mono * 0.75))
    path = _write_wav(
        tmp_path / "stereo.wav",
        stereo,
    )

    first = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )
    second = transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )

    assert first == second
    assert first["diagnostics"]["channels"] == 2


def test_full_mix_emits_honesty_warning(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "mix.wav",
        _sine_pulse(70.0),
    )

    result = transcribe_percussion_audio(
        path,
        source_kind="full_mix",
    )
    warning = " ".join(result["warnings"]).lower()

    assert "full-mix" in warning
    assert "drums stem" in warning


def test_result_schema_is_bounded_finite_and_path_free(
    tmp_path: Path,
) -> None:
    path = _write_wav(
        tmp_path / "schema.wav",
        _sine_pulse(70.0),
    )

    result = transcribe_percussion_audio(
        path,
        source_kind="isolated_drums_v2",
        algorithm_version="baseline-onset-bands-v1.1",
    )

    assert set(result) == {
        "algorithmVersion",
        "sourceKind",
        "events",
        "warnings",
        "diagnostics",
    }
    assert result["sourceKind"] == "isolated_drums_v2"
    assert result["algorithmVersion"] == "baseline-onset-bands-v1.1"
    encoded = json.dumps(result, allow_nan=False)
    assert str(tmp_path) not in encoded
    assert "samples" not in encoded.lower()
    for index, event in enumerate(result["events"], 1):
        assert event["id"] == f"r{index:06d}"
        assert math.isfinite(event["timeSeconds"])
        assert event["timeSeconds"] >= 0
        assert 0 <= event["strength"] <= 1
        assert event["hits"]
        assert len(event["hits"]) <= 3
        for hit in event["hits"]:
            assert hit["kind"]
            assert 0 <= hit["confidence"] <= 1


def test_input_audio_is_never_modified(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "immutable.wav",
        _sine_pulse(70.0),
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    transcribe_percussion_audio(
        path,
        source_kind="drums_stem",
    )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "source_kind,algorithm_version",
    [
        ("Full Mix", DEFAULT_ALGORITHM_VERSION),
        ("../drums", DEFAULT_ALGORITHM_VERSION),
        ("drums_stem", "bad version"),
        ("drums_stem", "../algorithm"),
    ],
)
def test_unsafe_open_identifiers_are_rejected(
    tmp_path: Path,
    source_kind: str,
    algorithm_version: str,
) -> None:
    path = _write_wav(
        tmp_path / "safe.wav",
        _sine_pulse(70.0),
    )

    with pytest.raises(PercussionTranscriptionError, match="invalid"):
        transcribe_percussion_audio(
            path,
            source_kind=source_kind,
            algorithm_version=algorithm_version,
        )


def test_missing_nonfile_and_symlink_inputs_are_rejected_without_path_leak(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "secret-missing.wav"
    directory = tmp_path / "secret-directory.wav"
    directory.mkdir()
    target = _write_wav(
        tmp_path / "target.wav",
        _sine_pulse(70.0),
    )
    symlink = tmp_path / "secret-link.wav"
    symlink.symlink_to(target)

    for path in (missing, directory, symlink):
        with pytest.raises(PercussionTranscriptionError) as caught:
            transcribe_percussion_audio(
                path,
                source_kind="drums_stem",
            )
        assert str(tmp_path) not in str(caught.value)


def test_non_wav_and_corrupt_input_are_rejected_safely(
    tmp_path: Path,
) -> None:
    flac = tmp_path / "audio.flac"
    sf.write(
        flac,
        _sine_pulse(70.0),
        SR,
        format="FLAC",
    )
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wave file")

    for path in (flac, corrupt):
        with pytest.raises(PercussionTranscriptionError) as caught:
            transcribe_percussion_audio(
                path,
                source_kind="drums_stem",
            )
        assert str(tmp_path) not in str(caught.value)


def test_empty_nonfinite_channels_and_sample_rates_are_rejected(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.wav"
    sf.write(
        empty,
        np.zeros(0, dtype=np.float32),
        SR,
        subtype="FLOAT",
    )
    nonfinite = tmp_path / "nonfinite.wav"
    sf.write(
        nonfinite,
        np.asarray([0.0, np.nan, 0.0], dtype=np.float32),
        SR,
        subtype="FLOAT",
    )
    channels = tmp_path / "channels.wav"
    sf.write(
        channels,
        np.zeros((SR // 10, 3), dtype=np.float32),
        SR,
        subtype="FLOAT",
    )
    rate = tmp_path / "rate.wav"
    sf.write(
        rate,
        np.zeros(4_000, dtype=np.float32),
        4_000,
        subtype="FLOAT",
    )

    for path in (empty, nonfinite, channels, rate):
        with pytest.raises(PercussionTranscriptionError):
            transcribe_percussion_audio(
                path,
                source_kind="drums_stem",
            )
