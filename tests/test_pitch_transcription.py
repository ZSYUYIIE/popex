from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app import pitch_transcription as pitch

SAMPLE_RATE = 22_050


def tone(
    frequency: float,
    duration: float,
    *,
    amplitude: float = 0.3,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    count = int(round(duration * sample_rate))
    times = np.arange(count, dtype=np.float64) / sample_rate
    values = amplitude * np.sin(2.0 * np.pi * frequency * times)
    fade = min(int(round(0.02 * sample_rate)), count // 4)
    if fade:
        envelope = np.ones(count, dtype=np.float64)
        envelope[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=True)
        values *= envelope
    return values.astype(np.float32)


def write_wav(
    path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = "PCM_16",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, format="WAV", subtype=subtype)
    return path


def assert_finite_json(value: Any) -> None:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_sustained_a4_produces_one_raw_event_and_frame_evidence(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "a4.wav", tone(440.0, 1.2))
    result = pitch.transcribe_pitched_audio(path, source_kind="vocals")
    assert list(result) == [
        "algorithmVersion",
        "sourceKind",
        "events",
        "warnings",
        "diagnostics",
    ]
    assert result["algorithmVersion"] == "baseline-pyin-v1"
    assert result["sourceKind"] == "vocals"
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert set(event) == {
        "id",
        "sourceKind",
        "startSeconds",
        "endSeconds",
        "midiNote",
        "midiPitch",
        "frequencyHz",
        "noteName",
        "confidence",
        "velocity",
        "warnings",
    }
    assert event["id"] == "p000001"
    assert event["sourceKind"] == "vocals"
    assert event["startSeconds"] == pytest.approx(0.0, abs=0.04)
    assert event["endSeconds"] == pytest.approx(1.2, abs=0.05)
    assert event["midiNote"] == 69
    assert event["midiPitch"] == pytest.approx(69.0, abs=0.15)
    assert event["frequencyHz"] == pytest.approx(440.0, rel=0.015)
    assert event["noteName"] == "A4"
    assert 0.0 <= event["confidence"] <= 1.0
    assert 1 <= event["velocity"] <= 127
    assert (
        result["diagnostics"]["eventing"]["eventEvidence"][0][
            "voicedFrameCount"
        ]
        >= 3
    )
    frames = result["diagnostics"]["pyin"]["frameEvidence"]
    assert frames
    assert all(
        set(frame)
        == {
            "index",
            "timeSeconds",
            "voiced",
            "accepted",
            "frequencyHz",
            "midiPitch",
            "smoothedMidiPitch",
            "voicedProbability",
            "rms",
        }
        for frame in frames
    )
    accepted = [frame for frame in frames if frame["accepted"]]
    assert accepted
    assert all(frame["frequencyHz"] is not None for frame in accepted)
    assert all(frame["midiPitch"] is not None for frame in accepted)
    assert [frame["timeSeconds"] for frame in frames] == sorted(
        frame["timeSeconds"] for frame in frames
    )
    assert_finite_json(result)


def test_detuned_tone_retains_fractional_midi_pitch(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "detuned.wav", tone(445.0, 1.0))
    result = pitch.transcribe_pitched_audio(path, source_kind="vocals")
    event = result["events"][0]
    expected = 69.0 + 12.0 * math.log2(445.0 / 440.0)
    assert event["midiPitch"] == pytest.approx(expected, abs=0.15)
    assert not math.isclose(
        event["midiPitch"],
        float(event["midiNote"]),
        abs_tol=0.02,
    )
    accepted = [
        frame
        for frame in result["diagnostics"]["pyin"]["frameEvidence"]
        if frame["accepted"]
    ]
    assert any(
        not math.isclose(
            frame["midiPitch"],
            round(frame["midiPitch"]),
            abs_tol=0.02,
        )
        for frame in accepted
    )


def test_custom_algorithm_version_is_preserved_and_invalid_version_rejected(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "version.wav", tone(440.0, 0.7))
    result = pitch.transcribe_pitched_audio(
        path,
        source_kind="vocals",
        algorithm_version="baseline-pyin-v1.1",
    )
    assert result["algorithmVersion"] == "baseline-pyin-v1.1"
    with pytest.raises(pitch.PitchedTranscriptionError, match="algorithm version"):
        pitch.transcribe_pitched_audio(
            path,
            source_kind="vocals",
            algorithm_version="../unsafe",
        )


def test_two_separated_tones_produce_two_ordered_events(tmp_path: Path) -> None:
    audio = np.concatenate(
        [
            tone(440.0, 0.7),
            np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32),
            tone(523.251, 0.7),
        ]
    )
    result = pitch.transcribe_pitched_audio(
        write_wav(tmp_path / "two.wav", audio),
        source_kind="vocal",
    )
    assert [event["midiNote"] for event in result["events"]] == [69, 72]
    assert result["events"][0]["endSeconds"] < result["events"][1][
        "startSeconds"
    ]
    assert [event["id"] for event in result["events"]] == [
        "p000001",
        "p000002",
    ]


def test_material_pitch_change_splits_without_beat_quantization(
    tmp_path: Path,
) -> None:
    audio = np.concatenate([tone(440.0, 0.7), tone(523.251, 0.7)])
    result = pitch.transcribe_pitched_audio(
        write_wav(tmp_path / "change.wav", audio),
        source_kind="vocals",
    )
    assert [event["midiNote"] for event in result["events"]] == [69, 72]
    boundary = result["events"][0]["endSeconds"]
    assert boundary == pytest.approx(0.7, abs=0.08)
    assert not math.isclose(
        boundary,
        round(boundary * 4) / 4,
        abs_tol=1e-6,
    )
    assert result["diagnostics"]["eventing"]["timingQuantized"] is False


def test_silence_returns_no_fabricated_events(tmp_path: Path) -> None:
    path = write_wav(
        tmp_path / "silence.wav",
        np.zeros(SAMPLE_RATE, dtype=np.float32),
    )
    result = pitch.transcribe_pitched_audio(path, source_kind="vocals")
    assert result["events"] == []
    assert any("silent or too quiet" in warning for warning in result["warnings"])
    assert result["diagnostics"]["pyin"]["frameEvidence"] == []


def test_short_noise_spike_does_not_become_note_event(tmp_path: Path) -> None:
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    rng = np.random.default_rng(17)
    audio[5000:5100] = rng.normal(0.0, 0.8, 100).astype(np.float32)
    result = pitch.transcribe_pitched_audio(
        write_wav(tmp_path / "spike.wav", audio),
        source_kind="vocals",
    )
    assert result["events"] == []
    assert any(
        "too little reliable voicing" in warning
        or "short voiced glitches" in warning
        for warning in result["warnings"]
    )


def test_bass_source_range_detects_e2(tmp_path: Path) -> None:
    result = pitch.transcribe_pitched_audio(
        write_wav(tmp_path / "bass.wav", tone(82.4069, 1.5)),
        source_kind="bass",
    )
    assert len(result["events"]) == 1
    assert result["events"][0]["midiNote"] == 40
    assert result["events"][0]["noteName"] == "E2"
    range_info = result["diagnostics"]["range"]
    assert range_info["fminHz"] == pytest.approx(41.203, rel=0.01)
    assert range_info["fmaxHz"] == pytest.approx(523.251, rel=0.01)


def test_stereo_input_is_deterministic_and_input_is_unchanged(
    tmp_path: Path,
) -> None:
    mono = tone(440.0, 1.0)
    stereo = np.column_stack((mono, mono * 0.75)).astype(np.float32)
    path = write_wav(tmp_path / "stereo.wav", stereo)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    first = pitch.transcribe_pitched_audio(path, source_kind="lead_vocal")
    second = pitch.transcribe_pitched_audio(path, source_kind="lead_vocal")
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first == second
    assert before == after
    assert first["diagnostics"]["audio"]["channels"] == 2
    assert first["events"][0]["sourceKind"] == "lead_vocal"


def test_full_mix_polyphony_emits_honesty_warning(tmp_path: Path) -> None:
    audio = tone(440.0, 1.2) + 0.8 * tone(659.255, 1.2)
    result = pitch.transcribe_pitched_audio(
        write_wav(tmp_path / "poly.wav", audio),
        source_kind="full_mix",
    )
    assert any(
        "dominant pitch line" in warning and "not polyphonic" in warning
        for warning in result["warnings"]
    )
    assert_finite_json(result)


def test_custom_range_is_preserved_as_diagnostic_evidence(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "custom.wav", tone(440.0, 0.9))
    result = pitch.transcribe_pitched_audio(
        path,
        source_kind="custom_source",
        fmin_hz=400.0,
        fmax_hz=500.0,
    )
    assert result["diagnostics"]["range"] == {
        "fminHz": 400.0,
        "fmaxHz": 500.0,
        "usedSourceDefault": False,
    }
    assert result["events"][0]["midiPitch"] == pytest.approx(69.0, abs=0.15)


@pytest.mark.parametrize(
    "source_kind",
    [
        "",
        " vocals",
        "vocals ",
        "VOCALS",
        "../vocals",
        "vocal/path",
        "vocal.kind",
        "a" * 65,
    ],
)
def test_invalid_source_kind_is_rejected_without_path(
    source_kind: str,
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "input.wav", tone(440.0, 0.5))
    with pytest.raises(
        pitch.PitchedTranscriptionError,
        match="source kind is invalid",
    ) as caught:
        pitch.transcribe_pitched_audio(path, source_kind=source_kind)
    assert str(path) not in str(caught.value)


@pytest.mark.parametrize(
    ("fmin_hz", "fmax_hz"),
    [
        (500.0, 400.0),
        (0.0, 500.0),
        (float("nan"), 500.0),
        (100.0, float("inf")),
        (True, 500.0),
    ],
)
def test_invalid_frequency_range_is_rejected(
    tmp_path: Path,
    fmin_hz: Any,
    fmax_hz: Any,
) -> None:
    path = write_wav(tmp_path / "input.wav", tone(440.0, 0.5))
    with pytest.raises(pitch.PitchedTranscriptionError, match="frequency"):
        pitch.transcribe_pitched_audio(
            path,
            source_kind="vocals",
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )


def test_missing_nonfile_and_symlink_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(pitch.PitchedTranscriptionError, match="missing"):
        pitch.transcribe_pitched_audio(missing, source_kind="vocals")

    directory = tmp_path / "directory.wav"
    directory.mkdir()
    with pytest.raises(pitch.PitchedTranscriptionError, match="regular WAV"):
        pitch.transcribe_pitched_audio(directory, source_kind="vocals")

    target = write_wav(tmp_path / "target.wav", tone(440.0, 0.5))
    link = tmp_path / "link.wav"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(pitch.PitchedTranscriptionError, match="symbolic links"):
        pitch.transcribe_pitched_audio(link, source_kind="vocals")


def test_non_wav_empty_nonfinite_channels_and_sample_rate_are_rejected(
    tmp_path: Path,
) -> None:
    disguised = tmp_path / "disguised.wav"
    sf.write(disguised, tone(440.0, 0.5), SAMPLE_RATE, format="FLAC")
    with pytest.raises(pitch.PitchedTranscriptionError, match="not a WAV"):
        pitch.transcribe_pitched_audio(disguised, source_kind="vocals")

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(pitch.PitchedTranscriptionError, match="unreadable|empty"):
        pitch.transcribe_pitched_audio(empty, source_kind="vocals")

    nonfinite = tmp_path / "nonfinite.wav"
    write_wav(
        nonfinite,
        np.asarray([0.0, np.nan, 0.0], dtype=np.float32),
        sample_rate=8_000,
        subtype="FLOAT",
    )
    with pytest.raises(pitch.PitchedTranscriptionError, match="invalid numerical"):
        pitch.transcribe_pitched_audio(nonfinite, source_kind="vocals")

    three_channels = np.zeros((8_000, 3), dtype=np.float32)
    with pytest.raises(pitch.PitchedTranscriptionError, match="mono or stereo"):
        pitch.transcribe_pitched_audio(
            write_wav(
                tmp_path / "three.wav",
                three_channels,
                sample_rate=8_000,
            ),
            source_kind="vocals",
        )

    low_rate = write_wav(
        tmp_path / "low.wav",
        tone(440.0, 0.5, sample_rate=4_000),
        sample_rate=4_000,
    )
    with pytest.raises(pitch.PitchedTranscriptionError, match="sample rate"):
        pitch.transcribe_pitched_audio(low_rate, source_kind="vocals")


def test_duration_limit_rejects_before_loading_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_wav(
        tmp_path / "long.wav",
        np.zeros(8_000, dtype=np.float32),
        sample_rate=8_000,
    )
    original = sf.info(path)

    class FakeInfo:
        format = "WAV"
        channels = 1
        samplerate = 8_000
        frames = int((pitch._MAX_DURATION_SECONDS + 1) * samplerate)

    monkeypatch.setattr(pitch.sf, "info", lambda _path: FakeInfo())
    read_called = False

    def fail_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal read_called
        read_called = True
        raise AssertionError("read should not run")

    monkeypatch.setattr(pitch.sf, "read", fail_read)
    with pytest.raises(
        pitch.PitchedTranscriptionError,
        match="duration or memory limit",
    ):
        pitch.transcribe_pitched_audio(path, source_kind="vocals")
    assert read_called is False
    assert original.frames == 8_000


def test_analysis_failure_is_generic_and_path_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_wav(tmp_path / "input.wav", tone(440.0, 0.5))

    def fail_pyin(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"failure at {path}")

    monkeypatch.setattr(pitch.librosa, "pyin", fail_pyin)
    with pytest.raises(pitch.PitchedTranscriptionError) as caught:
        pitch.transcribe_pitched_audio(path, source_kind="vocals")
    assert str(caught.value) == (
        "Pitched transcription could not analyze this WAV reliably."
    )
    assert str(path) not in str(caught.value)


def test_output_contains_no_paths_arrays_or_notation_claims(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "input.wav", tone(440.0, 0.8))
    result = pitch.transcribe_pitched_audio(path, source_kind="other")
    encoded = json.dumps(result, allow_nan=False)
    assert str(tmp_path) not in encoded
    assert ".wav" not in encoded
    assert "score" not in encoded.lower()
    assert "measure" not in encoded.lower()
    assert "tablature" not in encoded.lower()
    assert result["diagnostics"]["eventing"]["timingQuantized"] is False

    def walk(value: Any) -> None:
        assert not isinstance(value, (Path, np.ndarray, np.generic))
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
