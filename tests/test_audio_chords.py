from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app import audio_chords
from app.audio_chords import (
    ANALYSIS_SAMPLE_RATE,
    AUDIO_CHORDS_VERSION,
    AudioChordsError,
    audio_chords_file_name,
    build_audio_chords_artifact,
    load_audio,
    load_audio_chords_artifact,
    recognize_chords,
    validate_audio_chords_artifact,
    write_audio_chords_artifact,
)

SR = ANALYSIS_SAMPLE_RATE
BEAT = 0.5  # 120 BPM
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
JOB_ID = "ab" * 16


def tone(midi: float, seconds: float, level: float = 0.2) -> np.ndarray:
    """A plucked-sounding tone with a few harmonics. Synthetic; no recording."""
    time = np.arange(int(seconds * SR)) / SR
    frequency = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(np.sin(2 * np.pi * frequency * k * time) / k for k in (1, 2, 3, 4))
    envelope = np.exp(-1.5 * time) * np.minimum(1.0, time / 0.01)
    return (level * wave * envelope).astype("float32")


def chord_audio(symbol: str, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """``(accompaniment, bass)`` for one chord held for ``seconds``."""
    root = NOTE[symbol.rstrip("m")]
    third = 3 if symbol.endswith("m") else 4
    accompaniment = sum(tone(60 + (root + step) % 12, seconds) for step in (0, third, 7))
    return accompaniment.astype("float32"), tone(36 + root, seconds, level=0.3)


def progression(bars: list[str | None], *, meter: int = 4) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Accompaniment and bass stems for one chord (or silence) per bar, and the beat times."""
    parts, basses = [], []
    for symbol in bars:
        for _ in range(meter):
            if symbol is None:
                parts.append(np.zeros(int(BEAT * SR), dtype="float32"))
                basses.append(np.zeros(int(BEAT * SR), dtype="float32"))
            else:
                accompaniment, bass = chord_audio(symbol, BEAT)
                parts.append(accompaniment)
                basses.append(bass)
    beats = [index * BEAT for index in range(len(bars) * meter + 1)]
    return np.concatenate(parts), np.concatenate(basses), beats


def drums(seconds: float, seed: int = 3) -> np.ndarray:
    """Noise bursts on every beat: percussive, no pitch."""
    generator = np.random.default_rng(seed)
    signal = np.zeros(int(seconds * SR), dtype="float32")
    burst = int(0.05 * SR)
    for start in np.arange(0, seconds - 0.06, BEAT):
        index = int(start * SR)
        signal[index : index + burst] += (generator.standard_normal(burst) * np.linspace(1, 0, burst) * 0.5).astype("float32")
    return signal


def bar_symbols(result: dict, *, meter: int = 4) -> list[str | None]:
    """The chord on each bar when one chord holds the whole bar."""
    per_beat: list[str | None] = []
    for segment in result["segments"]:
        per_beat += [segment["symbol"]] * segment["beats"]
    bars = []
    for index in range(0, len(per_beat) - meter + 1, meter):
        cell = set(per_beat[index : index + meter])
        bars.append(cell.pop() if len(cell) == 1 else "mixed")
    return bars


# ---------------------------------------------------------------------------
# Recognition on synthetic audio


def test_stems_give_the_chord_on_every_bar():
    bars = ["Am", "Am", "C", "C", "G", "G", "G", "G", "F", "F", "Dm", "Dm", "Em", "Em", "C", "C"]
    accompaniment, bass, beats = progression(bars)
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)

    assert result["source"] == "stems"
    assert bar_symbols(result) == bars
    assert all(segment["confidence"] >= 0.6 for segment in result["segments"])
    assert result["beats"] == 64


def test_full_mix_alone_still_finds_the_chords_under_drums():
    bars = ["C", "C", "G", "G", "Am", "Am", "F", "F", "C", "C", "G", "G"]
    accompaniment, bass, beats = progression(bars)
    mix = accompaniment + bass + drums(len(accompaniment) / SR)
    result = recognize_chords(beats, meter=4, mix=mix)

    assert result["source"] == "mix"
    assert bar_symbols(result) == bars


def test_major_and_minor_on_the_same_root_are_told_apart():
    bars = ["A", "A", "D", "D", "E", "E", "A", "A", "F#m", "F#m", "D", "D"]
    accompaniment, bass, beats = progression(bars)
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    assert bar_symbols(result) == bars
    assert result["scaleTonic"] == "A"


def test_flat_keys_are_spelled_with_flats():
    bars = ["A#", "A#", "D#", "D#", "F", "F", "A#", "A#", "Gm", "Gm", "D#", "D#"]
    accompaniment, bass, beats = progression(bars)
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    assert bar_symbols(result) == ["Bb", "Bb", "Eb", "Eb", "F", "F", "Bb", "Bb", "Gm", "Gm", "Eb", "Eb"]
    assert result["scaleTonic"] == "Bb"


def test_drum_only_and_silent_bars_get_no_chord():
    bars = [None, None, None, None, "C", "C", "G", "G", "Am", "Am", "F", "F", "C", "C", "G", "G"]
    accompaniment, bass, beats = progression(bars)
    mix = accompaniment + bass + drums(len(accompaniment) / SR)
    result = recognize_chords(beats, meter=4, mix=mix, accompaniment=accompaniment, bass=bass)
    symbols = bar_symbols(result)

    assert result["source"] == "stems+mix"
    assert symbols[:3] == [None, None, None]
    assert symbols[5:] == bars[5:]
    opening = result["segments"][0]
    assert opening["noChord"] is True and opening["symbol"] is None and opening["confidence"] is None


def test_quiet_stems_fall_back_to_the_full_mix():
    # Bars 5-8 have no accompaniment or bass stem, but the mix carries a sung
    # line outlining the chord (here: the triad an octave higher).
    bars = ["C", "C", "G", "G", "Am", "Am", "F", "F", "C", "C", "G", "G"]
    accompaniment, bass, beats = progression(bars)
    voice = accompaniment.copy()
    quiet = slice(4 * 4 * int(BEAT * SR), 8 * 4 * int(BEAT * SR))
    mix = accompaniment + bass + drums(len(accompaniment) / SR)
    accompaniment = accompaniment.copy()
    bass = bass.copy()
    accompaniment[quiet] = 0.0
    bass[quiet] = 0.0
    mix[quiet] = voice[quiet] + drums(len(voice) / SR)[quiet]
    result = recognize_chords(beats, meter=4, mix=mix, accompaniment=accompaniment, bass=bass)

    assert bar_symbols(result) == bars
    shares = [segment["stemShare"] for segment in result["segments"] if segment["symbol"] in ("Am", "F")]
    assert shares and max(shares) < 0.2


def test_a_chord_change_on_the_half_bar_is_followed():
    # Two chords in bar 2: beats 1-2 C, beats 3-4 G.
    accompaniment, bass, beats = progression(["C", "C", "Am", "Am", "F", "F"])
    half = 2 * int(BEAT * SR)
    start = 4 * int(BEAT * SR) + half
    g_accompaniment, g_bass = chord_audio("G", BEAT)
    for offset in (0, int(BEAT * SR)):
        accompaniment[start + offset : start + offset + len(g_accompaniment)] = g_accompaniment
        bass[start + offset : start + offset + len(g_bass)] = g_bass
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    per_beat = [segment["symbol"] for segment in result["segments"] for _ in range(segment["beats"])]
    assert per_beat[4:8] == ["C", "C", "G", "G"]


def test_recognition_is_deterministic():
    accompaniment, bass, beats = progression(["C", "C", "F", "F", "G", "G", "C", "C"])
    first = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    second = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    assert first == second


def test_segments_tile_the_grid_without_gaps_or_overlaps():
    accompaniment, bass, beats = progression(["C", "G", "Am", "F", "C", "G", "Am", "F"])
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    segments = result["segments"]
    assert segments[0]["startSeconds"] == 0.0
    assert segments[-1]["endSeconds"] == pytest.approx(beats[-1], abs=1e-3)
    for previous, current in zip(segments, segments[1:]):
        assert current["startSeconds"] == previous["endSeconds"]
    assert sum(segment["beats"] for segment in segments) == result["beats"]
    assert [segment["id"] for segment in segments] == [f"c{index + 1:06d}" for index in range(len(segments))]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"meter": 0},
        {"meter": 4.0},
        {"beats": [0.0, 0.5, 1.0]},
        {"beats": [0.0, 0.5, 0.5, 1.0, 1.5, 2.0]},
        {"beats": [0.0, 0.5, float("nan"), 1.5, 2.0, 2.5]},
        {"no_audio": True},
        {"no_bass": True},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    accompaniment, bass, beats = progression(["C", "C"])
    arguments = {"meter": kwargs.get("meter", 4), "accompaniment": accompaniment, "bass": bass}
    if kwargs.get("no_audio"):
        arguments.update(accompaniment=None, bass=None)
    if kwargs.get("no_bass"):
        arguments.update(bass=None)
    with pytest.raises(AudioChordsError):
        recognize_chords(kwargs.get("beats", beats), **arguments)


def test_load_audio_reads_mono_at_the_analysis_rate(tmp_path):
    stereo = np.stack([tone(69, 1.0), tone(69, 1.0)], axis=1)
    path = tmp_path / "clip.wav"
    sf.write(path, np.repeat(stereo, 2, axis=0), SR * 2)
    loaded = load_audio(path)
    assert loaded.dtype == np.float32 and loaded.ndim == 1
    assert abs(len(loaded) - SR) <= 2
    with pytest.raises(AudioChordsError):
        load_audio(tmp_path / "missing.wav")


# ---------------------------------------------------------------------------
# Saved result


class _Settings:
    def __init__(self, root: Path) -> None:
        self.exports_dir = root


def make_artifact(harmony_created_at: str = "2026-11-16T00:00:00+00:00") -> dict:
    accompaniment, bass, beats = progression(["C", "C", None, None, "G", "G"])
    result = recognize_chords(beats, meter=4, accompaniment=accompaniment, bass=bass)
    return build_audio_chords_artifact(
        result,
        job_id=JOB_ID,
        created_at="2026-11-16T00:00:01+00:00",
        harmony_created_at=harmony_created_at,
        analysis_version="baseline-librosa-v1",
        grid_version="beat-grid-v1",
    )


@pytest.fixture(scope="module")
def artifact() -> dict:
    return make_artifact()


def test_artifact_is_versioned_and_round_trips(tmp_path, artifact):
    settings = _Settings(tmp_path)
    (tmp_path / JOB_ID).mkdir()
    relative = write_audio_chords_artifact(settings, artifact)

    assert artifact["chordsVersion"] == AUDIO_CHORDS_VERSION
    assert relative == audio_chords_file_name(artifact["harmonyCreatedAt"])
    assert relative.startswith("chords/audio-chords.") and relative.endswith(".json")
    loaded = load_audio_chords_artifact(settings, JOB_ID, artifact["harmonyCreatedAt"])
    assert loaded == artifact
    assert load_audio_chords_artifact(settings, JOB_ID, "another-harmony-run") is None
    assert not list((tmp_path / JOB_ID / "chords").glob("*.tmp"))
    text = (tmp_path / JOB_ID / relative).read_text(encoding="utf-8")
    assert str(tmp_path) not in text


def test_only_the_two_newest_results_are_kept(tmp_path, artifact):
    import os

    settings = _Settings(tmp_path)
    (tmp_path / JOB_ID).mkdir()
    names = []
    for index in range(4):
        version = {**artifact, "harmonyCreatedAt": f"run-{index}"}
        name = write_audio_chords_artifact(settings, version)
        names.append(name)
        os.utime(tmp_path / JOB_ID / name, (1_700_000_000 + index, 1_700_000_000 + index))
    kept = sorted(path.name for path in (tmp_path / JOB_ID / "chords").iterdir())
    assert kept == sorted(Path(name).name for name in names[-2:])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(schemaVersion=2),
        lambda doc: doc.update(artifactType="other"),
        lambda doc: doc.update(jobId="../../etc"),
        lambda doc: doc.update(source="guess"),
        lambda doc: doc.update(scaleTonic="H"),
        lambda doc: doc.update(beats=0),
        lambda doc: doc.update(extra=True),
        lambda doc: doc.pop("gridVersion"),
        lambda doc: doc["segments"][0].update(symbol="Cmaj7"),
        lambda doc: doc["segments"][0].update(symbol="C/E"),
        lambda doc: doc["segments"][0].update(confidence=0.1),
        lambda doc: doc["segments"][0].update(confidence=None),
        lambda doc: doc["segments"][0].update(noChord=True),
        lambda doc: doc["segments"][0].update(startSeconds=float("nan")),
        lambda doc: doc["segments"][0].update(endSeconds=0.0),
        lambda doc: doc["segments"][0].update(id="c000009"),
        lambda doc: doc["segments"][0].update(beats=10_000),
        lambda doc: doc["segments"][1].update(startSeconds=0.0),
        lambda doc: doc["segments"][0].update(path="C:/Users/someone"),
    ],
)
def test_tampered_artifacts_are_rejected(artifact, mutate):
    document = json.loads(json.dumps(artifact))
    mutate(document)
    with pytest.raises(AudioChordsError):
        validate_audio_chords_artifact(document)


def test_unreadable_or_foreign_files_are_refused(tmp_path, artifact):
    settings = _Settings(tmp_path)
    (tmp_path / JOB_ID).mkdir()
    relative = write_audio_chords_artifact(settings, artifact)
    path = tmp_path / JOB_ID / relative

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AudioChordsError):
        load_audio_chords_artifact(settings, JOB_ID, artifact["harmonyCreatedAt"])
    other_job = {**artifact, "jobId": "cd" * 16}
    path.write_text(json.dumps(other_job), encoding="utf-8")
    with pytest.raises(AudioChordsError):
        load_audio_chords_artifact(settings, JOB_ID, artifact["harmonyCreatedAt"])
    with pytest.raises(AudioChordsError):
        load_audio_chords_artifact(settings, "not-a-job", artifact["harmonyCreatedAt"])


def test_named_chords_always_meet_the_confidence_floor(artifact):
    for segment in artifact["segments"]:
        if segment["symbol"] is not None:
            assert segment["confidence"] >= audio_chords.MIN_NAMED_CONFIDENCE
        else:
            assert segment["confidence"] is None
