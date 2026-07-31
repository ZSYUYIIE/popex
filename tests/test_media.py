import json
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.media import (
    MediaProcessingError,
    friendly_error,
    generated_source_name,
    normalize_audio,
    probe_media,
)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com",),
        max_duration_seconds=30,
        max_filesize_mb=10,
        max_upload_mb=10,
        audio_quality="192",
        ffmpeg_binary="ffmpeg-test",
        ffprobe_binary="ffprobe-test",
    )


def test_generated_safe_filename():
    name = generated_source_name(".mp3")
    assert name.startswith("source-")
    assert name.endswith(".mp3")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name


def test_normalized_wav_creation_with_mocked_subprocess(
    tmp_path: Path, monkeypatch
):
    config = settings(tmp_path)
    source = tmp_path / "source.mp3"
    output = tmp_path / "analysis.wav"
    source.write_bytes(b"source")

    monkeypatch.setattr("app.media.shutil.which", lambda value: f"/mock/{value}")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"RIFFsynthetic wav")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("app.media.subprocess.run", fake_run)
    normalize_audio(source, output, config)

    assert output.read_bytes() == b"RIFFsynthetic wav"


def test_probe_media_with_mocked_ffprobe(tmp_path: Path, monkeypatch):
    config = settings(tmp_path)
    source = tmp_path / "tone.wav"
    source.write_bytes(b"RIFF")
    monkeypatch.setattr("app.media.shutil.which", lambda value: f"/mock/{value}")

    payload = {
        "format": {"duration": "1.25", "format_name": "wav"},
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": "44100",
                "channels": 2,
            }
        ],
    }

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr("app.media.subprocess.run", fake_run)

    assert probe_media(source, config) == {
        "duration_seconds": 1.25,
        "format_name": "wav",
        "sample_rate": 44100,
        "channel_count": 2,
    }


def test_failed_ffmpeg_error_hides_local_path(tmp_path: Path, monkeypatch):
    config = settings(tmp_path)
    source = tmp_path / "private" / "song.mp3"
    output = tmp_path / "private" / "analysis.wav"
    source.parent.mkdir()
    source.write_bytes(b"x")
    monkeypatch.setattr("app.media.shutil.which", lambda value: f"/mock/{value}")

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr=f"{source}: Invalid data found when processing input",
        )

    monkeypatch.setattr("app.media.subprocess.run", fake_run)

    with pytest.raises(MediaProcessingError) as exc:
        normalize_audio(source, output, config)

    assert "Invalid data" in str(exc.value)
    assert str(tmp_path) not in str(exc.value)
    assert "song.mp3" not in str(exc.value)


def test_friendly_error_redacts_windows_path(tmp_path: Path):
    config = settings(tmp_path)
    message = friendly_error(
        r"C:\Users\Example\Music\secret.mp3: invalid data",
        settings=config,
    )
    assert "C:\\Users" not in message
    assert "secret.mp3" not in message
    assert "invalid data" in message
