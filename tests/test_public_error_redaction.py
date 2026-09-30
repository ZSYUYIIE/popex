from pathlib import Path

import pytest

from app.separation_service import _sanitize_public_text
from app.media import friendly_error, redact_local_paths
from tests.test_stem_api import make_settings


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_separation_error_redacts_entire_known_root_descendant(
    tmp_path: Path, separator: str
) -> None:
    settings = make_settings(tmp_path)
    private_path = str(settings.data_dir) + separator + "private" + separator + "runtime-lock.json"
    result = _sanitize_public_text(
        f"Input unavailable at {private_path}: retry after checking permissions.",
        settings=settings,
        fallback="Unavailable.",
        limit=800,
    )
    assert "retry after checking permissions" in result
    assert "private" not in result
    assert "runtime-lock.json" not in result
    assert str(settings.data_dir) not in result


@pytest.mark.parametrize("path", [
    r"C:\Users\Private Music\secret-take.wav",
    r"c:/Users/Private Music/secret-take.wav",
    r"\\Server\Private Share\secret-take.wav",
    "/tmp/private music/secret-take.wav",
    r"C:\Users\Private Music/secret-take.wav",
])
@pytest.mark.parametrize("quoted", [False, True])
def test_spaced_paths_never_leak_private_basename(
    tmp_path: Path, path: str, quoted: bool
) -> None:
    settings = make_settings(tmp_path)
    location = f'"{path}"' if quoted else path
    value = f"Cannot read {location}: invalid data; retry with supported audio."
    media = friendly_error(value, settings=settings)
    separation = _sanitize_public_text(
        value, settings=settings, fallback="Unavailable.", limit=800
    )
    for text in (media, separation):
        assert "secret-take.wav" not in text
        assert "Private Music" not in text
        assert "private music" not in text
        assert "Private Share" not in text
        assert "invalid data" in text
        assert "retry with supported audio" in text


def test_known_windows_root_case_and_prefix_do_not_expose_descendants() -> None:
    roots = (Path(r"C:\Private Root"), Path(r"C:\Private Root\secret-take.wav"))
    result = redact_local_paths(
        r"c:\PRIVATE ROOT\secret-take.wav: invalid data; C:\Private Root-Backup\other.wav: unreadable",
        paths=roots,
    )
    assert "secret-take.wav" not in result
    assert "other.wav" not in result
    assert "PRIVATE ROOT" not in result
    assert "invalid data" in result
    assert "unreadable" in result


def test_multiple_paths_cannot_strip_the_next_drive_prefix(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = str(settings.data_dir) + r"\private\first-secret.wav"
    value = first + r" and C:\Private Music\second-secret.wav: invalid data"
    for text in (
        friendly_error(value, settings=settings),
        _sanitize_public_text(value, settings=settings, fallback="Unavailable.", limit=800),
    ):
        assert "first-secret.wav" not in text
        assert "second-secret.wav" not in text
        assert "Private Music" not in text
        assert "invalid data" in text


@pytest.mark.parametrize("path", [
    r"C:\Private Music\session; secret-take,01.wav",
    "/tmp/private music/session; secret-take,01.wav",
    r"\\Server\Private Share\session; secret-take,01.wav",
    r"C:\Private Music\O'Connor[session]\secret-take.wav",
    "/tmp/private music/O'Connor[session]/secret-take.wav",
])
@pytest.mark.parametrize("quoted", [False, True])
def test_punctuation_rich_filenames_remain_whole(path: str, quoted: bool) -> None:
    location = f'"{path}"' if quoted else path
    value = f'Cannot read {location}: invalid data; retry later.'
    result = redact_local_paths(value)
    assert "secret-take" not in result
    assert "Private" not in result
    assert "private music" not in result
    assert "invalid data; retry later" in result


def test_error_redaction_preserves_plain_diagnostics_and_removes_credentials(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    value = "Audio has invalid data; retry after checking permissions."
    assert friendly_error(value, settings=settings) == value
    assert _sanitize_public_text(value, settings=settings, fallback="Unavailable.", limit=800) == value
    for value in (
        r"C:\Private Root\secret-take.wav: invalid data; token=synthetic-token",
        '"/tmp/private music/secret-take.wav": invalid data; authorization=synthetic-credential',
    ):
        text = _sanitize_public_text(value, settings=settings, fallback="Unavailable.", limit=800)
        assert "synthetic-token" not in text
        assert "synthetic-credential" not in text
        assert "secret-take.wav" not in text
        assert "invalid data" in text
    assert len(_sanitize_public_text("diagnostic " * 1000, settings=settings,
                                   fallback="Unavailable.", limit=800)) <= 800
