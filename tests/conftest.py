"""Shared test helpers."""

from __future__ import annotations

import functools
import os
import tempfile

import pytest


@functools.lru_cache(maxsize=1)
def symlinks_available() -> bool:
    """Return whether this process may create file and directory symlinks.

    Unprivileged Windows refuses both (WinError 1314) unless Developer Mode is
    on. Fixtures that need a symlink skip explicitly instead of failing, while
    the containment checks they exercise still run wherever symlinks can be
    created, which includes Linux CI (enforced by test_symlink_capability).
    """
    with tempfile.TemporaryDirectory() as root:
        target_dir = os.path.join(root, "target-dir")
        target_file = os.path.join(root, "target-file")
        os.mkdir(target_dir)
        with open(target_file, "w", encoding="utf-8"):
            pass
        try:
            os.symlink(target_dir, os.path.join(root, "dir-link"), target_is_directory=True)
            os.symlink(target_file, os.path.join(root, "file-link"))
        except (OSError, NotImplementedError):
            return False
    return True


SYMLINK_SKIP_REASON = "Creating symlinks needs extra privileges on this platform (Windows Developer Mode)."


@pytest.fixture
def require_symlinks() -> None:
    """Skip a test whose fixture must create a symlink when the platform forbids it."""
    if not symlinks_available():
        pytest.skip(SYMLINK_SKIP_REASON)
