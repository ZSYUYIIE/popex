"""Symlink fixtures may only be skipped where the platform truly forbids them."""

from __future__ import annotations

import os

from conftest import symlinks_available


def test_symlink_fixtures_run_wherever_symlinks_can_exist() -> None:
    # Linux and macOS CI must run every symlink containment test; only
    # unprivileged Windows may skip them.
    if os.name != "nt":
        assert symlinks_available()
