from pathlib import Path

import pytest

from app.separation_service import _sanitize_public_text
from tests.test_stem_api import make_settings


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_separation_error_redacts_entire_known_root_descendant(
    tmp_path: Path, separator: str
) -> None:
    settings = make_settings(tmp_path)
    private_path = str(settings.data_dir) + separator + "private" + separator + "runtime-lock.json"
    result = _sanitize_public_text(
        f"Input unavailable at {private_path}; retry after checking permissions.",
        settings=settings,
        fallback="Unavailable.",
        limit=800,
    )
    assert "retry after checking permissions" in result
    assert "private" not in result
    assert "runtime-lock.json" not in result
    assert str(settings.data_dir) not in result
