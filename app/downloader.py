"""Compatibility exports for the original URL extraction module."""

from app.media import (
    MediaProcessingError as ExtractionError,
    MediaResult as ExtractionResult,
    process_url as extract_audio,
)

__all__ = ["ExtractionError", "ExtractionResult", "extract_audio"]
