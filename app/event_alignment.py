"""Advisory alignment of raw transcription events to analysis timing.

Issue #47 implementation is in progress. Raw event timestamps remain authoritative;
this module will add only derived alignment candidates.
"""

from __future__ import annotations


class EventAlignmentError(RuntimeError):
    """Raised when raw events or timing evidence are malformed."""
