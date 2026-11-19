"""Bar-aligned song sections: where the arrangement changes, and what repeats.

Sections are found on the score's own bar grid, so every boundary is a bar
line. For each bar PopEx measures the loudness of each separated stem (when
stems exist), the harmony (chroma) and the timbre (MFCC) of the mix. A
checkerboard novelty curve over the bar-to-bar similarity matrix marks where
the sound changes; sections are at least four bars long. Sections that sound
alike share a letter.

Letters mean "these bars sound alike". They are not verse or chorus names:
PopEx does not claim to know a song's form. Each section also says which
stems are clearly playing in it, measured against that recording's own
levels, so a musician can tell the drums-only intro from the full chorus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SECTIONS_VERSION = "bar-sections-v1"
STEM_ORDER = ("vocals", "drums", "bass", "other")
STEM_LABELS = {"vocals": "vocals", "drums": "drums", "bass": "bass", "other": "accompaniment"}
_MIN_BARS = 4
_KERNEL_BARS = 4
_PEAK_RADIUS = 3
_MIN_NOVELTY = 0.2
_SAME_LABEL_DISTANCE = 2.2
_ACTIVE_BELOW_PEAK_DB = 15.0
_SILENT_DB = -55.0
MIN_BARS_FOR_SECTIONS = 16
_MAX_SECTIONS = 64
_SAMPLE_RATE = 22050


class SectionError(ValueError):
    """Section evidence or a saved section summary is invalid."""


def bar_features(
    bar_spans: Sequence[tuple[float, float]],
    *,
    mix: np.ndarray,
    stems: Mapping[str, np.ndarray] | None = None,
    sample_rate: int = _SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return per-bar features and per-bar stem levels in dB (or ``None``).

    ``bar_spans`` are ``(startSeconds, endSeconds)`` per bar; a pickup bar may
    start before zero and is clipped. Audio is mono at ``sample_rate``.
    """
    import librosa

    def window(signal: np.ndarray, start: float, end: float) -> np.ndarray:
        first = max(0, int(round(start * sample_rate)))
        last = max(first + 1, int(round(end * sample_rate)))
        piece = signal[first:last]
        return piece if piece.size >= 2048 else np.pad(piece, (0, 2048 - piece.size))

    levels = None
    if stems:
        levels = np.array(
            [
                [
                    20.0 * np.log10(float(np.sqrt(np.mean(window(stems[name], a, b) ** 2))) + 1e-6)
                    if name in stems
                    else -120.0
                    for name in STEM_ORDER
                ]
                for a, b in bar_spans
            ]
        )
    rows = []
    for a, b in bar_spans:
        piece = window(mix, a, b)
        loudness = 20.0 * np.log10(float(np.sqrt(np.mean(piece**2))) + 1e-6)
        if np.max(np.abs(piece)) < 1e-5:
            chroma = np.zeros(12)
            mfcc = np.zeros(12)
        else:
            chroma = librosa.feature.chroma_stft(y=piece, sr=sample_rate, tuning=0.0).mean(axis=1)
            chroma = chroma / (np.linalg.norm(chroma) + 1e-9)
            mfcc = librosa.feature.mfcc(y=piece, sr=sample_rate, n_mfcc=13).mean(axis=1)[1:]
        rows.append(np.concatenate([[loudness], chroma, mfcc]))
    return np.array(rows), levels


def find_sections(
    features: np.ndarray,
    stem_levels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Split bars into sections and letter the ones that sound alike.

    Returns ``{"version", "sections", "note"}``. ``sections`` lists
    ``startMeasure`` (0-based, inclusive), ``endMeasure`` (exclusive),
    ``label``, ``boundaryStrength`` and ``heard`` (stems clearly playing).
    """
    bar_count = int(features.shape[0]) if features.ndim == 2 else 0
    if bar_count < MIN_BARS_FOR_SECTIONS:
        return {
            "version": SECTIONS_VERSION,
            "sections": [],
            "note": f"Sections need at least {MIN_BARS_FOR_SECTIONS} bars; this recording is shorter.",
        }
    columns = [features]
    weights = [np.concatenate([[1.0], np.full(12, 0.5), np.full(features.shape[1] - 13, 1.0)])]
    if stem_levels is not None:
        columns.append(np.maximum(stem_levels, -90.0))
        weights.append(np.full(stem_levels.shape[1], 2.0))
    raw = np.hstack(columns)
    scale = raw.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (raw - raw.mean(axis=0)) / scale * np.concatenate(weights)

    distance = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    typical = float(np.median(distance)) or 1.0
    similarity = np.exp(-distance / typical)
    kernel = np.kron(np.array([[1.0, -1.0], [-1.0, 1.0]]), np.ones((_KERNEL_BARS, _KERNEL_BARS)))
    novelty = np.zeros(bar_count)
    for index in range(_KERNEL_BARS, bar_count - _KERNEL_BARS + 1):
        block = similarity[index - _KERNEL_BARS : index + _KERNEL_BARS, index - _KERNEL_BARS : index + _KERNEL_BARS]
        novelty[index] = float((block * kernel).sum())
    novelty = np.maximum(novelty, 0.0)
    peak = float(novelty.max())
    if peak <= 1e-9:
        return {
            "version": SECTIONS_VERSION,
            "sections": [],
            "note": "The sound does not change enough between bars to mark sections.",
        }
    novelty /= peak

    candidates = [
        index
        for index in range(1, bar_count)
        if novelty[index] >= _MIN_NOVELTY
        and novelty[index] == novelty[max(0, index - _PEAK_RADIUS) : index + _PEAK_RADIUS + 1].max()
    ]
    bounds = [0]
    for index in sorted(candidates, key=lambda item: -novelty[item]):
        if all(abs(index - other) >= _MIN_BARS for other in bounds) and bar_count - index >= _MIN_BARS:
            bounds.append(index)
        if len(bounds) >= _MAX_SECTIONS:
            break
    bounds = sorted(bounds) + [bar_count]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    means = [z[start:end].mean(axis=0) for start, end in spans]
    prototypes: list[np.ndarray] = []
    labels: list[str] = []
    for mean in means:
        gaps = [float(np.linalg.norm(mean - prototype)) for prototype in prototypes]
        if gaps and min(gaps) < _SAME_LABEL_DISTANCE:
            labels.append(_letter(int(np.argmin(gaps))))
        else:
            prototypes.append(mean)
            labels.append(_letter(len(prototypes) - 1))

    peaks = None
    if stem_levels is not None:
        peaks = np.percentile(stem_levels, 90, axis=0)
    sections: list[dict[str, Any]] = []
    for (start, end), label in zip(spans, labels):
        heard = None
        if stem_levels is not None and peaks is not None:
            level = stem_levels[start:end].mean(axis=0)
            heard = [
                name
                for column, name in enumerate(STEM_ORDER)
                if level[column] > _SILENT_DB and level[column] >= peaks[column] - _ACTIVE_BELOW_PEAK_DB
            ]
        if sections and sections[-1]["label"] == label and sections[-1]["heard"] == heard:
            sections[-1]["endMeasure"] = end
            continue
        sections.append(
            {
                "startMeasure": start,
                "endMeasure": end,
                "label": label,
                "boundaryStrength": round(float(novelty[start]) if start else 1.0, 3),
                "heard": heard,
            }
        )
    return {"version": SECTIONS_VERSION, "sections": sections, "note": None}


def public_sections(document: Mapping[str, Any]) -> dict[str, Any] | None:
    """Sections of a saved score for the review page, with times and plain text.

    ``None`` for scores saved before sections existed.
    """
    value = document.get("sections")
    if not isinstance(value, Mapping):
        return None
    measures = document["measures"]
    items = []
    for item in value["sections"]:
        first = measures[item["startMeasure"]]
        last = measures[item["endMeasure"] - 1]
        items.append(
            {
                "label": item["label"],
                "startMeasure": item["startMeasure"],
                "endMeasure": item["endMeasure"],
                "bars": item["endMeasure"] - item["startMeasure"],
                "startSeconds": round(max(0.0, float(first["startSeconds"])), 3),
                "endSeconds": round(float(last["endSeconds"]), 3),
                "heard": item["heard"],
                "heardText": describe_heard(item["heard"]),
                "boundaryStrength": item["boundaryStrength"],
            }
        )
    return {
        "version": value["version"],
        "note": value["note"],
        "sections": items,
        "explanation": (
            "Letters mark bars that sound alike; they are not verse or chorus names."
        ),
    }


def score_section_form(document: Mapping[str, Any]) -> str | None:
    """The section letters of a score in order, e.g. ``"A B C B"``, or ``None``."""
    value = document.get("sections")
    if not isinstance(value, Mapping) or not value.get("sections"):
        return None
    return " ".join(item["label"] for item in value["sections"])[:400]


def section_note(note: str) -> dict[str, Any]:
    """A section summary that explains why there are no sections."""
    return {"version": SECTIONS_VERSION, "sections": [], "note": note}


def _letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"Z{index - 25}"


def describe_heard(heard: Sequence[str] | None) -> str | None:
    """Musician-facing text for the stems heard in a section."""
    if heard is None:
        return None
    if not heard:
        return "quiet"
    names = [STEM_LABELS[name] for name in STEM_ORDER if name in heard]
    if len(names) == len(STEM_ORDER):
        return "full band"
    if len(names) == 1:
        return f"{names[0]} only"
    return ", ".join(names)


def validate_sections(value: Any, measure_count: int) -> dict[str, Any]:
    """Validate a saved section summary against the score's bar count."""
    if not isinstance(value, Mapping) or set(value) != {"version", "sections", "note"}:
        raise SectionError("sections is malformed.")
    if value["version"] != SECTIONS_VERSION:
        raise SectionError("sections.version is unsupported.")
    note = value["note"]
    if note is not None and (not isinstance(note, str) or not 0 < len(note) <= 300):
        raise SectionError("sections.note is invalid.")
    items = value["sections"]
    if not isinstance(items, list) or len(items) > _MAX_SECTIONS:
        raise SectionError("sections.sections is invalid.")
    if bool(items) == (note is not None):
        raise SectionError("Sections carry either sections or a note.")
    cursor = 0
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or set(item) != {
            "startMeasure", "endMeasure", "label", "boundaryStrength", "heard"
        }:
            raise SectionError(f"sections[{index}] is malformed.")
        start, end = item["startMeasure"], item["endMeasure"]
        if any(type(x) is not int for x in (start, end)) or start != cursor or end <= start:
            raise SectionError(f"sections[{index}] does not follow the previous section.")
        cursor = end
        label = item["label"]
        if not isinstance(label, str) or not (
            (len(label) == 1 and "A" <= label <= "Z")
            or (label.startswith("Z") and label[1:].isdigit() and 1 <= len(label) <= 3)
        ):
            raise SectionError(f"sections[{index}].label is invalid.")
        strength = item["boundaryStrength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            raise SectionError(f"sections[{index}].boundaryStrength is invalid.")
        heard = item["heard"]
        if heard is not None and (
            not isinstance(heard, list)
            or len(set(heard)) != len(heard)
            or any(name not in STEM_ORDER for name in heard)
        ):
            raise SectionError(f"sections[{index}].heard is invalid.")
    if items and cursor != measure_count:
        raise SectionError("Sections do not cover every bar.")
    return {"version": value["version"], "sections": [dict(item) for item in items], "note": note}
