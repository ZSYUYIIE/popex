"""A time map between seconds and beats, built from tracked beats.

A draft score used to place every bar on one constant tempo. A tempo that is
off by one percent slides the bar lines by a whole bar over a four-minute
recording, and a recording whose tempo moves never fits at all. A
:class:`BeatGrid` follows the beats the analysis actually tracked instead:

- when the tracked beats sit on one straight line, the grid is that line
  (``steady``), which removes tracker jitter from a click-track recording;
- otherwise the grid follows locally smoothed beats (``following``);
- detected onsets may nudge the grid onto the played beat, because a beat
  tracker tends to report beats a few milliseconds late;
- the first beat of the bar comes from the kick and snare pattern when the
  drums show one, otherwise from confident analysis downbeats, otherwise the
  grid simply starts on the first tracked beat and says so.

Nothing here changes raw event times. The module is deterministic and uses the
standard library only.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Sequence
from typing import Any

BEAT_GRID_VERSION = "beat-grid-v1"
GRID_MODES = frozenset({"steady", "following"})
BAR_PHASE_SOURCES = frozenset({"drums", "analysis", "none"})

MIN_TRACKED_BEATS = 8
_MAX_BEATS = 20000
_MIN_BEAT_SECONDS = 0.2  # 300 BPM
_MAX_BEAT_SECONDS = 3.0  # 20 BPM
# A tracked beat sequence is "steady" when one straight line explains it.
_STEADY_RESIDUAL_STD_SECONDS = 0.03
_STEADY_MAX_RESIDUAL_BEATS = 0.35
_SMOOTH_HALF_WINDOW = 8
# Onsets this close to a grid beat (in beats) vote on where the beat really is.
_NEAR_BEAT = 0.2
_MIN_ONSET_VOTES = 16
_MAX_ONSET_SHIFT_BEATS = 0.2
# Bar-phase evidence.
_MIN_DRUM_VOTES = 8
_MIN_DRUM_MARGIN = 0.15
_MIN_ANALYSIS_DOWNBEATS = 4
_MIN_ANALYSIS_SHARE = 0.8
_MIN_ANALYSIS_METER_CONFIDENCE = 0.5
# A downbeat this soon after the start (in beats) begins bar 1; a later first
# downbeat means the recording opens with a pickup, which gets its own bar.
_LEAD_IN_TOLERANCE_BEATS = 0.5


class BeatGrid:
    """Monotonic map between seconds and beats. Beat 0 is beat 1 of bar 1."""

    __slots__ = (
        "times",
        "meter",
        "mode",
        "tempo_bpm",
        "bar_phase_source",
        "bar_phase_confidence",
        "onset_shift_seconds",
        "tracked_beats",
    )

    def __init__(
        self,
        times: Sequence[float],
        *,
        meter: int,
        mode: str,
        tempo_bpm: float,
        bar_phase_source: str,
        bar_phase_confidence: float | None,
        onset_shift_seconds: float,
        tracked_beats: int,
    ) -> None:
        self.times = tuple(float(value) for value in times)
        if len(self.times) < 2 or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("Grid beat times must be strictly increasing.")
        if mode not in GRID_MODES or bar_phase_source not in BAR_PHASE_SOURCES:
            raise ValueError("Unknown grid mode or bar-phase source.")
        self.meter = int(meter)
        self.mode = mode
        self.tempo_bpm = float(tempo_bpm)
        self.bar_phase_source = bar_phase_source
        self.bar_phase_confidence = bar_phase_confidence
        self.onset_shift_seconds = float(onset_shift_seconds)
        self.tracked_beats = int(tracked_beats)

    def beat_at(self, seconds: float) -> float:
        """Beat position of a time; extrapolates past either end."""
        times = self.times
        index = bisect.bisect_right(times, seconds) - 1
        index = min(max(index, 0), len(times) - 2)
        span = times[index + 1] - times[index]
        return index + (seconds - times[index]) / span

    def seconds_at(self, beat: float) -> float:
        """Time of a beat position; extrapolates past either end."""
        times = self.times
        index = min(max(int(math.floor(beat)), 0), len(times) - 2)
        return times[index] + (beat - index) * (times[index + 1] - times[index])

    def summary(self) -> dict[str, Any]:
        """Compact, serializable description of how the grid was built."""
        return {
            "version": BEAT_GRID_VERSION,
            "mode": self.mode,
            "barPhaseSource": self.bar_phase_source,
            "barPhaseConfidence": (
                None if self.bar_phase_confidence is None else round(self.bar_phase_confidence, 3)
            ),
            "onsetShiftSeconds": round(self.onset_shift_seconds, 4),
            "trackedBeats": self.tracked_beats,
            "firstBeatSeconds": round(self.times[0], 4),
        }


def _clean_beats(values: Any) -> list[float] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if not MIN_TRACKED_BEATS <= len(values) <= _MAX_BEATS:
        return None
    beats: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            return None
        if beats and number <= beats[-1]:
            return None
        beats.append(number)
    return beats


def _fill_gaps(beats: list[float]) -> list[float] | None:
    """Insert evenly spaced beats where the tracker skipped some."""
    median = statistics.median(b - a for a, b in zip(beats, beats[1:]))
    if not _MIN_BEAT_SECONDS <= median <= _MAX_BEAT_SECONDS:
        return None
    filled = [beats[0]]
    for previous, current in zip(beats, beats[1:]):
        steps = max(1, int(round((current - previous) / median)))
        if steps > 64:
            return None
        for step in range(1, steps + 1):
            filled.append(previous + (current - previous) * step / steps)
        if len(filled) > _MAX_BEATS:
            return None
    return filled


def _line_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares ``y = intercept + slope * x``."""
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread <= 0.0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    return mean_y - slope * mean_x, slope


def _smooth(beats: list[float]) -> tuple[list[float], str]:
    """Return ``(grid beats, mode)`` for cleaned, gap-filled tracked beats."""
    indices = list(range(len(beats)))
    intercept, slope = _line_fit(indices, beats)
    if slope > 0.0:
        residuals = [time - (intercept + slope * index) for index, time in zip(indices, beats)]
        spread = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        if (
            spread <= _STEADY_RESIDUAL_STD_SECONDS
            and max(abs(value) for value in residuals) <= _STEADY_MAX_RESIDUAL_BEATS * slope
        ):
            return [intercept + slope * index for index in indices], "steady"
    smoothed: list[float] = []
    for index in indices:
        low = max(0, index - _SMOOTH_HALF_WINDOW)
        high = min(len(beats), index + _SMOOTH_HALF_WINDOW + 1)
        local_intercept, local_slope = _line_fit(indices[low:high], beats[low:high])
        smoothed.append(local_intercept + local_slope * index)
    if any(b <= a for a, b in zip(smoothed, smoothed[1:])):
        return list(beats), "following"
    return smoothed, "following"


def _position(times: Sequence[float], seconds: float) -> float:
    index = bisect.bisect_right(times, seconds) - 1
    index = min(max(index, 0), len(times) - 2)
    return index + (seconds - times[index]) / (times[index + 1] - times[index])


def _onset_shift(times: list[float], onsets: Sequence[float] | None) -> float:
    """Median offset, in beats, of onsets that fall near a grid beat."""
    if not onsets:
        return 0.0
    offsets = []
    for onset in onsets:
        if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset):
            continue
        if not times[0] - 1.0 <= onset <= times[-1] + 1.0:
            continue
        position = _position(times, float(onset))
        offset = position - round(position)
        if abs(offset) <= _NEAR_BEAT:
            offsets.append(offset)
    if len(offsets) < _MIN_ONSET_VOTES:
        return 0.0
    shift = statistics.median(offsets)
    return shift if abs(shift) <= _MAX_ONSET_SHIFT_BEATS else 0.0


def _drum_phase(times: list[float], meter: int, drum_hits: Sequence[tuple[float, str]] | None) -> tuple[int, float] | None:
    """Bar phase from a backbeat: snare on beats 2 and 4, kick favouring beat 1.

    Returns ``(index of a downbeat among the grid beats modulo meter,
    margin)`` or ``None`` when the drums do not settle it.
    """
    if meter != 4 or not drum_hits:
        return None
    low = [0.0] * meter
    mid = [0.0] * meter
    for seconds, voice in drum_hits:
        if voice not in ("low", "mid") or not math.isfinite(seconds):
            continue
        position = _position(times, seconds)
        nearest = round(position)
        if abs(position - nearest) > _NEAR_BEAT:
            continue
        (low if voice == "low" else mid)[nearest % meter] += 1.0
    low_total, mid_total = sum(low), sum(mid)
    if low_total < _MIN_DRUM_VOTES and mid_total < _MIN_DRUM_VOTES:
        return None
    scores = []
    for phase in range(meter):
        backbeat = 0.0
        if mid_total >= _MIN_DRUM_VOTES:
            backbeat = (mid[(phase + 1) % 4] + mid[(phase + 3) % 4] - mid[phase] - mid[(phase + 2) % 4]) / mid_total
        downbeat = 0.0
        if low_total >= _MIN_DRUM_VOTES:
            downbeat = (low[phase] - low[(phase + 2) % 4]) / low_total
        scores.append((backbeat + downbeat, phase))
    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
    margin = ranked[0][0] - ranked[1][0]
    if ranked[0][0] <= 0.0 or margin < _MIN_DRUM_MARGIN:
        return None
    return ranked[0][1], min(1.0, margin)


def _analysis_phase(
    times: list[float],
    meter: int,
    downbeats: Sequence[float] | None,
    meter_confidence: float | None,
) -> tuple[int, float] | None:
    if (
        not downbeats
        or meter_confidence is None
        or meter_confidence < _MIN_ANALYSIS_METER_CONFIDENCE
    ):
        return None
    votes = [0] * meter
    for seconds in downbeats:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            continue
        position = _position(times, float(seconds))
        nearest = round(position)
        if abs(position - nearest) <= _NEAR_BEAT:
            votes[nearest % meter] += 1
    total = sum(votes)
    if total < _MIN_ANALYSIS_DOWNBEATS:
        return None
    best = max(range(meter), key=lambda phase: (votes[phase], -phase))
    share = votes[best] / total
    return (best, share) if share >= _MIN_ANALYSIS_SHARE else None


def build_beat_grid(
    tracked_beats: Any,
    *,
    meter: int,
    end_seconds: float,
    onsets: Sequence[float] | None = None,
    drum_hits: Sequence[tuple[float, str]] | None = None,
    analysis_downbeats: Sequence[float] | None = None,
    analysis_meter_confidence: float | None = None,
) -> BeatGrid | None:
    """Build a grid from tracked beats, or return ``None`` when they cannot carry one.

    ``drum_hits`` is ``(seconds, "low" | "mid")`` for kick-like and snare-like
    hits. ``onsets`` are any detected attack times used to settle the grid on
    the played beat.
    """
    if isinstance(meter, bool) or not isinstance(meter, int) or not 1 <= meter <= 12:
        return None
    if isinstance(end_seconds, bool) or not isinstance(end_seconds, (int, float)) or not math.isfinite(end_seconds):
        return None
    beats = _clean_beats(tracked_beats)
    if beats is None:
        return None
    tracked_count = len(beats)
    beats = _fill_gaps(beats)
    if beats is None:
        return None
    grid, mode = _smooth(beats)

    shift_beats = _onset_shift(grid, onsets)
    shift_seconds = 0.0
    if shift_beats:
        intervals = [b - a for a, b in zip(grid, grid[1:])]
        intervals.append(intervals[-1])
        shift_seconds = shift_beats * statistics.median(intervals)
        grid = [time + shift_beats * interval for time, interval in zip(grid, intervals)]
        if any(b <= a for a, b in zip(grid, grid[1:])):
            return None

    phase = _drum_phase(grid, meter, drum_hits)
    source = "drums"
    if phase is None:
        phase = _analysis_phase(grid, meter, analysis_downbeats, analysis_meter_confidence)
        source = "analysis"
    if phase is None:
        phase, source = (0, None), "none"
    phase_index, phase_confidence = phase

    first_interval = statistics.median(b - a for a, b in zip(grid[:9], grid[1:9]))
    last_interval = statistics.median(b - a for a, b in zip(grid[-9:], grid[-8:]))
    if not (
        _MIN_BEAT_SECONDS <= first_interval <= _MAX_BEAT_SECONDS
        and _MIN_BEAT_SECONDS <= last_interval <= _MAX_BEAT_SECONDS
    ):
        return None

    def time_of(index: int) -> float:
        if index < 0:
            return grid[0] + index * first_interval
        if index >= len(grid):
            return grid[-1] + (index - len(grid) + 1) * last_interval
        return grid[index]

    # First downbeat at or just before the start of the recording.
    tolerance = _LEAD_IN_TOLERANCE_BEATS * first_interval
    origin = phase_index
    while time_of(origin - meter) >= -tolerance:
        origin -= meter
    while time_of(origin) < -tolerance:
        origin += meter
    if time_of(origin) > tolerance:
        origin -= meter  # the recording opens with a pickup into this downbeat

    final_index = origin + 1
    limit = max(float(end_seconds), grid[-1]) + meter * last_interval
    while time_of(final_index) <= limit:
        final_index += 1
        if final_index - origin > _MAX_BEATS:
            return None
    final_index = origin + meter * int(math.ceil((final_index - origin) / meter))
    times = [time_of(index) for index in range(origin, final_index + 1)]

    tempo = 60.0 / statistics.median(b - a for a, b in zip(grid, grid[1:]))
    if mode == "steady":
        tempo = 60.0 / ((grid[-1] - grid[0]) / (len(grid) - 1))
    if not 20.0 <= tempo <= 300.0:
        return None
    return BeatGrid(
        times,
        meter=meter,
        mode=mode,
        tempo_bpm=tempo,
        bar_phase_source=source,
        bar_phase_confidence=phase_confidence,
        onset_shift_seconds=shift_seconds,
        tracked_beats=tracked_count,
    )


__all__ = [
    "BAR_PHASE_SOURCES",
    "BEAT_GRID_VERSION",
    "GRID_MODES",
    "BeatGrid",
    "build_beat_grid",
]
