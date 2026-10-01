"""Playable string and fret positions for already-transcribed pitched notes.

Tablature is a separate layer from pitch transcription: this module never
changes a note's pitch, timing, or identity. It only proposes where a note
could be played on a fretted instrument, so the suggestion stays editable.

Standard tunings (string 1 is the highest-sounding string, as in MusicXML):

======  ==========================  =====  ==================
Id      Strings 1..n (MIDI)         Frets  Max fretted span
======  ==========================  =====  ==================
guitar  E4 B3 G3 D3 A2 E2           21     4 frets
        (64 59 55 50 45 40)
bass    G2 D2 A1 E1 (43 38 33 28)   20     4 frets
======  ==========================  =====  ==================

Algorithm (deterministic Viterbi over onset groups):

- notes that start on the same quantized beat form one group (a chord);
- every playable position set for a group uses distinct strings and keeps
  the fretted notes within the span limit (open strings are always free);
- a group's own cost prefers low positions and compact shapes:
  ``0.1 * mean fret + 0.5 * span + 0.15 * frets above 12``;
- the fretting hand covers a window of ``max span + 1`` frets starting at
  fret ``w``; a shape fits every window that contains its fretted notes
  (open strings fit anywhere), and each window keeps its cheapest shape;
- moving the hand between groups costs ``1.0 + 0.6 * |window change|``
  (nothing when it stays put), plus ``0.05`` per string of average string
  movement, so notes inside one position cost nothing to reach;
- notes outside the instrument's range are ``out_of_range``; when a chord
  cannot be played at all, its least confident notes are marked
  ``unplayable`` one by one until the rest fit. Nothing is dropped,
  transposed, or re-voiced.

Limitations: standard tuning only; held notes do not block strings for later
onsets; no capo, alternate tunings, or techniques.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TABLATURE_VERSION = "tab-fingering-v1"
TAB_STATUSES = frozenset({"assigned", "out_of_range", "unplayable"})
_MAX_CANDIDATES_PER_GROUP = 48
_MAX_COMBINATIONS = 4096
_MAX_TAB_NOTES = 5000


class TablatureError(RuntimeError):
    """Tablature positions could not be computed safely."""


@dataclass(frozen=True, slots=True)
class TabInstrument:
    id: str
    label: str
    strings: tuple[int, ...]  # open-string MIDI notes, string 1 first
    frets: int
    max_span: int
    tuning_name: str

    @property
    def lowest(self) -> int:
        return min(self.strings)

    @property
    def highest(self) -> int:
        return max(self.strings) + self.frets


TAB_INSTRUMENTS: dict[str, TabInstrument] = {
    "bass": TabInstrument(
        id="bass",
        label="Bass",
        strings=(43, 38, 33, 28),
        frets=20,
        max_span=4,
        tuning_name="Standard 4-string (E A D G)",
    ),
    "guitar": TabInstrument(
        id="guitar",
        label="Guitar",
        strings=(64, 59, 55, 50, 45, 40),
        frets=21,
        max_span=4,
        tuning_name="Standard 6-string (E A D G B E)",
    ),
}
TAB_INSTRUMENT_ORDER = ("bass", "guitar")


def positions_for(midi_note: int, instrument: TabInstrument) -> list[tuple[int, int]]:
    """Return every ``(string, fret)`` that sounds ``midi_note``, string 1 first."""
    return [
        (index + 1, midi_note - open_note)
        for index, open_note in enumerate(instrument.strings)
        if 0 <= midi_note - open_note <= instrument.frets
    ]


@dataclass(frozen=True, slots=True)
class _Candidate:
    positions: tuple[tuple[int, int], ...]  # aligned with the group's playable notes
    cost: float
    mean_string: float


def _shape_cost(positions: Sequence[tuple[int, int]], instrument: TabInstrument) -> float | None:
    frets = [fret for _string, fret in positions]
    fretted = [fret for fret in frets if fret > 0]
    span = max(fretted) - min(fretted) if fretted else 0
    if span > instrument.max_span:
        return None
    mean_fret = sum(frets) / len(frets)
    above_twelve = sum(max(0, fret - 12) for fret in frets)
    return 0.1 * mean_fret + 0.5 * span + 0.15 * above_twelve


def _group_candidates(
    midi_notes: Sequence[int], instrument: TabInstrument
) -> list[_Candidate]:
    options = [positions_for(note, instrument) for note in midi_notes]
    candidates: list[_Candidate] = []
    for count, combination in enumerate(itertools.product(*options)):
        if count >= _MAX_COMBINATIONS:
            break
        strings = [string for string, _fret in combination]
        if len(set(strings)) != len(strings):
            continue
        cost = _shape_cost(combination, instrument)
        if cost is None:
            continue
        candidates.append(
            _Candidate(
                positions=tuple(combination),
                cost=cost,
                mean_string=sum(strings) / len(strings),
            )
        )
    candidates.sort(key=lambda item: (item.cost, item.positions))
    return candidates[:_MAX_CANDIDATES_PER_GROUP]


def assign_tablature(
    notes: Sequence[Mapping[str, Any]],
    instrument_id: str,
) -> dict[str, dict[str, Any]]:
    """Return ``{note id: {status, string, fret}}`` for one instrument.

    Each note needs ``id``, ``midiNote``, ``quantizedBeat`` and ``confidence``.
    """
    instrument = TAB_INSTRUMENTS.get(instrument_id)
    if instrument is None:
        raise TablatureError("Unsupported tablature instrument.")
    if len(notes) > _MAX_TAB_NOTES:
        raise TablatureError("Too many notes for tablature.")
    result: dict[str, dict[str, Any]] = {}
    groups: dict[float, list[Mapping[str, Any]]] = {}
    for note in notes:
        note_id = note.get("id")
        midi = note.get("midiNote")
        beat = note.get("quantizedBeat")
        confidence = note.get("confidence")
        if (
            not isinstance(note_id, str)
            or note_id in result
            or isinstance(midi, bool)
            or not isinstance(midi, int)
            or isinstance(beat, bool)
            or not isinstance(beat, (int, float))
            or not math.isfinite(float(beat))
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
        ):
            raise TablatureError("A tablature note is malformed.")
        if positions_for(midi, instrument):
            groups.setdefault(float(beat), []).append(note)
            result[note_id] = {"status": "assigned", "string": None, "fret": None}
        else:
            result[note_id] = {"status": "out_of_range", "string": None, "fret": None}

    chain: list[tuple[list[Mapping[str, Any]], list[_Candidate]]] = []
    for beat in sorted(groups):
        playable = sorted(
            groups[beat], key=lambda item: (-float(item["confidence"]), item["midiNote"], item["id"])
        )
        candidates: list[_Candidate] = []
        while playable:
            if len(playable) <= len(instrument.strings):
                ordered = sorted(playable, key=lambda item: (item["midiNote"], item["id"]))
                candidates = _group_candidates([item["midiNote"] for item in ordered], instrument)
                if candidates:
                    playable = ordered
                    break
            dropped = playable.pop()  # least confident note gives way first
            result[dropped["id"]]["status"] = "unplayable"
        if candidates:
            chain.append((playable, candidates))

    if not chain:
        return result
    windows = range(1, instrument.frets + 1)

    def by_window(candidates: list[_Candidate]) -> dict[int, _Candidate]:
        best_shape: dict[int, _Candidate] = {}
        for candidate in candidates:  # already sorted cheapest first
            fretted = [fret for _string, fret in candidate.positions if fret > 0]
            low = max(fretted) - instrument.max_span if fretted else 1
            high = min(fretted) if fretted else instrument.frets
            for window in range(max(1, low), high + 1):
                best_shape.setdefault(window, candidate)
        return best_shape

    shapes = [by_window(candidates) for _notes, candidates in chain]
    best = {window: shape.cost for window, shape in shapes[0].items()}
    back: list[dict[int, int]] = [{}]
    for index in range(1, len(chain)):
        row: dict[int, float] = {}
        row_back: dict[int, int] = {}
        for window, shape in shapes[index].items():
            total, origin = min(
                (
                    cost
                    + (1.0 + 0.6 * abs(window - prior) if window != prior else 0.0)
                    + 0.05 * abs(shapes[index - 1][prior].mean_string - shape.mean_string),
                    prior,
                )
                for prior, cost in best.items()
            )
            row[window] = total + shape.cost
            row_back[window] = origin
        best = row
        back.append(row_back)
    window = min(best, key=lambda key: (best[key], key))
    for index in range(len(chain) - 1, -1, -1):
        group_notes, _candidates = chain[index]
        shape = shapes[index][window]
        for note, (string, fret) in zip(group_notes, shape.positions):
            result[note["id"]] = {"status": "assigned", "string": string, "fret": fret}
        if index:
            window = back[index][window]
    return result


def tab_position_is_consistent(midi_note: int, instrument_id: str, string: int, fret: int) -> bool:
    """Return whether ``string``/``fret`` sounds exactly ``midi_note``."""
    instrument = TAB_INSTRUMENTS.get(instrument_id)
    if instrument is None or not 1 <= string <= len(instrument.strings):
        return False
    return 0 <= fret <= instrument.frets and instrument.strings[string - 1] + fret == midi_note


__all__ = [
    "TABLATURE_VERSION",
    "TAB_INSTRUMENTS",
    "TAB_INSTRUMENT_ORDER",
    "TAB_STATUSES",
    "TabInstrument",
    "TablatureError",
    "assign_tablature",
    "positions_for",
    "tab_position_is_consistent",
]
