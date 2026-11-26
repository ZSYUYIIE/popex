"""Ranked scale-collection (mode) candidates for one draft score.

This is a separate, versioned layer on top of the score's own evidence. It
does not change the global analysis baseline (24 Ionian/Aeolian profiles) or
the harmony stage. The recording is never forced into one scale: candidates
are ranked with confidence, relative modes that share the same notes are
reported as ambiguous when tonic evidence is weak, and local regions show
where the evidence suggests a different centre or collection.

Evidence:
- a pitch-class histogram of quantized notes, weighted by duration (beats)
  and note confidence;
- bass emphasis from the separated bass-stem line, else the full mix's
  lowest line (when the mix was split into lines), else the lowest note that
  starts each bar, weighted the same way;
- the analysis chroma mean, when available.

Scoring (documented): each collection is a probability distribution over the
12 pitch classes. Members share the mass in proportion to 1.0, with 2.0 on
the tonic, 1.4 on a perfect fifth and 1.2 on a third when they are members;
every non-member keeps a floor of 0.01 before normalization. A candidate's
score is the cross-entropy ``sum(h * log q)`` of the evidence histogram ``h``
under that distribution, plus ``0.3 * bass share of the tonic``. Notes outside
a collection therefore cost it heavily, and a smaller collection (pentatonic)
wins only when the evidence really stays inside it. Confidence is the margin
to the best clearly different reading (other notes, or the same notes with
another tonic), ``margin / 0.15``, scaled by the amount of note evidence (full
at 16 weighted beats), clipped to 0–1.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

TONAL_CONTEXT_VERSION = "modal-collections-v3"
KEY_SIGNATURE_MIN_CONFIDENCE = 0.5

# collection -> (label, intervals, diatonic parent offset in semitones from
# the tonic down to its parent major tonic, MusicXML/MIDI mode family)
COLLECTIONS: dict[str, tuple[str, tuple[int, ...], int, str]] = {
    "ionian": ("Ionian (major)", (0, 2, 4, 5, 7, 9, 11), 0, "major"),
    "dorian": ("Dorian", (0, 2, 3, 5, 7, 9, 10), 2, "dorian"),
    "phrygian": ("Phrygian", (0, 1, 3, 5, 7, 8, 10), 4, "phrygian"),
    "lydian": ("Lydian", (0, 2, 4, 6, 7, 9, 11), 5, "lydian"),
    "mixolydian": ("Mixolydian", (0, 2, 4, 5, 7, 9, 10), 7, "mixolydian"),
    "aeolian": ("Aeolian (natural minor)", (0, 2, 3, 5, 7, 8, 10), 9, "minor"),
    "locrian": ("Locrian", (0, 1, 3, 5, 6, 8, 10), 11, "locrian"),
    "harmonic_minor": ("Harmonic minor", (0, 2, 3, 5, 7, 8, 11), 9, "minor"),
    "melodic_minor": ("Melodic minor", (0, 2, 3, 5, 7, 9, 11), 9, "minor"),
    "major_pentatonic": ("Major pentatonic", (0, 2, 4, 7, 9), 0, "major"),
    "minor_pentatonic": ("Minor pentatonic", (0, 3, 5, 7, 10), 9, "minor"),
    "blues": ("Blues", (0, 3, 5, 6, 7, 10), 9, "minor"),
}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
# Fifths for each major-key tonic pitch class (flats preferred from Db to F).
_MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
_TONIC_WEIGHT, _FIFTH_WEIGHT, _THIRD_WEIGHT = 2.0, 1.4, 1.2
_NON_MEMBER_FLOOR = 0.01
_BASS_PRIOR = 0.3
_MARGIN_SCALE = 0.15
_FULL_EVIDENCE_BEATS = 16.0
_NOTE_SHARE, _CHROMA_SHARE = 0.6, 0.4
_REGION_BARS, _REGION_HOP = 8, 4
_REGION_MIN_NOTES, _REGION_MIN_BEATS = 6, 4.0
_MAX_CANDIDATES, _MAX_REGIONS = 8, 32
# Chords heard in the audio: their tones join the note histogram and their
# roots act as tonic evidence, because a song spends most time on its home chord.
_CHORD_SHARE = 0.4
_CHORD_ROOT_PRIOR = 0.6
# Notes that tell two same-tonic collections apart must carry this share of the
# histogram before the difference counts. A draft transcription scatters a few
# percent onto almost every pitch class, so less than this is treated as noise.
_DECISIVE_SHARE = 0.06
_COLLECTION_PREFERENCE = {
    "ionian": 0, "aeolian": 1, "mixolydian": 2, "dorian": 3, "harmonic_minor": 4,
    "melodic_minor": 5, "lydian": 6, "phrygian": 7, "major_pentatonic": 8,
    "minor_pentatonic": 9, "blues": 10, "locrian": 11,
}
_CHORD_SYMBOL = re.compile(r"([A-G])([#b]?)(.*)")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def key_signature(root: int, collection: str) -> dict[str, Any]:
    """Return the MusicXML ``fifths``/``mode`` for one tonal candidate."""
    _label, _intervals, parent_offset, mode = COLLECTIONS[collection]
    return {"fifths": _MAJOR_FIFTHS[(root - parent_offset) % 12], "mode": mode}


def key_signature_for(
    centre_confidence: float, signature_confidence: float, root: int, collection: str
) -> dict[str, Any] | None:
    """The key signature a score writes, or ``None``.

    A confident tonal centre writes its signature and mode. When the centre is
    uncertain but the notes clearly fit one signature, the signature is written
    with mode ``none``, so the score does not claim major or minor.
    """
    if centre_confidence >= KEY_SIGNATURE_MIN_CONFIDENCE:
        return key_signature(root, collection)
    if signature_confidence >= KEY_SIGNATURE_MIN_CONFIDENCE:
        return {"fifths": key_signature(root, collection)["fifths"], "mode": "none"}
    return None


def root_name(root: int, collection: str) -> str:
    fifths = key_signature(root, collection)["fifths"]
    return (_FLAT_NAMES if fifths < 0 else _SHARP_NAMES)[root % 12]


def _template(intervals: Sequence[int]) -> list[float]:
    template = [_NON_MEMBER_FLOOR] * 12
    for interval in intervals:
        template[interval] = 1.0
    template[0] = _TONIC_WEIGHT
    if 7 in intervals:
        template[7] = _FIFTH_WEIGHT
    for third in (3, 4):
        if third in intervals:
            template[third] = _THIRD_WEIGHT
    return template


def _log_distribution(template: Sequence[float]) -> list[float]:
    total = sum(template)
    return [math.log(value / total) for value in template]


_TEMPLATES = {name: _log_distribution(_template(spec[1])) for name, spec in COLLECTIONS.items()}


def _cross_entropy(values: Sequence[float], log_template: Sequence[float]) -> float:
    return sum(value * log_value for value, log_value in zip(values, log_template))


def _normalize(values: Sequence[float]) -> list[float] | None:
    total = sum(values)
    return None if total <= 1e-12 else [value / total for value in values]


def _rank(
    histogram: Sequence[float],
    bass: Sequence[float] | None,
    chord_roots: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    bass_share = _normalize(bass) if bass is not None else None
    root_share = _normalize(chord_roots) if chord_roots is not None else None
    ranked = []
    for collection, template in _TEMPLATES.items():
        for root in range(12):
            rotated = [template[(pc - root) % 12] for pc in range(12)]
            score = _cross_entropy(histogram, rotated)
            if bass_share is not None:
                score += _BASS_PRIOR * bass_share[root]
            if root_share is not None:
                score += _CHORD_ROOT_PRIOR * root_share[root]
            ranked.append({"root": root, "collection": collection, "score": score})
    ranked.sort(key=lambda item: (-item["score"], item["collection"], item["root"]))
    return _prefer_common_collection(ranked, histogram)


def _undecided(first: Mapping[str, Any], second: Mapping[str, Any], histogram: Sequence[float]) -> bool:
    """Same tonic, and the notes that tell the two collections apart are barely heard."""
    if first["root"] != second["root"]:
        return False
    differing = _pitch_set(first["root"], first["collection"]) ^ _pitch_set(
        second["root"], second["collection"]
    )
    return sum(histogram[pc] for pc in differing) < _DECISIVE_SHARE


def _prefer_common_collection(ranked: list[dict[str, Any]], histogram: Sequence[float]) -> list[dict[str, Any]]:
    """Among readings the audio cannot tell apart, lead with the most common one.

    A song that never sounds the seventh degree fits G Ionian and G Mixolydian
    equally; ranking them on noise would make the key signature flip between
    runs. Ionian, then Aeolian, then the other modes is the usual way such a
    score is written.
    """
    best = ranked[0]
    if len(COLLECTIONS[best["collection"]][1]) != 7:
        return ranked  # a pentatonic or blues reading is a finding, not a tie
    tied = [
        item
        for item in ranked
        if len(COLLECTIONS[item["collection"]][1]) == 7
        and _undecided(best, item, histogram)
        and best["score"] - item["score"] < _MARGIN_SCALE
    ]
    leader = min(tied, key=lambda item: (_COLLECTION_PREFERENCE.get(item["collection"], 99), -item["score"]))
    if leader is best:
        return ranked
    return [leader] + [item for item in ranked if item is not leader]


def _pitch_set(root: int, collection: str) -> frozenset[int]:
    return frozenset((root + interval) % 12 for interval in COLLECTIONS[collection][1])


def _describe(item: Mapping[str, Any], confidence: float) -> dict[str, Any]:
    root, collection = item["root"], item["collection"]
    return {
        "tonalCenter": root_name(root, collection),
        "rootPitchClass": root,
        "collection": collection,
        "displayName": f"{root_name(root, collection)} {COLLECTIONS[collection][0]}",
        "score": round(float(item["score"]), 4),
        "confidence": round(confidence, 3),
    }


def _assess(histogram, bass, evidence_beats, chord_roots=None):
    """Return ranked candidates with confidences, plus ambiguity notes."""
    ranked = _rank(histogram, bass, chord_roots)
    best = ranked[0]
    best_set = _pitch_set(best["root"], best["collection"])
    # Margin to the best clearly different reading: another tonic, or the same
    # tonic with different notes that are actually heard. Readings the audio
    # cannot tell apart do not lower the confidence in the tonal centre.
    # A same-tonic reading that would be written with the same key signature
    # (G major pentatonic against G Ionian) is a refinement, not a rival.
    best_signature = key_signature(best["root"], best["collection"])["fifths"]
    rival = next(
        item for item in ranked[1:]
        if item["root"] != best["root"]
        or (
            _pitch_set(item["root"], item["collection"]) != best_set
            and not _undecided(best, item, histogram)
            and key_signature(item["root"], item["collection"])["fifths"] != best_signature
        )
    )
    evidence = min(1.0, evidence_beats / _FULL_EVIDENCE_BEATS)
    confidence = max(0.0, min(1.0, (best["score"] - rival["score"]) / _MARGIN_SCALE)) * evidence
    candidates, seen = [], set()
    for item in ranked:
        key = (item["root"], item["collection"])
        if key in seen:
            continue
        seen.add(key)
        relative = max(0.0, min(1.0, confidence - (best["score"] - item["score"]) / _MARGIN_SCALE))
        candidates.append(_describe(item, confidence if not candidates else relative))
        if len(candidates) == _MAX_CANDIDATES:
            break
    ambiguous = [
        _describe(item, 0.0)["displayName"]
        for item in ranked[1:12]
        if item["root"] != best["root"]
        and _pitch_set(item["root"], item["collection"]) == best_set
        and best["score"] - item["score"] < _MARGIN_SCALE * 0.5
    ]
    # The key signature depends only on which notes are used, so readings that
    # share it (C Ionian, A Aeolian, F Lydian) do not make it less certain.
    signature_rival = next(
        (
            item for item in ranked[1:]
            if key_signature(item["root"], item["collection"])["fifths"] != best_signature
        ),
        None,
    )
    signature_confidence = (
        evidence
        if signature_rival is None
        else max(0.0, min(1.0, (best["score"] - signature_rival["score"]) / _MARGIN_SCALE)) * evidence
    )
    return candidates, ambiguous[:3], best, signature_confidence


def _histograms(notes: Sequence[Mapping[str, Any]], measures_bass: Sequence[Mapping[str, Any]]):
    histogram = [0.0] * 12
    bass = [0.0] * 12
    beats = 0.0
    for note in notes:
        weight = max(0.0, float(note["quantizedDurationBeats"])) * max(0.05, float(note["confidence"]))
        histogram[note["midiNote"] % 12] += weight
        beats += float(note["quantizedDurationBeats"])
    for note in measures_bass:
        weight = max(0.0, float(note["quantizedDurationBeats"])) * max(0.05, float(note["confidence"]))
        bass[note["midiNote"] % 12] += weight
    return histogram, bass, beats


def _bass_notes(measures: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], str]:
    stem = [note for measure in measures for note in measure["notes"] if note.get("sourceKind") == "bass"]
    if stem:
        return stem, "bass-stem line"
    mix_line = [note for measure in measures for note in measure["notes"] if note.get("sourceKind") == "mix_bass"]
    if mix_line:
        return mix_line, "lowest line of the full mix"
    lowest = []
    for measure in measures:
        if measure["notes"]:
            first_beat = min(note["quantizedBeat"] for note in measure["notes"])
            starting = [note for note in measure["notes"] if note["quantizedBeat"] == first_beat]
            lowest.append(min(starting, key=lambda note: note["midiNote"]))
    return lowest, "lowest note at the start of each bar"


def _chord_tones(symbol: str) -> tuple[int, tuple[int, ...]] | None:
    """Root and triad tones of a chord symbol; extensions and basses are ignored."""
    match = _CHORD_SYMBOL.fullmatch(symbol.strip())
    if match is None:
        return None
    root = (_LETTERS[match.group(1)] + {"#": 1, "b": -1, "": 0}[match.group(2)]) % 12
    rest = match.group(3).split("/")[0]
    if rest.startswith(("sus", "5", "dim", "aug", "+", "o")):
        return None
    minor = rest.startswith(("m", "-")) and not rest.startswith("maj")
    return root, (0, 3, 7) if minor else (0, 4, 7)


def _chord_evidence(measures: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]] | None:
    """Tone and root histograms from the resolved harmony mapped onto bars."""
    tones = [0.0] * 12
    roots = [0.0] * 12
    for measure in measures:
        for entry in measure.get("harmony") or ():
            if not isinstance(entry, Mapping) or entry.get("unresolved") or not isinstance(entry.get("symbol"), str):
                continue
            parsed = _chord_tones(entry["symbol"])
            if parsed is None:
                continue
            try:
                span = max(0.0, float(entry["endBeat"]) - float(entry["startBeat"]))
                confidence = float(entry.get("confidence") or 0.5)
            except (TypeError, ValueError, KeyError):
                continue
            weight = span * max(0.05, min(1.0, confidence))
            root, intervals = parsed
            for interval in intervals:
                tones[(root + interval) % 12] += weight
            roots[root] += weight
    return (tones, roots) if sum(roots) > 1e-9 else None


def build_tonal_context(
    measures: Sequence[Mapping[str, Any]],
    *,
    chroma_mean: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return the persisted ``tonality`` layer for one score's measures.

    Chord symbols already mapped onto the bars (``measure["harmony"]``) count
    as evidence when present: their tones join the histogram and their roots
    support the tonal centre.
    """
    notes = [note for measure in measures for note in measure["notes"]]
    bass_notes, bass_source = _bass_notes(measures)
    note_hist, bass_hist, beats = _histograms(notes, bass_notes)
    chroma = None
    if chroma_mean is not None and len(chroma_mean) == 12:
        values = [float(value) for value in chroma_mean]
        if all(math.isfinite(value) and value >= 0 for value in values):
            chroma = _normalize(values)
    note_share = _normalize(note_hist)
    if note_share and chroma:
        histogram = [_NOTE_SHARE * a + _CHROMA_SHARE * b for a, b in zip(note_share, chroma)]
    else:
        histogram = note_share or chroma
    chords = _chord_evidence(measures)
    chord_roots = None
    if chords is not None and histogram is not None:
        tone_share = _normalize(chords[0])
        histogram = [(1 - _CHORD_SHARE) * a + _CHORD_SHARE * b for a, b in zip(histogram, tone_share)]
        chord_roots = chords[1]
    evidence = {
        "noteCount": len(notes),
        "weightedBeats": round(beats, 3),
        "bassSource": bass_source if bass_notes else None,
        "usesAnalysisChroma": chroma is not None,
    }
    if histogram is None or not notes:
        return {
            "version": TONAL_CONTEXT_VERSION,
            "evidence": evidence,
            "primaryCandidate": None,
            "candidates": [],
            "ambiguousWith": [],
            "localRegions": [],
            "chromaticismScore": None,
            "keySignature": None,
            "signatureConfidence": None,
            "notes": ["No transcribed notes were available, so no scale or mode is suggested."],
        }
    candidates, ambiguous, best, signature_confidence = _assess(
        histogram, bass_hist if bass_notes else None, beats, chord_roots
    )
    primary = candidates[0]
    members = _pitch_set(best["root"], best["collection"])
    total = sum(note_hist)
    outside = sum(weight for pc, weight in enumerate(note_hist) if pc not in members)
    chromaticism = round(outside / total, 3) if total > 1e-12 else None

    regions = []
    for start in range(0, max(1, len(measures) - _REGION_BARS + 1), _REGION_HOP):
        window = measures[start : start + _REGION_BARS]
        window_notes = [note for measure in window for note in measure["notes"]]
        window_bass, _source = _bass_notes(window)
        hist, bass, window_beats = _histograms(window_notes, window_bass)
        normalized = _normalize(hist)
        if (
            normalized is None
            or len(window_notes) < _REGION_MIN_NOTES
            or window_beats < _REGION_MIN_BEATS
        ):
            continue
        local_candidates, _ambiguous, local_best, _signature = _assess(
            normalized, bass if window_bass else None, window_beats
        )
        local = local_candidates[0]
        end = start + len(window) - 1
        same_as_previous = (
            regions
            and regions[-1]["rootPitchClass"] == local["rootPitchClass"]
            and regions[-1]["collection"] == local["collection"]
            and regions[-1]["endMeasure"] >= start - 1
        )
        if same_as_previous:
            regions[-1]["endMeasure"] = end
            regions[-1]["confidence"] = round(min(regions[-1]["confidence"], local["confidence"]), 3)
            continue
        regions.append(
            {
                "startMeasure": start,
                "endMeasure": end,
                "tonalCenter": local["tonalCenter"],
                "rootPitchClass": local["rootPitchClass"],
                "collection": local["collection"],
                "displayName": local["displayName"],
                "confidence": local["confidence"],
                "differsFromWhole": (local["rootPitchClass"], local["collection"])
                != (primary["rootPitchClass"], primary["collection"]),
            }
        )
        if len(regions) >= _MAX_REGIONS:
            break

    notes_out = []
    if chord_roots is not None:
        notes_out.append("The chord symbols on the bars count as evidence for the tonal centre.")
    if ambiguous:
        notes_out.append(
            f"{primary['displayName']} shares its notes with {', '.join(ambiguous)}; "
            "tonic evidence is weak, so treat the mode as a suggestion."
        )
    signature = key_signature_for(primary["confidence"], signature_confidence, best["root"], best["collection"])
    if signature is None:
        notes_out.append(
            "Confidence is below 0.50, so no key signature is written; accidentals are explicit."
        )
    elif signature["mode"] == "none":
        notes_out.append(
            "The notes clearly fit this key signature, but the tonal centre is uncertain, "
            "so the key is written without naming a mode."
        )
    if any(region["differsFromWhole"] for region in regions):
        notes_out.append(
            "Some bars suggest a different centre or collection: possible modulation, "
            "modal mixture, or borrowed harmony to review."
        )
    return {
        "version": TONAL_CONTEXT_VERSION,
        "evidence": evidence,
        "primaryCandidate": primary,
        "candidates": candidates,
        "ambiguousWith": ambiguous,
        "localRegions": regions,
        "chromaticismScore": chromaticism,
        "keySignature": signature,
        "signatureConfidence": round(signature_confidence, 3),
        "notes": notes_out,
    }


__all__ = [
    "COLLECTIONS",
    "KEY_SIGNATURE_MIN_CONFIDENCE",
    "TONAL_CONTEXT_VERSION",
    "build_tonal_context",
    "key_signature",
    "key_signature_for",
    "root_name",
]


def score_key_name(document: Mapping[str, Any]) -> str | None:
    """The score's suggested key in words (for example "G Ionian (major)"), or ``None``.

    Only a candidate confident enough to set the key signature and its mode is
    named; a signature written without a mode names no key.
    """
    tonality = document.get("tonality")
    signature = tonality.get("keySignature") if isinstance(tonality, Mapping) else None
    if not isinstance(signature, Mapping) or signature.get("mode") == "none":
        return None
    primary = tonality.get("primaryCandidate")
    name = primary.get("displayName") if isinstance(primary, Mapping) else None
    return name[:64] if isinstance(name, str) and name else None
