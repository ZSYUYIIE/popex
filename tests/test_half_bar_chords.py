"""Half-bar chord changes for bars without one clear chord (Cycle 19)."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import pytest

from app.score_artifacts import ScoreArtifactValidationError, _validate_chord_changes
from app.score_construction import ScoreConstructionError, build_score_document, score_to_musicxml_text
from app.score_pipeline import _map_harmony, score_outdated_reason
from test_frontend_score import SUMMARY, _run_node

SPB = 0.5  # 120 BPM, so a 4/4 bar lasts two seconds


def segment(seg_id: str, start: float, end: float, symbol: str | None) -> dict:
    return {
        "id": seg_id,
        "rawStartSeconds": start,
        "rawEndSeconds": end,
        "unresolved": symbol is None,
        "primaryCandidate": None if symbol is None else {"symbol": symbol, "confidence": 0.8},
    }


def bars(count: int, meter: int = 4) -> list[dict]:
    length = meter * SPB
    return [{"startSeconds": i * length, "endSeconds": (i + 1) * length} for i in range(count)]


def test_a_bar_that_changes_chord_mid_bar_gets_two_half_bar_symbols() -> None:
    measures = bars(1)
    stats = _map_harmony(
        measures, [segment("a", 0.0, 1.0, "Am"), segment("f", 1.0, 2.0, "F")],
        seconds_per_beat=SPB, beats_per_measure=4,
    )
    assert measures[0]["chordSymbol"] is None
    assert measures[0]["chordChanges"] == [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F"}]
    assert (stats["split"], stats["ambiguous"]) == (1, 0)


def test_a_clear_whole_bar_chord_needs_no_split() -> None:
    measures = bars(1)
    _map_harmony(measures, [segment("c", 0.0, 2.0, "C")], seconds_per_beat=SPB, beats_per_measure=4)
    assert (measures[0]["chordSymbol"], measures[0]["chordChanges"]) == ("C", [])


def test_an_unclear_half_keeps_the_bar_empty_and_counted_as_ambiguous() -> None:
    measures = bars(1)
    # Am fills the first half, but the second half is split evenly between Am and F.
    stats = _map_harmony(
        measures, [segment("a", 0.0, 1.5, "Am"), segment("f", 1.5, 2.0, "F")],
        seconds_per_beat=SPB, beats_per_measure=4,
    )
    assert (measures[0]["chordSymbol"], measures[0]["chordChanges"]) == (None, [])
    assert (stats["split"], stats["ambiguous"]) == (0, 1)


def test_unresolved_windows_and_odd_meters_never_split() -> None:
    measures = bars(1)
    _map_harmony(
        measures, [segment("a", 0.0, 1.0, "Am"), segment("u", 1.0, 2.0, None)],
        seconds_per_beat=SPB, beats_per_measure=4,
    )
    assert measures[0]["chordChanges"] == []
    waltz = bars(1, meter=3)
    _map_harmony(
        waltz, [segment("a", 0.0, 0.75, "Am"), segment("f", 0.75, 1.5, "F")],
        seconds_per_beat=SPB, beats_per_measure=3,
    )
    assert (waltz[0]["chordSymbol"], waltz[0]["chordChanges"]) == (None, [])


def _two_bar_document(changes: list[dict], chord: str | None = None) -> dict:
    document = build_score_document(
        [{"id": "n1", "startSeconds": 0.0, "endSeconds": 0.4, "midiNote": 60, "confidence": 0.9},
         {"id": "n2", "startSeconds": 2.0, "endSeconds": 2.4, "midiNote": 62, "confidence": 0.9}],
        tempo_bpm=120.0,
    )
    document["measures"][0]["chordChanges"] = changes
    document["measures"][0]["chordSymbol"] = chord
    return document


def test_musicxml_places_the_second_chord_at_the_middle_of_the_bar() -> None:
    document = _two_bar_document([{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "Fmaj7(#11)"}])
    root = ET.fromstring(score_to_musicxml_text(document))
    first = root.find(".//measure")
    harmony = first.findall("harmony")
    assert [h.findtext("root/root-step") for h in harmony] == ["A"]
    assert harmony[0].find("offset") is None
    # Text the parser cannot read stays words, still at beat 3 (two quarters = 960 divisions).
    direction = [d for d in first.findall("direction") if d.find("direction-type/words") is not None]
    assert [(d.findtext("direction-type/words"), d.findtext("offset")) for d in direction] == [("Fmaj7(#11)", "960")]
    parsed = _two_bar_document([{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F"}])
    offsets = [h.findtext("offset") for h in ET.fromstring(score_to_musicxml_text(parsed)).iter("harmony")]
    assert offsets == [None, "960"]


def test_a_whole_bar_symbol_such_as_a_correction_wins_over_half_bar_changes() -> None:
    document = _two_bar_document([{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F"}], chord="Dm7")
    harmony = list(ET.fromstring(score_to_musicxml_text(document)).iter("harmony"))
    assert [(h.findtext("root/root-step"), h.findtext("kind")) for h in harmony] == [("D", "minor-seventh")]


@pytest.mark.parametrize(
    "changes",
    [
        [{"beat": 0.0, "symbol": "A"}, {"beat": 1.0, "symbol": "B"}, {"beat": 2.0, "symbol": "C"}],
        [{"beat": 2.0, "symbol": "A"}, {"beat": 0.0, "symbol": "B"}],
        [{"beat": 0.0, "symbol": "A"}, {"beat": 4.0, "symbol": "B"}],
        [{"beat": 0.0, "symbol": "A", "extra": 1}],
        "Am F",
    ],
)
def test_exporter_rejects_malformed_chord_changes(changes) -> None:
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(_two_bar_document(changes))


def _evidence(*entries: tuple[str, float, float]) -> list[dict]:
    return [
        {"segmentId": f"s{i}", "startBeat": start, "endBeat": end, "symbol": symbol,
         "confidence": 0.8, "unresolved": False}
        for i, (symbol, start, end) in enumerate(entries)
    ]


def test_saved_changes_must_trace_back_to_harmonic_evidence() -> None:
    harmony = _evidence(("Am", 0.0, 2.0), ("F", 2.0, 4.0))
    ok = {"chordChanges": [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F"}]}
    _validate_chord_changes(ok, harmony, 4, "m")
    _validate_chord_changes({"chordChanges": []}, harmony, 4, "m")
    for bad in (
        [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "G"}],  # no evidence for G
        [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "Am"}],  # not a change
        [{"beat": 0.0, "symbol": "Am"}, {"beat": 1.0, "symbol": "F"}],  # not the middle
        [{"beat": 0.0, "symbol": "Am"}],  # one half only
    ):
        with pytest.raises(ScoreArtifactValidationError):
            _validate_chord_changes({"chordChanges": bad}, harmony, 4, "m")
    with pytest.raises(ScoreArtifactValidationError):
        _validate_chord_changes(ok, _evidence(("Am", 0.0, 1.5), ("F", 2.0, 3.0)), 3, "m")


def test_half_bar_changes_validate_against_musicxml_schema() -> None:
    xsd = os.environ.get("POPEX_MUSICXML_XSD")
    if not xsd:
        pytest.skip("set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
    from lxml import etree

    schema = etree.XMLSchema(etree.parse(xsd))
    for changes in (
        [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F/C"}],
        [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "Fmaj7(#11)"}],
    ):
        tree = etree.fromstring(score_to_musicxml_text(_two_bar_document(changes)).encode("utf-8"))
        assert schema.validate(tree), schema.error_log


def test_older_scores_with_harmony_are_out_of_date() -> None:
    record = {"score_version": "score-pipeline-v6", "score_note_count": 4, "harmony_status": "completed"}
    assert score_outdated_reason(record) == "half-bar-chords"
    assert score_outdated_reason({**record, "harmony_status": "not_started"}) is None
    assert score_outdated_reason({**record, "score_version": "score-pipeline-v7"}) is None


def test_review_table_and_roll_show_both_chords() -> None:
    detail = {
        "available": True, "createdAt": SUMMARY["createdAt"], "layers": {}, "counts": {}, "warnings": [],
        "timing": {"tempoBpm": 120, "beatsPerMeasure": 4},
        "measures": [
            {"measureIndex": 0, "startSeconds": 0, "endSeconds": 2, "chordSymbol": None,
             "chordChanges": [{"beat": 0, "symbol": "Am"}, {"beat": 2, "symbol": "F"}],
             "notes": [{"id": "n", "midiNote": 60, "quantizedBeat": 0, "quantizedDurationBeats": 1,
                        "confidence": 0.9, "sourceKind": "vocals"}], "harmony": [], "percussionHits": []},
        ],
    }
    result = _run_node(
        f"""
t.setDetail("split", {json.dumps(detail)});
const html=t.renderScore({{id:"split",score:{json.dumps(SUMMARY)}}});
const stale=t.renderScore({{id:"s",score:{{...{json.dumps(SUMMARY)}, stale:true, staleReason:"half-bar-chords"}}}});
console.log(JSON.stringify({{html,stale}}));
"""
    )
    html = result["html"]
    assert '<span class="chord-split">Am <span class="detail-note">then F from beat 3</span></span>' in html
    assert ">Am</text>" in html and ">F</text>" in html  # both chords in the piano-roll lane
    assert "built before bars could show two chords" in result["stale"]
