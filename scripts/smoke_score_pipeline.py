"""End-to-end synthetic smoke test of the local score workflow.

Generates a short synthetic recording (a sine-tone melody over a kick, snare
and hi-hat pattern), then drives the real application through upload, FFmpeg
normalization, analysis, raw transcription, interpretation, harmony and
score construction. It asks for guitar tablature of the melody line, then
checks that the saved draft score holds pitched notes, guitar tablature and a
separate percussion part, and that the MIDI and MusicXML exports carry them. Only synthetic audio is used; nothing is downloaded.

Usage::

    python scripts/smoke_score_pipeline.py [--keep DIR]

Requires FFmpeg and ffprobe on ``PATH`` (or ``POPEX_FFMPEG_BINARY`` /
``POPEX_FFPROBE_BINARY``). Prints a JSON summary and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 44_100
TEMPO_BPM = 120.0
BARS = 4


def synthetic_recording() -> bytes:
    """Return WAV bytes: a tone melody over a simple rock drum pattern."""
    beat = 60.0 / TEMPO_BPM
    total = BARS * 4 * beat + 0.5
    samples = int(total * SAMPLE_RATE)
    audio = np.zeros(samples, dtype=np.float64)
    rng = np.random.default_rng(7)

    def add(start: float, signal: np.ndarray) -> None:
        begin = int(start * SAMPLE_RATE)
        end = min(samples, begin + len(signal))
        audio[begin:end] += signal[: end - begin]

    def envelope(length: float, decay: float) -> np.ndarray:
        n = int(length * SAMPLE_RATE)
        return np.exp(-np.arange(n) / SAMPLE_RATE / decay)

    for bar in range(BARS):
        for step in range(8):
            start = (bar * 4 + step / 2) * beat
            # Closed hi-hat on every eighth: short high-passed noise.
            noise = rng.standard_normal(int(0.05 * SAMPLE_RATE))
            hat = np.diff(noise, prepend=0.0) * envelope(0.05, 0.012) * 0.18
            add(start, hat)
            if step in (0, 4):
                n = int(0.25 * SAMPLE_RATE)
                time = np.arange(n) / SAMPLE_RATE
                kick = np.sin(2 * np.pi * (55 + 90 * np.exp(-time / 0.03)) * time)
                add(start, kick * envelope(0.25, 0.08) * 0.9)
            if step in (2, 6):
                snare = rng.standard_normal(int(0.18 * SAMPLE_RATE)) * envelope(0.18, 0.05)
                tone = np.sin(2 * np.pi * 190 * np.arange(len(snare)) / SAMPLE_RATE)
                add(start, (snare * 0.45 + tone * envelope(0.18, 0.04) * 0.3))
    melody = [69, 72, 76, 72, 67, 71, 74, 71]  # A4 C5 E5 C5 G4 B4 D5 B4
    for index, midi in enumerate(melody):
        start = index * 2 * beat
        length = 2 * beat * 0.9
        n = int(length * SAMPLE_RATE)
        time = np.arange(n) / SAMPLE_RATE
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        fade = np.minimum(1.0, np.minimum(time / 0.02, (length - time) / 0.05))
        add(start, np.sin(2 * np.pi * freq * time) * fade * 0.25)
    audio /= max(1.0, float(np.max(np.abs(audio))) / 0.9)
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _channel_ten_note_ons(data: bytes) -> int:
    length = struct.unpack(">I", data[18:22])[0]
    track = data[22 : 22 + length]
    position, count = 0, 0
    while position < len(track):
        while track[position] & 0x80:
            position += 1
        position += 1
        status = track[position]
        if status == 0xFF:
            position += 3 + track[position + 2]
            continue
        if status == 0x99 and track[position + 2] > 0:
            count += 1
        position += 3
    return count


def run(data_dir: Path) -> dict:
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = replace(Settings.from_env(), data_dir=data_dir)
    summary: dict = {}
    with TestClient(create_app(settings)) as client:
        def job() -> dict:
            return next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)

        def step(name: str, url: str, stage_key: str) -> None:
            response = client.post(url)
            if response.status_code != 202:
                raise SystemExit(f"{name} was rejected: {response.status_code} {response.text}")
            stage = job().get(stage_key) or {}
            if stage.get("status") != "completed":
                raise SystemExit(f"{name} did not complete: {json.dumps(stage)[:600]}")

        upload = client.post(
            "/api/uploads",
            files={"file": ("synthetic-drums-and-tones.wav", synthetic_recording(), "audio/wav")},
        )
        if upload.status_code != 202:
            raise SystemExit(f"Upload was rejected: {upload.status_code} {upload.text}")
        job_id = upload.json()["id"]
        current = job()
        if current.get("analysis", {}).get("status") != "completed":
            raise SystemExit(f"Preparation/analysis did not complete: {json.dumps(current)[:800]}")
        step("Transcription", f"/api/jobs/{job_id}/transcribe", "transcription")
        step("Interpretation", f"/api/jobs/{job_id}/interpret", "interpretation")
        step("Harmony", f"/api/jobs/{job_id}/harmonize", "harmony")
        choice = client.put(
            f"/api/jobs/{job_id}/score/tablature", json={"bass": "bass", "guitar": "full_mix"}
        )
        if choice.status_code != 200:
            raise SystemExit(f"Tablature choice was rejected: {choice.status_code} {choice.text}")
        step("Score", f"/api/jobs/{job_id}/score/construct", "score")

        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        base = f"/api/jobs/{job_id}/score/saved/download?format="
        musicxml = client.get(base + "musicxml")
        midi = client.get(base + "midi")
        if musicxml.status_code != 200 or midi.status_code != 200:
            raise SystemExit("Saved score downloads failed.")
        root = ET.fromstring(musicxml.content)
        parts = [part.get("id") for part in root.findall("part")]
        drum_notes = root.findall("part[@id='P2']/measure/note")
        tab_notes = [
            note
            for note in root.findall("part[@id='P4']/measure/note")
            if note.find("notations/technical/fret") is not None
        ]
        summary = {
            "jobId": job_id,
            "scoreVersion": details["version"],
            "layers": {name: layer["status"] for name, layer in details["layers"].items()},
            "voiceSource": details["percussion"]["voiceSource"],
            "voices": {
                voice["broadVoice"]: voice["hitCount"] for voice in details["percussion"]["voices"]
            },
            "counts": details["counts"],
            "musicxmlParts": parts,
            "musicxmlDrumNotes": len(drum_notes),
            "tablature": details["tablature"],
            "musicxmlGuitarTabNotes": len(tab_notes),
            "midiChannel10NoteOns": _channel_ten_note_ons(midi.content),
            "warnings": details["warnings"],
        }
    problems = []
    if summary["layers"].get("percussion") != "included":
        problems.append("percussion layer is not included")
    if summary["counts"]["notes"] < 1:
        problems.append("no pitched notes were notated")
    if "P2" not in summary["musicxmlParts"] or not summary["musicxmlDrumNotes"]:
        problems.append("MusicXML has no percussion part")
    if summary["midiChannel10NoteOns"] != summary["counts"]["notatedPercussionHits"]:
        problems.append("MIDI channel-10 hits do not match notated hits")
    if summary["layers"].get("tablature") != "included":
        problems.append("tablature layer is not included")
    if summary["musicxmlGuitarTabNotes"] != summary["counts"]["fingeredTabNotes"]:
        problems.append("MusicXML TAB notes do not match fingered notes")
    summary["problems"] = problems
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=Path, help="keep the data directory here")
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
        summary = run(args.keep.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="popex-smoke-") as directory:
            summary = run(Path(directory))
    print(json.dumps(summary, indent=2))
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    os.environ.setdefault("STEM_SEPARATION_ENABLED", "false")
    raise SystemExit(main())
