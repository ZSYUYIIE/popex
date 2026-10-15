"""Song sections from the analysis stage (Cycle 22)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from app.analysis import STRUCTURE_METHOD, _analyze_structure

APP_JS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"


def _render_analysis(details: dict) -> dict:
    """Render the analysis fold for each saved analysis detail in a minimal DOM stub."""
    script = f"""
const fs = require("fs");
const stub = () => ({{value: "", files: [], dataset: {{}}, textContent: "", innerHTML: "",
  classList: {{add() {{}}, remove() {{}}, toggle() {{}}}}, addEventListener() {{}},
  setAttribute() {{}}, getAttribute() {{ return null; }}}});
global.document = {{querySelector: stub, documentElement: stub()}};
global.window = {{location: {{origin: "https://popex.local"}}}};
global.fetch = async () => ({{ok: true, status: 200, json: async () => []}});
global.setTimeout = () => 1;
const source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8").replace(/^initTheme\\(\\);.*$/m, "")
  + ";globalThis.__t={{renderAnalysis, analysisCache}};";
eval(source);
const t = globalThis.__t, out = {{}};
for (const [id, detail] of Object.entries({json.dumps(details)})) {{
  t.analysisCache.set(id, detail);
  out[id] = t.renderAnalysis({{id, analysis: {{status: "completed"}}, preparation: {{status: "completed"}}}});
}}
console.log(JSON.stringify(out));
"""
    completed = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(completed.stdout)

SR = 22050


def chord(freqs: list[float], seconds: float, noise: float = 0.0) -> np.ndarray:
    times = np.arange(int(seconds * SR)) / SR
    signal = sum(np.sin(2 * np.pi * f * times) for f in freqs) / len(freqs)
    if noise:
        signal = signal + noise * np.random.default_rng(0).standard_normal(times.size)
    return 0.3 * signal


def test_repeating_parts_share_a_letter_and_boundaries_land_near_the_changes() -> None:
    a = chord([261.6, 329.6, 392.0], 16)
    b = chord([220.0, 277.2, 329.6, 440.0], 16, noise=0.05)
    c = chord([174.6, 220.0, 261.6], 16)
    audio = np.concatenate([a, b, a, b, c])
    result = _analyze_structure(audio, SR, [], audio.size / SR)
    assert result["method"] == STRUCTURE_METHOD and result["note"] is None
    sections = result["sections"]
    assert [item["label"] for item in sections] == ["A", "B", "A", "B", "C"]
    starts = [item["startSeconds"] for item in sections]
    assert all(abs(start - expected) <= 1.5 for start, expected in zip(starts, [0, 16, 32, 48, 64]))
    assert sections[0]["boundaryStrength"] is None
    assert all(0 < item["boundaryStrength"] <= 1 for item in sections[1:])
    assert sections[-1]["endSeconds"] == audio.size / SR


def test_boundaries_snap_to_a_nearby_beat() -> None:
    audio = np.concatenate([chord([261.6, 329.6, 392.0], 20), chord([220.0, 277.2, 329.6], 20, noise=0.05)])
    unsnapped = _analyze_structure(audio, SR, [], audio.size / SR)["sections"][1]["startSeconds"]
    beat = round(unsnapped + 0.4, 3)
    snapped = _analyze_structure(audio, SR, [beat], audio.size / SR)["sections"][1]["startSeconds"]
    assert snapped == beat


def test_short_or_uniform_recordings_say_so_instead_of_guessing() -> None:
    short = chord([261.6, 329.6], 20)
    result = _analyze_structure(short, SR, [], short.size / SR)
    assert result["sections"] == [] and "too short" in result["note"]
    uniform = chord([261.6, 329.6, 392.0], 45)
    result = _analyze_structure(uniform, SR, [], uniform.size / SR)
    assert result["sections"] == [] and result["note"] == "No clear section changes were found."


def test_analysis_fold_draws_one_proportional_strip() -> None:
    detail = {"structure": {"method": STRUCTURE_METHOD, "note": None, "sections": [
        {"startSeconds": 0, "endSeconds": 16, "label": "A", "boundaryStrength": None},
        {"startSeconds": 16, "endSeconds": 40, "label": "B", "boundaryStrength": 0.9},
        {"startSeconds": 40, "endSeconds": 56, "label": "<b>", "boundaryStrength": 0.7},
    ]}}
    result = _render_analysis({
        "j": detail,
        "k": {"structure": {"sections": [], "note": "The recording is too short to find sections."}},
    })
    html = result["j"]
    assert '<ol class="section-strip" aria-label="Song sections. Letters mark sections that sound alike.">' in html
    assert '<li class="section-a" style="flex-grow:16.00"><span class="section-label">A</span><span class="section-time">0:00-0:16</span></li>' in html
    assert 'style="flex-grow:24.00"' in html
    assert "<b>" not in html and 'class="section-x"' in html  # unexpected labels are neutralised
    assert "not verse or chorus names" in html
    assert '<p class="detail-note">The recording is too short to find sections.</p>' in result["k"]
