from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "app" / "static" / "app.js"
STYLES_CSS = ROOT / "app" / "static" / "styles.css"


def _run_node(body: str) -> dict:
    app_path = json.dumps(str(APP_JS))
    script = f"""
const fs = require("fs");
const listeners = new Map();
const elements = new Map();
function element(selector) {{
  if (!elements.has(selector)) {{
    elements.set(selector, {{
      selector, value: "", files: [], disabled: false, focused: false,
      textContent: "", innerHTML: "", dataset: {{}}, attributes: {{}},
      classList: {{add() {{}}, remove() {{}}, toggle() {{}}}},
      addEventListener(type, handler) {{listeners.set(`${{selector}}:${{type}}`, handler);}},
      setAttribute(name, value) {{this.attributes[name] = String(value);}},
      getAttribute(name) {{return this.attributes[name] ?? null;}},
      reportValidity() {{return true;}},
      focus() {{this.focused = true;}},
      scrollIntoView() {{}},
    }});
  }}
  return elements.get(selector);
}}
global.document = {{querySelector: element}};
global.window = {{location: {{origin: "https://popex.local"}}}};
let fetchImpl = async () => ({{ok: true, status: 200, json: async () => []}});
global.fetch = (...args) => fetchImpl(...args);
global.setTimeout = () => 1;
global.clearTimeout = () => {{}};
const source = fs.readFileSync({app_path}, "utf8").replace("updateFilePresentation();loadJobs();","") + `
;globalThis.__popexScoreTest={{
  renderScore, renderJob, deriveState, loadJobs, hydrateCompletedScores,
  invalidateStaleScoreDetails,
  setDetail:(id,value)=>scoreCache.set(id,value),
  hasDetail:(id)=>scoreCache.has(id),
  getDetail:(id)=>scoreCache.get(id),
  setFetch:(value)=>{{fetchImpl=value;}},
  getElement:element,
  getListener:(selector,type)=>listeners.get(selector+":"+type)
}};`;
eval(source);
(async () => {{
  await Promise.resolve();
  const t = globalThis.__popexScoreTest;
  {body}
}})().catch(error => {{
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}});
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


SUMMARY = {
    "enabled": True,
    "status": "completed",
    "stage": "completed",
    "progress": 100,
    "available": True,
    "stale": False,
    "createdAt": "2026-09-30T10:00:00+00:00",
    "counts": {"measures": 2, "notes": 3, "chordSymbols": 1, "warnings": 2},
    "canStart": False,
    "canRebuild": True,
    "startUrl": "/api/jobs/done/score/construct",
    "detailsUrl": "/api/jobs/done/score/saved?includeMeasures=false",
    "fullDetailsUrl": "/api/jobs/done/score/saved?includeMeasures=true",
    "downloadUrls": {
        "midi": "/api/jobs/done/score/saved/download?format=midi",
        "musicxml": "/api/jobs/done/score/saved/download?format=musicxml",
        "json": "/api/jobs/done/score/saved/download?format=json",
    },
}

DETAIL = {
    "available": True,
    "stale": False,
    "createdAt": "2026-09-30T10:00:00+00:00",
    "timing": {"tempoBpm": 120.0, "beatsPerMeasure": 4, "meterSource": "fallback-4/4"},
    "counts": {"measures": 2, "notes": 3, "chordSymbols": 1},
    "layers": {
        "pitchedNotes": {"status": "included", "note": "Quantized from raw events."},
        "chordSymbols": {"status": "included", "note": "From resolved candidates."},
        "partLabels": {"status": "omitted", "note": "No editable interpretation."},
        "percussion": {"status": "omitted", "note": "Not notated yet."},
        "tablature": {"status": "omitted", "note": "Planned later."},
    },
    "parts": [],
    "warnings": ["Review the chord symbols.", "Tempo is estimated."],
    "measures": [
        {
            "measureIndex": 0,
            "chordSymbol": "C<script>",
            "notes": [
                {"id": "p_c", "noteName": "C4", "quantizedBeat": 0.0,
                 "quantizedDurationBeats": 2.0, "confidence": 0.9, "partId": None},
                {"id": "p_e", "noteName": "E4", "quantizedBeat": 2.5,
                 "quantizedDurationBeats": 0.5, "confidence": 0.3, "partId": None},
            ],
            "harmony": [
                {"segmentId": "seg_1", "startBeat": 0.0, "endBeat": 4.0,
                 "symbol": "C", "confidence": 0.8, "unresolved": False},
                {"segmentId": "seg_2", "startBeat": 2.0, "endBeat": 4.0,
                 "symbol": None, "confidence": None, "unresolved": True},
            ],
        },
        {"measureIndex": 1, "chordSymbol": None, "notes": [], "harmony": []},
    ],
}


def test_panel_is_absent_without_a_score_contract() -> None:
    result = _run_node(
        """
console.log(JSON.stringify({values:[
  t.renderScore({id:"a"}), t.renderScore({id:"b",score:null}),
  t.renderScore({id:"c",score:[]}), t.renderScore({id:"d",score:"x"})
]}));
"""
    )
    assert result["values"] == ["", "", "", ""]


def test_build_retry_and_rebuild_actions_render_as_buttons() -> None:
    result = _run_node(
        """
const start=t.renderScore({id:"n",score:{enabled:true,status:"not_started",available:false,
  canStart:true,startUrl:"/api/jobs/n/score/construct",counts:{}}});
const retry=t.renderScore({id:"f",score:{enabled:true,status:"failed",available:false,
  canStart:true,startUrl:"/api/jobs/f/score/construct",counts:{},error:"Evidence changed."}});
const blocked=t.renderScore({id:"b",score:{enabled:true,status:"not_started",available:false,
  canStart:false,startUrl:"/api/jobs/b/score/construct",counts:{}}});
console.log(JSON.stringify({start,retry,blocked}));
"""
    )
    assert '<button type="button" data-action="score"' in result["start"]
    assert ">Build score</button>" in result["start"]
    assert 'data-force="false"' in result["start"]
    assert "Download MIDI" not in result["start"]
    assert ">Retry score</button>" in result["retry"]
    assert 'data-retry="true"' in result["retry"]
    assert "could not be built" in result["retry"]
    assert "Evidence changed." in result["retry"]
    assert "<button" not in result["blocked"]


def test_completed_score_renders_review_table_layers_and_downloads() -> None:
    result = _run_node(
        f"""
t.setDetail("done", {json.dumps(DETAIL)});
const html=t.renderScore({{id:"done",score:{json.dumps(SUMMARY)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert "Draft score" in html
    assert '<span aria-hidden="true">✓</span>Score ready' in html
    assert ">Rebuild score</button>" in html and 'data-force="true"' in html
    for label, fmt in (("Download MIDI", "midi"), ("Download MusicXML", "musicxml"),
                       ("Download score data (JSON)", "json")):
        assert f'href="/api/jobs/done/score/saved/download?format={fmt}" download>{label}</a>' in html
    assert "<dt>Tempo</dt><dd>120 BPM</dd>" in html
    assert "<dt>Time signature</dt><dd>4/4 (assumed)</dd>" in html
    assert "<strong>Tablature:</strong>" in html and "Not included" in html
    assert "<caption>" in html and '<th scope="col">Chord symbol</th>' in html
    assert '<div class="table-scroll" role="region" aria-label="Measure-by-measure review table" tabindex="0">' in html
    assert '<th scope="row">1</th>' in html
    assert "C&lt;script&gt;" in html and "<script>" not in html
    assert "C4 · beat 1 · 2 beats" in html
    assert "E4 · beat 3.5 · 0.5 beats" in html
    assert '<span class="score-low">low confidence</span>' in html
    assert "C from beat 1 (80%)" in html and "unresolved from beat 3" in html
    assert "No notes detected" in html
    assert "Review the chord symbols." in html


def test_stale_processing_and_failed_states_are_honest() -> None:
    stale = {**SUMMARY, "stale": True}
    rebuilding = {**SUMMARY, "status": "processing", "stage": "mapping_harmony",
                  "progress": 50, "canRebuild": False}
    failed = {**SUMMARY, "status": "failed", "canRebuild": False, "canStart": True,
              "error": "Could not read /home/user/secret.json"}
    result = _run_node(
        f"""
const stale=t.renderScore({{id:"s",score:{json.dumps(stale)}}});
const busy=t.renderScore({{id:"p",score:{json.dumps(rebuilding)}}});
const failed=t.renderScore({{id:"f",score:{json.dumps(failed)}}});
const state=t.deriveState({{preparation:{{status:"completed"}},analysis:{{status:"completed"}},
  transcription:{{status:"completed"}},score:{json.dumps(stale)}}});
console.log(JSON.stringify({{stale,busy,failed,state}}));
"""
    )
    assert "Out of date" in result["stale"]
    assert "This score is out of date." in result["stale"]
    assert ">Rebuild score</button>" in result["stale"]
    assert '<progress value="50" max="100" aria-label="Draft score: Placing chord symbols, 50% complete">' in result["busy"]
    assert "previous draft score remains available" in result["busy"]
    assert "<button" not in result["busy"]
    assert "Download MIDI" in result["busy"]
    assert "previous draft score is still available" in result["failed"]
    assert ">Retry score</button>" in result["failed"]
    assert "/home/user" not in result["failed"]
    assert result["state"]["label"] == "Score out of date"
    assert result["state"]["tone"] == "warning"


def test_unsafe_urls_are_never_rendered() -> None:
    unsafe = {**SUMMARY, "startUrl": "https://evil.example/x", "downloadUrls": {
        "midi": "//evil.example/a.mid", "musicxml": "/api/../../etc", "json": "javascript:alert(1)"}}
    result = _run_node(
        f"""
const html=t.renderScore({{id:"u",score:{json.dumps(unsafe)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    assert "evil.example" not in result["html"]
    assert "javascript:" not in result["html"]
    assert "<button" not in result["html"]
    assert "Download" not in result["html"]


def test_rebuild_click_posts_force_and_clears_cached_detail() -> None:
    result = _run_node(
        """
const requested=[];
t.setFetch(async (url, options={}) => {requested.push({url,method:options.method||"GET"});
  return {ok:true,status:200,json:async()=>[]};});
t.setDetail("job",{available:true});
const button={dataset:{action:"score",jobId:"job",startUrl:"/api/jobs/job/score/construct",
  force:"true",retry:"false"},disabled:false,textContent:"Rebuild score",focus(){}};
await t.getListener("#jobs","click")({target:{closest:()=>button}});
console.log(JSON.stringify({requested,text:button.textContent,cached:t.hasDetail("job"),
  message:t.getElement("#jobs-message").textContent}));
"""
    )
    assert result["requested"][0] == {
        "url": "/api/jobs/job/score/construct?force=true",
        "method": "POST",
    }
    assert result["text"] == "Started"
    assert result["cached"] is False
    assert "previous draft score remains available" in result["message"]


def test_failed_start_restores_button_and_reports_error() -> None:
    result = _run_node(
        """
t.setFetch(async () => ({ok:false,status:409,json:async()=>({detail:"Score construction is already running."})}));
const button={dataset:{action:"score",jobId:"job",startUrl:"/api/jobs/job/score/construct",
  force:"false",retry:"false"},disabled:false,textContent:"Build score",focused:false,focus(){this.focused=true}};
await t.getListener("#jobs","click")({target:{closest:()=>button}});
console.log(JSON.stringify({text:button.textContent,disabled:button.disabled,focused:button.focused,
  message:t.getElement("#jobs-message").textContent}));
"""
    )
    assert result["text"] == "Build score"
    assert result["disabled"] is False
    assert result["focused"] is True
    assert "already running" in result["message"]


def test_hydration_loads_full_details_and_invalidates_on_new_build() -> None:
    result = _run_node(
        f"""
const requested=[];
t.setFetch(async (url) => {{requested.push(url);return {{ok:true,status:200,json:async()=>({json.dumps(DETAIL)})}};}});
const jobs=[{{id:"done",score:{json.dumps(SUMMARY)}}}];
const failures=await t.hydrateCompletedScores(jobs);
const loaded=t.hasDetail("done");
t.invalidateStaleScoreDetails([{{id:"done",score:{{...{json.dumps(SUMMARY)},createdAt:"2026-10-01T00:00:00+00:00"}}}}]);
const afterRebuild=t.hasDetail("done");
t.setDetail("gone",{{}});
t.invalidateStaleScoreDetails([]);
t.setFetch(async () => {{throw new Error("offline")}});
const failed=await t.hydrateCompletedScores(jobs);
console.log(JSON.stringify({{requested,failures,loaded,afterRebuild,gone:t.hasDetail("gone"),
  failed,cachedFailure:t.getDetail("done"),message:t.getElement("#jobs-message").textContent,
  html:t.renderScore(jobs[0])}}));
"""
    )
    assert result["requested"] == ["/api/jobs/done/score/saved?includeMeasures=true"]
    assert result["failures"] == 0 and result["loaded"] is True
    assert result["afterRebuild"] is False
    assert result["gone"] is False
    assert result["failed"] == 1 and result["cachedFailure"] is None
    assert "could not be loaded" in result["message"]
    assert "The saved draft score could not be loaded." in result["html"]


def test_job_card_includes_score_panel_and_polls_while_building() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "${renderHarmony(job)}${renderScore(job)}${renderFiles(files)}" in source
    assert 'job.score?.status==="processing"' in source
    styles = STYLES_CSS.read_text(encoding="utf-8")
    assert ".score-measures table" in styles and ".table-scroll" in styles
    # The scroll box must not widen the page on narrow screens.
    rule = re.search(r"\.table-scroll\s*\{([^}]*)\}", styles).group(1).replace(" ", "")
    assert "overflow-x:auto" in rule and "width:0" in rule and "min-width:100%" in rule


def test_upstream_work_in_progress_outranks_saved_score_state() -> None:
    result = _run_node(
        """
const base={preparation:{status:"completed"},analysis:{status:"completed"},
  transcription:{status:"completed"},score:{status:"completed",stale:false}};
console.log(JSON.stringify({
  ready:t.deriveState(base).label,
  harmony:t.deriveState({...base,harmony:{status:"processing"}}).label,
  building:t.deriveState({...base,score:{status:"processing"},harmony:{status:"processing"}}).label
}));
"""
    )
    assert result == {
        "ready": "Draft score ready",
        "harmony": "Inferring harmony",
        "building": "Building score",
    }



def test_failed_detail_load_is_retried_after_a_new_saved_score() -> None:
    result = _run_node(
        f"""
let fail=true;
t.setFetch(async () => {{if(fail)throw new Error("offline");return {{ok:true,status:200,json:async()=>({json.dumps(DETAIL)})}};}});
const jobs=[{{id:"done",score:{json.dumps(SUMMARY)}}}];
await t.hydrateCompletedScores(jobs);
t.invalidateStaleScoreDetails(jobs);
const keptForSameScore=t.hasDetail("done")&&t.getDetail("done")===null;
fail=false;
const rebuilt=[{{id:"done",score:{{...{json.dumps(SUMMARY)},createdAt:"2026-10-01T00:00:00+00:00"}}}}];
t.invalidateStaleScoreDetails(rebuilt);
const cleared=!t.hasDetail("done");
await t.hydrateCompletedScores(rebuilt);
console.log(JSON.stringify({{keptForSameScore,cleared,loaded:t.getDetail("done")!==null}}));
"""
    )
    assert result == {"keptForSameScore": True, "cleared": True, "loaded": True}


DRUM_DETAIL = {
    **DETAIL,
    "percussion": {
        "voiceSource": "raw-hit-kinds",
        "voices": [
            {"broadVoice": "low_drum", "label": "Low drum", "displayStep": "F",
             "displayOctave": 4, "notehead": "normal", "gmNote": 36, "hitCount": 2},
            {"broadVoice": "unresolved_percussion", "label": "Unresolved percussion",
             "displayStep": "B", "displayOctave": 4, "notehead": "triangle",
             "gmNote": 76, "hitCount": 1},
        ],
    },
    "counts": {**DETAIL["counts"], "percussionHits": 4, "notatedPercussionHits": 3,
               "collapsedPercussionHits": 1, "unresolvedPercussionHits": 1,
               "offGridPercussionHits": 0, "unplacedPercussionHits": 2},
    "measures": [
        {**DETAIL["measures"][0], "percussionHits": [
            {"broadVoice": "low_drum", "quantizedBeat": 0.0, "notation": "notated"},
            {"broadVoice": "low_drum", "quantizedBeat": 2.0, "notation": "notated"},
            {"broadVoice": "low_drum", "quantizedBeat": 2.0, "notation": "collapsed"},
            {"broadVoice": "unresolved_percussion", "quantizedBeat": 3.5,
             "notation": "notated"},
        ]},
        {**DETAIL["measures"][1], "percussionHits": []},
    ],
}


def test_drum_notation_summary_and_per_bar_hits_render() -> None:
    result = _run_node(
        f"""
t.setDetail("drums", {json.dumps(DRUM_DETAIL)});
const html=t.renderScore({{id:"drums",score:{json.dumps(SUMMARY)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert "<dt>Drum hits</dt><dd>3</dd>" in html
    assert "<strong>Drum notation by voice</strong>" in html
    assert "Voices come from the detected hit types" in html
    assert "<li>Low drum: 2 hits</li>" in html
    assert "Unresolved percussion: 1 hit · <span class=\"score-low\">not assigned to a drum</span>" in html
    assert "1 unresolved · 2 unplaced (no confident rhythm placement)" in html
    assert "1 merged with a hit in the same slot" in html
    assert '<th scope="col">Drum hits</th>' in html
    assert "<li>Low drum · beats 1, 3</li>" in html
    assert 'Unresolved percussion <span class="score-low">(unresolved)</span> · beat 4.5' in html
    assert "drum hits, and harmonic evidence per measure" in html


def test_schema_one_detail_has_no_drum_column() -> None:
    result = _run_node(
        f"""
t.setDetail("old", {json.dumps(DETAIL)});
const html=t.renderScore({{id:"old",score:{json.dumps(SUMMARY)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    assert "Drum hits" not in result["html"]
    assert "Drum notation by voice" not in result["html"]


def test_score_built_before_drum_notation_explains_the_rebuild() -> None:
    stale = {**SUMMARY, "stale": True, "staleReason": "drum-notation"}
    result = _run_node(
        f"""
const html=t.renderScore({{id:"s",score:{json.dumps(stale)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    assert "This score is out of date." in result["html"]
    assert "built before drum notation was available" in result["html"]
    assert ">Rebuild score</button>" in result["html"]


ROLL_DETAIL = {
    **DETAIL,
    "timing": {"tempoBpm": 120, "beatsPerMeasure": 4, "meterSource": "analysis"},
    "measures": [
        {"measureIndex": 0, "chordSymbol": "Am", "harmony": [],
         "notes": [
             {"id": "v", "noteName": "A4", "midiNote": 69, "quantizedBeat": 0, "quantizedDurationBeats": 1,
              "confidence": 0.9, "sourceKind": "vocals"},
             {"id": "b", "noteName": "A2", "midiNote": 45, "quantizedBeat": 0, "quantizedDurationBeats": 2,
              "confidence": 0.4, "sourceKind": "bass"},
         ],
         "percussionHits": [
             {"broadVoice": "low_drum", "quantizedBeat": 0, "notation": "notated"},
             {"broadVoice": "unresolved_percussion", "quantizedBeat": 1, "notation": "notated"},
             {"broadVoice": "mid_drum", "quantizedBeat": 2, "notation": "collapsed"},
         ]},
        {"measureIndex": 1, "chordSymbol": None, "harmony": [], "notes": [], "percussionHits": []},
    ],
}


def test_score_panel_draws_a_piano_roll_from_saved_data() -> None:
    result = _run_node(
        f"""
t.setDetail("roll", {json.dumps(ROLL_DETAIL)});
const html=t.renderScore({{id:"roll",score:{json.dumps(SUMMARY)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert '<div class="score-roll roll-frame" data-review-job="roll"' in html
    assert 'role="region" aria-label="Piano roll, scroll sideways to see later bars" tabindex="0"' in html
    assert 'aria-label="Piano roll of 2 bars: 2 notes (Lead vocal, Bass), 2 drum hits."' in html
    assert 'class="roll-note vocal"' in html and 'class="roll-note bass low"' in html
    assert '<text class="roll-chord"' in html and ">Am</text>" in html
    assert 'class="roll-hit kick"' in html and 'class="roll-hit unresolved"' in html
    assert html.count("class=\"roll-hit") == 2  # collapsed duplicates are not drawn twice
    assert "<span><i class=\"vocal\" aria-hidden=\"true\"></i>Lead vocal</span>" in html
    # The roll comes before the long text review sections.
    assert html.index("score-roll") < html.index("score-insights")


def test_hero_roll_uses_a_labelled_demo_until_a_real_score_has_enough_notes() -> None:
    result = _run_node(
        f"""
t.setDetail("tiny", {json.dumps(ROLL_DETAIL)});
renderHeroRoll([{{id:"tiny",title:"Tiny take"}}]);
const demo=t.getElement("#hero-roll").innerHTML, demoCaption=t.getElement("#hero-roll-caption").textContent;
const rich=JSON.parse(JSON.stringify({json.dumps(ROLL_DETAIL)}));
rich.createdAt="2026-10-01T00:00:00+00:00";
for(let i=0;i<20;i++)rich.measures[0].notes.push({{id:"n"+i,midiNote:60+(i%12),quantizedBeat:i%4,quantizedDurationBeats:.5,confidence:.9,sourceKind:"other"}});
t.setDetail("rich", rich);
renderHeroRoll([{{id:"rich",title:"Studio take"}}]);
console.log(JSON.stringify({{demo,demoCaption,real:t.getElement("#hero-roll").innerHTML,
  caption:t.getElement("#hero-roll-caption").textContent}}));
"""
    )
    assert "Demo phrase" in result["demo"] and "synthetic demo phrase" in result["demoCaption"]
    assert "Studio take" in result["real"] and 'preserveAspectRatio="xMidYMid meet"' in result["real"]
    assert result["caption"] == "From the draft score of “Studio take”. The playhead follows its tempo."


def test_theme_picker_and_assets_are_wired() -> None:
    template = (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")
    assert '<select id="theme-select" aria-label="Colour theme">' in template
    assert 'localStorage.getItem("popex-theme")' in template
    assert ':root[data-theme="light"]' in styles and "@media (prefers-color-scheme: light)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    for font in ("Geist-Variable.woff2", "GeistMono-Variable.woff2", "BricolageGrotesque-latin.woff2"):
        assert f"/static/fonts/{font}" in styles
        assert (ROOT / "app" / "static" / "fonts" / font).is_file()
    assert (ROOT / "app" / "static" / "fonts" / "LICENSE-Geist-OFL.txt").is_file()
    assert (ROOT / "app" / "static" / "fonts" / "LICENSE-BricolageGrotesque-OFL.txt").is_file()
    visible = template + (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "\u2014" not in visible and "\u2013" not in visible  # no em or en dashes in UI copy
