from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "app" / "static" / "app.js"


def _run_node(body: str) -> dict:
    app_path = json.dumps(str(APP_JS))
    script = f"""
const fs = require("fs");
const listeners = new Map();
const elements = new Map();
function element(selector) {{
  if (!elements.has(selector)) {{
    elements.set(selector, {{
      selector,
      value: "",
      files: [],
      disabled: false,
      textContent: "",
      innerHTML: "",
      dataset: {{}},
      classList: {{add() {{}}, remove() {{}}, toggle() {{}}}},
      addEventListener(type, handler) {{listeners.set(`${{selector}}:${{type}}`, handler);}},
      setAttribute() {{}},
      reportValidity() {{return true;}},
      focus() {{}},
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
const source = fs.readFileSync({app_path}, "utf8") + `
;globalThis.__popexInterpretationTest={{
  renderInterpretation,
  renderInterpretationReview,
  interpretationCounts,
  interpretationStageText,
  deriveState,
  safeRelativeUrl,
  hydrateCompletedInterpretations,
  loadJobs,
  setDetail:(id,value)=>interpretationCache.set(id,value),
  clearDetails:()=>interpretationCache.clear(),
  setFetch:(value)=>{{fetchImpl=value;}},
  getElement:element,
  getListener:(selector,type)=>listeners.get(selector+":"+type)
}};`;
eval(source);
(async () => {{
  await Promise.resolve();
  await Promise.resolve();
  const t = globalThis.__popexInterpretationTest;
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
    )
    return json.loads(completed.stdout)


def test_javascript_syntax() -> None:
    subprocess.run(["node", "--check", str(APP_JS)], cwd=ROOT, check=True)


def test_panel_is_absent_without_live_contract() -> None:
    result = _run_node(
        """
const values = [
  t.renderInterpretation({id:"j1"}),
  t.renderInterpretation({id:"j2",interpretation:null}),
  t.renderInterpretation({id:"j3",interpretation:"future"}),
  t.renderInterpretation({id:"j4",interpretation:[]})
];
console.log(JSON.stringify({values}));
"""
    )
    assert result["values"] == ["", "", "", ""]


def test_interpret_retry_and_reinterpret_actions_render() -> None:
    result = _run_node(
        """
const start = t.renderInterpretation({id:"new",interpretation:{
  enabled:true,status:"not_started",progress:0,available:false,canStart:true,
  startUrl:"/api/jobs/new/interpret",counts:{}
}});
const retry = t.renderInterpretation({id:"failed",interpretation:{
  enabled:true,status:"failed",progress:50,available:false,canStart:true,
  startUrl:"/api/jobs/failed/interpret",counts:{},error:"bounded failure"
}});
const again = t.renderInterpretation({id:"done",interpretation:{
  enabled:true,status:"completed",progress:100,available:true,canReinterpret:true,
  startUrl:"/api/jobs/done/interpret",counts:{parts:1,phrases:1,pitched:1,percussion:0,warnings:0}
}});
console.log(JSON.stringify({start,retry,again}));
"""
    )
    assert 'data-action="interpretation"' in result["start"]
    assert ">Interpret</button>" in result["start"]
    assert 'data-force="false"' in result["start"]
    assert "Retry interpretation" in result["retry"]
    assert 'data-retry="true"' in result["retry"]
    assert "Re-interpret" in result["again"]
    assert 'data-force="true"' in result["again"]


def test_reinterpret_click_adds_force_true_and_preserves_existing_actions() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    for action in ('data-action="analysis"', 'data-action="separation"', 'data-action="transcription"'):
        assert action in source
    assert 'data-action="interpretation"' in source

    result = _run_node(
        """
const requested = [];
t.setFetch(async (url, options={}) => {
  requested.push({url,method:options.method||"GET"});
  return {ok:true,status:200,json:async()=>[]};
});
const button = {
  dataset:{
    action:"interpretation",jobId:"job",startUrl:"/api/jobs/job/interpret",
    force:"true",retry:"false"
  },
  disabled:false,
  textContent:"Re-interpret"
};
const listener = t.getListener("#jobs","click");
await listener({target:{closest:()=>button}});
console.log(JSON.stringify({requested,buttonText:button.textContent}));
"""
    )
    assert result["requested"][0] == {
        "url": "/api/jobs/job/interpret?force=true",
        "method": "POST",
    }
    assert result["buttonText"] == "Started"


def test_processing_interpretation_keeps_polling() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'job.interpretation?.status==="processing"' in source

    result = _run_node(
        """
let delay = null;
global.setTimeout = (_fn, value) => {delay = value; return 9;};
t.setFetch(async () => ({ok:true,status:200,json:async()=>[{
  id:"job",status:"completed",preparation_status:"completed",files:[],
  analysis:{status:"completed"},
  transcription:{status:"completed"},
  interpretation:{status:"processing",progress:55,available:false,counts:{}}
}]}));
await t.loadJobs();
console.log(JSON.stringify({delay}));
"""
    )
    assert result["delay"] == 1500


def test_exact_backend_interpretation_stages_are_musician_facing() -> None:
    result = _run_node(
        """
const stages = Object.fromEntries([
  "loading_raw_transcription",
  "loading_analysis_timing",
  "interpreting_pitched_parts",
  "interpreting_percussion",
  "interpreting_rhythm",
  "assembling_interpretation_draft",
  "validating_interpretation_draft",
  "saving_interpretation_draft"
].map(stage => [stage,t.interpretationStageText(stage,"processing")]));
console.log(JSON.stringify({stages}));
"""
    )
    assert result["stages"] == {
        "loading_raw_transcription": "Loading raw events",
        "loading_analysis_timing": "Loading timing evidence",
        "interpreting_pitched_parts": "Grouping pitched parts",
        "interpreting_percussion": "Grouping percussion voices",
        "interpreting_rhythm": "Building rhythm hypotheses",
        "assembling_interpretation_draft": "Assembling editable draft",
        "validating_interpretation_draft": "Validating editable structure",
        "saving_interpretation_draft": "Saving editable draft",
    }


def test_detail_counts_override_summary_and_derive_unresolved_fallbacks() -> None:
    result = _run_node(
        """
const explicit = t.interpretationCounts(
  {counts:{parts:99,phrases:99,pitched:99,percussion:99,warnings:99}},
  {counts:{parts:2,voices:3,measures:1,phrases:4,pitched:5,percussion:6,warnings:7,unassignedPitched:8,unplacedPercussion:9}}
);
const fallback = t.interpretationCounts(
  {counts:{}},
  {
    parts:[{},{}],voices:[{}],measures:[],phrases:[{}],
    pitchedItems:[
      {interpretationType:"note",placementStatus:"placed"},
      {interpretationType:"unassigned",placementStatus:"unassigned"}
    ],
    percussionItems:[{placementStatus:"unassigned"}],warnings:["a"]
  }
);
console.log(JSON.stringify({explicit,fallback}));
"""
    )
    assert result["explicit"] == {
        "parts": 2,
        "voices": 3,
        "measures": 1,
        "phrases": 4,
        "pitched": 5,
        "percussion": 6,
        "warnings": 7,
        "unassignedPitched": 8,
        "unplacedPercussion": 9,
    }
    assert result["fallback"] == {
        "parts": 2,
        "voices": 1,
        "measures": 0,
        "phrases": 1,
        "pitched": 2,
        "percussion": 1,
        "warnings": 1,
        "unassignedPitched": 1,
        "unplacedPercussion": 1,
    }


def test_completed_review_shows_structure_raw_vs_interpreted_and_uncertainty() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{parts:2,voices:3,measures:1,phrases:1,pitched:1,percussion:1,warnings:1,unassignedPitched:1,unplacedPercussion:0},
  sourceKinds:["vocals","drums"],
  algorithms:{rhythmInterpretation:{version:"rhythm-v1"},pitchedPartInference:{version:"pitch-v1"}},
  warnings:["Review <carefully>"],
  parts:[
    {label:"Lead <voice>",sourceKind:"vocals",role:"pitched",voiceIds:["v1","v2"]},
    {label:"Drums",sourceKind:"drums",role:"percussion",voiceIds:["d1"]}
  ],
  phrases:[{rawStartSeconds:0.125,rawEndSeconds:0.875,confidence:0.55}],
  pitchedItems:[{
    id:"p",interpretationType:"unassigned",placementStatus:"unassigned",sourceKind:"vocals",
    rawStartSeconds:0.125,rawEndSeconds:0.5,interpretedStartSeconds:0.25,interpretedDurationSeconds:0.25,
    pitch:{noteName:"C#4",midiPitch:61.23},confidence:0.45,alternatives:[{},{}]
  }],
  percussionItems:[{
    id:"r",placementStatus:"placed",rawStartSeconds:0.5,rawEndSeconds:0.5,
    interpretedStartSeconds:0.5,interpretedDurationSeconds:0,confidence:0.8,alternatives:[{}],
    hits:[
      {rawKind:"kick",broadVoice:"low_drum"},
      {rawKind:"closed_hihat",broadVoice:"closed_high_frequency"}
    ]
  }]
});
const html = t.renderInterpretation({id:"job",interpretation:{
  status:"completed",available:true,progress:100,counts:{parts:99},canReinterpret:true,
  startUrl:"/api/jobs/job/interpret",
  fullDetailsUrl:"/api/jobs/job/interpretation?includeItems=true",
  downloadUrl:"/api/jobs/job/interpretation/download"
}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "Editable interpretation" in html
    assert "<dt>Parts</dt><dd>2</dd>" in html
    assert "<dt>Voices</dt><dd>3</dd>" in html
    assert "<dt>Unassigned pitched</dt><dd>1</dd>" in html
    assert "Lead &lt;voice&gt;" in html
    assert "raw 0.125–0.500 s" in html
    assert "interpreted 0.250 s + 0.250 s" in html
    assert "C#4" in html
    assert "Unassigned" in html
    assert "2 alternatives" in html
    assert "Kick → Low Drum + Closed Hihat → Closed High Frequency" in html
    assert "1 alternative" in html
    assert "Review &lt;carefully&gt;" in html
    assert "Rhythm Interpretation" in html and "rhythm-v1" in html
    assert '<a href="/api/jobs/job/interpretation/download" download>' in html
    assert "raw vs interpreted timing" in html.lower()


def test_failed_or_processing_with_available_result_keeps_previous_review() -> None:
    result = _run_node(
        """
t.clearDetails();
const detail={
  counts:{parts:1,voices:1,measures:0,phrases:0,pitched:1,percussion:0,warnings:0,unassignedPitched:1,unplacedPercussion:0},
  parts:[{sourceKind:"vocals",role:"pitched",voiceIds:["v"]}],phrases:[],
  pitchedItems:[{interpretationType:"unassigned",placementStatus:"unassigned",sourceKind:"vocals",rawStartSeconds:0,rawEndSeconds:.5,pitch:{noteName:"A4"},confidence:.4,alternatives:[]}],
  percussionItems:[],warnings:[]
};
t.setDetail("failed",detail);t.setDetail("processing",detail);
const failed=t.renderInterpretation({id:"failed",interpretation:{
  status:"failed",available:true,progress:70,canStart:true,startUrl:"/api/jobs/failed/interpret",
  fullDetailsUrl:"/api/jobs/failed/interpretation?includeItems=true",downloadUrl:"/api/jobs/failed/interpretation/download",counts:{},error:"retry stopped"
}});
const processing=t.renderInterpretation({id:"processing",interpretation:{
  status:"processing",available:true,progress:40,
  fullDetailsUrl:"/api/jobs/processing/interpretation?includeItems=true",downloadUrl:"/api/jobs/processing/interpretation/download",counts:{}
}});
console.log(JSON.stringify({failed,processing}));
"""
    )
    assert "previous draft is still available" in result["failed"].lower()
    assert "Retry interpretation" in result["failed"]
    assert "A4" in result["failed"]
    assert "previous valid draft remains available" in result["processing"].lower()
    assert "A4" in result["processing"]


def test_completed_failed_and_processing_available_details_hydrate_full_urls() -> None:
    result = _run_node(
        """
t.clearDetails();
const requested=[];
t.setFetch(async url=>{
  requested.push(url);
  return {ok:true,status:200,json:async()=>({counts:{parts:1}})};
});
const jobs=[
  {id:"done",interpretation:{status:"completed",available:true,fullDetailsUrl:"/api/jobs/done/interpretation?includeItems=true"}},
  {id:"failed",interpretation:{status:"failed",available:true,fullDetailsUrl:"/api/jobs/failed/interpretation?includeItems=true"}},
  {id:"processing",interpretation:{status:"processing",available:true,fullDetailsUrl:"/api/jobs/processing/interpretation?includeItems=true"}},
  {id:"none",interpretation:{status:"failed",available:false,fullDetailsUrl:"/api/jobs/none/interpretation?includeItems=true"}}
];
const failures=await t.hydrateCompletedInterpretations(jobs);
console.log(JSON.stringify({requested,failures}));
"""
    )
    assert result["requested"] == [
        "/api/jobs/done/interpretation?includeItems=true",
        "/api/jobs/failed/interpretation?includeItems=true",
        "/api/jobs/processing/interpretation?includeItems=true",
    ]
    assert result["failures"] == 0


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("/api/jobs/j/interpret", True),
        ("/api/jobs/j/interpretation?includeItems=true", True),
        ("https://evil.example/api", False),
        ("//evil.example/api", False),
        ("/api/../secret", False),
        ("/api/%2e%2e/secret", False),
        ("/api/%252e%252e/secret", False),
        (r"/api\\secret", False),
        ("/api/%5csecret", False),
        ("/api/%255csecret", False),
    ],
)
def test_interpretation_urls_use_existing_strict_relative_url_guard(
    value: str,
    accepted: bool,
) -> None:
    result = _run_node(
        f"""
const value=t.safeRelativeUrl({json.dumps(value)});
console.log(JSON.stringify({{value}}));
"""
    )
    assert bool(result["value"]) is accepted


def test_untrusted_html_and_urls_are_not_injected() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{parts:1},sourceKinds:["vocals","<img>"],
  algorithms:{rhythm:{version:"<b>v1</b>"}},warnings:["<img src=x onerror=alert(1)>"],
  parts:[{label:"<svg onload=alert(1)>",sourceKind:"vocals",role:"pitched",voiceIds:[]}],
  phrases:[],pitchedItems:[],percussionItems:[]
});
const html=t.renderInterpretation({id:"job",interpretation:{
  status:"completed",available:true,progress:100,counts:{},canReinterpret:true,
  message:"<script>alert(1)</script>",startUrl:"/api/%252e%252e/private",
  fullDetailsUrl:"/api/jobs/job/interpretation?includeItems=true",
  downloadUrl:"https://evil.example/draft.json"
}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img" not in html
    assert "<svg" not in html
    assert "https://evil.example" not in html
    assert "%252e%252e" not in html
    assert "Re-interpret" not in html  # unsafe start URL disables the action


def test_job_state_tracks_interpretation_status() -> None:
    result = _run_node(
        """
const base={status:"completed",preparation_status:"completed",analysis:{status:"completed"},transcription:{status:"completed"}};
const processing=t.deriveState({...base,interpretation:{status:"processing"}});
const completed=t.deriveState({...base,interpretation:{status:"completed"}});
const failed=t.deriveState({...base,interpretation:{status:"failed"}});
console.log(JSON.stringify({processing,completed,failed}));
"""
    )
    assert result["processing"]["label"] == "Interpreting draft"
    assert result["completed"]["label"] == "Editable draft ready"
    assert result["failed"]["label"] == "Interpretation needs attention"


def test_no_premature_score_or_export_placeholders() -> None:
    source = APP_JS.read_text(encoding="utf-8").lower()
    for phrase in (
        "musicxml",
        "score engraving",
        "midi export",
        "tablature ready",
        "chord symbols ready",
        "publication-ready score",
    ):
        assert phrase not in source
