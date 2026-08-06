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
;globalThis.__popexTest={{
  renderTranscription,
  safeRelativeUrl,
  transcriptionCounts,
  transcriptionStageText,
  deriveState,
  hydrateCompletedTranscriptions,
  loadJobs,
  setDetail:(id,value)=>transcriptionCache.set(id,value),
  clearDetails:()=>transcriptionCache.clear(),
  setFetch:(value)=>{{fetchImpl=value;}},
  getElement:element,
  getListener:(selector,type)=>listeners.get(selector+":"+type)
}};`;
eval(source);
(async () => {{
  await Promise.resolve();
  await Promise.resolve();
  const t = globalThis.__popexTest;
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
  t.renderTranscription({id:"j1"}),
  t.renderTranscription({id:"j2",transcription:null}),
  t.renderTranscription({id:"j3",transcription:"future"}),
  t.renderTranscription({id:"j4",transcription:[]})
];
console.log(JSON.stringify({values}));
"""
    )
    assert result["values"] == ["", "", "", ""]


def test_start_retry_polling_and_existing_actions_remain() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'data-action="analysis"' in source
    assert 'data-action="separation"' in source
    assert 'data-action="transcription"' in source
    assert 'job.transcription?.status==="processing"' in source

    result = _run_node(
        """
const start = t.renderTranscription({
  id:"job-1",
  transcription:{
    enabled:true,status:"not_started",progress:0,counts:{pitched:0,percussion:0,aligned:0},
    canStart:true,startUrl:"/api/jobs/job-1/transcribe"
  }
});
const retry = t.renderTranscription({
  id:"job-2",
  transcription:{
    enabled:true,status:"failed",available:false,progress:20,
    counts:{pitched:0,percussion:0,aligned:0},
    canStart:true,startUrl:"/api/jobs/job-2/transcribe"
  }
});
let delay = null;
global.setTimeout = (_fn, value) => {delay = value; return 9;};
t.setFetch(async () => ({
  ok:true,status:200,
  json:async()=>[{id:"job-3",status:"completed",preparation_status:"completed",
    analysis:{status:"completed"},files:[],
    transcription:{status:"processing",progress:40,counts:{pitched:0,percussion:0,aligned:0}}}]
}));
await t.loadJobs();
console.log(JSON.stringify({start,retry,delay}));
"""
    )
    assert "Transcribe audio" in result["start"]
    assert 'data-action="transcription"' in result["start"]
    assert "Retry transcription" in result["retry"]
    assert result["delay"] == 1500


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("/api/jobs/j/transcribe", True),
        ("/api/jobs/j/details?format=json", True),
        ("https://evil.example/api", False),
        ("//evil.example/api", False),
        ("/api/../secret", False),
        ("/api/%2e%2e/secret", False),
        ("/api/%252e%252e/secret", False),
        (r"/api\\secret", False),
        ("/api/%5csecret", False),
        ("/api/%255csecret", False),
        ("/%252f%252fevil.example/path", False),
    ],
)
def test_safe_relative_url_rejects_traversal_and_backslashes(
    value: str,
    accepted: bool,
) -> None:
    result = _run_node(
        f"""
const value = t.safeRelativeUrl({json.dumps(value)});
console.log(JSON.stringify({{value}}));
"""
    )
    assert bool(result["value"]) is accepted


def test_alignment_fallback_counts_only_candidates_with_aligned_time() -> None:
    result = _run_node(
        """
const fallback = t.transcriptionCounts(
  {counts:{}},
  {
    pitchedNoteEvents:[{},{}],
    percussionEvents:[{}],
    alignmentCandidates:[
      {eventId:"p1",alignedTimeSeconds:1.0},
      {eventId:"p2",beatIndex:2},
      {eventId:"r1",alignedTimeSeconds:null},
      null
    ]
  }
);
const explicit = t.transcriptionCounts(
  {counts:{pitched:8,percussion:7,aligned:6}},
  {counts:{aligned:4},alignmentCandidates:[]}
);
console.log(JSON.stringify({fallback,explicit}));
"""
    )
    assert result["fallback"] == {"pitched": 2, "percussion": 1, "aligned": 2}
    assert result["explicit"]["aligned"] == 4


def test_exact_merged_pipeline_stages_are_musician_facing() -> None:
    result = _run_node(
        """
const stages = Object.fromEntries([
  "selecting_transcription_inputs",
  "detecting_pitched_events",
  "detecting_percussion_events",
  "aligning_transcription_events"
].map(stage => [stage,t.transcriptionStageText(stage,"processing")]));
console.log(JSON.stringify({stages}));
"""
    )
    assert result["stages"] == {
        "selecting_transcription_inputs": "Selecting transcription inputs",
        "detecting_pitched_events": "Detecting note candidates",
        "detecting_percussion_events": "Detecting percussion events",
        "aligning_transcription_events": "Building advisory alignment",
    }


def test_completed_details_render_counts_warnings_sources_algorithms_and_download() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  sourceKinds:["vocals","bass","vocals"],
  algorithms:{
    pitched:{version:"pitch-v1"},
    percussion:"drum-v1",
    unsafe:"<script>"
  },
  warnings:["Review <carefully>","second warning"],
  pitchedNoteEvents:[{},{}],
  percussionEvents:[{}],
  alignmentCandidates:[
    {alignedTimeSeconds:1},
    {beatIndex:2}
  ]
});
const html = t.renderTranscription({
  id:"job",
  transcription:{
    status:"completed",available:true,progress:100,counts:{},
    detailsUrl:"/api/jobs/job/transcription",
    downloadUrl:"/api/jobs/job/transcription/download"
  }
});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "<h4" in html and "Raw transcription" in html
    assert "<dt>Note candidates</dt><dd>2</dd>" in html
    assert "<dt>Percussion events</dt><dd>1</dd>" in html
    assert "<dt>Advisory alignment</dt><dd>1</dd>" in html
    assert html.index("Bass") < html.index("Vocals")
    assert "Review &lt;carefully&gt;" in html
    assert "Pitch" in html and "pitch-v1" in html
    assert "Percussion" in html and "drum-v1" in html
    assert '<a href="/api/jobs/job/transcription/download" download>' in html
    assert "<script>" not in html


def test_completed_and_failed_available_details_hydrate_from_safe_urls() -> None:
    result = _run_node(
        """
t.clearDetails();
const requested = [];
t.setFetch(async url => {
  requested.push(url);
  return {ok:true,status:200,json:async()=>({counts:{pitched:1,percussion:2,aligned:3}})};
});
const jobs = [
  {id:"completed",status:"completed",preparation_status:"completed",analysis:{status:"completed"},files:[],
   transcription:{status:"completed",detailsUrl:"/api/jobs/completed/transcription",counts:{}}},
  {id:"failed",status:"completed",preparation_status:"completed",analysis:{status:"completed"},files:[],
   transcription:{status:"failed",available:true,detailsUrl:"/api/jobs/failed/transcription",counts:{}}},
  {id:"no-artifact",status:"completed",preparation_status:"completed",analysis:{status:"completed"},files:[],
   transcription:{status:"failed",available:false,detailsUrl:"/api/jobs/no-artifact/transcription",counts:{}}}
];
const failures = await t.hydrateCompletedTranscriptions(jobs);
console.log(JSON.stringify({requested,failures}));
"""
    )
    assert result["requested"] == [
        "/api/jobs/completed/transcription",
        "/api/jobs/failed/transcription",
    ]
    assert result["failures"] == 0


def test_untrusted_urls_and_html_are_not_injected() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  sourceKinds:["vocals","<img>"],
  algorithms:{pitched:{version:"<b>v1</b>"}},
  warnings:["<img src=x onerror=alert(1)>"]
});
const html = t.renderTranscription({
  id:"job",
  transcription:{
    status:"completed",available:true,progress:100,counts:{pitched:1,percussion:0,aligned:0},
    message:"<svg onload=alert(1)>",
    detailsUrl:"/api/%252e%252e/private",
    downloadUrl:"https://evil.example/raw.json"
  }
});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "<svg" not in html
    assert "&lt;svg onload=alert(1)&gt;" in html
    assert "<img" not in html
    assert "https://evil.example" not in html
    assert "%252e%252e" not in html


def test_no_notation_or_accuracy_placeholder_claims() -> None:
    source = APP_JS.read_text(encoding="utf-8").lower()
    for phrase in (
        "sheet music",
        "publication-ready",
        "publication ready",
        "tablature ready",
        "notation ready",
        "accuracy guaranteed",
    ):
        assert phrase not in source
