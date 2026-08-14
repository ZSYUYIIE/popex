from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


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
      selector,
      value: "",
      files: [],
      disabled: false,
      focused: false,
      textContent: "",
      innerHTML: "",
      dataset: {{}},
      attributes: {{}},
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
;globalThis.__popexHarmonyTest={{
  renderHarmony,
  renderHarmonyReview,
  renderHarmonySegment,
  harmonyCounts,
  harmonyStageText,
  deriveState,
  safeRelativeUrl,
  hydrateCompletedAnalyses,
  hydrateCompletedSeparations,
  hydrateCompletedTranscriptions,
  hydrateCompletedInterpretations,
  hydrateCompletedHarmonies,
  loadJobs,
  setDetail:(id,value)=>harmonyCache.set(id,value),
  clearDetails:()=>harmonyCache.clear(),
  setFetch:(value)=>{{fetchImpl=value;}},
  getElement:element,
  getListener:(selector,type)=>listeners.get(selector+":"+type),
  getRevision:()=>jobsRevision,
  getPollHandle:()=>pollHandle,
  getCacheKeys:()=>({{
    analysis:[...analysisCache.keys()],
    separation:[...separationCache.keys()],
    transcription:[...transcriptionCache.keys()],
    interpretation:[...interpretationCache.keys()],
    harmony:[...harmonyCache.keys()]
  }})
}};`;
eval(source);
(async () => {{
  await Promise.resolve();
  await Promise.resolve();
  const t = globalThis.__popexHarmonyTest;
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
  t.renderHarmony({id:"j1"}),
  t.renderHarmony({id:"j2",harmony:null}),
  t.renderHarmony({id:"j3",harmony:"future"}),
  t.renderHarmony({id:"j4",harmony:[]})
];
console.log(JSON.stringify({values}));
"""
    )
    assert result["values"] == ["", "", "", ""]


def test_harmonize_retry_and_reharmonize_actions_render() -> None:
    result = _run_node(
        """
const start = t.renderHarmony({id:"new",harmony:{
  enabled:true,status:"not_started",progress:0,available:false,canStart:true,
  startUrl:"/api/jobs/new/harmonize",counts:{}
}});
const retry = t.renderHarmony({id:"failed",harmony:{
  enabled:true,status:"failed",progress:50,available:false,canStart:true,
  startUrl:"/api/jobs/failed/harmonize",counts:{},error:"bounded failure"
}});
const again = t.renderHarmony({id:"done",harmony:{
  enabled:true,status:"completed",progress:100,available:true,canReharmonize:true,
  startUrl:"/api/jobs/done/harmonize",counts:{events:3,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:0}
}});
console.log(JSON.stringify({start,retry,again}));
"""
    )
    assert 'data-action="harmony"' in result["start"]
    assert ">Harmonize</button>" in result["start"]
    assert 'data-force="false"' in result["start"]
    assert "Retry harmony" in result["retry"]
    assert 'data-retry="true"' in result["retry"]
    assert "Re-harmonize" in result["again"]
    assert 'data-force="true"' in result["again"]


def test_reharmonize_click_adds_force_true_and_preserves_existing_actions() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    for action in (
        'data-action="analysis"',
        'data-action="separation"',
        'data-action="transcription"',
        'data-action="interpretation"',
    ):
        assert action in source
    assert 'data-action="harmony"' in source

    result = _run_node(
        """
const requested = [];
t.setFetch(async (url, options={}) => {
  requested.push({url,method:options.method||"GET"});
  return {ok:true,status:200,json:async()=>[]};
});
const button = {
  dataset:{
    action:"harmony",jobId:"job",startUrl:"/api/jobs/job/harmonize?source=review",
    force:"true",retry:"false"
  },
  disabled:false,
  textContent:"Re-harmonize",
  focus(){}
};
const listener = t.getListener("#jobs","click");
await listener({target:{closest:()=>button}});
console.log(JSON.stringify({requested,buttonText:button.textContent}));
"""
    )
    assert result["requested"][0] == {
        "url": "/api/jobs/job/harmonize?source=review&force=true",
        "method": "POST",
    }
    assert result["buttonText"] == "Started"


def test_retry_click_does_not_force_and_unsafe_start_url_is_rejected() -> None:
    result = _run_node(
        """
const requested=[];
t.setFetch(async (url,options={})=>{
  requested.push({url,method:options.method||"GET"});
  return {ok:true,status:200,json:async()=>[]};
});
const listener=t.getListener("#jobs","click");
const retry={dataset:{action:"harmony",jobId:"retry",startUrl:"/api/jobs/retry/harmonize",force:"false",retry:"true"},disabled:false,textContent:"Retry harmony",focus(){this.focused=true;}};
await listener({target:{closest:()=>retry}});
const unsafe={dataset:{action:"harmony",jobId:"unsafe",startUrl:"https://evil.example/harmonize",force:"false",retry:"false"},disabled:false,textContent:"Harmonize",focused:false,focus(){this.focused=true;}};
await listener({target:{closest:()=>unsafe}});
console.log(JSON.stringify({requested,retryText:retry.textContent,unsafeText:unsafe.textContent,unsafeDisabled:unsafe.disabled,unsafeFocused:unsafe.focused,message:t.getElement("#jobs-message").textContent}));
"""
    )
    assert result["requested"][0] == {
        "url": "/api/jobs/retry/harmonize",
        "method": "POST",
    }
    assert len(result["requested"]) == 2  # second request is the job refresh
    assert result["retryText"] == "Started"
    assert result["unsafeText"] == "Harmonize"
    assert result["unsafeDisabled"] is False
    assert result["unsafeFocused"] is True
    assert "action URL is unavailable" in result["message"]


def test_processing_harmony_keeps_polling_and_change_announcements_track_it() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'job.harmony?.status==="processing"' in source
    assert "job.harmony?.stage" in source
    assert "job.harmony?.message" in source

    result = _run_node(
        """
let delay=null;
global.setTimeout=(_fn,value)=>{delay=value;return 9;};
let round=0;
t.setFetch(async()=>({ok:true,status:200,json:async()=>[{
  id:"job",title:"Song",status:"completed",stage:"completed",message:"Ready",
  preparation_status:"completed",files:[],analysis:{status:"completed"},
  transcription:{status:"completed"},
  harmony:{status:"processing",stage:round++?"saving_harmonic_context":"inferring_harmonic_context",message:round>1?"Saving harmonic context.":"Inferring harmonic candidates.",progress:55,available:false,counts:{}}
}]}));
await t.loadJobs();
await t.loadJobs();
console.log(JSON.stringify({delay,message:t.getElement("#jobs-message").textContent}));
"""
    )
    assert result["delay"] == 1500
    assert "Song: Inferring harmony." in result["message"]


def test_exact_backend_harmony_stages_are_musician_facing() -> None:
    result = _run_node(
        """
const stages=Object.fromEntries([
  "loading_raw_transcription",
  "loading_analysis_context",
  "loading_optional_interpretation",
  "inferring_harmonic_context",
  "validating_harmonic_context",
  "saving_harmonic_context"
].map(stage=>[stage,t.harmonyStageText(stage,"processing")]));
console.log(JSON.stringify({stages}));
"""
    )
    assert result["stages"] == {
        "loading_raw_transcription": "Loading raw pitch evidence",
        "loading_analysis_context": "Loading timing and tonal evidence",
        "loading_optional_interpretation": "Checking editable-part context",
        "inferring_harmonic_context": "Inferring harmonic candidates",
        "validating_harmonic_context": "Validating harmonic evidence",
        "saving_harmonic_context": "Saving harmonic context",
    }


def test_detail_counts_override_summary_and_have_safe_array_fallbacks() -> None:
    result = _run_node(
        """
const explicit=t.harmonyCounts(
  {counts:{events:99,segments:99,resolved:99,unresolved:99,unresolvedEvents:99,warnings:99}},
  {counts:{events:6,segments:4,resolved:3,unresolved:1,unresolvedEvents:2,warnings:5},rawEvidence:[{}],segments:[{}],warnings:[]}
);
const fallback=t.harmonyCounts(
  {counts:{}},
  {
    rawEvidence:[{id:"a"},{id:"b"},{id:"c"}],
    segments:[
      {unresolved:false,primaryCandidate:{}},
      {unresolved:true,primaryCandidate:null},
      null
    ],
    unresolvedEventIds:["b"],warnings:["one","two"]
  }
);
const malformed=t.harmonyCounts({counts:{events:true,segments:"4",resolved:null,unresolved:2.5,unresolvedEvents:-1,warnings:Infinity}},null);
console.log(JSON.stringify({explicit,fallback,malformed}));
"""
    )
    assert result["explicit"] == {
        "events": 6,
        "segments": 4,
        "resolved": 3,
        "unresolved": 1,
        "unresolvedEvents": 2,
        "warnings": 5,
    }
    assert result["fallback"] == {
        "events": 3,
        "segments": 2,
        "resolved": 1,
        "unresolved": 1,
        "unresolvedEvents": 1,
        "warnings": 2,
    }
    assert result["malformed"] == {
        "events": 0,
        "segments": 0,
        "resolved": 0,
        "unresolved": 0,
        "unresolvedEvents": 0,
        "warnings": 0,
    }


def test_completed_review_shows_local_evidence_alternatives_and_inversion() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{events:4,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:1},
  harmonyVersion:"harmonic-context-v1",createdAt:"2026-08-14T04:15:00+00:00",
  usedInterpretationContext:true,
  tonalContext:{tonalCenter:"C",collection:"ionian",displayName:"C <major>",confidence:.72,advisoryOnly:true},
  algorithms:{harmonyInference:{version:"pitch-class-window-v1"}},
  warnings:["Review <carefully>"],
  rawEvidence:[
    {id:"p_c",sourceKind:"vocals",rawStartSeconds:.125,rawEndSeconds:.8,midiNote:60,midiPitch:60.12,pitchClass:0,pitchName:"C",confidence:.91},
    {id:"p_bass_e",sourceKind:"bass",rawStartSeconds:.125,rawEndSeconds:.8,midiNote:52,midiPitch:52.08,pitchClass:4,pitchName:"E",confidence:.95},
    {id:"p_g",sourceKind:"other",rawStartSeconds:.2,rawEndSeconds:.8,midiNote:67,midiPitch:67.04,pitchClass:7,pitchName:"G",confidence:.86},
    {id:"n",sourceKind:"other",rawStartSeconds:.3,rawEndSeconds:.5,midiNote:66,midiPitch:66.03,pitchClass:6,pitchName:"F#",confidence:.2}
  ],
  segments:[{
    id:"segment_0001",rawStartSeconds:.125,rawEndSeconds:.875,windowMode:"beat",beatIndex:2,
    supportingEventIds:["p_c","p_bass_e","p_g","n"],sourceKinds:["vocals","bass","other"],
    partIds:["part_lead"],voiceIds:["voice_one"],unassignedContextEventIds:["n"],
    observedPitchClasses:[
      {pitchClass:0,pitchName:"C",weight:1.2,weightRatio:.45},
      {pitchClass:4,pitchName:"E",weight:1,weightRatio:.35},
      {pitchClass:7,pitchName:"G",weight:.5,weightRatio:.2}
    ],
    primaryCandidate:{
      rootPitchClass:0,root:"C",quality:"major",symbol:"C",pitchClasses:[0,4,7],score:.83,
      templateCoverage:1,chordToneWeightRatio:.92,nonChordToneRatio:.08,rootWeightRatio:.45,
      tonalContextSupport:.72,evidenceEventIds:["p_c","p_bass_e","p_g"],confidence:.78,
      inversionCandidate:{bassPitchClass:4,bassPitchName:"E",position:"first_inversion",confidence:.84,sourceEventIds:["p_bass_e"]}
    },
    alternatives:[
      {rootPitchClass:9,root:"A",quality:"minor_seventh",symbol:"Am7",pitchClasses:[9,0,4,7],score:.62,templateCoverage:.75,chordToneWeightRatio:.8,nonChordToneRatio:.2,rootWeightRatio:0,tonalContextSupport:.1,evidenceEventIds:["p_c","p_bass_e","p_g"],confidence:.48}
    ],
    unresolved:false,warnings:["Local <ambiguity>"]
  }],
  unresolvedEventIds:[]
});
const html=t.renderHarmony({id:"job",harmony:{
  status:"completed",available:true,progress:100,counts:{events:99},canReharmonize:true,
  version:"harmonic-context-v1",attemptVersion:"harmonic-context-v1",createdAt:"2026-08-14T04:15:00+00:00",
  usedInterpretationContext:true,startUrl:"/api/jobs/job/harmonize",
  fullDetailsUrl:"/api/jobs/job/harmony?includeSegments=true",downloadUrl:"/api/jobs/job/harmony/download"
}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "Harmonic context" in html
    assert "<dt>Raw pitch events</dt><dd>4</dd>" in html
    assert "<dt>Local windows</dt><dd>1</dd>" in html
    assert "<dt>Resolved windows</dt><dd>1</dd>" in html
    assert "Used editable-part context" in html
    assert "C &lt;major&gt;" in html
    assert "advisory" in html.lower()
    assert "raw 0.125–0.875 s" in html
    assert "Beat 3" in html
    assert "<strong>C</strong>" in html
    assert "Major" in html
    assert "78% confidence" in html
    assert "8% non-chord evidence" in html
    assert "C 45%" in html and "E 35%" in html and "G 20%" in html
    assert "Am7" in html and "48% confidence" in html
    assert "Bass-supported inversion candidate" in html
    assert "E · First Inversion" in html
    assert "p_bass_e" in html
    assert "Part Lead" in html and "Voice One" in html
    assert "p_c" in html and "MIDI 60.12" in html
    assert "Review &lt;carefully&gt;" in html
    assert "Local &lt;ambiguity&gt;" in html
    assert "Harmony Inference" in html and "pitch-class-window-v1" in html
    assert '<a href="/api/jobs/job/harmony/download" download>' in html


def test_unresolved_window_and_raw_evidence_remain_explicit() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{events:1,segments:1,resolved:0,unresolved:1,unresolvedEvents:1,warnings:0},
  rawEvidence:[{id:"lonely",sourceKind:"full_mix",rawStartSeconds:1,rawEndSeconds:1.5,midiPitch:61.23,pitchName:"C#",confidence:.31}],
  segments:[{id:"segment_1",rawStartSeconds:1,rawEndSeconds:2,windowMode:"absolute_time",supportingEventIds:["lonely"],sourceKinds:["full_mix"],partIds:[],voiceIds:[],unassignedContextEventIds:["lonely"],observedPitchClasses:[{pitchClass:1,pitchName:"C#",weight:.2,weightRatio:1}],primaryCandidate:null,alternatives:[{rootPitchClass:1,root:"C#",quality:"power",symbol:"C#5",pitchClasses:[1,8],confidence:.25,nonChordToneRatio:0,evidenceEventIds:["lonely"]}],unresolved:true,warnings:["Insufficient evidence"]}],
  unresolvedEventIds:["lonely"],warnings:[]
});
const html=t.renderHarmony({id:"job",harmony:{status:"completed",available:true,counts:{},fullDetailsUrl:"/api/jobs/job/harmony?includeSegments=true"}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "Unresolved local window" in html
    assert "Absolute-time window" in html
    assert "No primary candidate" in html
    assert "C#5" in html
    assert "Insufficient evidence" in html
    assert "lonely" in html
    assert "MIDI 61.23" in html


def test_failed_or_processing_with_available_result_keeps_previous_review() -> None:
    result = _run_node(
        """
t.clearDetails();
const detail={counts:{events:3,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:0},rawEvidence:[{id:"c",sourceKind:"full_mix",rawStartSeconds:0,rawEndSeconds:.5,midiPitch:60,pitchName:"C",confidence:.8}],segments:[{id:"s",rawStartSeconds:0,rawEndSeconds:1,windowMode:"beat",beatIndex:0,supportingEventIds:["c"],sourceKinds:["full_mix"],partIds:[],voiceIds:[],unassignedContextEventIds:[],observedPitchClasses:[{pitchClass:0,pitchName:"C",weight:1,weightRatio:1}],primaryCandidate:{root:"C",quality:"major",symbol:"C",confidence:.7,nonChordToneRatio:0,evidenceEventIds:["c"]},alternatives:[],unresolved:false,warnings:[]}],warnings:[]};
t.setDetail("failed",detail);t.setDetail("processing",detail);
const failed=t.renderHarmony({id:"failed",harmony:{status:"failed",available:true,progress:70,canStart:true,startUrl:"/api/jobs/failed/harmonize",fullDetailsUrl:"/api/jobs/failed/harmony?includeSegments=true",downloadUrl:"/api/jobs/failed/harmony/download",counts:{},error:"retry stopped"}});
const processing=t.renderHarmony({id:"processing",harmony:{status:"processing",available:true,progress:40,attemptVersion:"v2",version:"v1",fullDetailsUrl:"/api/jobs/processing/harmony?includeSegments=true",downloadUrl:"/api/jobs/processing/harmony/download",counts:{}}});
console.log(JSON.stringify({failed,processing}));
"""
    )
    assert "previous harmonic result is still available" in result["failed"].lower()
    assert "Retry harmony" in result["failed"]
    assert "<strong>C</strong>" in result["failed"]
    assert "previous valid harmonic result remains available" in result["processing"].lower()
    assert "<strong>C</strong>" in result["processing"]
    assert "Attempt version" in result["processing"]


def test_completed_failed_and_processing_available_details_hydrate_full_urls() -> None:
    result = _run_node(
        """
t.clearDetails();
const requested=[];
t.setFetch(async url=>{
  requested.push(url);
  return {ok:true,status:200,json:async()=>({counts:{segments:1},segments:[],rawEvidence:[]})};
});
const jobs=[
  {id:"done",harmony:{status:"completed",available:true,fullDetailsUrl:"/api/jobs/done/harmony?includeSegments=true"}},
  {id:"failed",harmony:{status:"failed",available:true,fullDetailsUrl:"/api/jobs/failed/harmony?includeSegments=true"}},
  {id:"processing",harmony:{status:"processing",available:true,fullDetailsUrl:"/api/jobs/processing/harmony?includeSegments=true"}},
  {id:"none",harmony:{status:"failed",available:false,fullDetailsUrl:"/api/jobs/none/harmony?includeSegments=true"}},
  {id:"unsafe",harmony:{status:"completed",available:true,fullDetailsUrl:"https://evil.example/harmony"}}
];
const failures=await t.hydrateCompletedHarmonies(jobs);
console.log(JSON.stringify({requested,failures}));
"""
    )
    assert result["requested"] == [
        "/api/jobs/done/harmony?includeSegments=true",
        "/api/jobs/failed/harmony?includeSegments=true",
        "/api/jobs/processing/harmony?includeSegments=true",
    ]
    assert result["failures"] == 1


def test_loading_and_failed_detail_states_are_local_and_recoverable() -> None:
    result = _run_node(
        """
t.clearDetails();
const loading=t.renderHarmony({id:"loading",harmony:{status:"completed",available:true,counts:{},fullDetailsUrl:"/api/jobs/loading/harmony?includeSegments=true"}});
t.setDetail("failed",null);
const failed=t.renderHarmony({id:"failed",harmony:{status:"completed",available:true,counts:{events:3},fullDetailsUrl:"/api/jobs/failed/harmony?includeSegments=true",downloadUrl:"/api/jobs/failed/harmony/download"}});
console.log(JSON.stringify({loading,failed}));
"""
    )
    assert "Loading harmonic-context details" in result["loading"]
    assert "details could not be loaded" in result["failed"]
    assert "Raw transcription and earlier artifacts remain available" in result["failed"]
    assert "Download harmonic context JSON" in result["failed"]


def test_untrusted_strings_urls_numbers_and_malformed_arrays_are_sanitized() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{events:"bad",segments:-1,resolved:true,unresolved:2.5,unresolvedEvents:Infinity,warnings:NaN},
  harmonyVersion:"<img src=x onerror=alert(1)>",createdAt:"javascript:alert(1)",
  tonalContext:{displayName:"api_key=secret",tonalCenter:"<C>",collection:"../../ionian",confidence:Infinity},
  algorithms:{"<script>":{version:"https://evil.example"}},
  warnings:["<svg onload=alert(1)>","debug at /home/user/private.wav","token=secret"],
  rawEvidence:[null,{id:"<img>",sourceKind:"../evil",rawStartSeconds:-1,rawEndSeconds:Infinity,midiPitch:999,pitchName:"<C>",confidence:2}],
  segments:[null,{id:"<script>",rawStartSeconds:-1,rawEndSeconds:Infinity,windowMode:"../beat",supportingEventIds:["<img>"],sourceKinds:["<svg>"],partIds:["../../part"],voiceIds:["token=secret"],unassignedContextEventIds:[],observedPitchClasses:[{pitchName:"<b>C</b>",weightRatio:2}],primaryCandidate:{symbol:"<img onerror=1>",root:"<C>",quality:"../major",confidence:2,nonChordToneRatio:-1,evidenceEventIds:["<img>"]},alternatives:"bad",unresolved:false,warnings:["https://evil.example"]}],
  unresolvedEventIds:"bad"
});
const html=t.renderHarmony({id:"job",harmony:{status:"completed",available:true,counts:{},startUrl:"https://evil.example/start",downloadUrl:"https://evil.example/download",fullDetailsUrl:"/api/%252e%252e/private"}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "<img" not in html
    assert "<script" not in html
    assert "<svg" not in html
    assert "evil.example" not in html
    assert "%252e%252e" not in html
    assert "/home/user" not in html
    assert "token=secret" not in html
    assert "api_key=secret" not in html
    assert "Infinity" not in html
    assert "NaN" not in html
    assert "javascript:" not in html


def test_unknown_status_and_detached_malformed_detail_do_not_crash() -> None:
    result = _run_node(
        """
t.clearDetails();
const unknown=t.renderHarmony({id:"unknown",harmony:{status:"future_state",progress:"bad",counts:null,message:{bad:true}}});
t.setDetail("bounded",{counts:{},rawEvidence:Array.from({length:80},(_,i)=>({id:`e_${i}`,sourceKind:"other",rawStartSeconds:i,rawEndSeconds:i+.5,midiPitch:60,pitchName:"C",confidence:.5})),segments:Array.from({length:80},(_,i)=>({id:`s_${i}`,rawStartSeconds:i,rawEndSeconds:i+1,windowMode:"absolute_time",supportingEventIds:[`e_${i}`],sourceKinds:["other"],partIds:[],voiceIds:[],unassignedContextEventIds:[],observedPitchClasses:[{pitchName:"C",weightRatio:1}],primaryCandidate:null,alternatives:[],unresolved:true,warnings:[]})),warnings:[]});
const bounded=t.renderHarmony({id:"bounded",harmony:{status:"completed",available:true,counts:{},fullDetailsUrl:"/api/jobs/bounded/harmony?includeSegments=true"}});
console.log(JSON.stringify({unknown,bounded,segments:(bounded.match(/Unresolved local window/g)||[]).length}));
"""
    )
    assert "Status update" in result["unknown"]
    assert result["segments"] == 48


def test_derive_state_prioritizes_active_harmony_and_reports_outcomes() -> None:
    result = _run_node(
        """
const base={preparation_status:"completed",analysis:{status:"completed"},transcription:{status:"completed"}};
const processing=t.deriveState({...base,harmony:{status:"processing"}});
const completed=t.deriveState({...base,harmony:{status:"completed"}});
const failed=t.deriveState({...base,harmony:{status:"failed"}});
const unknown=t.deriveState({...base,harmony:{status:"future"}});
console.log(JSON.stringify({processing,completed,failed,unknown}));
"""
    )
    assert result["processing"]["label"] == "Inferring harmony"
    assert result["completed"]["label"] == "Harmonic context ready"
    assert result["failed"]["label"] == "Harmony needs attention"
    assert result["unknown"]["label"] == "Harmony status update"


def test_null_numeric_evidence_does_not_invent_zero_precision() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  counts:{events:1,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:0},
  tonalContext:{displayName:"C major",confidence:null},
  rawEvidence:[{id:"raw",sourceKind:"other",rawStartSeconds:null,rawEndSeconds:null,midiPitch:null,pitchName:"C",confidence:null}],
  segments:[{
    id:"segment",rawStartSeconds:null,rawEndSeconds:null,windowMode:"beat",beatIndex:null,
    supportingEventIds:["raw"],sourceKinds:["other"],partIds:[],voiceIds:[],unassignedContextEventIds:[],
    observedPitchClasses:[{pitchName:"C",weightRatio:null}],
    primaryCandidate:{symbol:"C",root:"C",quality:"major",confidence:null,nonChordToneRatio:null,evidenceEventIds:["raw"]},
    alternatives:[],unresolved:false,warnings:[]
  }],warnings:[]
});
const html=t.renderHarmony({id:"job",harmony:{status:"completed",available:true,counts:{},fullDetailsUrl:"/api/jobs/job/harmony?includeSegments=true"}});
console.log(JSON.stringify({html}));
"""
    )
    html = result["html"]
    assert "raw 0.000 s" not in html
    assert "MIDI 0.00" not in html
    assert "0% confidence" not in html
    assert "Beat 1" not in html
    assert "raw unresolved" in html
    assert "MIDI unresolved" in html
    assert "not available" in html


def test_out_of_order_harmony_hydration_does_not_resurrect_disappeared_job() -> None:
    result = _run_node(
        """
t.clearDetails();
let resolveDetail=null;
t.setFetch(async url=>{
  if(url==="/api/jobs")return {ok:true,status:200,json:async()=>[]};
  if(url==="/api/jobs/old/harmony?includeSegments=true")return await new Promise(resolve=>{resolveDetail=()=>resolve({ok:true,status:200,json:async()=>({counts:{events:1},rawEvidence:[],segments:[]})});});
  throw new Error("unexpected request "+url);
});
const stale=[{id:"old",status:"completed",preparation_status:"completed",files:[],analysis:{status:"completed"},transcription:{status:"completed"},harmony:{status:"completed",available:true,counts:{},fullDetailsUrl:"/api/jobs/old/harmony?includeSegments=true"}}];
const revision=t.getRevision();
const pending=t.hydrateCompletedHarmonies(stale,revision);
await Promise.resolve();
await t.loadJobs();
const newer=t.getElement("#jobs").innerHTML;
resolveDetail();
await pending;
const after=t.getElement("#jobs").innerHTML;
console.log(JSON.stringify({newer,after}));
"""
    )
    assert "No saved audio yet" in result["newer"]
    assert "No saved audio yet" in result["after"]
    assert "harmony-old" not in result["after"]


def test_new_harmony_artifact_identity_invalidates_success_cache() -> None:
    result = _run_node(
        """
t.clearDetails();
t.setDetail("job",{
  harmonyVersion:"v1",createdAt:"2026-08-14T04:00:00+00:00",counts:{events:1,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:0},
  rawEvidence:[{id:"old",sourceKind:"other",rawStartSeconds:0,rawEndSeconds:.5,midiPitch:60,pitchName:"C",confidence:.8}],
  segments:[{id:"old_segment",rawStartSeconds:0,rawEndSeconds:1,windowMode:"absolute_time",supportingEventIds:["old"],sourceKinds:["other"],partIds:[],voiceIds:[],unassignedContextEventIds:[],observedPitchClasses:[{pitchName:"C",weightRatio:1}],primaryCandidate:{symbol:"C",root:"C",quality:"major",confidence:.8,nonChordToneRatio:0,evidenceEventIds:["old"]},alternatives:[],unresolved:false,warnings:[]}],warnings:[]
});
const requested=[];
t.setFetch(async url=>{
  requested.push(url);
  if(url==="/api/jobs")return {ok:true,status:200,json:async()=>[{id:"job",status:"completed",preparation_status:"completed",files:[],analysis:{status:"completed"},transcription:{status:"completed"},harmony:{status:"completed",available:true,version:"v2",createdAt:"2026-08-14T05:00:00+00:00",counts:{},fullDetailsUrl:"/api/jobs/job/harmony?includeSegments=true"}}]};
  if(url==="/api/jobs/job/harmony?includeSegments=true")return {ok:true,status:200,json:async()=>({harmonyVersion:"v2",createdAt:"2026-08-14T05:00:00+00:00",counts:{events:1,segments:1,resolved:1,unresolved:0,unresolvedEvents:0,warnings:0},rawEvidence:[{id:"new",sourceKind:"other",rawStartSeconds:0,rawEndSeconds:.5,midiPitch:62,pitchName:"D",confidence:.8}],segments:[{id:"new_segment",rawStartSeconds:0,rawEndSeconds:1,windowMode:"absolute_time",supportingEventIds:["new"],sourceKinds:["other"],partIds:[],voiceIds:[],unassignedContextEventIds:[],observedPitchClasses:[{pitchName:"D",weightRatio:1}],primaryCandidate:{symbol:"D",root:"D",quality:"major",confidence:.8,nonChordToneRatio:0,evidenceEventIds:["new"]},alternatives:[],unresolved:false,warnings:[]}],warnings:[]})};
  throw new Error("unexpected request "+url);
});
await t.loadJobs();
const html=t.getElement("#jobs").innerHTML;
console.log(JSON.stringify({requested,html}));
"""
    )
    assert "/api/jobs/job/harmony?includeSegments=true" in result["requested"]
    assert "<strong>D</strong>" in result["html"]
    assert "<strong>C</strong>" not in result["html"]


def test_duplicate_harmony_action_is_ignored_while_first_request_is_pending() -> None:
    result = _run_node(
        """
const posts=[];
const releases=[];
t.setFetch((url,options={})=>{
  if(options.method==="POST"){
    posts.push(url);
    return new Promise(resolve=>releases.push(()=>resolve({ok:true,status:200,json:async()=>({})})));
  }
  return Promise.resolve({ok:true,status:200,json:async()=>[]});
});
const button={dataset:{action:"harmony",jobId:"job",startUrl:"/api/jobs/job/harmonize",force:"false",retry:"false"},disabled:false,textContent:"Harmonize",focus(){}};
const listener=t.getListener("#jobs","click");
const first=listener({target:{closest:()=>button}});
await Promise.resolve();
const second=listener({target:{closest:()=>button}});
await Promise.resolve();
const beforeRelease=posts.length;
for(const release of releases)release();
await Promise.all([first,second]);
console.log(JSON.stringify({beforeRelease,posts}));
"""
    )
    assert result["beforeRelease"] == 1
    assert result["posts"] == ["/api/jobs/job/harmonize"]


def test_failed_harmony_request_reenables_action_and_restores_focus() -> None:
    result = _run_node(
        """
t.setFetch(async (url,options={})=>({ok:false,status:500,json:async()=>({detail:"bounded failure"})}));
const button={dataset:{action:"harmony",jobId:"job",startUrl:"/api/jobs/job/harmonize",force:"false",retry:"false"},disabled:false,focused:false,textContent:"Harmonize",focus(){this.focused=true;}};
const listener=t.getListener("#jobs","click");
await listener({target:{closest:()=>button}});
console.log(JSON.stringify({disabled:button.disabled,focused:button.focused,text:button.textContent,message:t.getElement("#jobs-message").textContent}));
"""
    )
    assert result["disabled"] is False
    assert result["focused"] is True
    assert result["text"] == "Harmonize"
    assert result["message"] == "bounded failure"


def test_newer_load_wins_when_older_success_resolves_last() -> None:
    result = _run_node(
        """
const requests=[];
const timerCalls=[];
global.setTimeout=(_fn,delay)=>{timerCalls.push(delay);return timerCalls.length;};
global.clearTimeout=()=>{};
t.setFetch(url=>{
  if(url!=="/api/jobs")throw new Error("unexpected request "+url);
  return new Promise((resolve,reject)=>requests.push({resolve,reject}));
});
const older=t.loadJobs({announce:true});
const newer=t.loadJobs({announce:true});
requests[1].resolve({ok:true,status:200,json:async()=>[{
  id:"new",title:"New job",status:"processing",stage:"processing",message:"New work",
  preparation_status:"processing",files:[]
}]});
await newer;
const snapshot=()=>({
  html:t.getElement("#jobs").innerHTML,
  message:t.getElement("#jobs-message").textContent,
  busy:t.getElement("#jobs").getAttribute("aria-busy"),
  refreshDisabled:t.getElement("#refresh-button").disabled,
  poll:t.getPollHandle(),
  timerCalls:[...timerCalls],
  caches:t.getCacheKeys()
});
const before=snapshot();
requests[0].resolve({ok:true,status:200,json:async()=>[{
  id:"old",title:"Old job",status:"completed",stage:"completed",message:"Old work",
  preparation_status:"completed",files:[],analysis:{status:"not_started"}
}]});
await older;
const after=snapshot();
console.log(JSON.stringify({before,after}));
"""
    )
    assert result["before"] == result["after"]
    assert "New job" in result["after"]["html"]
    assert "Old job" not in result["after"]["html"]
    assert result["after"]["message"] == "Recent audio is up to date."
    assert result["after"]["busy"] == "false"
    assert result["after"]["refreshDisabled"] is False
    assert result["after"]["timerCalls"] == [1500]
    assert result["after"]["caches"] == {
        "analysis": [],
        "separation": [],
        "transcription": [],
        "interpretation": [],
        "harmony": [],
    }


def test_newer_load_wins_when_older_failure_resolves_last() -> None:
    result = _run_node(
        """
const requests=[];
const timerCalls=[];
global.setTimeout=(_fn,delay)=>{timerCalls.push(delay);return timerCalls.length;};
global.clearTimeout=()=>{};
t.setFetch(url=>{
  if(url!=="/api/jobs")throw new Error("unexpected request "+url);
  return new Promise((resolve,reject)=>requests.push({resolve,reject}));
});
const older=t.loadJobs({announce:true});
const newer=t.loadJobs({announce:true});
requests[1].resolve({ok:true,status:200,json:async()=>[{
  id:"new",title:"New job",status:"processing",stage:"processing",message:"New work",
  preparation_status:"processing",files:[]
}]});
await newer;
const snapshot=()=>({
  html:t.getElement("#jobs").innerHTML,
  message:t.getElement("#jobs-message").textContent,
  busy:t.getElement("#jobs").getAttribute("aria-busy"),
  refreshDisabled:t.getElement("#refresh-button").disabled,
  poll:t.getPollHandle(),
  timerCalls:[...timerCalls],
  caches:t.getCacheKeys()
});
const before=snapshot();
requests[0].reject(new Error("stale request failed"));
await older;
const after=snapshot();
console.log(JSON.stringify({before,after}));
"""
    )
    assert result["before"] == result["after"]
    assert "New job" in result["after"]["html"]
    assert "Recent audio could not be loaded" not in result["after"]["html"]
    assert "stale request failed" not in result["after"]["message"]
    assert result["after"]["busy"] == "false"
    assert result["after"]["refreshDisabled"] is False
    assert result["after"]["timerCalls"] == [1500]


def test_superseded_inflight_hydrators_cannot_cache_or_rerender() -> None:
    result = _run_node(
        """
await t.loadJobs();
const cases=[
  {
    name:"analysis",url:"/detail/analysis",hydrate:t.hydrateCompletedAnalyses,
    jobs:[{id:"old-analysis",analysis:{status:"completed",endpoint:"/detail/analysis"}}],
    payload:{warnings:["old analysis"]},cache:"analysis"
  },
  {
    name:"separation",url:"/detail/separation",hydrate:t.hydrateCompletedSeparations,
    jobs:[{id:"old-separation",separation:{status:"completed",detailsUrl:"/detail/separation"}}],
    payload:{stems:[]},cache:"separation"
  },
  {
    name:"transcription",url:"/detail/transcription",hydrate:t.hydrateCompletedTranscriptions,
    jobs:[{id:"old-transcription",transcription:{status:"completed",detailsUrl:"/detail/transcription"}}],
    payload:{counts:{}},cache:"transcription"
  },
  {
    name:"interpretation",url:"/detail/interpretation",hydrate:t.hydrateCompletedInterpretations,
    jobs:[{id:"old-interpretation",interpretation:{status:"completed",available:true,fullDetailsUrl:"/detail/interpretation"}}],
    payload:{counts:{}},cache:"interpretation"
  },
  {
    name:"harmony",url:"/detail/harmony",hydrate:t.hydrateCompletedHarmonies,
    jobs:[{id:"old-harmony",harmony:{status:"completed",available:true,fullDetailsUrl:"/detail/harmony"}}],
    payload:{counts:{},rawEvidence:[],segments:[]},cache:"harmony"
  }
];
const results=[];
for(const item of cases){
  const revision=t.getRevision();
  let resolveDetail=null;
  t.setFetch(url=>{
    if(url===item.url)return new Promise(resolve=>{resolveDetail=resolve;});
    if(url==="/api/jobs")return Promise.resolve({ok:true,status:200,json:async()=>[{
      id:`new-${item.name}`,title:`New ${item.name}`,status:"processing",stage:"processing",message:"New work",
      preparation_status:"processing",files:[]
    }]});
    throw new Error("unexpected request "+url);
  });
  const pending=item.hydrate(item.jobs,revision);
  await Promise.resolve();
  await t.loadJobs({announce:true});
  const snapshot=()=>({
    html:t.getElement("#jobs").innerHTML,
    message:t.getElement("#jobs-message").textContent,
    busy:t.getElement("#jobs").getAttribute("aria-busy"),
    refreshDisabled:t.getElement("#refresh-button").disabled,
    poll:t.getPollHandle(),
    caches:t.getCacheKeys()
  });
  const before=snapshot();
  resolveDetail({ok:true,status:200,json:async()=>item.payload});
  await pending;
  const after=snapshot();
  results.push({name:item.name,cache:item.cache,oldId:`old-${item.name}`,before,after});
}
console.log(JSON.stringify({results}));
"""
    )
    assert [item["name"] for item in result["results"]] == [
        "analysis",
        "separation",
        "transcription",
        "interpretation",
        "harmony",
    ]
    for item in result["results"]:
        assert item["before"] == item["after"]
        assert f"New {item['name']}" in item["after"]["html"]
        assert item["oldId"] not in item["after"]["caches"][item["cache"]]
        assert item["after"]["busy"] == "false"
        assert item["after"]["refreshDisabled"] is False


def test_harmony_segment_css_stacks_and_wraps_for_narrow_reflow() -> None:
    source = STYLES_CSS.read_text(encoding="utf-8")
    assert re.search(
        r"\.harmony-segments\s*>\s*li\s*\{[^}]*display\s*:\s*grid",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"\.harmony-segments\s*>\s*li\s*>\s*\*\s*\{[^}]*overflow-wrap\s*:\s*anywhere",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"\.harmony-segments\s+\.candidate-list\s+li\s*\{[^}]*flex-wrap\s*:\s*wrap",
        source,
        re.DOTALL,
    )


def test_no_final_score_export_or_voicing_claims() -> None:
    source = APP_JS.read_text(encoding="utf-8").lower()
    for phrase in (
        "roman numeral",
        "final chord chart",
        "exact voicing",
        "guitar voicing",
        "tablature",
        "musicxml",
        "midi export",
        "engraving",
        "publication-ready",
        "publication ready",
    ):
        assert phrase not in source
