"""Recording playback follows the timing of the displayed saved score."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from test_harmony_api import make_settings
from test_score_corrections import build, client_for, saved, score_file_digest
from test_score_percussion import create_drum_job
from test_score_notation import notation_node

ROOT = Path(__file__).resolve().parents[1]


def test_notation_can_request_compact_exact_saved_measure_timing(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        digest = score_file_digest(settings, job_id)
        document = saved(client, job_id)
        base = f"/api/jobs/{job_id}/score/saved"
        assert "measureTiming" not in client.get(base).json()
        response = client.get(base + "?includeMeasureTiming=true")
        assert response.status_code == 200
        assert response.json()["measureTiming"] == [
            {key: measure[key] for key in ("measureIndex", "startSeconds", "endSeconds")}
            for measure in document["measures"]
        ]
        assert "measures" not in response.json()
        assert score_file_digest(settings, job_id) == digest


def test_local_notation_has_explicit_recording_review_controls(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        page = client.get(f"/jobs/{job_id}/notation")
    assert page.status_code == 200
    assert 'id="notation-audio"' in page.text
    assert "Play shown bars" in page.text
    assert "Repeat shown bars" in page.text
    assert "score-notation-audio.js?v=" in page.text
    assert "autoplay" not in page.text


@pytest.mark.parametrize("flag", ["yes", "1", "", "TRUE"])
def test_compact_measure_timing_flag_is_strict(tmp_path, flag):
    with client_for(make_settings(tmp_path)) as client:
        assert client.get(f"/api/jobs/{'0' * 32}/score/saved?includeMeasureTiming={flag}").status_code == 422


def audio_node(body):
    script = r'''
const fs=require("fs");
class Element {
  constructor(){this.listeners={};this.children=[];this.value="";this.textContent="";this.disabled=false;this.hidden=false;this.checked=false;}
  addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
  fire(name){for(const fn of this.listeners[name]||[])fn();}
  replaceChildren(){this.children=[];}
  add(item){this.children.push(item);}
}
class Audio extends Element {
  constructor(){super();this.currentTime=0;this.duration=20;this.paused=true;this.ended=false;this.playbackRate=1;this.playCalls=0;this.loads=0;this.defer=false;this.pending=[];this.reject=false;}
  pause(){this.paused=true;this.fire("pause");}
  play(){this.playCalls++;this.paused=false;this.ended=false;this.fire("play");if(this.reject)return Promise.reject(new Error("simulated audio error"));return this.defer?new Promise((resolve,reject)=>this.pending.push({resolve,reject})):Promise.resolve();}
  load(){this.loads++;this.currentTime=0;}
  removeAttribute(key){delete this[key];}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,id==="notation-audio"?new Audio():new Element());return elements.get(id);};
global.Option=class{constructor(text,value){this.textContent=text;this.value=value;}};
eval(fs.readFileSync(__FILE__,"utf8"));
const {Review,snapshot,barAt,measureCue}=PopexNotationAudio;
const audio=get("notation-audio"),review=new Review(get),cue=[];
const value={jobId:"a".repeat(32),sourceUrl:"/api/jobs/"+"a".repeat(32)+"/files/analysis.wav",start:1,end:3,measures:[
  {measureIndex:0,startSeconds:-0.5,endSeconds:1.5},
  {measureIndex:1,startSeconds:1.5,endSeconds:4},
  {measureIndex:2,startSeconds:5,endSeconds:7}
]};
const report=()=>({position:get("notation-audio-position").textContent,message:get("notation-audio-status").textContent});
(async()=>{__BODY__ review.stop(false);})().catch(error=>{review.stop(false);console.error(error);process.exit(1);});
'''.replace("__FILE__", json.dumps(str(ROOT / "app/static/score-notation-audio.js"))).replace("__BODY__", body)
    result = subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def test_audio_snapshot_is_bound_to_exact_job_and_saved_timing():
    result = audio_node('''
const original=snapshot(value);value.measures[0].endSeconds=99;
const urls=["https://evil.test/x.wav","//evil.test/x.wav","/api/jobs/"+"b".repeat(32)+"/files/analysis.wav",value.sourceUrl+"?next=http://evil.test",value.sourceUrl.replace("analysis.wav","../analysis.wav")];
console.log(JSON.stringify({start:original.measures[0].startSeconds,end:original.measures[0].endSeconds,gap:barAt(original.measures,4.5),boundary:barAt(original.measures,1.5).measureIndex,refused:urls.every(sourceUrl=>snapshot({...value,sourceUrl})===null)}));
''')
    assert result == {"start": 0, "end": 1.5, "gap": None, "boundary": 1, "refused": True}


@pytest.mark.parametrize("setup", [
    "value.end=129;",
    "value.measures[1].startSeconds=1;",
    "value.measures[1].endSeconds=NaN;",
    "value.measures[1].measureIndex=0;",
    "value.measures.pop();",
    "value.measures[0].endSeconds=0;",
])
def test_audio_snapshot_refuses_unsafe_timing(setup):
    assert audio_node(setup + "console.log(JSON.stringify({refused:snapshot(value)===null}));") == {"refused": True}


def test_audio_no_autoplay_exact_bar_seek_and_bounded_stop():
    result = audio_node('''
review.setScore(value,(index,playing)=>cue.push([index,playing]));
const noAutoplay=audio.playCalls===0;
await review.playRange(2);const start=audio.currentTime;
audio.currentTime=2;audio.fire("timeupdate");const playing=report();
audio.currentTime=4;audio.fire("timeupdate");
console.log(JSON.stringify({noAutoplay,start,playing,paused:audio.paused,range:review.range,cue,message:report().message}));
''')
    assert result["noAutoplay"] and result["start"] == 1.5
    assert result["playing"]["position"].startswith("Playing bar 2")
    assert result["paused"] and result["range"] is None
    assert [1, True] in result["cue"] and "finished" in result["message"]


def test_audio_gap_outside_seek_repeat_and_new_snapshot_pause():
    result = audio_node('''
review.setScore(value);get("notation-audio-repeat").checked=true;
await review.playRange();audio.currentTime=4.5;audio.fire("timeupdate");const gap=report().position;
audio.currentTime=7;audio.fire("timeupdate");await Promise.resolve();
const repeated={time:audio.currentTime,calls:audio.playCalls,paused:audio.paused};
review.setScore({...value,start:2,end:3});const paused=audio.paused;
audio.currentTime=0;audio.fire("timeupdate");
console.log(JSON.stringify({gap,repeated,paused,outside:report().position,choices:get("notation-audio-bar").children.map(x=>x.value)}));
''')
    assert "timing gap" in result["gap"]
    assert result["repeated"] == {"time": 0, "calls": 2, "paused": False}
    assert result["paused"] and "outside" in result["outside"]
    assert result["choices"] == ["2", "3"]


def test_audio_stop_wins_over_pending_play_and_old_failure_cannot_stop_new_play():
    result = audio_node('''
review.setScore(value);audio.defer=true;
const old=review.playRange();review.stop();audio.pending[0].resolve();await old;
const stopped=audio.paused;
const a=review.playRange(1),b=review.playRange(2);
audio.pending[1].reject(new Error("old request failed"));await a;
audio.pending[2].resolve();await b;
console.log(JSON.stringify({stopped,paused:audio.paused,start:audio.currentTime,message:report().message}));
''')
    assert result["stopped"] and not result["paused"]
    assert result["start"] == 1.5 and result["message"].startswith("Playing bar 2")


def test_audio_error_and_short_recording_leave_notation_context_retryable():
    result = audio_node('''
review.setScore(value);audio.reject=true;await review.playRange();const failed=report().message;
audio.reject=false;audio.duration=1;await review.playRange(3);const short=report().message;
audio.duration=20;audio.fire("error");const error=report().message;
review.setScore(value);await review.playRange(1);
console.log(JSON.stringify({failed,short,error,retry:!audio.paused,loads:audio.loads}));
''')
    assert "could not start" in result["failed"]
    assert "after the recording ends" in result["short"]
    assert "Notation and downloads are unchanged" in result["error"]
    assert result["retry"] and result["loads"] == 2


def test_bar_cursor_uses_actual_indices_without_automatic_scroll():
    result = audio_node('''
const calls=[];const iterator={CurrentMeasureIndex:0,EndReached:false};
const cursor={Iterator:iterator,hide(){calls.push("hide");},show(){calls.push("show:"+iterator.CurrentMeasureIndex);},next(){iterator.CurrentMeasureIndex++;},reset(){calls.push("reset");iterator.CurrentMeasureIndex=0;}};
const update=measureCue({cursor});update(2,true);update(1,true);update(null,false);
measureCue({cursor:{hide(){},get Iterator(){throw new Error("bad cursor");}}})(2,true);
console.log(JSON.stringify({calls}));
''')
    assert "show:2" in result["calls"] and "show:1" in result["calls"] and "reset" in result["calls"]


def test_audio_script_syntax():
    subprocess.run(["node", "--check", str(ROOT / "app/static/score-notation-audio.js")], check=True, capture_output=True)


def test_failed_render_never_rebinds_the_previous_audio_snapshot():
    result = notation_node('''
await t.render({preventDefault(){}});
const first=t.element("notation-paper").children[0];
failRender=true;t.element("notation-view").value="original";
await t.render({preventDefault(){}});
console.log(JSON.stringify({calls:audioCalls.length,job:audioCalls[0].jobId,start:audioCalls[0].start,end:audioCalls[0].end,preserved:first===t.element("notation-paper").children[0]}));
''', audio_setup='''
const audioCalls=[];
global.PopexNotationAudio={Review:class{setScore(value){audioCalls.push(value);}},measureCue:()=>()=>{}};
''')
    assert result == {"calls": 1, "job": "a" * 32, "start": 1, "end": 2, "preserved": True}


def test_audio_initialization_failure_cannot_remove_a_successful_svg():
    result = notation_node('''
await t.render({preventDefault(){}});
console.log(JSON.stringify({svg:!!t.element("notation-paper").children[0]?.querySelector("svg"),disabled:t.element("notation-audio-controls").disabled,message:t.element("notation-audio-status").textContent}));
''', audio_setup='''
global.PopexNotationAudio={Review:class{setScore(){throw new Error("audio unavailable");}stop(){}},measureCue:()=>()=>{}};
''')
    assert result["svg"] and result["disabled"]
    assert "notation and downloads remain usable" in result["message"]
