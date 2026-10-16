"""Local score notation reuses saved MusicXML without changing predictions."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import make_settings
from test_score_corrections import add, build, client_for, saved, score_file_digest
from test_score_percussion import create_drum_job

ROOT = Path(__file__).resolve().parents[1]


def test_saved_score_has_local_notation_page(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        page = client.get(f"/jobs/{job_id}/notation")
    assert page.status_code == 200
    assert "Render notation" in page.text
    assert "score-notation.js" in page.text
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["referrer-policy"] == "no-referrer"
    policy = page.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "connect-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy and "base-uri 'none'" in policy
    assert "unsafe-eval" not in policy
    assert not re.search(r"<script(?![^>]*src=)", page.text)
    assert not re.search(r"(?:src|href)=\"https?://", page.text)
    assert "/static/score-notation.js?v=" in page.text
    assert 'id="notation-paper" aria-hidden="true"' in page.text


@pytest.mark.parametrize("job_id", ["no-such-job", "0" * 32, "http:example.com"])
def test_notation_page_rejects_unknown_or_invalid_recordings(tmp_path, job_id):
    with client_for(make_settings(tmp_path)) as client:
        assert client.get(f"/jobs/{job_id}/notation").status_code == 404


def test_pinned_renderer_and_complete_notices_are_served_locally(tmp_path):
    vendor = ROOT / "app/static/vendor/osmd-2.2.1"
    manifest = json.loads((vendor / "manifest.json").read_text())
    asset = (vendor / manifest["file"]).read_bytes()
    assert len(asset) == manifest["bytes"] == 1420444
    assert hashlib.sha256(asset).hexdigest() == manifest["sha256"] == "224ee5bee61d31ff3d07ade025f931a6cecd394cb54d2e39b1e7af2f0c7f1f5d"
    assert b"sourceMappingURL" not in asset
    notices = (vendor / manifest["notices"]).read_text(encoding="utf-8")
    for value in ("PhonicScore", "Mohit Muthanna", "Tim Perry", "Tomasz Ciborski", "Basarat Ali Syed",
                  "Stuart Knightley", "Calvin Metcalf", "Barnesandnoble", "Vitaly Puzrin",
                  "Jean-loup Gailly", "Mark Adler", "Simon Tatham", "zlib notice", "MIT option selected"):
        assert value in notices
    with client_for(make_settings(tmp_path)) as client:
        response = client.get("/static/vendor/osmd-2.2.1/opensheetmusicdisplay.min.js")
        assert response.status_code == 200 and response.content == asset
        license_response = client.get("/static/vendor/osmd-2.2.1/opensheetmusicdisplay.min.js.LICENSE.txt")
        assert license_response.status_code == 200 and "PhonicScore" in license_response.text
    assert "-text" in (ROOT / ".gitattributes").read_text()
    assert "recursive-include app/static *" in (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include app/templates *" in (ROOT / "MANIFEST.in").read_text()


def test_saved_export_headers_describe_actual_snapshot_and_reads_preserve_files(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        digest = score_file_digest(settings, job_id)
        assert add(client, job_id, 0, {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 62}).status_code == 200
        for view in ("corrected", "original"):
            detail = saved(client, job_id, view)
            for format in ("musicxml", "midi", "json"):
                response = client.get(f"/api/jobs/{job_id}/score/saved/download?format={format}&view={view}")
                assert response.status_code == 200
                assert response.headers["x-popex-score-created-at"] == detail["createdAt"]
                assert response.headers["x-popex-correction-revision"] == "1"
                assert response.headers["x-popex-score-view"] == view
                assert response.headers["cache-control"] == "no-store"
        assert score_file_digest(settings, job_id) == digest


def test_main_score_action_links_only_saved_scores_to_local_notation():
    result = _run_node(f'''
const ready=t.renderScore({{id:"done",score:{json.dumps(SUMMARY)}}});
const waiting=t.renderScore({{id:"waiting",score:{{...{json.dumps(SUMMARY)},available:false}}}});
console.log(JSON.stringify({{ready,waiting}}));
''')
    assert 'href="/jobs/done/notation">Open notation</a>' in result["ready"]
    assert "Open notation" not in result["waiting"]


def notation_node(body, audio_setup=""):
    """Unit-test the view controller, not a substitute for real OSMD browser QA."""
    script = r'''
const fs=require("fs");
const nodes=new Map();
class Element {
  constructor(tag="div"){this.tag=tag;this.value="";this.children=[];this.options=[];this.dataset={};this.listeners={};this.textContent="";this.disabled=false;this.hidden=false;this.className="";this.isConnected=true;this.focused=false;}
  addEventListener(kind,handler){this.listeners[kind]=handler;}
  setAttribute(name,value){this[name]=value;}
  reportValidity(){return true;}
  contains(item){return (this.elements||[]).includes(item);}
  focus(){global.document.activeElement=this;this.focused=true;}
  replaceChildren(...items){for(const old of this.children)old.parent=null;this.children=[];this.options=[];for(const item of items)this.append(item);}
  append(item){item.parent=this;this.children.push(item);if(this.tag==="select")this.options.push(item);}
  add(item){this.append(item);}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(item=>item!==this);}
  querySelector(selector){return this.children.find(item=>item.tag===selector)||null;}
  get selectedIndex(){return this.options.findIndex(option=>option.value===this.value);}
}
function element(id){if(!nodes.has(id))nodes.set(id,new Element(id==="notation-part"||id==="notation-view"?"select":"div"));return nodes.get(id);}
global.document={getElementById:element,createElement:tag=>new Element(tag)};
global.Option=class extends Element{constructor(label,value){super("option");this.textContent=label;this.value=value;}};
global.location={pathname:"/jobs/"+"a".repeat(32)+"/notation"};
element("notation-view").value="corrected";
element("notation-from").value="1";element("notation-through").value="2";element("notation-zoom").value="1";
element("notation-controls").elements=["notation-view","notation-part","notation-from","notation-through","notation-zoom","notation-render"].map(element);
let failRender=false,forceRevision=null,xml="<?xml version='1.0'?><score-partwise/>",rendererCalls=0,parts=[{id:"drums",name:"Drums"}];
const requested=[];
const details=()=>({available:true,createdAt:"2026-10-16T00:00:00Z",counts:{measures:2},corrections:{revision:3},exportParts:parts,warnings:["Review drums."]});
global.fetch=async url=>{
  requested.push(url);
  if(url.includes("/download?"))return new Response(xml,{headers:{"X-PopEx-Score-Created-At":details().createdAt,"X-PopEx-Correction-Revision":String(forceRevision??3),"X-PopEx-Score-View":new URL(url,"http://local").searchParams.get("view")}});
  return new Response(JSON.stringify(url.includes("/score/saved")?details():{title:"<img src=x onerror=evil>",version:{label:"Live",kind:"live"}}));
};
global.DOMParser=class {parseFromString(){return {documentElement:{localName:"score-partwise"},querySelector:()=>null,querySelectorAll:()=>[]};}};
global.opensheetmusicdisplay={OpenSheetMusicDisplay:class{constructor(stage,options){this.stage=stage;this.options=options;rendererCalls++;}async load(){}render(){if(failRender)throw new Error("simulated renderer failure");this.stage.append(new Element("svg"));}}};
__AUDIO_SETUP__
const file=__FILE__;
let source=fs.readFileSync(file,"utf8").replace("if (match) reload();","globalThis.t={reload,render,boundedXml,endpoint,element}; if (match) reload();");
eval(source);
(async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve)); __BODY__})().catch(error=>{console.error(error);process.exit(1);});
'''.replace("__FILE__", json.dumps(str(ROOT / "app/static/score-notation.js"))).replace("__BODY__", body).replace("__AUDIO_SETUP__", audio_setup)
    result = subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def test_notation_controller_part_view_local_fetch_and_failure_preservation():
    result = notation_node('''
t.element("notation-part").value="drums";
await t.render({preventDefault(){}});
const first=t.element("notation-paper").children[0];
failRender=true;await t.render({preventDefault(){}});
console.log(JSON.stringify({requested,title:t.element("recording-title").textContent,partName:t.element("notation-part").options[1].textContent,preserved:first===t.element("notation-paper").children[0],message:t.element("notation-status").textContent,rendererCalls}));
''')
    assert all(url.startswith("/api/jobs/" + "a" * 32) for url in result["requested"])
    assert any("part=drums" in url and "view=corrected" in url for url in result["requested"])
    assert result["title"].startswith("<img")  # assigned as textContent, never interpreted
    assert result["partName"] == "Drums"
    assert result["preserved"] and result["rendererCalls"] == 2
    assert "previous render remains shown" in result["message"]


@pytest.mark.parametrize("setup,expected", [
    ("forceRevision=4;", "score changed while loading"),
    ("xml='<!DOCTYPE score-partwise><score-partwise/>';", "unsupported declarations"),
    ("t.element('notation-through').value='129';", "1 to 128 bars"),
    ("t.element('notation-part').value='drums';parts=[];", "part no longer exists"),
])
def test_notation_controller_refuses_stale_unsafe_or_invalid_selection(setup, expected):
    result = notation_node(setup + '''await t.render({preventDefault(){}});console.log(JSON.stringify({message:t.element("notation-status").textContent,rendererCalls}));''')
    assert expected.lower() in result["message"].lower()
    assert result["rendererCalls"] == 0


def test_notation_controller_caps_chunked_xml_without_content_length():
    result = notation_node('''
let refused=false;
try{await t.boundedXml(new Response(new Uint8Array(8*1024*1024+1)));}catch(error){refused=error.message.includes("too large");}
console.log(JSON.stringify({refused}));
''')
    assert result["refused"]


def test_local_notation_script_syntax():
    subprocess.run(["node", "--check", str(ROOT / "app/static/score-notation.js")], check=True, capture_output=True)


def test_notation_render_restores_keyboard_focus_after_busy_state():
    result = notation_node('''
document.activeElement=t.element("notation-render");
await t.render({preventDefault(){}});
console.log(JSON.stringify({focused:t.element("notation-render").focused,disabled:t.element("notation-render").disabled}));
''')
    assert result == {"focused": True, "disabled": False}
