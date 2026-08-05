#!/usr/bin/env python3
"""Validate PopEx's real Linux CPU stem path and print path-free evidence."""
from __future__ import annotations

import argparse, hashlib, io, json, math, os, platform, resource, shutil, stat, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import numpy as np
import soundfile as sf

EXPECTED_PROFILE="linux-x86_64-cpu-cpython313"
EXPECTED_PROTOCOL_VERSION=1
EXPECTED_WORKER_VERSION="1.0.0"
EXPECTED_DEMUCS_VERSION="4.1.0"
EXPECTED_TORCH_VERSION="2.13.0+cpu"
EXPECTED_MODEL_REPOSITORY="adefossez/HTDemucs"
EXPECTED_MODEL_REVISION="bf35a81b663819a8255c8fefee17f9d812b786b5"
EXPECTED_CHECKPOINT_FILE="955717e8.safetensors"
EXPECTED_CHECKPOINT_SIZE_BYTES=84_025_440
EXPECTED_CHECKPOINT_SHA256="d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
EXPECTED_STEM_KINDS=("vocals","bass","drums","other")
SAMPLE_RATE=44_100
SYNTHETIC_DURATION_SECONDS=4.0
SYNTHETIC_JOB_ID=hashlib.sha256(b"popex-linux-real-model-e2e-v1").hexdigest()[:32]
_SAFE_ERROR_KEYS=frozenset({"status","code","phase","runtimeCode","workerCode","exitCode"})
_SAFE_SUMMARY_KEYS=frozenset({"schemaVersion","runtimeProfile","workerVersion","demucsVersion","torchVersion","modelRevision","checkpointSizeBytes","checkpointSha256","jobStatus","stemKinds","stemMetrics","elapsedPreparationSeconds","elapsedInferenceSeconds","peakProcessRssMiB"})

class E2EValidationError(RuntimeError):
    def __init__(self,code:str,*,phase:str,evidence:Mapping[str,Any]|None=None):
        self.code=_code(code); self.phase=_code(phase).lower(); self.evidence=_evidence(evidence or {}); super().__init__(self.code)
    def safe_payload(self): return {"status":"error","code":self.code,"phase":self.phase,**self.evidence}

class TimedRuntimeClient:
    def __init__(self,delegate):
        self.delegate=delegate; self.prepare_calls=0; self.inference_calls=0; self.preparation_seconds=0.0; self.inference_seconds=0.0; self.last_model_result=None; self.last_error_evidence={}
    def _call(self,fn,**kw):
        try:return fn(**kw)
        except BaseException as exc:self.last_error_evidence=_runtime_evidence(exc);raise
    def runtime_probe(self):return self._call(self.delegate.runtime_probe)
    def model_probe(self):
        result=self._call(self.delegate.model_probe);self.last_model_result=result;return result
    def prepare_model(self,*,allow_model_download:bool):
        self.prepare_calls+=1;start=time.monotonic()
        try:result=self._call(self.delegate.prepare_model,allow_model_download=allow_model_download)
        finally:self.preparation_seconds+=time.monotonic()-start
        self.last_model_result=result;return result
    def __call__(self,**kw):
        self.inference_calls+=1;start=time.monotonic()
        try:return self._call(self.delegate,**kw)
        finally:self.inference_seconds+=time.monotonic()-start

def build_parser():
    p=argparse.ArgumentParser();p.add_argument("--worker",required=True);p.add_argument("--runtime-lock",required=True);p.add_argument("--cache-root",required=True);p.add_argument("--data-dir",required=True);p.add_argument("--expected-profile",required=True);p.add_argument("--timeout-seconds",type=int,default=3600);return p

def validate_trusted_inputs(args):
    if args.expected_profile!=EXPECTED_PROFILE:raise E2EValidationError("UNAPPROVED_RUNTIME_PROFILE",phase="input")
    if not _supported_platform():raise E2EValidationError("UNSUPPORTED_VALIDATION_PLATFORM",phase="input")
    if not 60<=args.timeout_seconds<=7200:raise E2EValidationError("INVALID_TIMEOUT",phase="input")
    worker=_file(args.worker,True);lock=_file(args.runtime_lock,False);cache=_root(args.cache_root);data=_root(args.data_dir);runtime=lock.parent
    if not _inside(worker,runtime):raise E2EValidationError("WORKER_OUTSIDE_RUNTIME",phase="input")
    for a,b in ((cache,data),(cache,runtime),(data,runtime)):
        if a==b or _inside(a,b) or _inside(b,a):raise E2EValidationError("TRUSTED_ROOTS_OVERLAP",phase="input")
    return SimpleNamespace(worker=worker,runtime_lock=lock,cache_root=cache,data_dir=data,runtime_root=runtime)

def generate_synthetic_wav(path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);n=int(SAMPLE_RATE*SYNTHETIC_DURATION_SECONDS);t=np.arange(n)/SAMPLE_RATE;rng=np.random.default_rng(20260805)
    bass=(.35+.65*(np.mod(t,.5)<.32))*(.24*np.sin(2*np.pi*110*t)+.08*np.sin(2*np.pi*220*t));chord=np.zeros(n)
    for i,fs in enumerate(((261.6256,329.6276,391.9954),(220.,261.6256,329.6276),(174.6141,220.,261.6256),(195.9977,246.9417,293.6648))):
        a=i*SAMPLE_RATE;b=min(n,a+SAMPLE_RATE);env=np.sin(np.linspace(0,np.pi,b-a))**.35
        for f in fs:chord[a:b]+=.075*env*np.sin(2*np.pi*f*t[a:b])
    melody=np.zeros(n);notes=(440.,523.2511,659.2551,587.3295,493.8833,392.,440.,659.2551);step=n//len(notes)
    for i,f in enumerate(notes):
        a=i*step;b=n if i==len(notes)-1 else (i+1)*step;env=np.sin(np.linspace(0,np.pi,b-a))**.8;melody[a:b]=.14*env*(np.sin(2*np.pi*f*t[a:b])+.18*np.sin(4*np.pi*f*t[a:b]))
    drums=np.zeros(n)
    for beat in np.arange(0,SYNTHETIC_DURATION_SECONDS,.5):
        a=int(beat*SAMPLE_RATE);m=min(int(.16*SAMPLE_RATE),n-a);u=np.arange(m)/SAMPLE_RATE;drums[a:a+m]+=.34*np.exp(-28*u)*np.sin(2*np.pi*(72-30*u)*u)
    for beat in np.arange(.5,SYNTHETIC_DURATION_SECONDS,1.):
        a=int(beat*SAMPLE_RATE);m=min(int(.11*SAMPLE_RATE),n-a);u=np.arange(m)/SAMPLE_RATE;drums[a:a+m]+=.12*np.exp(-38*u)*rng.standard_normal(m)
    for beat in np.arange(.25,SYNTHETIC_DURATION_SECONDS,.25):
        a=int(beat*SAMPLE_RATE);m=min(int(.035*SAMPLE_RATE),n-a);u=np.arange(m)/SAMPLE_RATE;drums[a:a+m]+=.035*np.exp(-90*u)*rng.standard_normal(m)
    audio=np.column_stack((bass+.88*chord+1.08*melody+drums,.92*bass+1.08*chord+.9*melody+np.roll(drums,29)));fade=int(.02*SAMPLE_RATE);r=np.linspace(0,1,fade);audio[:fade]*=r[:,None];audio[-fade:]*=r[::-1,None];peak=float(np.max(np.abs(audio)))
    if not math.isfinite(peak) or peak<=0:raise E2EValidationError("SYNTHETIC_AUDIO_INVALID",phase="audio")
    audio=np.asarray(audio*(.82/peak),dtype=np.float32);sf.write(path,audio,SAMPLE_RATE,subtype="PCM_16",format="WAV");info=sf.info(str(path))
    if info.samplerate!=SAMPLE_RATE or info.channels!=2 or info.frames!=n:raise E2EValidationError("SYNTHETIC_WAV_INVALID",phase="audio")
    return {"sampleRate":SAMPLE_RATE,"channels":2,"frames":n,"durationSeconds":n/SAMPLE_RATE}

def build_safe_summary(**v):
    metrics=[{"kind":str(x["kind"]),"durationSeconds":round(float(x["durationSeconds"]),6),"sizeBytes":int(x["sizeBytes"])} for x in v["stem_metrics"]]
    s={"schemaVersion":1,"runtimeProfile":v["runtime_profile"],"workerVersion":v["worker_version"],"demucsVersion":v["demucs_version"],"torchVersion":v["torch_version"],"modelRevision":v["model_revision"],"checkpointSizeBytes":int(v["checkpoint_size_bytes"]),"checkpointSha256":v["checkpoint_sha256"],"jobStatus":v["job_status"],"stemKinds":[x["kind"] for x in metrics],"stemMetrics":metrics,"elapsedPreparationSeconds":round(float(v["preparation_seconds"]),3),"elapsedInferenceSeconds":round(float(v["inference_seconds"]),3),"peakProcessRssMiB":round(float(v["peak_rss_mib"]),1) if v["peak_rss_mib"] is not None else None}
    if set(s)!=_SAFE_SUMMARY_KEYS or s["runtimeProfile"]!=EXPECTED_PROFILE or s["workerVersion"]!=EXPECTED_WORKER_VERSION or s["demucsVersion"]!=EXPECTED_DEMUCS_VERSION or s["torchVersion"]!=EXPECTED_TORCH_VERSION or s["modelRevision"]!=EXPECTED_MODEL_REVISION or s["checkpointSizeBytes"]!=EXPECTED_CHECKPOINT_SIZE_BYTES or s["checkpointSha256"]!=EXPECTED_CHECKPOINT_SHA256 or s["jobStatus"]!="completed" or tuple(s["stemKinds"])!=EXPECTED_STEM_KINDS:raise E2EValidationError("SAFE_SUMMARY_INVALID",phase="summary")
    if any(x["durationSeconds"]<=0 or x["sizeBytes"]<=0 for x in metrics) or any(c in json.dumps(s) for c in ("/","\\")):raise E2EValidationError("SAFE_SUMMARY_INVALID",phase="summary")
    return s

def run_validation(args):
    paths=validate_trusted_inputs(args);secrets=tuple(v for k in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN") if (v:=os.getenv(k)) and len(v)>=4)
    from fastapi.testclient import TestClient
    from app import db
    from app.config import Settings
    from app.main import create_app
    from app.separation import STEM_MANIFEST_RELATIVE_PATH,STEM_MANIFEST_SCHEMA_VERSION,load_stem_manifest
    from app.separation_runtime import SeparationRuntimeClient
    settings=Settings(data_dir=paths.data_dir,allowed_hosts=("example.invalid",),max_duration_seconds=60,max_filesize_mb=16,max_upload_mb=16,audio_quality="192",ffmpeg_binary="ffmpeg",ffprobe_binary="ffprobe",audio_analysis_enabled=True,stem_separation_enabled=True,stem_separation_version="demucs-worker-v3",stem_separation_worker_executable=paths.worker,stem_separation_runtime_lock=paths.runtime_lock,stem_separation_cache_dir=paths.cache_root,stem_separation_runtime_profile=EXPECTED_PROFILE,stem_separation_device="cpu",stem_separation_timeout_seconds=args.timeout_seconds)
    runtime=TimedRuntimeClient(SeparationRuntimeClient(paths.worker,paths.cache_root,runtime_lock_path=paths.runtime_lock,expected_protocol_version=1,expected_runtime_profile=EXPECTED_PROFILE,command_timeouts={"prepare-model":min(args.timeout_seconds,1800),"verify-model":min(args.timeout_seconds,600),"separate":args.timeout_seconds}))
    app=create_app(settings=settings,separation_runtime_client=runtime)
    if paths.cache_root.exists():raise E2EValidationError("CACHE_CREATED_BEFORE_APP_START",phase="startup")
    with TestClient(app) as client:
        job=SYNTHETIC_JOB_ID;db.create_job(settings.database_path,job,source_type="upload",original_filename="synthetic-e2e.wav");root=settings.exports_dir/job;root.mkdir();analysis=root/"analysis.wav";generate_synthetic_wav(analysis);source=root/"source-synthetic.wav";shutil.copyfile(analysis,source);(root/"metadata.json").write_text('{"source_type":"synthetic"}',encoding="utf-8");(root/"analysis").mkdir();(root/"analysis"/"audio-analysis.json").write_text('{"schemaVersion":1}',encoding="utf-8")
        db.update_job(settings.database_path,job,status="completed",stage="completed",progress=100,message="Synthetic source and analysis are ready.",title="Synthetic E2E mixture",duration_seconds=4.0,source_format="wav",sample_rate=SAMPLE_RATE,channel_count=2,source_file_name=source.name,normalized_file_name=analysis.name,metadata_file_name="metadata.json",preparation_status="completed",analysis_status="completed",analysis_version="synthetic-e2e-v1",analysis_json_file_name="analysis/audio-analysis.json",analyzed_at="2026-08-05T00:00:00+00:00")
        initial=client.get(f"/api/jobs/{job}").json();cap=initial.get("separation",{}).get("runtime",{})
        if cap.get("state")!="download_required" or cap.get("profile")!=EXPECTED_PROFILE or cap.get("modelSource")!=EXPECTED_MODEL_REPOSITORY or cap.get("modelRevision")!=EXPECTED_MODEL_REVISION or cap.get("checkpointSizeBytes")!=EXPECTED_CHECKPOINT_SIZE_BYTES or initial["separation"].get("canStart") is not True:raise E2EValidationError("INITIAL_CAPABILITY_MISMATCH",phase="startup",evidence=runtime.last_error_evidence)
        _no_model(paths.cache_root,False);_safe(initial,paths,secrets)
        response=client.post(f"/api/jobs/{job}/separate",json={"allowModelDownload":True});_status(response.status_code,202,"CONSENT_REQUEST_FAILED",runtime.last_error_evidence);_safe(response.json(),paths,secrets)
        final=client.get(f"/api/jobs/{job}").json();_safe(final,paths,secrets)
        if final.get("separation",{}).get("status")!="completed" or runtime.prepare_calls!=1 or runtime.inference_calls!=1:raise E2EValidationError("SEPARATION_NOT_COMPLETED",phase="separation",evidence=runtime.last_error_evidence)
        record=db.get_job(settings.database_path,job)
        if not record or record.get("separation_status")!="completed" or record.get("stem_manifest_file_name")!=STEM_MANIFEST_RELATIVE_PATH:raise E2EValidationError("SQLITE_STATE_MISMATCH",phase="persistence")
        result=load_stem_manifest(job,settings);p=result.provenance if result else None
        if not result or result.payload.get("schemaVersion")!=STEM_MANIFEST_SCHEMA_VERSION or not p or (p.runtime_profile,p.worker_version,p.demucs_version,p.torch_version,p.model_repository,p.model_revision,p.checkpoint_file,p.checkpoint_sha256,p.device)!=(EXPECTED_PROFILE,EXPECTED_WORKER_VERSION,EXPECTED_DEMUCS_VERSION,EXPECTED_TORCH_VERSION,EXPECTED_MODEL_REPOSITORY,EXPECTED_MODEL_REVISION,EXPECTED_CHECKPOINT_FILE,EXPECTED_CHECKPOINT_SHA256,"cpu"):raise E2EValidationError("MANIFEST_PROVENANCE_MISMATCH",phase="manifest")
        model=runtime.last_model_result or runtime.model_probe()
        if model.checkpoint_size_bytes!=EXPECTED_CHECKPOINT_SIZE_BYTES or model.checkpoint_sha256!=EXPECTED_CHECKPOINT_SHA256 or model.offline_ready is not True:raise E2EValidationError("MODEL_READINESS_MISMATCH",phase="model")
        _checkpoint(paths.cache_root)
        details=client.get(f"/api/jobs/{job}/stems");_status(details.status_code,200,"STEM_DETAILS_FAILED");_safe(details.json(),paths,secrets)
        metrics=[];hashes=[];manifest={x.kind:x for x in result.stems};jobroot=root.resolve()
        for kind in EXPECTED_STEM_KINDS:
            preview=client.get(f"/api/jobs/{job}/stems/{kind}/preview");download=client.get(f"/api/jobs/{job}/stems/{kind}/download");_status(preview.status_code,200,"STEM_PREVIEW_FAILED");_status(download.status_code,200,"STEM_DOWNLOAD_FAILED")
            if preview.content!=download.content or not preview.headers.get("content-type","").startswith("audio/wav") or "content-disposition" in preview.headers or f"{kind}.wav" not in download.headers.get("content-disposition",""):raise E2EValidationError("STEM_ENDPOINT_MISMATCH",phase="artifacts")
            duration,sr,ch=_wav(preview.content);artifact=manifest[kind];physical=(jobroot/artifact.file_name).resolve();info=physical.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or not _inside(physical,jobroot) or len(preview.content)!=artifact.size_bytes or sr!=artifact.sample_rate or ch!=artifact.channels:raise E2EValidationError("STEM_ARTIFACT_MISMATCH",phase="artifacts")
            hashes.append(hashlib.sha256(preview.content).hexdigest());metrics.append({"kind":kind,"durationSeconds":duration,"sizeBytes":len(preview.content)})
        if len(set(hashes))!=4:raise E2EValidationError("STEM_OUTPUTS_NOT_DISTINCT",phase="artifacts")
        for scan in (paths.runtime_root,paths.cache_root):
            if any(x.name in {"analysis.wav","source-synthetic.wav"} or job in x.parts for x in scan.rglob("*")):raise E2EValidationError("SOURCE_AUDIO_LEFT_APP_DATA",phase="privacy")
        summary=build_safe_summary(runtime_profile=p.runtime_profile,worker_version=p.worker_version,demucs_version=p.demucs_version,torch_version=p.torch_version,model_revision=p.model_revision,checkpoint_size_bytes=model.checkpoint_size_bytes,checkpoint_sha256=p.checkpoint_sha256,job_status=record["separation_status"],stem_metrics=metrics,preparation_seconds=runtime.preparation_seconds,inference_seconds=runtime.inference_seconds,peak_rss_mib=_rss());_safe(summary,paths,secrets);return summary

def main(argv:Sequence[str]|None=None):
    try:print(json.dumps(run_validation(build_parser().parse_args(argv)),sort_keys=True,separators=(",",":")));return 0
    except E2EValidationError as exc:print(json.dumps(exc.safe_payload(),sort_keys=True,separators=(",",":")),file=sys.stderr);return 2
    except BaseException:print('{"code":"E2E_INTERNAL_ERROR","phase":"internal","status":"error"}',file=sys.stderr);return 3

def _path(raw):
    if not raw or "\0" in raw or raw.startswith("~"):raise E2EValidationError("INVALID_TRUSTED_PATH",phase="input")
    p=Path(raw)
    if not p.is_absolute() or Path(os.path.normpath(raw))!=p or p.resolve(strict=False)!=p:raise E2EValidationError("INVALID_TRUSTED_PATH",phase="input")
    return p
def _file(raw,executable):
    p=_path(raw)
    try:i=p.lstat()
    except OSError:raise E2EValidationError("TRUSTED_FILE_UNAVAILABLE",phase="input") from None
    if stat.S_ISLNK(i.st_mode) or not stat.S_ISREG(i.st_mode) or (executable and not os.access(p,os.X_OK)):raise E2EValidationError("TRUSTED_FILE_UNSAFE",phase="input")
    return p
def _root(raw):
    p=_path(raw)
    if p==Path(p.anchor):raise E2EValidationError("TRUSTED_ROOT_TOO_BROAD",phase="input")
    if p.exists():
        if p.is_symlink() or not p.is_dir() or any(p.iterdir()):raise E2EValidationError("TRUSTED_ROOT_NOT_EMPTY",phase="input")
    elif not p.parent.is_dir() or p.parent.is_symlink() or p.parent.resolve()!=p.parent:raise E2EValidationError("TRUSTED_ROOT_PARENT_UNSAFE",phase="input")
    return p
def _supported_platform():return platform.system()=="Linux" and platform.machine().lower() in {"x86_64","amd64"} and platform.python_implementation()=="CPython" and sys.version_info[:2]==(3,13)
def _inside(p,r):
    try:p.relative_to(r);return True
    except ValueError:return False
def _no_model(cache,required):
    checkpoints=list(cache.rglob(EXPECTED_CHECKPOINT_FILE));ready=list(cache.rglob("htdemucs-bf35a81b-v1.json"));legacy=[x for pat in ("*.th","*.ckpt") for x in cache.rglob(pat)]
    if legacy or (required and (not checkpoints or not ready)) or (not required and (checkpoints or ready or list(cache.rglob("*.safetensors")))):raise E2EValidationError("MODEL_ASSET_STATE_INVALID",phase="model")
def _checkpoint(cache):
    _no_model(cache,True);root=cache.resolve();targets={x.resolve() for x in cache.rglob(EXPECTED_CHECKPOINT_FILE)}
    if len(targets)!=1:raise E2EValidationError("CHECKPOINT_IDENTITY_AMBIGUOUS",phase="model")
    p=targets.pop()
    if not _inside(p,root) or p.stat().st_size!=EXPECTED_CHECKPOINT_SIZE_BYTES or hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED_CHECKPOINT_SHA256:raise E2EValidationError("CHECKPOINT_VALIDATION_FAILED",phase="model")
def _wav(data):
    try:i=sf.info(io.BytesIO(data));sr=int(i.samplerate);ch=int(i.channels);frames=int(i.frames);duration=frames/sr
    except Exception:raise E2EValidationError("INVALID_STEM_WAV",phase="artifacts") from None
    if str(i.format).upper()!="WAV" or min(sr,ch,frames)<=0 or not math.isfinite(duration):raise E2EValidationError("INVALID_STEM_WAV",phase="artifacts")
    return duration,sr,ch
def _safe(payload,paths,secrets):
    text=json.dumps(payload,sort_keys=True);forbidden=(str(paths.worker),str(paths.runtime_lock),str(paths.cache_root),str(paths.data_dir),str(paths.runtime_root),*secrets)
    if any(x and x in text for x in forbidden) or "http://" in text.lower() or "https://" in text.lower() or "bearer " in text.lower():raise E2EValidationError("SENSITIVE_VALUE_EXPOSED",phase="privacy")
def _status(actual,expected,code,evidence=None):
    if actual!=expected:raise E2EValidationError(code,phase="separation",evidence=evidence)
def _code(v):return "".join(c if c.isalnum() or c=="_" else "_" for c in str(v).upper()).strip("_")[:96] or "UNKNOWN"
def _evidence(v):
    out={}
    for k in ("runtimeCode","workerCode"):
        if isinstance(v.get(k),str):out[k]=_code(v[k])
    if type(v.get("exitCode")) is int:out["exitCode"]=v["exitCode"]
    return out
def _runtime_evidence(exc):
    d=getattr(exc,"detail",None);return _evidence({"runtimeCode":getattr(exc,"code",None),"workerCode":getattr(d,"worker_code",None),"exitCode":getattr(d,"exit_code",None)})
def _rss():
    try:return max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)/1024
    except Exception:return None
if __name__=="__main__":raise SystemExit(main())
