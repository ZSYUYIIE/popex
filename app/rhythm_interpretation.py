"""Conservative rhythm hypotheses; raw times stay authoritative."""
from __future__ import annotations
import json,math,re
from dataclasses import dataclass
from typing import Any
V=RHYTHM_INTERPRETATION_VERSION="conservative-grid-v1";ID=re.compile(r"[\w.-]{1,128}");SL=re.compile(r"[a-z0-9][a-z0-9_-]{0,63}");BAD=re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|users|tmp|var|etc|mnt|private|opt|usr)/|\w+://)",re.I);D=((.25,"sixteenth",4),(1/3,"eighth_triplet","3T"),(.5,"eighth",2),(2/3,"quarter_triplet","3T"),(.75,"dotted_eighth",4),(1,"quarter",1),(1.5,"dotted_quarter",2),(2,"half",1),(3,"dotted_half",1),(4,"whole",1))
class RhythmInterpretationError(RuntimeError):pass
@dataclass(frozen=True)
class RhythmInterpretationResult:
 version:str;meter_candidates:tuple[dict[str,Any],...];measures:tuple[dict[str,Any],...];event_interpretations:tuple[dict[str,Any],...];rest_candidates:tuple[dict[str,Any],...];warnings:tuple[str,...];diagnostics:dict[str,Any]
def interpret_rhythm(pitched_events,percussion_events,alignment_candidates,timing,*,version=V):
 if not isinstance(version,str) or not re.fullmatch(r"[\w.+-]{1,128}",version):E("version")
 p,r,ix=_ev(pitched_events,percussion_events);a=_al(alignment_candidates,ix);t=_tm(timing);_ck(a,t);mc=_mc(t);ms=_ms(t,mc);nx=_nx(p);bd={x["endSeconds"] for x in ms[:-1]};out=[];rp=amb=0
 for n,e in enumerate(sorted(p+r,key=lambda x:(x["on"],x["typ"]!="pitched",x["id"])),1):
  q=a.get(e["id"]);pl,ok=_pl(e,q);rp+=ok;dh=[];dok=e["typ"]=="percussion"
  if e["typ"]=="pitched":dh,dok=_du(e,nx.get((e["src"],e["id"])),t,q);amb+=len(dh)>1
  cont=[] if e["typ"]!="pitched" else [{"kind":"tie_or_continuation","boundaryType":"measure" if x in bd else "beat","boundaryTimeSeconds":x,"confidence":U(min(dh[0]["confidence"] if dh else 0,q["confidence"] if q else .25)),"resolved":False,"warnings":["Continuation is provisional."]} for x in sorted(set(t["b"])|bd) if e["st"]<x<e["en"]][:4]
  raw={"startSeconds":e["st"],"endSeconds":e["en"],"durationSeconds":Q(e["en"]-e["st"])} if e["typ"]=="pitched" else {"timeSeconds":e["on"]};cs=[x["confidence"] for x in pl if x["kind"]!="unresolved"]+([dh[0]["confidence"]] if dh else [])
  out.append({"id":f"rh{n:06d}","eventId":e["id"],"sourceEventIds":[e["id"]],"eventType":e["typ"],"sourceKind":e["src"],"rawTiming":raw,"placementHypotheses":pl,"durationHypotheses":dh,"continuationHypotheses":cont,"unresolved":not ok or not dok,"confidence":U(min(cs) if cs else 0),"warnings":[]})
 rests=_rs(p,t);un=len(out)-rp;w=[]
 if not t["b"]:w.append("Beat evidence is unavailable; raw time is preserved.")
 if not ms:w.append("Measures remain unresolved.")
 if un:w.append(f"{un} event placement(s) remain unresolved.")
 if amb:w.append(f"{amb} pitched event(s) retain duration alternatives.")
 z={"pitchedEventCount":len(p),"percussionEventCount":len(r),"alignmentCandidateCount":len(a),"resolvedPlacementCount":rp,"unresolvedPlacementCount":un,"ambiguousDurationCount":amb,"meterCandidateCount":len(mc),"measureCount":len(ms),"restCandidateCount":len(rests),"placementConfidenceThreshold":.55,"durationConfidenceThreshold":.6,"rawTimingAuthoritative":True,"timingMode":"measured" if ms else "beat_relative" if t["b"] else "absolute_time"};o=RhythmInterpretationResult(version,tuple(mc),tuple(ms),tuple(out),tuple(rests),tuple(w),z);json.dumps(o.__dict__,allow_nan=False);return o
def _ev(pv,rv):
 if not isinstance(pv,(list,tuple)) or not isinstance(rv,(list,tuple)) or len(pv)+len(rv)>100000:E("events")
 ix={};p=[];r=[]
 for m in pv:
  i=_id(m,ix);s=_sl(m);x=N(m.get("startSeconds"),0);y=N(m.get("endSeconds"),0)
  if y<=x:E("range")
  e={"id":i,"typ":"pitched","src":s,"st":x,"en":y,"on":x,"c":C(m.get("confidence"),.5)};p.append(e);ix[i]=e
 for m in rv:
  i=_id(m,ix);s=_sl(m);x=N(m.get("timeSeconds"),0);e={"id":i,"typ":"percussion","src":s,"on":x,"c":C(m.get("confidence",m.get("strength")),.5)};r.append(e);ix[i]=e
 return p,r,ix
def _id(m,ix):
 if not isinstance(m,dict):E("mapping")
 i=m.get("id")
 if not isinstance(i,str) or not ID.fullmatch(i) or "/" in i or "\\" in i or i.startswith("."):E("id")
 if i in ix:E("unique")
 return i
def _sl(m):
 s=m.get("sourceKind")
 if not isinstance(s,str) or not SL.fullmatch(s):E("slug")
 return s
def _al(v,ix):
 if not isinstance(v,(list,tuple)) or len(v)>200000:E("alignment")
 o={}
 for m in v:
  if not isinstance(m,dict):E("mapping")
  i=m.get("eventId")
  if i not in ix:E("existing")
  if i in o:E("at most one")
  e=ix[i]
  if m.get("eventType")!=e["typ"]:E("type")
  rt=N(m.get("rawTimeSeconds"),0)
  if rt!=e["on"]:E("preserve")
  q={"confidence":C(m.get("confidence"),0)};W(m.get("warnings",[]));h="alignedTimeSeconds" in m;g={"beatIndex","subdivision","subdivisionIndex"}&m.keys()
  if h:
   if "offsetSeconds" not in m or len(g)!=3:E("partial")
   at=N(m["alignedTimeSeconds"],0);off=N(m["offsetSeconds"])
   if not math.isclose(rt-at,off,abs_tol=1e-9):E("offset")
   q.update(alignedTimeSeconds=at,offsetSeconds=off,beatIndex=I(m["beatIndex"],0),subdivision=m["subdivision"],subdivisionIndex=I(m["subdivisionIndex"],0))
  elif g or "offsetSeconds" in m:E("partial")
  if ("measureIndex" in m)!=("beatInMeasure" in m):E("measure")
  if "measureIndex" in m:q.update(measureIndex=I(m["measureIndex"],0),beatInMeasure=I(m["beatInMeasure"],1))
  o[i]=q
 return o
def _tm(m):
 if not isinstance(m,dict):E("timing")
 b=T(m.get("beatsSeconds",[]));d=T(m.get("downbeatsSeconds",[]));mt=m.get("meter")
 if mt is not None:mt=I(mt,2,12)
 st=m.get("tempoStable")
 if st is not None and type(st) is not bool:E("tempo")
 return {"b":b,"d":d,"mt":mt,"mc":OC(m.get("meterConfidence")),"bc":OC(m.get("beatConfidence")),"stable":st}
def _ck(a,t):
 for q in a.values():
  if "alignedTimeSeconds" not in q:continue
  i=q["beatIndex"]
  if i>=len(t["b"]):E("beatIndex")
  sd=q["subdivision"];si=q["subdivisionIndex"]
  if isinstance(sd,int):
   if sd<1 or si>=sd:E("subdivision")
   ex=t["b"][i] if i==len(t["b"])-1 else t["b"][i]+(t["b"][i+1]-t["b"][i])*si/sd
   if not math.isclose(q["alignedTimeSeconds"],ex,abs_tol=1e-7):E("grid placement")
  if "measureIndex" in q:
   if not t["d"]:E("downbeat")
   mt=t["mt"]
   if not mt:E("measure")
def _mc(t):
 if t["mt"] is None:return []
 c=t["mc"] if t["mc"] is not None else .25;return [{"meter":t["mt"],"confidence":c,"evidence":["timing_meter"],"resolved":c>=.55,"warnings":[] if c>=.55 else ["Meter evidence is weak."]}]
def _ms(t,m):
 if not m or not m[0]["resolved"] or len(t["d"])<2:return []
 k=m[0]["meter"];look={round(x,9):i for i,x in enumerate(t["b"])};o=[]
 for a,b in zip(t["d"],t["d"][1:]):
  i=look.get(round(a,9));j=look.get(round(b,9))
  if i is not None and j is not None and j-i==k:o.append({"id":f"m{len(o)+1:06d}","index":len(o),"startSeconds":a,"endSeconds":b,"meter":k,"beatIndices":list(range(i,j)),"confidence":m[0]["confidence"],"evidence":["observed_downbeats","observed_beats"],"warnings":[]})
 return o
def _pl(e,a):
 rt=e["on"]
 if not a or "alignedTimeSeconds" not in a:return [{"kind":"unresolved","rawTimeSeconds":rt,"confidence":a["confidence"] if a else 0}],False
 g={"kind":"grid","status":"resolved" if a["confidence"]>=.55 else "uncertain","rawTimeSeconds":rt,"alignedTimeSeconds":a["alignedTimeSeconds"],"offsetSeconds":a["offsetSeconds"],"beatIndex":a["beatIndex"],"subdivision":a["subdivision"],"subdivisionIndex":a["subdivisionIndex"],"confidence":a["confidence"]}
 return ([g] if a["confidence"]>=.55 else [g,{"kind":"unresolved","rawTimeSeconds":rt,"confidence":U(1-a["confidence"])}]),a["confidence"]>=.55
def _du(e,nx,t,a):
 raw=e["en"]-e["st"]
 if len(t["b"])<2:return [{"kind":"absolute_duration","durationSeconds":Q(raw),"confidence":min(e["c"],.35),"unresolved":True}],False
 beat=min((abs((t["b"][i]+t["b"][i+1])/2-e["st"]),t["b"][i+1]-t["b"][i]) for i in range(len(t["b"])-1))[1];r=(min(raw,nx-e["st"]) if nx and nx>e["st"] else raw)/beat;o=[]
 for n,l,s in D:
  c=.65*max(0,1-abs(n-r)/max(.25,n))+.2*e["c"]+.15*(t["bc"] or .35)
  if t["stable"] is False:c=min(c,.58)
  o.append({"kind":"grid_duration","label":l,"durationBeats":n,"durationSeconds":n*beat,"subdivision":s,"confidence":U(c),"rawDurationSeconds":Q(raw),"warnings":[]})
 o.sort(key=lambda x:(-x["confidence"],abs(x["durationBeats"]-r)));keep=[x for x in o if x["confidence"]>=o[0]["confidence"]-.2][:3]
 if len(keep)==1 and o[0]["confidence"]<.8:keep.append(o[1])
 ok=o[0]["confidence"]>=.6 and not(len(keep)>1 and o[0]["confidence"]-keep[1]["confidence"]<.12);return keep,ok
def _nx(p):
 g={};o={}
 for e in p:g.setdefault(e["src"],[]).append(e)
 for s,z in g.items():
  z.sort(key=lambda e:e["st"])
  for i,e in enumerate(z):o[s,e["id"]]=z[i+1]["st"] if i+1<len(z) else None
 return o
def _rs(p,t):
 g={};o=[]
 for e in p:g.setdefault(e["src"],[]).append(e)
 for s,z in g.items():
  z.sort(key=lambda e:e["st"])
  for a,b in zip(z,z[1:]):
   x,y=a["en"],b["st"];gap=y-x
   if gap<=0:continue
   beat=(t["b"][1]-t["b"][0]) if len(t["b"])>1 else None
   if beat and gap/beat<.55 or not beat and gap<.5:continue
   c=.65*min(a["c"],b["c"]);hs=[{"kind":"absolute_duration","durationSeconds":Q(gap),"confidence":min(c,.35),"unresolved":True}] if not beat else [{"kind":"grid_duration","label":l,"durationBeats":n,"durationSeconds":n*beat,"subdivision":sd,"confidence":U(c*max(0,1-abs(n-gap/beat)/max(.25,n)))} for n,l,sd in sorted(D,key=lambda q:abs(q[0]-gap/beat))[:2]];it={"id":f"rest{len(o)+1:06d}","sourceKind":s,"afterEventId":a["id"],"beforeEventId":b["id"],"sourceEventIds":[a["id"],b["id"]],"rawGap":{"startSeconds":x,"endSeconds":y,"durationSeconds":Q(gap)},"durationHypotheses":hs,"confidence":U(c),"resolved":c>=.6,"warnings":[]}
   if s in {"full_mix","other"}:it.update(confidence=min(it["confidence"],.45),resolved=False,warnings=["Source coverage is uncertain."])
   o.append(it)
 return o
def N(v,lo=None):
 if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)) or lo is not None and v<lo:E("number")
 return float(v)
def I(v,lo=0,hi=None):
 if type(v) is not int or v<lo or hi is not None and v>hi:E("integer")
 return v
def C(v,d):return d if v is None else U(N(v,0))
def OC(v):return None if v is None else C(v,0)
def T(v):
 if not isinstance(v,(list,tuple)) or len(v)>200000:E("timing")
 x=tuple(N(q,0) for q in v)
 if any(b<=a for a,b in zip(x,x[1:])):E("timing")
 return x
def W(v):
 if not isinstance(v,(list,tuple)) or len(v)>6:E("warnings")
 for x in v:
  if not isinstance(x,str) or not x or len(x)>200 or BAD.search(x):E("paths")
def U(x):return max(0.,min(1.,float(x)))
def Q(x):return round(float(x),12)
def E(x):raise RhythmInterpretationError(x)
