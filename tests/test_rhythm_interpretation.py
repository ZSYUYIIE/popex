from __future__ import annotations
import copy,json,math
import pytest
from app.rhythm_interpretation import *

def timing(**x):
 d={"tempoBpm":120.,"tempoConfidence":.9,"tempoStable":True,"beatsSeconds":[0,.5,1,1.5,2,2.5,3,3.5,4],"beatConfidence":.9,"downbeatsSeconds":[0,2,4],"meter":4,"meterConfidence":.85};d.update(x);return d
def p(i,a,b,s="vocals",c=.9):return {"id":i,"sourceKind":s,"startSeconds":a,"endSeconds":b,"midiNote":69,"midiPitch":69.,"frequencyHz":440.,"noteName":"A4","confidence":c,"warnings":[]}
def r(i,t):return {"id":i,"sourceKind":"drums","timeSeconds":t,"strength":.9,"hits":[{"kind":"kick","confidence":.8}]}
def al(i,k,t,g=None,c=.9,**x):
 d={"eventId":i,"eventType":k,"rawTimeSeconds":t,"confidence":c,"warnings":[]}
 if g is not None:d.update(alignedTimeSeconds=g,offsetSeconds=t-g,beatIndex=round(g/.5),subdivision=1,subdivisionIndex=0)
 d.update(x);return d
def item(z,i):return next(x for x in z.event_interpretations if x["eventId"]==i)

def test_quarter_duration_and_raw_time():
 es=[p("p1",0,.5),p("p2",.5,1),p("p3",1,1.5)];z=interpret_rhythm(es,[],[al(e["id"],"pitched",e["startSeconds"],e["startSeconds"]) for e in es],timing())
 assert z.version==RHYTHM_INTERPRETATION_VERSION
 assert all(x["durationHypotheses"][0]["label"]=="quarter" and x["durationHypotheses"][0]["confidence"]>=.75 for x in z.event_interpretations)
 assert item(z,"p1")["rawTiming"]["durationSeconds"]==.5
def test_uncertain_grid_retains_unresolved_alternative():
 z=interpret_rhythm([p("p1",.137,.48)],[],[al("p1","pitched",.137,.125,.42,subdivision=4,subdivisionIndex=1)],timing());x=item(z,"p1")
 assert x["rawTiming"]["startSeconds"]==.137 and x["placementHypotheses"][0]["status"]=="uncertain" and x["placementHypotheses"][1]["kind"]=="unresolved" and x["unresolved"]
def test_aligned_and_unaligned_remain_distinct():
 z=interpret_rhythm([p("p1",0,.5),p("p2",.73,1.1)],[],[al("p1","pitched",0,0),al("p2","pitched",.73,None,0)],timing())
 assert item(z,"p1")["placementHypotheses"][0]["kind"]=="grid" and item(z,"p2")["placementHypotheses"][0]["kind"]=="unresolved"
def test_measure_crossing_continuation():
 z=interpret_rhythm([p("p1",1.75,2.25)],[],[al("p1","pitched",1.75,1.75,beatIndex=3,subdivision=2,subdivisionIndex=1)],timing())
 assert any(x["boundaryType"]=="measure" and x["boundaryTimeSeconds"]==2 for x in item(z,"p1")["continuationHypotheses"])
def test_gap_creates_rest_and_full_mix_is_provisional():
 z=interpret_rhythm([p("p1",0,.5),p("p2",1.5,2)],[],[],timing());q=z.rest_candidates[0]
 assert q["rawGap"]["durationSeconds"]==1 and 1<=len(q["durationHypotheses"])<=2
 z=interpret_rhythm([p("p1",0,.5,"full_mix"),p("p2",1.5,2,"full_mix")],[],[],timing());assert not z.rest_candidates[0]["resolved"] and z.rest_candidates[0]["confidence"]<=.45
def test_weak_meter_and_no_beats_fallbacks():
 z=interpret_rhythm([p("p1",.5,1)],[],[al("p1","pitched",.5,.5)],timing(meterConfidence=.2));assert not z.measures and z.diagnostics["timingMode"]=="beat_relative"
 z=interpret_rhythm([p("p1",.2,.7)],[],[],timing(beatsSeconds=[],downbeatsSeconds=[],meter=None,meterConfidence=None));assert item(z,"p1")["durationHypotheses"][0]["kind"]=="absolute_duration" and z.diagnostics["timingMode"]=="absolute_time"
def test_simultaneous_percussion_not_collapsed():
 z=interpret_rhythm([], [r("r1",.5),r("r2",.5)],[al("r1","percussion",.5,.5),al("r2","percussion",.5,.5)],timing())
 assert [x["eventId"] for x in z.event_interpretations]==["r1","r2"] and all(x["rawTiming"]["timeSeconds"]==.5 for x in z.event_interpretations)
def test_irregular_timing_stays_ambiguous():
 t=timing(beatsSeconds=[0,.41,1.02,1.43,2.08],downbeatsSeconds=[],meter=None,meterConfidence=None,tempoStable=False,beatConfidence=.55);z=interpret_rhythm([p("p1",.41,.92)],[],[al("p1","pitched",.41,.41,.6)],t);x=item(z,"p1")
 assert len(x["durationHypotheses"])>=2 and x["durationHypotheses"][0]["confidence"]<=.58 and x["unresolved"]
def test_determinism_nonmutation_and_safe_output():
 args=([p("p1",0,.5)],[r("r1",.5)],[al("p1","pitched",0,0),al("r1","percussion",.5,.5)],timing());before=copy.deepcopy(args);a=interpret_rhythm(*args);b=interpret_rhythm(*copy.deepcopy(args));assert args==before and a==b;json.dumps(a.__dict__,allow_nan=False)
 e=p("p1",0,.5);e["privatePath"]="/home/user/x.wav";assert "privatePath" not in json.dumps(interpret_rhythm([e],[],[],timing()).__dict__)
def test_shuffled_input_stable_ids():
 es=[p("p2",.5,1),p("p1",0,.5)];aa=[al("p2","pitched",.5,.5),al("p1","pitched",0,0)];x=interpret_rhythm(es,[],aa,timing());y=interpret_rhythm(es[::-1],[],aa[::-1],timing());assert x==y and [q["id"] for q in x.event_interpretations]==["rh000001","rh000002"]

@pytest.mark.parametrize("bad",["","bad id","../p","p/1","p\\1","x"*129])
def test_bad_ids(bad):
 with pytest.raises(RhythmInterpretationError):interpret_rhythm([p(bad,0,.5)],[],[],timing())
@pytest.mark.parametrize("bad",[math.nan,math.inf,-math.inf,True,".5",-.1])
def test_bad_times(bad):
 with pytest.raises(RhythmInterpretationError):interpret_rhythm([p("p",bad,.5)],[],[],timing())
@pytest.mark.parametrize("bad",["Vocals","bad source","../x","x/y"])
def test_bad_slugs(bad):
 with pytest.raises(RhythmInterpretationError):interpret_rhythm([p("p",0,.5,bad)],[],[],timing())
@pytest.mark.parametrize("beats",[[0,.5,.5],[0,1,.5],[0,math.nan],[0,True],"bad"])
def test_bad_timing(beats):
 with pytest.raises(RhythmInterpretationError):interpret_rhythm([],[],[],timing(beatsSeconds=beats))
def test_duplicate_and_invalid_references():
 with pytest.raises(RhythmInterpretationError,match="unique"):interpret_rhythm([p("x",0,.5)],[r("x",.5)],[],timing())
 with pytest.raises(RhythmInterpretationError,match="existing"):interpret_rhythm([],[],[al("x","pitched",0,0)],timing())
 with pytest.raises(RhythmInterpretationError,match="at most one"):
  q=al("p","pitched",0,0);interpret_rhythm([p("p",0,.5)],[],[q,q],timing())
def test_alignment_integrity_and_path_safety():
 with pytest.raises(RhythmInterpretationError,match="preserve"):interpret_rhythm([p("p",0,.5)],[],[al("p","pitched",.1,0)],timing())
 q=al("p","pitched",0,0);q["beatIndex"]=999
 with pytest.raises(RhythmInterpretationError,match="beatIndex"):interpret_rhythm([p("p",0,.5)],[],[q],timing())
 q=al("p","pitched",.13,.13,subdivision=4,subdivisionIndex=1)
 with pytest.raises(RhythmInterpretationError,match="grid placement"):interpret_rhythm([p("p",.13,.5)],[],[q],timing())
 q=al("p","pitched",0,0);q["warnings"]=["at /home/user/x.wav"]
 with pytest.raises(RhythmInterpretationError,match="paths"):interpret_rhythm([p("p",0,.5)],[],[q],timing())
def test_measure_needs_downbeats_and_bounds():
 q=al("p","pitched",.5,.5,measureIndex=0,beatInMeasure=2)
 with pytest.raises(RhythmInterpretationError,match="downbeat"):interpret_rhythm([p("p",.5,1)],[],[q],timing(downbeatsSeconds=[]))
 es=[p(f"p{i}",i*.51,i*.51+.31) for i in range(20)];z=interpret_rhythm(es,[],[],timing(tempoStable=False));assert len(z.warnings)<=32 and all(len(x["durationHypotheses"])<=3 for x in z.event_interpretations)

def test_pitched_overlap_and_drum_assignment_shapes_remain_composable():
 from app.pitched_part_inference import infer_pitched_parts
 from app.percussion_interpretation import interpret_percussion
 pitched=[p("overlap-a",0,.75),p("overlap-b",.25,1)]
 percussion=[{"id":"drum-event","sourceKind":"drums","timeSeconds":.5,"strength":.9,"hits":[{"kind":"kick","confidence":.9},{"kind":"closed_hihat","confidence":.8}],"warnings":["Two broad hit families remain simultaneous."],"rawFeatureSummary":{}}]
 aligns=[al("overlap-a","pitched",0,0),al("overlap-b","pitched",.25,.25,subdivision=2,subdivisionIndex=1),al("drum-event","percussion",.5,.5)]
 rhythm=interpret_rhythm(pitched,percussion,aligns,timing())
 parts=infer_pitched_parts(pitched,aligns[:2])
 drums=interpret_percussion(percussion,aligns[2:])
 assert {x["eventId"] for x in rhythm.event_interpretations}=={"overlap-a","overlap-b","drum-event"}
 assert len(parts.assignments)==2 and len({x["eventId"] for x in parts.assignments})==2
 assert len(drums.assignments)==2 and {x["eventId"] for x in drums.assignments}=={"drum-event"}
 placement=item(rhythm,"drum-event")["placementHypotheses"]
 assert all(a["rawTimeSeconds"]==.5 for a in drums.assignments)
 assert placement[0]["alignedTimeSeconds"]==.5

def test_result_is_directly_retainable_without_discarding_uncertainty():
 z=interpret_rhythm([p("p1",.137,.48)],[],[al("p1","pitched",.137,.125,.42,subdivision=4,subdivisionIndex=1,warnings=["Timing evidence remains uncertain."])],timing())
 x=item(z,"p1")
 retained={"sourceEventIds":x["sourceEventIds"],"rawTiming":x["rawTiming"],"placementHypotheses":x["placementHypotheses"],"durationHypotheses":x["durationHypotheses"],"continuationHypotheses":x["continuationHypotheses"],"warnings":x["warnings"],"unresolved":x["unresolved"]}
 assert retained["sourceEventIds"]==["p1"] and retained["rawTiming"]["startSeconds"]==.137
 assert len(retained["placementHypotheses"])==2 and retained["unresolved"] is True
 json.dumps(retained,allow_nan=False)
