from __future__ import annotations
import hashlib, json, os, sys, types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]; WR=ROOT/'runtimes'/'demucs_worker'; sys.path.insert(0,str(WR/'src'))
from popex_demucs_worker import cli,commands,constants,model_artifacts,paths,probes,separation_command
LOCK={'demucs':'4.1.0','torch':'test-torch','huggingface_hub':'test-hub','safetensors':'test-safe','PyYAML':'test-yaml'}; PROFILE='test-linux-cpu-v1'; OPT={'demucs','demucs.api','demucs.apply','demucs.hf','torch','safetensors','huggingface_hub','yaml'}
RUN_ID='0123456789abcdef0123456789abcdef'; OTHER_RUN_ID='fedcba9876543210fedcba9876543210'

@pytest.fixture(autouse=True)
def lock(monkeypatch,tmp_path):
 f=tmp_path/'runtime-lock.json';f.write_text(json.dumps({'schemaVersion':1,'runtimeProfile':PROFILE,'workerVersion':constants.WORKER_VERSION,'packages':LOCK}));monkeypatch.setenv(constants.RUNTIME_LOCK_ENV,str(f))
 def version(n):
  if n=='popex-demucs-worker': return constants.WORKER_VERSION
  if n in LOCK:return LOCK[n]
  raise probes.metadata.PackageNotFoundError(n)
 monkeypatch.setattr(probes.metadata,'version',version)

@pytest.fixture
def model(monkeypatch):
 data=b'approved'; dig=hashlib.sha256(data).hexdigest()
 for m in(probes,model_artifacts,separation_command):monkeypatch.setattr(m,'CHECKPOINT_SIZE_BYTES',len(data));monkeypatch.setattr(m,'CHECKPOINT_SHA256',dig)
 return data,dig

def run(capsys,*args):
 code=cli.main(list(args)); out=capsys.readouterr(); lines=out.out.splitlines(); assert len(lines)==1
 return code,json.loads(lines[0]),out

def yamlmod(monkeypatch):
 m=types.ModuleType('yaml');m.safe_load=json.loads;monkeypatch.setitem(sys.modules,'yaml',m)

def assets(root,data):
 p=root/'hub'/'models--adefossez--HTDemucs'/'snapshots'/constants.MODEL_REVISION;p.mkdir(parents=True,exist_ok=True)
 b=p/constants.BAG_FILE;b.write_text(json.dumps({'models':[constants.BAG_SIGNATURE]}),encoding='utf-8')
 c=p/constants.CHECKPOINT_FILE;c.write_bytes(data);return b,c

def manifest(root,data,**over):
 b,c=assets(root,data);profile,v,_=probes.require_compatible_runtime();d=hashlib.sha256(data).hexdigest()
 p={'schemaVersion':1,'protocolVersion':1,'runtimeProfile':profile,'workerVersion':constants.WORKER_VERSION,'demucsVersion':'4.1.0','torchVersion':v['torch'],'huggingfaceHubVersion':v['huggingface_hub'],'packageVersions':v,'modelRepository':constants.MODEL_REPOSITORY,'modelRevision':constants.MODEL_REVISION,'bagFile':constants.BAG_FILE,'bagModelSignatures':[constants.BAG_SIGNATURE],'checkpointFile':constants.CHECKPOINT_FILE,'checkpointSizeBytes':len(data),'checkpointSha256':d,'verifiedAt':'2026-08-03T00:00:00Z','cacheAssets':{'bag':b.relative_to(root).as_posix(),'checkpoint':c.relative_to(root).as_posix()},'offlineReady':True,'warnings':[]};p.update(over)
 f=root/'readiness'/'htdemucs-bf35a81b-v1.json';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(p),encoding='utf-8');return p

def hub(monkeypatch,fn):
 m=types.ModuleType('huggingface_hub');m.hf_hub_download=fn;monkeypatch.setitem(sys.modules,'huggingface_hub',m)

def downloader(data,calls,bag=None):
 def f(**kw):
  calls.append(dict(kw));p=Path(kw['cache_dir'])/'snapshots'/kw['revision'];p.mkdir(parents=True,exist_ok=True);t=p/kw['filename']
  if kw['filename']==constants.BAG_FILE:t.write_text(json.dumps(bag or {'models':[constants.BAG_SIGNATURE]}),encoding='utf-8')
  else:t.write_bytes(data)
  return str(t)
 return f

def ready(monkeypatch,root,data,calls=None):
 calls=[] if calls is None else calls;yamlmod(monkeypatch);hub(monkeypatch,downloader(data,calls));return commands.prepare_model(str(root)),calls

class HTDemucs:
 __module__='demucs.htdemucs';audio_channels=2;samplerate=44100;sources=['drums','bass','other','vocals']
class Bag:
 def __init__(self,models,**kw):self.audio_channels=models[0].audio_channels;self.samplerate=models[0].samplerate;self.sources=list(models[0].sources)
class Sep:
 def __init__(self,*,model,device,progress):self._load_model()
 def separate_audio_file(self,p):return b'x',{'drums':b'd','bass':b'b','other':b'o','vocals':b'v'}
def demucs(monkeypatch,mdl=None,sep=Sep):
 root=types.ModuleType('demucs');root.__path__=[];hf=types.ModuleType('demucs.hf');ap=types.ModuleType('demucs.apply');api=types.ModuleType('demucs.api')
 hf.load_safetensors_model=lambda p:mdl or HTDemucs();ap.BagOfModels=Bag;api.Separator=sep
 api.save_audio=lambda audio,path,**kw:Path(path).write_bytes(audio)
 for n,m in {'demucs':root,'demucs.hf':hf,'demucs.apply':ap,'demucs.api':api}.items():monkeypatch.setitem(sys.modules,n,m)

def sep_args(cache,work,inp='analysis.wav',out=f'stems/runs/{RUN_ID}/worker-output',device='cpu'):
 return ('--protocol-version','1','separate','--cache-root',str(cache),'--workspace-root',str(work),'--input-relative',inp,'--output-relative',out,'--device',device)

def test_package_metadata_and_entry_point():
 import tomllib
 p=tomllib.loads((WR/'pyproject.toml').read_text());assert p['project']['name']=='popex-demucs-worker';assert p['project']['dependencies']==[];assert p['project']['scripts']['popex-demucs-worker']=='popex_demucs_worker.cli:main'

def test_runtime_probe_lazy_exact_envelope_and_single_stdout(capsys,monkeypatch):
 for n in OPT:monkeypatch.delitem(sys.modules,n,raising=False)
 code,e,o=run(capsys,'--protocol-version','1','runtime-probe');assert code==0 and o.out.count('\n')==1
 assert e=={'protocolVersion':1,'command':'runtime-probe','status':'ok','result':e['result'],'warnings':[]};assert e['result']['compatible'];assert not OPT&sys.modules.keys()

def test_model_probe_lazy_and_readiness(capsys,monkeypatch,tmp_path,model):
 manifest(tmp_path,model[0])
 for n in OPT:monkeypatch.delitem(sys.modules,n,raising=False)
 code,e,_=run(capsys,'--protocol-version','1','model-probe','--cache-root',str(tmp_path));assert code==0 and e['result']['offlineReady'];assert not OPT&sys.modules.keys()
 assert e['result']=={
  'runtimeProfile':PROFILE,
  'workerVersion':constants.WORKER_VERSION,
  'demucsVersion':LOCK['demucs'],
  'torchVersion':LOCK['torch'],
  'huggingfaceHubVersion':LOCK['huggingface_hub'],
  'modelRepository':constants.MODEL_REPOSITORY,
  'modelRevision':constants.MODEL_REVISION,
  'checkpointFile':constants.CHECKPOINT_FILE,
  'checkpointSizeBytes':len(model[0]),
  'checkpointSha256':model[1],
  'verifiedAt':'2026-08-03T00:00:00Z',
  'offlineReady':True,
  'readinessManifest':constants.READINESS_RELATIVE_PATH,
 }

@pytest.mark.parametrize('args,exitcode,err',[(('--protocol-version','2','runtime-probe'),30,'UNSUPPORTED_PROTOCOL'),(('--protocol-version','1','model-probe','--cache-root','relative'),30,'INVALID_PATH')])
def test_invalid_protocol_and_root(capsys,args,exitcode,err):
 code,e,_=run(capsys,*args);assert(code,e['error']['code'])==(exitcode,err)

def test_runtime_mismatch_and_missing_manifest_exit_codes(capsys,monkeypatch,tmp_path):
 old=probes.metadata.version;monkeypatch.setattr(probes.metadata,'version',lambda n:'4.0.0' if n=='demucs' else old(n));code,e,_=run(capsys,'--protocol-version','1','runtime-probe');assert(code,e['error']['code'])==(10,'RUNTIME_INCOMPATIBLE')
 monkeypatch.setattr(probes.metadata,'version',old);code,e,_=run(capsys,'--protocol-version','1','model-probe','--cache-root',str(tmp_path));assert(code,e['error']['code'])==(20,'MODEL_DOWNLOAD_REQUIRED')

def test_prepare_exact_downloads_atomic_manifest_and_privacy(monkeypatch,tmp_path,model):
 calls=[];yamlmod(monkeypatch);hub(monkeypatch,downloader(model[0],calls));os.environ['HF_TOKEN']='hf_secret';r=commands.prepare_model(str(tmp_path))
 assert {x['filename'] for x in calls}=={constants.BAG_FILE,constants.CHECKPOINT_FILE};assert all(x['repo_id']==constants.MODEL_REPOSITORY and x['revision']==constants.MODEL_REVISION and x['token'] is False and x['local_files_only'] is False for x in calls)
 assert r['checkpointSha256']==model[1] and 'HF_TOKEN' not in os.environ;assert os.environ['HF_HUB_DISABLE_TELEMETRY']=='1';f=tmp_path/'readiness'/'htdemucs-bf35a81b-v1.json';assert f.is_file() and not list(f.parent.glob('*.tmp'))

@pytest.mark.parametrize('kind,exitcode,err',[('network',22,'MODEL_DOWNLOAD_FAILED'),('yaml',21,'BAG_SIGNATURE_MISMATCH'),('size',21,'CHECKPOINT_SIZE_MISMATCH'),('hash',21,'CHECKPOINT_HASH_MISMATCH')])
def test_prepare_failures(capsys,monkeypatch,tmp_path,model,kind,exitcode,err):
 yamlmod(monkeypatch);data=model[0]
 if kind=='network':hub(monkeypatch,lambda **kw:(_ for _ in()).throw(OSError('offline')))
 elif kind=='yaml':hub(monkeypatch,downloader(data,[],{'models':['wrong']}))
 elif kind=='size':hub(monkeypatch,downloader(data+b'x',[]))
 else:
  monkeypatch.setattr(model_artifacts,'CHECKPOINT_SHA256','0'*64);hub(monkeypatch,downloader(data,[]))
 code,e,_=run(capsys,'--protocol-version','1','prepare-model','--cache-root',str(tmp_path));assert(code,e['error']['code'])==(exitcode,err);assert not (tmp_path/'readiness'/'htdemucs-bf35a81b-v1.json').exists()

def test_verify_offline_local_only_no_http(capsys,monkeypatch,tmp_path,model):
 calls=[];ready(monkeypatch,tmp_path,model[0],calls);calls.clear();hub(monkeypatch,downloader(model[0],calls));code,e,_=run(capsys,'--protocol-version','1','verify-model','--cache-root',str(tmp_path));assert code==0 and e['result']['offlineReady'];assert os.environ['HF_HUB_OFFLINE']=='1';assert len(calls)==2 and all(x['local_files_only'] is True and x['token'] is False for x in calls)

def test_manifest_schema_containment_and_symlink_rejection(capsys,tmp_path,model):
 p=manifest(tmp_path,model[0]);outside=tmp_path.parent/'outside.bin';outside.write_bytes(model[0]);link=tmp_path/'hub'/'escape';link.symlink_to(outside);p['cacheAssets']['checkpoint']=link.relative_to(tmp_path).as_posix();(tmp_path/'readiness'/'htdemucs-bf35a81b-v1.json').write_text(json.dumps(p))
 code,e,_=run(capsys,'--protocol-version','1','model-probe','--cache-root',str(tmp_path));assert(code,e['error']['code'])==(21,'MODEL_ASSET_INVALID');outside.unlink()

@pytest.mark.parametrize('inp,out,device,err',[('source.wav',f'stems/runs/{RUN_ID}/worker-output','cpu','INVALID_INPUT'),('analysis.wav','../old','cpu','INVALID_OUTPUT'),('analysis.wav',f'stems/runs/{RUN_ID}/worker-output','tpu','INVALID_DEVICE')])
def test_separate_path_and_device_guards(capsys,monkeypatch,tmp_path,model,inp,out,device,err):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();(w/'analysis.wav').write_bytes(b'a');ready(monkeypatch,c,model[0]);code,e,_=run(capsys,*sep_args(c,w,inp,out,device));assert(code,e['error']['code'])==(30,err)

def test_exact_lowercase_hex_run_id_is_accepted(capsys,monkeypatch,tmp_path,model):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();(w/'analysis.wav').write_bytes(b'a');ready(monkeypatch,c,model[0]);demucs(monkeypatch)
 code,e,_=run(capsys,*sep_args(c,w,out=f'stems/runs/{RUN_ID}/worker-output'))
 assert code==0 and e['result']['outputs']==['vocals.wav','bass.wav','drums.wav','other.wav']

@pytest.mark.parametrize('run_id',['abc','a'*33,'A'*32,'g'*32,''])
def test_invalid_run_ids_are_rejected(capsys,monkeypatch,tmp_path,model,run_id):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();(w/'analysis.wav').write_bytes(b'a');ready(monkeypatch,c,model[0])
 code,e,_=run(capsys,*sep_args(c,w,out=f'stems/runs/{run_id}/worker-output'))
 assert(code,e['error']['code'])==(30,'INVALID_OUTPUT')

@pytest.mark.parametrize('mdl,err',[(type('Wrong',(),{'audio_channels':2,'samplerate':44100,'sources':list(constants.EXPECTED_SOURCES)})(),'MODEL_FAMILY_MISMATCH'),(type('HTDemucs',(),{'__module__':'demucs.htdemucs','audio_channels':1,'samplerate':44100,'sources':list(constants.EXPECTED_SOURCES)})(),'MODEL_CHANNEL_MISMATCH'),(type('HTDemucs',(),{'__module__':'demucs.htdemucs','audio_channels':2,'samplerate':48000,'sources':list(constants.EXPECTED_SOURCES)})(),'MODEL_SAMPLE_RATE_MISMATCH'),(type('HTDemucs',(),{'__module__':'demucs.htdemucs','audio_channels':2,'samplerate':44100,'sources':['vocals','bass','drums','other']})(),'MODEL_SOURCE_MISMATCH')])
def test_model_contract_validation(capsys,monkeypatch,tmp_path,model,mdl,err):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();(w/'analysis.wav').write_bytes(b'a');ready(monkeypatch,c,model[0]);demucs(monkeypatch,mdl);code,e,_=run(capsys,*sep_args(c,w));assert(code,e['error']['code'])==(40,err)

def test_four_outputs_provenance_and_no_job_manifest(capsys,monkeypatch,tmp_path,model):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();(w/'analysis.wav').write_bytes(b'a');ready(monkeypatch,c,model[0]);demucs(monkeypatch);code,e,_=run(capsys,*sep_args(c,w));r=e['result'];assert code==0 and r['outputs']==['vocals.wav','bass.wav','drums.wav','other.wav'];assert r['modelRevision']==constants.MODEL_REVISION and r['checkpointSha256']==model[1];o=w/'stems'/'runs'/RUN_ID/'worker-output';assert sorted(x.name for x in o.iterdir())==sorted(r['outputs']);assert not (w/'stems/stem-separation.json').exists()

def test_stdout_and_credential_redaction(capsys,monkeypatch,tmp_path):
 yamlmod(monkeypatch)
 def leak(**kw):print('https://user:secret@x.test hf_secret /private/cache');print('https://user:secret@x.test hf_secret /private/cache',file=sys.stderr);raise OSError('https://x hf_secret /private/cache')
 hub(monkeypatch,leak);os.environ['HF_TOKEN']='hf_inherited';code,e,o=run(capsys,'--protocol-version','1','prepare-model','--cache-root',str(tmp_path));allout=o.out+o.err;assert code==22 and e['error']['code']=='MODEL_DOWNLOAD_FAILED';assert all(x not in allout for x in ('secret','hf_','/private/cache','user:'));assert 'HF_TOKEN' not in os.environ

@pytest.mark.parametrize('exc,code,err',[(KeyboardInterrupt(),41,'CANCELLED'),(TimeoutError(),42,'WORKER_TIMEOUT'),(RuntimeError('x'),50,'INTERNAL_ERROR')])
def test_stable_exception_mapping(capsys,monkeypatch,tmp_path,exc,code,err):
 monkeypatch.setattr(commands,'prepare_model',lambda root:(_ for _ in()).throw(exc));got,e,_=run(capsys,'--protocol-version','1','prepare-model','--cache-root',str(tmp_path));assert(got,e['error']['code'])==(code,err)

def test_separation_cancellation_timeout_and_artifact_preservation(capsys,monkeypatch,tmp_path,model):
 c=tmp_path/'c';w=tmp_path/'w';c.mkdir();w.mkdir();a=w/'analysis.wav';m=w/'metadata.json';old=w/'stems'/'runs'/OTHER_RUN_ID/'worker-output';old.mkdir(parents=True);s=old/'vocals.wav';a.write_bytes(b'a');m.write_text('{}');s.write_bytes(b'old');ready(monkeypatch,c,model[0]);before={p:p.read_bytes() for p in(a,m,s)}
 code,e,_=run(capsys,*sep_args(c,w,out=f'stems/runs/{OTHER_RUN_ID}/worker-output'));assert(code,e['error']['code'])==(30,'INVALID_OUTPUT');assert {p:p.read_bytes() for p in before}==before
 class Stop(Sep):
  def separate_audio_file(self,p):raise KeyboardInterrupt()
 demucs(monkeypatch,sep=Stop);code,e,_=run(capsys,*sep_args(c,w,out=f'stems/runs/{RUN_ID}/worker-output'));assert(code,e['error']['code'])==(41,'CANCELLED')

def test_bundled_unprovisioned_lock_fails_closed(capsys,monkeypatch):
 monkeypatch.delenv(constants.RUNTIME_LOCK_ENV,raising=False)
 for n in OPT:monkeypatch.delitem(sys.modules,n,raising=False)
 code,e,_=run(capsys,'--protocol-version','1','runtime-probe');assert(code,e['error']['code'])==(10,'RUNTIME_PROFILE_UNPROVISIONED');assert not OPT&sys.modules.keys()

def test_readiness_parent_symlink_escape_rejected(capsys,tmp_path):
 outside=tmp_path.parent/(tmp_path.name+'-readiness');outside.mkdir();(outside/'htdemucs-bf35a81b-v1.json').write_text('{}');(tmp_path/'readiness').symlink_to(outside,target_is_directory=True)
 code,e,_=run(capsys,'--protocol-version','1','model-probe','--cache-root',str(tmp_path));assert(code,e['error']['code'])==(21,'READINESS_MANIFEST_INVALID')
 for p in outside.iterdir():p.unlink()
 outside.rmdir()

def test_allow_nan_false_fallback(capsys,monkeypatch):
 monkeypatch.setattr(cli,'runtime_probe',lambda:{'bad':float('nan')});code,e,o=run(capsys,'--protocol-version','1','runtime-probe');assert(code,e['error']['code'])==(50,'INTERNAL_ERROR');assert o.out.count('\n')==1


def test_readiness_parent_swap_after_temporary_write_is_contained(monkeypatch,tmp_path,model):
 yamlmod(monkeypatch);hub(monkeypatch,downloader(model[0],[]));outside=tmp_path.parent/(tmp_path.name+'-outside');outside.mkdir()
 real_replace=paths._replace_child;swapped={'done':False}
 def swap(parent,source,destination,directory_fd):
  if not swapped['done']:
   swapped['done']=True;old=parent.with_name(parent.name+'-old');parent.rename(old);parent.symlink_to(outside,target_is_directory=True)
  return real_replace(parent,source,destination,directory_fd)
 monkeypatch.setattr(paths,'_replace_child',swap)
 with pytest.raises(Exception):commands.prepare_model(str(tmp_path))
 assert not (outside/constants.READINESS_RELATIVE_PATH.split('/')[-1]).exists()
 assert not list(outside.glob('*.tmp'))
 old=tmp_path/'readiness-old'
 assert not (old/constants.READINESS_RELATIVE_PATH.split('/')[-1]).exists()
 assert not list(old.glob('*.tmp'))


@pytest.mark.parametrize('kind',['symlink','directory'])
def test_preexisting_unsafe_readiness_target_is_never_overwritten(monkeypatch,tmp_path,model,kind):
 parent=tmp_path/'readiness';parent.mkdir();target=parent/Path(constants.READINESS_RELATIVE_PATH).name;outside=tmp_path/'outside-readiness.json';outside.write_bytes(b'outside')
 if kind=='symlink':target.symlink_to(outside)
 else:target.mkdir()
 yamlmod(monkeypatch);hub(monkeypatch,downloader(model[0],[]))
 with pytest.raises(Exception):commands.prepare_model(str(tmp_path))
 assert outside.read_bytes()==b'outside'
 if kind=='symlink':assert target.is_symlink()
 else:assert target.is_dir()
 assert not list(parent.glob('*.tmp'))


def test_passive_hash_detects_checkpoint_change_during_read(monkeypatch,tmp_path,model):
 ready(monkeypatch,tmp_path,model[0]);payload=json.loads((tmp_path/constants.READINESS_RELATIVE_PATH).read_text());checkpoint=tmp_path/payload['cacheAssets']['checkpoint'];real=model_artifacts._sha256
 def mutate(path):
  digest=real(path);path.write_bytes(bytes(x^0x33 for x in model[0]));return digest
 monkeypatch.setattr(model_artifacts,'_sha256',mutate)
 with pytest.raises(Exception) as caught:probes.model_probe(str(tmp_path))
 assert getattr(caught.value,'code',None)=='CHECKPOINT_CHANGED_DURING_VERIFICATION'


def test_post_publication_checkpoint_change_removes_new_readiness(monkeypatch,tmp_path,model):
 yamlmod(monkeypatch);hub(monkeypatch,downloader(model[0],[]));real=commands._revalidate_assets;calls={'count':0}
 def mutate_on_post(cache_root,verified):
  calls['count']+=1
  if calls['count']==2:verified.checkpoint.lexical_path.write_bytes(bytes(x^0x55 for x in model[0]))
  return real(cache_root,verified)
 monkeypatch.setattr(commands,'_revalidate_assets',mutate_on_post)
 with pytest.raises(Exception):commands.prepare_model(str(tmp_path))
 assert not (tmp_path/constants.READINESS_RELATIVE_PATH).exists()
 assert not list((tmp_path/'readiness').glob('*.tmp'))


def test_unchanged_valid_cache_still_probes_successfully(monkeypatch,tmp_path,model):
 result,_=ready(monkeypatch,tmp_path,model[0]);probe=probes.model_probe(str(tmp_path))
 assert result['offlineReady'] is True and probe['offlineReady'] is True
 assert probe['checkpointSha256']==model[1]


def test_checkpoint_hashing_streams_84mb_scale(monkeypatch,tmp_path):
 checkpoint=tmp_path/'large.safetensors';checkpoint.write_bytes(b'');checkpoint.open('r+b').truncate(84_025_440);real_open=Path.open;sizes=[]
 class Tracking:
  def __init__(self,handle):self.handle=handle
  def __enter__(self):self.handle.__enter__();return self
  def __exit__(self,*args):return self.handle.__exit__(*args)
  def read(self,size=-1):sizes.append(size);return self.handle.read(size)
 def tracked(path,*args,**kwargs):
  handle=real_open(path,*args,**kwargs)
  return Tracking(handle) if path==checkpoint and args and args[0]=='rb' else handle
 monkeypatch.setattr(Path,'open',tracked)
 digest=model_artifacts._sha256(checkpoint)
 assert digest==hashlib.sha256(b'\x00'*84_025_440).hexdigest()
 assert sizes and -1 not in sizes and max(sizes)<=1024*1024 and len(sizes)>2


def test_readiness_publication_fallback_without_directory_fd(monkeypatch,tmp_path,model):
 monkeypatch.setattr(paths,'_open_directory_fd',lambda parent,expected:None);result,_=ready(monkeypatch,tmp_path,model[0])
 assert result['offlineReady'] is True
 assert probes.model_probe(str(tmp_path))['offlineReady'] is True
 assert not list((tmp_path/'readiness').glob('*.tmp'))
