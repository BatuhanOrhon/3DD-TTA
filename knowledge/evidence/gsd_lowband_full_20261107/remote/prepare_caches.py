"""Prepare each B2 once before all measured arms; preserve actual states in a sealed shard ZIP."""
import argparse,hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
ROOT=Path('/content/studies/gsd-lowband-full-20261107');ASSETS=ROOT/'prepared/assets';HERE=Path(__file__).resolve().parent
DRIVE=Path('/content/drive/MyDrive/thesis/result/gsd_lowband_full_20261107')
sys.path.insert(0,str(ASSETS/'source_snapshot'))
import numpy as np
import torch
from scripts.diagnose_decoded_spectral import configuration,apply_backend_policy,verify_setup
from scripts import eval_multibackbone_guidance as shared,eval_multibackbone_source as source
from legacy_trajectory import rng_hash
from research_artifacts import CORRUPTIONS
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);args=p.parse_args()
selection=json.loads(args.selection.read_text());ids_all=selection['indices'];label=args.selection.stem
config=configuration(['--asset-root',str(ASSETS),'--asset-manifest-sha256',source.file_identity(ASSETS/'manifest.json')['sha256'],
 '--result-root',str(ROOT/'cache_preparation'/label),'--paired-state-root',str(ROOT/'paired_states'),'--arm','control','--benchmark-shard',
 '--count',str(len(ids_all)),'--index-manifest',str(args.selection),'--index-manifest-sha256',source.file_identity(args.selection)['sha256'],
 '--backend-policy','controlled','--corruptions',*CORRUPTIONS])
backend=apply_backend_policy('controlled');manifest=source.read_manifest(ASSETS,config['asset_manifest_sha256']);verify_setup(ASSETS,manifest,config)
dataset=source.dataset_manifest(manifest,config);config['dataset_root']=dataset['root']
arrays={c:source.load_array(source.selected_data_path(config,c),dataset['files']['data_'+c+'_5.npy'],config) for c in CORRUPTIONS}
config['method']=shared.ORIGINAL_METHOD;lion,graph,reconstruct,ops,native,resolved=shared.load_guidance_runtime(config,manifest);assert graph is None
before=dict(vae=source.state_hash(lion.vae),priors=source.state_hash(lion.priors))
class Prepared(Exception):pass
records=[];started=time.monotonic()
for c in CORRUPTIONS:
 for start in range(0,len(ids_all),2):
  ids=np.asarray(ids_all[start:start+2],dtype=np.int64);shared.seed_inference(0)
  raw=torch.from_numpy(np.array(arrays[c][ids],dtype=np.float32,copy=True));normalized,center,scale=ops.normalize(raw)
  prepared=ops.rotate_pointcloud(torch.from_numpy(ops.upsample_all(normalized.numpy(),2048)).float().cuda()*3.3885)
  path=ROOT/'paired_states/modelnet40_c'/c/(hashlib.sha256(ids.tobytes()).hexdigest()+'.npz')
  assert not path.exists(),'Preserve previous attempt; do not overwrite/resume cache'
  state,recorder=shared.paired_preparation(path,prepared);assert state is None and recorder is not None
  def record_stop(values):recorder(values);raise Prepared()
  tick=time.monotonic()
  try:reconstruct(prepared,lion,35 if c=='background' else 8,**config['guidance_protocol'],state_observer=record_stop)
  except Prepared:pass
  else:raise AssertionError('Preparation callback not reached')
  torch.cuda.synchronize()
  records.append(dict(corruption=c,ids=ids.tolist(),member=str(path.relative_to(ROOT)),path=str(path),**source.file_identity(path),preparation_seconds=time.monotonic()-tick,rng_after=rng_hash()))
  if len(records)%16==0:print('FULL_CACHE_PREPARED',label,len(records),len(CORRUPTIONS)*len(ids_all)//2,flush=True)
assert before==dict(vae=source.state_hash(lion.vae),priors=source.state_hash(lion.priors))
report=dict(status='complete',label=label,selection=selection,records=records,elapsed_seconds=time.monotonic()-started,backend=backend,native=native,lion_state=before,scope='common encoder/style/noise only; no reverse diffusion/decoder/classifier')
folder=ROOT/'cache_archives';folder.mkdir(exist_ok=True);archive=folder/('cache_'+label+'.zip')
entries={row['member']:{k:row[k] for k in ('bytes','sha256')} for row in records}
raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();report_name='records/cache_preparation_'+label+'.json';entries[report_name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for row in records:z.write(row['path'],row['member'])
 z.writestr(report_name,raw);z.writestr('package_manifest.json',json.dumps(entries,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
record=ROOT/('cache_preparation_'+label+'.json')
with record.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
target=DRIVE/'cache_archives'/archive.name;target.parent.mkdir(parents=True,exist_ok=True)
with archive.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
assert source.file_identity(archive)==source.file_identity(target)
print('FULL_CACHE_ARCHIVE_DRIVE_VERIFIED',json.dumps(dict(path=str(archive),drive=str(target),**source.file_identity(archive))),flush=True)
