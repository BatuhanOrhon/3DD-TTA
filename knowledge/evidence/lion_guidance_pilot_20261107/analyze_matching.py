import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()))
from lion_guidance_pilot import feature_matches
archive=Path('result/modelnet40_c/lion_guidance_pilot/raw/20261107T110949Z_b.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4116a682093c08681aa1ad5fc500bb37eac68bd3487a0d77e3de1bc248fe2bbf'
torch.set_num_threads(2)
records=[]
with zipfile.ZipFile(archive) as z:
 receipt=json.loads(z.read('source_receipt.json'))
 assert hashlib.sha256(Path('lion_guidance_pilot.py').read_bytes()).hexdigest()==receipt['files']['source_snapshot/lion_guidance_pilot.py']['sha256']
 for name in sorted(n for n in z.namelist() if n.endswith('/paired_state.json')):
  case=name.split('/')[0];r=json.loads(z.read(name));cache='paired_states/'+Path(r['path']).name
  with np.load(io.BytesIO(z.read(cache)),allow_pickle=False) as d:reference=torch.from_numpy(d['local_latent'].copy()).double().reshape(2,2048,4)
  qref=reference[...,:3];rf=reference[...,3:]
  values=torch.cdist(qref,qref).square().topk(9,largest=False).values[:,:,1:]
  spatial=values.flatten(1).median(1).values.clamp_min(1e-12)[:,None,None]
  centered=rf-rf.median(1,keepdim=True).values
  feature=centered.abs().median(1,keepdim=True).values
  scale=torch.where(feature<=1e-6,centered.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6),feature)
  scales=spatial,scale
  permutation=torch.randperm(2048,generator=torch.Generator().manual_seed(20261107))
  for arm in ('full','feature'):
   with np.load(io.BytesIO(z.read(case+'/'+arm+'.npz')),allow_pickle=False) as d:latents=torch.from_numpy(d['latents'].copy()).double().reshape(2,4,2048,4)
   for checkpoint in range(3):
    h=latents[:,checkpoint];q=h[...,:3];f=h[...,3:]
    for direction in ('forward','reverse'):
     a,b,fa,fb=(q,qref,f,rf) if direction=='forward' else (qref,q,rf,f)
     geometric=feature_matches(a,b,fa,fb,beta=0.,scales=scales)
     matched=feature_matches(a,b,fa,fb,beta=.1,scales=scales)
     sfa,sfb=(fa,rf[:,permutation]) if direction=='forward' else (rf[:,permutation],fb)
     shuffled=feature_matches(a,b,sfa,sfb,beta=.1,scales=scales)
     records.append(dict(case=case,arm=arm,checkpoint=checkpoint,direction=direction,n=geometric.numel(),
       changed_vs_geometry=int((matched!=geometric).sum()),shuffled_changed=int((shuffled!=geometric).sum()),
       real_vs_shuffled=int((matched!=shuffled).sum())))
  print('B_MATCH_ANALYSIS_PROGRESS',case,flush=True)
report=dict(scope='CPU float64 recomputation on archived baseline/feature clean checkpoints; not exact captured CUDA assignments or proof of correct matches',torch=torch.__version__,beta=.1,k=8,rows=records)
report['aggregate']={arm:{key:sum(r[key] for r in records if r['arm']==arm) for key in ('n','changed_vs_geometry','shuffled_changed','real_vs_shuffled')} for arm in ('full','feature')}
out=Path('result/modelnet40_c/lion_guidance_pilot/derived/20261107T110949Z_b')
(out/'matching_activity.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'matching_analysis.py').write_text(Path(__file__).read_text())
print(json.dumps(report['aggregate'],indent=2))
