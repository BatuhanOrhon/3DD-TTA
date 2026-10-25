import hashlib,json,sys,zipfile
from pathlib import Path
phases=Path('/content/studies/guidance-components/phases')
assert hashlib.sha256((phases/'deploy_assets_template.py').read_bytes()).hexdigest()=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0,str(phases))
from deploy_assets_template import ROOT,DRIVE,MANIFEST_SHA,SOURCE_SHA,verify,identity,checked_members,copy_checked,utc
archive=Path('/content/agent_b_development_local_acceptance.zip')
verify(archive,{'bytes': 36863, 'sha256': '7d03416df5afd023e38a09d90cebabd131d65b8befd5536fa7b2db0a355b3f74'})
with zipfile.ZipFile(archive) as z:
 checked_members(z)
 accepted=z.read('accepted_development.json');plan=z.read('frozen_confirmation_plan.json');validation=z.read('local_development_analysis/validation.json')
 a=json.loads(accepted);v=json.loads(validation);p=json.loads(plan)
 assert a['status']=='accepted' and v['status']=='complete_fixed_development_subsets'
 assert a['manifest_sha256']==v['manifest_sha256']==MANIFEST_SHA and a['source_snapshot_sha256']==SOURCE_SHA
 assert a['local_validation']==dict(bytes=len(validation),sha256=hashlib.sha256(validation).hexdigest())
 assert a['frozen_plan']==dict(bytes=len(plan),sha256=hashlib.sha256(plan).hexdigest())
 assert p['status']=='frozen_before_any_confirmation_scoring' and p['seeds']==[0,1,2]
 verify(a['phase_archive']['drive_path'],a['phase_archive'])
 assert not (ROOT/'accepted_development.json').exists() and not (ROOT/'frozen_confirmation_plan.json').exists()
 target=DRIVE/'phase-evidence'/('local_development_acceptance_'+utc())/archive.name
 copy_checked(archive,target,{'bytes': 36863, 'sha256': '7d03416df5afd023e38a09d90cebabd131d65b8befd5536fa7b2db0a355b3f74'})
 (ROOT/'accepted_development.json').write_bytes(accepted);(ROOT/'frozen_confirmation_plan.json').write_bytes(plan)
 receipt=dict(status='accepted',archive=dict(identity(archive),drive_path=str(target)),acceptance=identity(ROOT/'accepted_development.json'),frozen_plan=a['frozen_plan'])
 (ROOT/'accepted_development_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (target.parent/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_LOCAL_DEVELOPMENT_ACCEPTED',flush=True)
