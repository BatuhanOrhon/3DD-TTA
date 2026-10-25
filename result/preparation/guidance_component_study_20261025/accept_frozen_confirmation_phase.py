import hashlib,json,sys,zipfile
from pathlib import Path
phases=Path('/content/studies/guidance-components/confirmation_phases')
assert hashlib.sha256((phases/'deploy_assets_template.py').read_bytes()).hexdigest()=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0,str(phases))
from deploy_assets_template import ROOT,DRIVE,MANIFEST_SHA,SOURCE_SHA,verify,identity,checked_members,copy_checked,utc
archive=Path('/content/agent_b_frozen_confirmation_local_acceptance.zip')
verify(archive,{'bytes': 71057, 'sha256': 'f2b6b2e874b0cdb3c3d095ff46179f0c600e7443f1112ab4a357e26f1dfba704'})
with zipfile.ZipFile(archive) as z:
 checked_members(z);manifest=json.loads(z.read('local_acceptance_manifest.json'))
 assert set(z.namelist())==set(manifest)|{'local_acceptance_manifest.json'}
 for name,row in manifest.items():
  b=z.read(name);assert row==dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 accepted=z.read('accepted_frozen_confirmation.json');a=json.loads(accepted)
 v=z.read('local_frozen_confirmation_analysis/validation.json');c=z.read('local_frozen_confirmation_analysis/cache_roundtrip_validation.json')
 assert a['local_validation']==dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
 assert a['local_cache_validation']==dict(bytes=len(c),sha256=hashlib.sha256(c).hexdigest())
 assert a['status']=='accepted' and a['manifest_sha256']==MANIFEST_SHA and a['source_snapshot_sha256']==SOURCE_SHA
 assert a['frozen_plan_sha256']==identity(ROOT/'frozen_confirmation_plan.json')['sha256']=='c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45'
 assert json.loads(v)['status']=='complete_frozen_confirmation_subsets' and json.loads(v)['no_policy_reselection']
 assert json.loads(c)['status']=='verified_local_drive_roundtrip' and json.loads(c)['cache_files']==90
 for row in [a['phase_archive'],a['cache_archive']]:
  assert row['drive_path'].startswith(str(DRIVE)+'/');verify(row['drive_path'],row)
 assert not (ROOT/'accepted_frozen_confirmation.json').exists()
 target=DRIVE/'phase-evidence'/('local_frozen_confirmation_acceptance_'+utc())/archive.name
 copy_checked(archive,target,{'bytes': 71057, 'sha256': 'f2b6b2e874b0cdb3c3d095ff46179f0c600e7443f1112ab4a357e26f1dfba704'})
 (ROOT/'accepted_frozen_confirmation.json').write_bytes(accepted)
 receipt=dict(status='accepted',archive=dict(identity(archive),drive_path=str(target)),acceptance=identity(ROOT/'accepted_frozen_confirmation.json'),frozen_plan_sha256=a['frozen_plan_sha256'])
 (ROOT/'accepted_frozen_confirmation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (target.parent/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_LOCAL_FROZEN_CONFIRMATION_ACCEPTED',flush=True)
