from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
from pathlib import Path
root=Path('/content/studies/guidance-components')
archive=Path('/content/agent_b_smoke_local_acceptance.zip')
def identity(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert identity(archive)=={'bytes': 19261, 'sha256': 'a868587767a1943f7e0dec8641ae64e733d7cb56aa11aac51dc0e2682ad998d0'}
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 initial=z.read('smoke_validation_initial.json')
 derived=z.read('smoke_validation_derived.json')
 accepted=z.read('accepted_smoke.json')
 projection=z.read('local_smoke_analysis/timing_projection.json')
 a=json.loads(accepted);v=json.loads(derived)
 assert a['status']=='accepted' and a['manifest_sha256']==v['manifest_sha256']
 assert a['validation']==dict(bytes=len(derived),sha256=hashlib.sha256(derived).hexdigest())
 assert a['timing_projection']==dict(bytes=len(projection),sha256=hashlib.sha256(projection).hexdigest())
 assert (root/'smoke_validation.json').read_bytes()==initial
 assert not (root/'accepted_smoke.json').exists() and not (root/'smoke_validation_initial.json').exists()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 drive=Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/phase-evidence')/('local_smoke_acceptance_'+stamp)
 drive.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(archive,drive/archive.name)
 assert identity(drive/archive.name)==identity(archive)
 (root/'smoke_validation_initial.json').write_bytes(initial)
 (root/'smoke_validation.json').write_bytes(derived)
 (root/'accepted_smoke.json').write_bytes(accepted)
 (root/'timing_projection.json').write_bytes(projection)
 receipt=dict(status='accepted',archive=dict(identity(archive),drive_path=str(drive/archive.name)),validation=a['validation'],timing_projection=a['timing_projection'])
 (root/'accepted_smoke_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_LOCAL_SMOKE_ACCEPTED',flush=True)
