import hashlib,importlib.util,json,time,zipfile
from pathlib import Path
root=Path('/content/studies/gsd-lowband-full-20261107');path=root/'operator/technical_controller.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='58c466f291e20b02c16b23f8691a1dee46c57616b16902743913a288f07b40b2'
spec=importlib.util.spec_from_file_location('technical_process_helper',path);op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
plan=json.loads((root/'operator/plan.json').read_text());selection=root/'operator/technical.json';selection_sha=op.identity(selection)['sha256']
common=['--asset-root',str(root/'prepared/assets'),'--asset-manifest-sha256',plan['asset_manifest_sha256'],'--benchmark-shard','--count','2','--index-manifest',str(selection),'--index-manifest-sha256',selection_sha,'--paired-state-root',str(root/'paired_states'),'--backend-policy','controlled','--corruptions','gaussian','background']
report=dict(status='running',started_epoch=time.time(),jobs=[])
for label,flags in [('tech_control0',['--arm','control']),('tech_control1',['--arm','control']),('tech_half0',['--arm','live_control','--lowband-modes','100','--checkpoint-scd-weight','.5']),('tech_M100',['--arm','physical','--lowband-modes','100','--checkpoint-scd-weight','.5','--weight',str(.5*plan['frozen_alpha100'])])]:
 command=[op.PYTHON,'-u',str(op.SOURCE/'scripts/diagnose_decoded_spectral.py'),*common,'--result-root',str(root/'technical_runs'/label),'--drive-root','/content/drive/MyDrive/thesis/result/gsd_lowband_full_20261107/technical/'+label,*flags]
 result=op.run_process(label,command,timeout=600);report['jobs'].append(result);assert result['returncode']==0 and not result['timed_out']
 configs=[x for x in (root/'technical_runs'/label).rglob('config.json') if (x.parent/'bundle_manifest.json').is_file()];assert len(configs)==1
 c=json.loads(configs[0].read_text());assert c['diagnostic_complete'] and c['status']=='partial' and c['completed_batches']==dict(gaussian=1,background=1)
 archive=Path(str(configs[0].parent)+'.zip');op.verify_zip(archive,sealed=True);result.update(bundle=str(archive),archive=op.identity(archive))
report.update(status='complete',wall_seconds=time.time()-report['started_epoch'])
op.save(root/'technical_phase.json',report);op.save(op.DRIVE/'technical_phase.json',report)
package=root/'technical_raw.zip'
with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
 z.write(root/'technical_phase.json','technical_phase.json');z.write(root/'cache_archives/cache_technical.zip','cache_technical.zip')
 for r in report['jobs']:z.write(r['bundle'],r['label']+'/'+Path(r['bundle']).name)
with zipfile.ZipFile(package) as z:assert z.testzip() is None
import shutil
target=op.DRIVE/'technical_raw.zip'
with package.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
assert op.identity(package)==op.identity(target)
record=dict(path=str(package),drive=str(target),**op.identity(package),status='technical raw Drive verified')
transfer=op.split_archive(package,'technical_raw','inference',str(target));record['transfer_manifest']=str(transfer)
op.save(root/'technical_archive.json',record)
print(json.dumps(record),flush=True);print('LOWBAND_FULL_TECHNICAL_PHASE_COMPLETE',flush=True)
