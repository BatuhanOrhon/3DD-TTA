"""Reviewed file-only recovery after canonical setup; never launch inference here."""
import argparse,copy,hashlib,json,os,shutil,time,zipfile
from pathlib import Path
ROOT=Path('/content/studies/gsd-lowband-full-20261107')
DRIVE=Path('/content/drive/MyDrive/thesis/result/gsd_lowband_full_20261107')
LOST='shard002_half0'
STARTED=1794066564.0503306
COMMON_KEYS=('executed_source_files','native_extensions','native_bootstrap','asset_manifest_sha256','lion_state_before','classifier_state_before_sha256','runtime')

def identity(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def owned(path,base):
 path=Path(path).resolve();base=Path(base).resolve();assert base in path.parents and path!=base
 return path
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def install(raw,target,expected=None):
 target=Path(target);expected=expected or dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
 if target.exists():assert identity(target)==expected,'Preserve mismatched existing file: '+str(target);return
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert identity(target)==expected
def checked_zip(path):
 z=zipfile.ZipFile(path);names=z.namelist()
 assert len(names)==len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names)
 assert z.testzip() is None
 return z
def derive_journal(original,accepted):
 assert original['started_epoch']==STARTED and original['budget_seconds']==172800
 assert set(original['jobs'])==set(accepted)|{LOST}
 assert original['jobs'][LOST]['status']=='running' and LOST not in accepted
 result=copy.deepcopy(original);del result['jobs'][LOST]
 return result

def restore_native(stage):
 stage=Path(stage);proof=Path('/content/cli_setup_verification.json')
 assert json.loads(proof.read_text())['status']=='passed','Fresh canonical ten-cell proof required first'
 assert os.path.ismount('/content/drive') and DRIVE.is_dir(),'Mounted owned Drive backup is required before replacing native files'
 inventory=json.loads((stage/'inventory.json').read_text());capsule=stage/'native_capsule.zip'
 assert identity(capsule)==inventory['native_capsule']
 destination=ROOT/'recovery'/('native_before_restore_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));destination.mkdir(parents=True,exist_ok=False)
 install(proof.read_bytes(),destination/'new_vm_canonical_setup.json')
 replacements={};fresh={}
 with checked_zip(capsule) as z:
  manifest=json.loads(z.read('native_manifest.json'));assert len(manifest)==10
  for name,expected in manifest.items():
   target=Path(name);assert any(base in target.parents for base in (Path('/content/3DD-TTA'),Path('/usr/local/envs/3dd_tta_env'),Path('/root/.cache/torch_extensions')))
   assert not target.is_symlink() and all(not parent.is_symlink() for parent in target.parents)
   raw=z.read('native_payload/'+name.lstrip('/'));assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
   replacements[name]=(raw,expected);fresh[name]=dict(present=target.exists())
   if target.exists():
    install(target.read_bytes(),destination/'fresh_binary_backups'/name.lstrip('/'))
    fresh[name].update(identity=identity(target))
 save(destination/'fresh_state.json',dict(files=fresh,canonical_setup=identity(proof),original_capsule=identity(capsule)))
 members={str(path.relative_to(destination)):identity(path) for path in destination.rglob('*') if path.is_file()}
 archive=destination/'fresh_native_backup.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for name in sorted(members):z.write(destination/name,name)
  z.writestr('fresh_backup_manifest.json',json.dumps(members,sort_keys=True))
 drive_directory=DRIVE/'recovery'/destination.name;drive_directory.mkdir(parents=True,exist_ok=False)
 drive_archive=drive_directory/archive.name;expected_archive=identity(archive)
 install(archive.read_bytes(),drive_archive,expected_archive)
 with checked_zip(drive_archive) as z:
  assert set(z.namelist())==set(members)|{'fresh_backup_manifest.json'}
  assert json.loads(z.read('fresh_backup_manifest.json'))==members
  for name,expected in members.items():
   raw=z.read(name);assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
 assert identity(drive_archive)==expected_archive
 backup_receipt=dict(status='fresh native bytes/proof verified on Drive before any native replacement',archive=dict(path=str(drive_archive),**expected_archive),files=fresh)
 save(drive_directory/'backup.receipt.json',backup_receipt);save(destination/'backup.receipt.json',backup_receipt)
 for name,(raw,expected) in replacements.items():
  target=Path(name)
  if target.exists() and identity(target)!=expected:
   pending=target.with_name(target.name+'.recovery.pending');install(raw,pending,expected);os.replace(pending,target)
  else:install(raw,target,expected)
 save(destination/'receipt.json',dict(status='original ten native binaries restored; fresh-process strict runtime/model checks still pending',capsule=identity(capsule),files=manifest))
 return dict(status='file-only native restore complete; no imports or GPU work',receipt=str(destination/'receipt.json'))

def restore(stage,apply=False):
 stage=Path(stage);inventory=json.loads((stage/'inventory.json').read_text());labels=set(inventory['accepted'])
 assert len(labels)==11 and sum(label.endswith('_cache') for label in labels)==3 and LOST not in labels
 assert not (DRIVE/'job_receipts'/(LOST+'.json')).exists(),'Lost job may already have accepted evidence; audit before retry'
 assert not list((DRIVE/'runs'/LOST).rglob('final.zip')),'Preserve and audit possible completed/partial final ZIP before retry'
 original_raw=(DRIVE/'campaign_journal.json').read_bytes();original=json.loads(original_raw)
 derived=derive_journal(original,labels)
 assert original['binding']==inventory['binding']
 operator_common=json.loads((DRIVE/'common_identity.json').read_text())
 assert operator_common==inventory['operator_common']
 job_receipts={}
 for label,item in inventory['accepted'].items():
  receipt_path=DRIVE/'job_receipts'/(label+'.json');r=json.loads(receipt_path.read_text())
  assert r['status']=='accepted' and r['label']==label and r['binding']==original['binding']
  assert r['archive']==item['manifest']['archive'] and identity(r['archive']['drive'])==item['archive_identity']
  assert r['selection_sha256']==identity(stage/'operator'/('shard'+label[5:8]+'.json'))['sha256']
  if r['kind']=='inference':assert r['validation']['common']==operator_common
  job_receipts[label]=r
 assert identity(ROOT/'prepared/assets/manifest.json')==inventory['asset_manifest'],'Prepare the exact e2 asset namespace first'
 assert identity(ROOT/'prepared/assets/full_source_snapshot.zip')==inventory['source_snapshot']
 assert identity(ROOT/'prepared/assets/source_snapshot/scripts/diagnose_decoded_spectral.py')==inventory['driver']
 assert identity(ROOT/'calibration_frozen.json')==inventory['calibration']
 # Exact native bytes must be restored by the reviewed native-capsule step before resume.
 native=json.loads((stage/'native_manifest.json').read_text())
 for path,expected in native.items():assert identity(path)==expected,'Native capsule not restored: '+path
 for path,expected in inventory['absolute_required_files'].items():
  target=Path(path);assert not target.is_symlink() and all(not parent.is_symlink() for parent in target.parents),'Absolute-path symlink forbidden: '+path
  assert identity(target)==expected,'Canonical consumed path differs: '+path
 if not apply:
  return dict(status='verified file-only restore plan; no mutation',accepted_jobs=sorted(labels),inference_jobs_skipped=8,
   lost_retry=LOST,started_epoch=original['started_epoch'],budget_seconds=original['budget_seconds'])
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());archive_dir=DRIVE/'recovery'/('runtime_loss_20261107T1839Z_'+stamp)
 archive_dir.mkdir(parents=True,exist_ok=False)
 # Preserve the last Drive journal before any continuation journal or replay checkpoint can replace it.
 install(original_raw,archive_dir/'campaign_journal.original.json')
 save(archive_dir/'runtime_loss_attempt.json',dict(status='lost_unaccepted_attempt',label=LOST,loss_reported_utc='2026-11-07T18:39:00Z',
  last_reported_batches=93,total_batches=480,ephemeral_vm_files='unrecoverable unless separately archived; never inferred complete',
  source_journal=identity(archive_dir/'campaign_journal.original.json'),lost_entry=original['jobs'][LOST],
  accepted_inference_jobs_not_rerun=sorted(label for label in labels if not label.endswith('_cache')),
  retry_policy='Fresh UTC run_id, same joblabel and Drive prefix; preserve all old partial attempts and all identity gates'))
 partial=DRIVE/'runs'/LOST
 with zipfile.ZipFile(archive_dir/'lost_partial_drive_files.zip','x',zipfile.ZIP_DEFLATED) as z:
  manifest={}
  for path in sorted(partial.rglob('*')):
   if path.is_file():
    member=str(path.relative_to(partial));manifest[member]=identity(path);z.write(path,member)
  z.writestr('recovery_members.json',json.dumps(manifest,sort_keys=True))
 with checked_zip(archive_dir/'lost_partial_drive_files.zip') as z:
  for name,expected in manifest.items():
   raw=z.read(name);assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
 local_archive=ROOT/'recovery'/archive_dir.name;local_archive.mkdir(parents=True,exist_ok=False)
 for path in archive_dir.iterdir():
  if path.is_file():install(path.read_bytes(),local_archive/path.name)
 # These active paths must remain fresh. Keep any surviving attempt outside controller searches.
 for active in (ROOT/'raw_runs'/LOST,ROOT/'phase_logs'/(LOST+'.log'),ROOT/'phase_logs'/(LOST+'.process.json')):
  if active.exists():
   moved=local_archive/'old_vm_active_paths'/active.relative_to(ROOT);moved.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(active),str(moved))
 for path in (stage/'operator').iterdir():
  if path.is_file():install(path.read_bytes(),ROOT/'operator'/path.name)
 install((DRIVE/'common_identity.json').read_bytes(),ROOT/'common_identity.json')
 for label,item in inventory['accepted'].items():
  install((DRIVE/'job_receipts'/(label+'.json')).read_bytes(),ROOT/'job_receipts'/(label+'.json'))
  install((stage/'contexts'/label/'transfer_manifest.json').read_bytes(),ROOT/'transfers'/label/'transfer_manifest.json')
  install((stage/'contexts'/label/'ack.json').read_bytes(),ROOT/'acks'/(label+'.json'))
  if label.endswith('_cache'):
   archive=Path(item['manifest']['archive']['drive'])
   with checked_zip(archive) as z:
    entries=json.loads(z.read('package_manifest.json'));assert set(z.namelist())==set(entries)|{'package_manifest.json'}
    for name,expected in entries.items():install(z.read(name),owned(ROOT/name,ROOT),expected)
 # Do not recreate accepted VM bundle/ZIP/parts: exact ACK + permanent Drive SHA lets original reaper verify absence and recreate cleanup receipts.
 save(local_archive/'campaign_journal.derived.json',derived)
 save(local_archive/'restore_receipt.json',dict(status='restored; GPU/source/common preflight still required before resume',accepted_jobs=sorted(labels),
  skipped_inference_jobs=8,retry_label=LOST,original_journal=identity(archive_dir/'campaign_journal.original.json'),
  derived_journal=identity(local_archive/'campaign_journal.derived.json'),partial_drive_archive=identity(archive_dir/'lost_partial_drive_files.zip')))
 if (ROOT/'campaign_journal.json').exists():assert (ROOT/'campaign_journal.json').read_bytes()==original_raw
 pending=ROOT/'campaign_journal.recovery.pending.json';save(pending,derived);os.replace(pending,ROOT/'campaign_journal.json')
 if (ROOT/'science_started.json').exists():assert json.loads((ROOT/'science_started.json').read_text())['started_epoch']==STARTED
 else:save(ROOT/'science_started.json',dict(started_epoch=STARTED,reconstructed_from='immutable original Drive campaign journal; original technical clock unchanged'))
 return dict(status='file-only recovery restored; do not launch until strict preflight',record=str(local_archive/'restore_receipt.json'),accepted_inference_jobs_skipped=8)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);p.add_argument('--apply',action='store_true');p.add_argument('--restore-native',action='store_true');a=p.parse_args()
 assert not (a.apply and a.restore_native),'Separate native restore and context restore for review'
 print(json.dumps(restore_native(a.stage) if a.restore_native else restore(a.stage,a.apply),indent=2))
