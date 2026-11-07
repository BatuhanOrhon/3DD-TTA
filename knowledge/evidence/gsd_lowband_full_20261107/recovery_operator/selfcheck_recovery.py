"""Pure journal derivation and recovery-path tests; never resume a real job."""
import copy,hashlib,importlib.util,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('restore_context',Path(__file__).with_name('restore_context.py'))
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
accepted={f'shard00{s}_{a}' for s in range(3) for a in ('cache','control','M100','half0')}-{restore.LOST}
original=dict(started_epoch=1794066564.0503306,budget_seconds=172800,binding={'same':'frozen'},status='running',
 jobs={label:dict(status='acked_and_cleaned') for label in accepted})
original['jobs'][restore.LOST]=dict(status='running',started_epoch=1794076350.)
before=copy.deepcopy(original)
derived=restore.derive_journal(original,accepted)
assert original==before and set(derived['jobs'])==accepted
assert derived['started_epoch']==original['started_epoch'] and derived['budget_seconds']==172800
assert len(accepted)==11 and sum(not label.endswith('_cache') for label in accepted)==8
expected=copy.deepcopy(original);del expected['jobs'][restore.LOST];assert derived==expected
for changed in ('missing_lost','completed_lost','unknown_job','wrong_clock'):
 bad=copy.deepcopy(original)
 if changed=='missing_lost':del bad['jobs'][restore.LOST]
 elif changed=='completed_lost':bad['jobs'][restore.LOST]['status']='acked_and_cleaned'
 elif changed=='unknown_job':bad['jobs']['peer_run']={}
 else:bad['started_epoch']+=1
 try:restore.derive_journal(bad,accepted)
 except AssertionError:pass
 else:raise AssertionError('Unsafe journal accepted: '+changed)
assert restore.owned(restore.ROOT/'acks/shard000_control.json',restore.ROOT)==restore.ROOT/'acks/shard000_control.json'
for path in (restore.ROOT/'../peer',Path('/content/studies/peer/data')):
 try:restore.owned(path,restore.ROOT)
 except AssertionError:pass
 else:raise AssertionError('Unowned path accepted')
with tempfile.TemporaryDirectory(prefix='native-drive-before-replace-') as temporary:
 for scenario in ('drive_copy_failure','success','unmounted'):
  base=Path(temporary).resolve()/scenario;base.mkdir();stage=base/'stage';stage.mkdir();native=base/'native';native.mkdir()
  proof=base/'new_vm_proof.json';proof.write_text('{"status":"passed","synthetic_cpu_fixture":true}\n')
  restore.ROOT=base/'owned';restore.ROOT.mkdir();restore.DRIVE=base/'drive';restore.DRIVE.mkdir()
  manifest={};fresh={}
  for index in range(10):
   target=native/('fixture%d.so'%index);raw=('fresh-native-%d'%index).encode();target.write_bytes(raw);fresh[target]=raw
   old=('original-native-%d'%index).encode();manifest[str(target)]=dict(bytes=len(old),sha256=hashlib.sha256(old).hexdigest())
  capsule=stage/'native_capsule.zip'
  with zipfile.ZipFile(capsule,'x') as z:
   z.writestr('native_manifest.json',json.dumps(manifest))
   for index,target in enumerate(fresh):z.writestr('native_payload/'+str(target).lstrip('/'),('original-native-%d'%index).encode())
  (stage/'inventory.json').write_text(json.dumps(dict(native_capsule=restore.identity(capsule))))
  real_path=Path
  def mapped_path(value):
   if str(value)=='/content/cli_setup_verification.json':return proof
   if str(value)=='/content/3DD-TTA':return native
   return real_path(value)
  original_install=restore.install;original_replace=restore.os.replace;replaced=[]
  def install(raw,target,expected=None):
   if scenario=='drive_copy_failure' and restore.DRIVE in Path(target).parents:raise OSError('Synthetic Drive copy failure')
   return original_install(raw,target,expected)
  def replace(source,target):
   if Path(target) in fresh:
    archives=list((restore.DRIVE/'recovery').glob('*/fresh_native_backup.zip'));assert len(archives)==1,'Native replacement preceded durable Drive backup'
    with zipfile.ZipFile(archives[0]) as z:
     assert z.testzip() is None
     for file,raw in fresh.items():assert z.read('fresh_binary_backups/'+str(file).lstrip('/'))==raw
     assert z.read('new_vm_canonical_setup.json')==proof.read_bytes()
    replaced.append(Path(target))
   return original_replace(source,target)
  with patch.object(restore,'Path',mapped_path),patch.object(restore,'install',install),patch.object(restore.os,'replace',replace),patch.object(restore.os.path,'ismount',return_value=scenario!='unmounted'):
   if scenario=='success':
    result=restore.restore_native(stage);assert len(replaced)==10
    assert result['status'].startswith('file-only native restore complete')
   else:
    try:restore.restore_native(stage)
    except (OSError if scenario=='drive_copy_failure' else AssertionError):pass
    else:raise AssertionError('Expected Drive failure was bypassed: '+scenario)
    assert not replaced and all(path.read_bytes()==raw for path,raw in fresh.items()),'Drive failure changed fresh native bytes'
print('RECOVERY_CPU_JOURNAL_ZERO_ACCEPTED_RERUN_SELFCHECK_COMPLETE')
