"""Validate raw package/source/native/actual paired caches before CPU recount."""
import ast,hashlib,importlib.util,io,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'knowledge/evidence/decoded_physical_accuracy_20261104'))
import analyze_accuracy as analysis
from scripts.analyze_scan_trajectory import Bundle
spec=importlib.util.spec_from_file_location('deploy_safety',ROOT/'operator_v1/deploy_assets_template.py')
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
identity=analysis.identity
receipt=json.loads((ROOT/'archive_receipt.json').read_text())
assert receipt['status']=='complete' and receipt['raw_run_count']==20
expected={k:receipt[k] for k in ('bytes','sha256')}
archive=ROOT/'accuracy_raw_package.zip';assert identity(archive)==expected
with zipfile.ZipFile(archive) as z:
 safety.checked_members(z)
 manifest=json.loads(z.read('package_manifest.json'))
 assert set(z.namelist())==set(manifest)|{'package_manifest.json'}
 for name,entry in manifest.items():
  data=z.read(name);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==entry,name
DEST=ROOT/'downloaded';safety.extract_checked(archive,DEST)
source=json.loads((DEST/'deployment_operator/source_receipt.json').read_text())
assert source['source_commit'].startswith('cdbdfb1')
with zipfile.ZipFile(DEST/'deployment_operator/physical_source_overlay.zip') as z:
 assert set(z.namelist())==set(source['files']) and z.testzip() is None
 for name,entry in source['files'].items():
  data=z.read(name);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==entry
  if name.endswith('.py'):ast.parse(data,feature_version=(3,8))
asset=json.loads((DEST/'metadata/asset_manifest.json').read_text())
assert identity(DEST/'metadata/asset_manifest.json')==dict(bytes=181793,sha256='05b6fcde7c6982954594bec34fd6bd7d62b8b020f86129d2d8d16fafd8c0363c')
native=json.loads((DEST/'metadata/executed_native_manifest.json').read_text())
for entry in native.values():
 assert identity(DEST/entry['archive_member'])=={k:entry[k] for k in ('bytes','sha256')}
 assert (DEST/entry['archive_member']).read_bytes()[:4]==b'\x7fELF'
research=json.loads((DEST/'metadata/research_receipt.json').read_text())
assert research['status']=='research_complete' and research['science_elapsed_seconds']<3300
cache_report=json.loads((DEST/'metadata/cache_preparation.json').read_text())
assert cache_report['status']=='complete' and len(cache_report['records'])==240
for entry in cache_report['records']:
 p=DEST/'paired_states/modelnet40_c'/entry['corruption']/Path(entry['path']).name
 assert identity(p)=={k:entry[k] for k in ('bytes','sha256')}
assert len(list((DEST/'paired_states').rglob('*.npz')))==240
paths={};cases=0;cache_seen=set()
canonical_root=REPO/'result/modelnet40_c/decoded_physical_spectral_diagnostic';canonical_root.mkdir(parents=True,exist_ok=True)
for path in sorted((DEST/'raw_runs').glob('*.zip')):
 canonical=canonical_root/path.name
 with path.open('rb') as a,canonical.open('xb') as b:shutil.copyfileobj(a,b)
 assert identity(canonical)==identity(path)
 paths[path.name]=str(canonical)
 bundle=Bundle(canonical)
 try:
  config=bundle.json('config.json');assert config['asset_manifest']==asset
  for name,entry in config['executed_source_files'].items():
   if name.startswith('assets/'):assert entry==asset['files'][name.removeprefix('assets/')]
   else:
    assert name.startswith('repo/')
    member='source_snapshot/'+name.removeprefix('repo/');data=bundle.read(member)
    assert entry==asset['files'][member]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  for entry in list(config['native_extensions'].values())+list(config['native_bootstrap']['bindings'].values()):
   assert {k:entry[k] for k in ('bytes','sha256')}=={k:native[entry['path']][k] for k in ('bytes','sha256')}
  for c in config['corruptions']:
   for batch in range(config['count']//2):
    case='batches/%s/batch%04d/'%(c,batch);d=bundle.json(case+'diagnostics.json')
    if config['arm']=='control':assert 'decoded_spectral' not in d['trajectory']
    else:
     spectral=d['trajectory']['decoded_spectral'];mode='frozen_residual_surrogate' if config['freeze_decoder_residual'] else 'full_decoder_gradient'
     assert spectral['loss_gradient_mode']==mode and spectral['decoder_jacobian_in_loss']==(not config['freeze_decoder_residual'])
     assert spectral['reference_xyz_sha256']==d['prepared_x_sha256'] and len(spectral['records'])==3
     assert [r['loop_count'] for r in spectral['records']]==([9,18,27] if c=='background' else [2,4,6])
     assert all(np.isfinite(r['raw_loss']) and r['decoded_geometry_reused'] and not r['classifier_gradients'] and r['loss_gradient_mode']==mode for r in spectral['records'])
    cache=DEST/('paired_states/modelnet40_c/'+c if config['count']==8 else 'p0_paired_state')/Path(d['paired_state']['path']).name
    assert identity(cache)=={k:d['paired_state'][k] for k in ('bytes','sha256')}
    with np.load(cache,allow_pickle=False) as saved:
     metadata=json.loads(str(saved['metadata'].item()))
     assert metadata['input_sha256']==d['prepared_x_sha256']
     assert metadata['rng_after_sha256']==d['trajectory']['initial_rng_sha256']
     for key in ('shape_latent','local_latent','noise'):
      assert analysis.tensor_identity(saved[key])==metadata['tensor_hashes'][key]==d['trajectory']['initial_hashes'][key]
     reference=analysis.arrays(bundle,case+'reference.npz')
     assert np.array_equal(saved['local_latent'].reshape(2,2048,4)[:,:,:3],reference['initial_h0_xyz'])
    geometry=analysis.arrays(bundle,case+'terminal_geometry.npz')
    raw=geometry['raw_corrupted_xyz'].astype(np.float32);center=(raw.max(1,keepdims=True)+raw.min(1,keepdims=True))/2
    scale=(raw-center).reshape(2,-1).max(1).reshape(2,1,1)
    assert np.array_equal(center,geometry['normalization_center']) and np.array_equal(scale,geometry['normalization_scale'])
    assert d['scheduler']['final_alpha_cumprod']==1 and d['scheduler']['config']['set_alpha_to_one'] is True
    cache_seen.add(str(cache));cases+=1
 finally:bundle.close()
assert cases==964 and len(cache_seen)==241
stage_paths=json.loads((DEST/'metadata/stage_paths.json').read_text())
local_paths={s:{a:paths[Path(p).name] for a,p in row.items()} for s,row in stage_paths.items()}
with (ROOT/'stage_paths_local.json').open('x') as f:json.dump(local_paths,f,indent=2)
technical_remote=json.loads((DEST/'metadata/technical_summary.json').read_text())
technical=analysis.technical([paths[Path(technical_remote['runs'][a]['path']).name] for a in analysis.ARMS])
assert technical['passed'] and technical['matched_physical_metric_gate']['passed']
main=analysis.main_analysis(ROOT/'stage_paths_local.json')
for name,value in (('technical_local_recount.json',technical),('main_local_recount.json',main)):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
report=dict(status='accepted_raw',package=expected,source_commit=source['source_commit'],source_files=len(source['files']),
 native_payloads=len(native),runs=len(paths),cases=cases,actual_paired_caches=len(cache_seen),science_elapsed_seconds=research['science_elapsed_seconds'],
 main_numerical_attribution_gate=main['numerical_attribution_gate_passed'],main_metric_gate=main['matched_physical_metric_gate']['passed'])
with (ROOT/'local_acceptance.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
