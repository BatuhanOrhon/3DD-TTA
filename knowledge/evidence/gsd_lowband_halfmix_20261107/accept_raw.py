"""Accept immutable raw bytes and actual source/native/caches before interpretation."""
import argparse,ast,hashlib,importlib.util,json,shutil,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('archive_safety',REPO/'knowledge/evidence/chamfer_guidance_executed_code_20261031/historical_controls/remote/deploy_assets_template.py')
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
identity=safety.identity
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('receipt',type=Path);p.add_argument('destination',type=Path);args=p.parse_args()
receipt=json.loads(args.receipt.read_text());assert identity(args.archive)=={k:receipt[k] for k in ('bytes','sha256')}
with zipfile.ZipFile(args.archive) as z:
 safety.checked_members(z);manifest=json.loads(z.read('package_manifest.json'))
 assert set(z.namelist())==set(manifest)|{'package_manifest.json','collection.json'}
 for name,expected in manifest.items():
  data=z.read(name);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==expected,name
 assert json.loads(z.read('collection.json'))==receipt['collection']
safety.extract_checked(args.archive,args.destination)
root=args.destination
source=json.loads((root/'operator/source_receipt.json').read_text());assert source['source_commit']=='d6761c895ef6fa38d882bb2d4ca3f3205c1a1fa3'
with zipfile.ZipFile(root/'source/physical_source_overlay.zip') as z:
 safety.checked_members(z);assert set(z.namelist())==set(source['files'])
 for name,expected in source['files'].items():
  data=z.read(name);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==expected
  if name.endswith('.py'):ast.parse(data,feature_version=(3,8))
assert identity(root/'source/pinned_asset_base.zip')=={k:source['base_full_zip'][k] for k in ('bytes','sha256')}
plan=json.loads((root/'operator/study_plan.json').read_text())
for name,expected in plan['operator_identities'].items():assert identity(root/'operator'/name)==expected,name
binding=json.loads((root/'records/runtime_binding.json').read_text())
assert binding['plan']==plan and binding['asset_manifest']==identity(root/'source/asset_manifest.json')
assert binding['setup']==identity(root/'setup/cli_setup_verification.json')
asset=json.loads((root/'source/asset_manifest.json').read_text())
assert asset['source_overlay_receipt']==source and asset['source_snapshot_sha256']==source['overlay']['sha256']
collection=json.loads((root/'collection.json').read_text())
for name,expected in collection['native_payloads'].items():
 path=root/'native_payload'/name.lstrip('/');assert identity(path)==expected and path.read_bytes()[:4]==b'\x7fELF'
cache_report=json.loads((root/'records/cache_preparation.json').read_text())
assert cache_report['status']=='complete' and len(cache_report['records'])==126
assert len({(r['corruption'],tuple(r['ids'])) for r in cache_report['records']})==126
for row in cache_report['records']:
 path=root/'paired_states/modelnet40_c'/row['corruption']/Path(row['path']).name
 assert identity(path)=={k:row[k] for k in ('bytes','sha256')}
assert len(list((root/'paired_states').rglob('*.npz')))==collection['paired_cache_count']==126
assert len(list((root/'fresh_states').rglob('*.npz')))==collection['fresh_cache_count']
canonical=REPO/'result/modelnet40_c/decoded_physical_spectral_diagnostic';canonical.mkdir(parents=True,exist_ok=True)
archives=[]
for path in sorted((root/'raw_runs').rglob('*.zip')):
 target=canonical/path.name
 with path.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
 assert identity(path)==identity(target)
 archives.append(dict(package_path=str(path.relative_to(root)),canonical_path=str(target),**identity(path)))
report=dict(status='raw bytes accepted; scientific analysis separate',package=identity(args.archive),source_commit=source['source_commit'],asset_manifest=identity(root/'source/asset_manifest.json'),
 native_payload_count=len(collection['native_payloads']),paired_caches=126,fresh_caches=collection['fresh_cache_count'],archive_failures=collection['archive_failures'],raw_archives=archives)
with (root.parent/'local_acceptance.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_archives'},indent=2))
