"""CPU evidence collection only; retain exact cached preparation/noise bytes."""
import hashlib
import json
import sys
from pathlib import Path

phases = Path('/content/studies/guidance-components/phases')
assert hashlib.sha256((phases / 'deploy_assets_template.py').read_bytes()).hexdigest() == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0, str(phases))
from deploy_assets_template import ROOT, DRIVE, MANIFEST_SHA, SESSION, verify, copy_checked, seal, utc

receipt = json.loads((ROOT / 'development_phase_receipt.json').read_text())
validation = json.loads((ROOT / 'development_validation.json').read_text())
ledger = json.loads((ROOT / 'initial_gpu_phase_ledger.json').read_text())
assert ledger['active'] is None and ledger['session'] == SESSION
assert validation['status'] == 'validated' and receipt['state']['status'] == 'complete'
assert receipt['state']['manifest_sha256'] == MANIFEST_SHA
verify(receipt['archive']['drive_path'], receipt['archive'])
records = {}
for job in receipt['state']['jobs']:
    assert job['status'] == 'complete'
    for record in job['cache']:
        key = record['local_path']
        assert key not in records or records[key] == record
        records[key] = record
assert len(records) == 30
output = ROOT / ('paired_cache_evidence_' + utc())
output.mkdir(exist_ok=False)
for key, record in sorted(records.items()):
    path = Path(key)
    relative = path.relative_to(ROOT / 'paired_state_v1_seed0')
    assert path.suffix == '.npz' and not path.is_symlink() and '..' not in relative.parts
    assert Path(record['drive_path']).relative_to(DRIVE / 'paired_state_v1_seed0') == relative
    verify(record['drive_path'], record)
    copy_checked(path, output / 'paired_state_v1_seed0' / relative, record)
(output / 'cache_records.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
(output / 'development_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
archive = seal(output, DRIVE / 'phase-evidence' / output.name / 'paired_cache_evidence.zip')
result = dict(status='verified', session=SESSION, manifest_sha256=MANIFEST_SHA, cache_count=30, archive=archive)
(ROOT / 'paired_cache_evidence_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
(Path(archive['drive_path']).parent / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print('AGENT_B_PAIRED_CACHE_EVIDENCE', json.dumps(archive), flush=True)
print('AGENT_B_PAIRED_CACHE_COLLECTION_COMPLETE', flush=True)
