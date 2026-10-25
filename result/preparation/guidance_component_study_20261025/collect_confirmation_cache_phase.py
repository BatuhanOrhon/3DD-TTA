"""CPU-only immutable collection of the three completed confirmation caches."""
import hashlib
import json
import sys
from pathlib import Path

phases = Path('/content/studies/guidance-components/confirmation_phases')
assert hashlib.sha256((phases / 'deploy_assets_template.py').read_bytes()).hexdigest() == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0, str(phases))
from deploy_assets_template import ROOT, DRIVE, MANIFEST_SHA, SESSION, verify, copy_checked, seal, utc

receipt = json.loads((ROOT / 'frozen_confirmation_receipt.json').read_text())
validation = json.loads((ROOT / 'frozen_confirmation_validation.json').read_text())
ledger = json.loads((ROOT / 'frozen_confirmation_ledger.json').read_text())
assert ledger['active'] is None and ledger['session'] == SESSION
assert validation['status'] == 'validated' and receipt['state']['status'] == 'complete'
assert receipt['state']['manifest_sha256'] == MANIFEST_SHA
assert receipt['state']['frozen_plan_sha256'] == 'c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45'
assert len(receipt['state']['jobs']) == 12
verify(receipt['archive']['drive_path'], receipt['archive'])
records = {}
for job in receipt['state']['jobs']:
    assert job['status'] == 'complete' and type(job['seed']) is int and job['seed'] in (0, 1, 2)
    for record in job['cache']:
        key = record['local_path']
        assert key not in records or records[key] == record
        records[key] = record
assert len(records) == 90
output = ROOT / ('confirmation_cache_evidence_' + utc())
output.mkdir(exist_ok=False)
counts = {seed: 0 for seed in (0, 1, 2)}
for key, record in sorted(records.items()):
    path = Path(key)
    relative = path.relative_to(ROOT)
    matching = [seed for seed in counts if relative.parts[0] == 'paired_confirmation_heldout_v1_seed%d' % seed]
    assert len(matching) == 1 and path.suffix == '.npz' and not path.is_symlink() and '..' not in relative.parts
    assert Path(record['drive_path']).relative_to(DRIVE) == relative
    verify(record['drive_path'], record)
    copy_checked(path, output / relative, record)
    counts[matching[0]] += 1
assert set(counts.values()) == {30}
(output / 'cache_records.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
(output / 'frozen_confirmation_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
archive = seal(output, DRIVE / 'phase-evidence' / output.name / 'confirmation_cache_evidence.zip')
result = dict(status='verified', session=SESSION, manifest_sha256=MANIFEST_SHA, cache_count=90, per_seed_cache_count=counts, archive=archive)
(ROOT / 'confirmation_cache_evidence_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
(Path(archive['drive_path']).parent / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print('AGENT_B_CONFIRMATION_CACHE_EVIDENCE', json.dumps(archive), flush=True)
print('AGENT_B_CONFIRMATION_CACHE_COLLECTION_COMPLETE', flush=True)
