"""Recount the historical decoder-only control; never runs LION."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path('/Users/batuhan/Desktop/projects/3DD-TTA/result/modelnet40_c/shared_trajectory_decoder_control')
EXPECTED = {
    '20260920-131504_shared-decoder-s5-all15-seed0.zip': 'd2cefdb57a91b1d1beefc444a8475e19f55bd2e27f890033fd3d6f5644f5080f',
    '20260920-134840_shared-decoder-s5-all15-seed1.zip': 'bfafbe4c581da659bf46db34582ca8fa2ac4c733e1202286e490676cc4194fce',
    '20260920-142217_shared-decoder-s5-all15-seed2.zip': 'a826c867595598b2dadf2031dd964c6e265ecce9f23a7dc6f9a33f7b8917e8b0',
}
records, deltas = [], []
for name, expected in EXPECTED.items():
    path = ROOT / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, name
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        members = [n for n in archive.namelist() if n.endswith('/per_corruption.csv')]
        assert len(members) == 1
        rows = list(csv.DictReader(io.StringIO(archive.read(members[0]).decode())))
    assert len(rows) == 15 and len({r['corruption'] for r in rows}) == 15
    assert all(r['status'] == 'complete' and int(r['n_examples']) == 2468 for r in rows)
    original, updated = [], []
    for row in rows:
        n = int(row['n_examples'])
        a = int(row['original_style_n_correct']) / n
        b = int(row['updated_style_n_correct']) / n
        assert abs(a - float(row['original_style_accuracy'])) < 1e-12
        assert abs(b - float(row['updated_style_accuracy'])) < 1e-12
        assert abs(100 * (b - a) - float(row['paired_delta_pp'])) < 1e-10
        original.append(a)
        updated.append(b)
        deltas.append(b - a)
    records.append({'path': str(path), 'sha256': digest, 'rows': rows,
                    'original_macro_percent': 100 * statistics.mean(original),
                    'updated_macro_percent': 100 * statistics.mean(updated)})
result = {'status': 'passed', 'archives': records,
          'original_three_seed_macro_percent': statistics.mean(r['original_macro_percent'] for r in records),
          'updated_three_seed_macro_percent': statistics.mean(r['updated_macro_percent'] for r in records),
          'delta_pp': 100 * statistics.mean(deltas),
          'improved_rows': sum(d > 0 for d in deltas),
          'worse_rows': sum(d < 0 for d in deltas),
          'tied_rows': sum(d == 0 for d in deltas),
          'limits': 'Decoder-only intervention, not a global-prior experiment. No per-example predictions in these CSVs; no new CI inferred.'}
Path(__file__).with_name('decoder_control_recount.json').write_text(json.dumps(result, indent=2) + '\n')
print({k: v for k, v in result.items() if k != 'archives'})
