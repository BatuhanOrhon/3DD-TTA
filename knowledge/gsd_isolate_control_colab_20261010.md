# Legacy / isolate correction control — 2026-10-10

Run cells in order in the existing Colab GPU runtime with `3dd_tta_env`, data,
checkpoints and native extensions already prepared. Repository:
`/content/3DD-TTA`, branch `gsd-smooth-spectrum`, base `c18926f`.
No Git reset, checkout, install or push is performed here.

Upload the accompanying `legacy_isolate_control_20261010.zip`, not the Oct8 ZIP.
Cell 1 accepts an absent overlay, identical files, or the exact known Oct8
overlay (backed up before upgrading). Unknown existing edits stop the cell.

[Code] The independent variable is only isolate detection: self-only nodes
receive the existing 1000 penalty with `offdiag`. Raw-eigenvalue weighting convention, M800,
beta2, spectral1.17, SCD1, batch70, steps10/30, updated final style and DDIM
alpha boundary are identical. Count nonzero nonself edges; no epsilon added.

[Open] Seed0 is new and explicit; the historical run's seed is unknown.
These are seed-controlled separate runs, not hash-proven prepared-input pairs.
Each full arm has 37020 examples; two arms have 74040, plus 560 smoke examples.
Smoke checks integration, not accuracy. A single seed cannot establish a
reliable +1–2pp gain. Do not silently reduce batch70 after OOM.

On failure, retain the failed ZIP and use Cell 7 to download available evidence.
Do not rerun into existing output paths. Keep the session directory if the
runtime disconnects; Cell 2 refuses to overwrite its initialized session.
No parameter search or seed1/2 campaign is included.

## 1 — Source upload and verification (CPU)

```python
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, zipfile
from google.colab import files

repo = Path('/content/3DD-TTA')
uploaded = files.upload()  # Select ONLY legacy_isolate_control_20261010.zip
assert len(uploaded) == 1
payload_zip = next(iter(uploaded.values()))
assert hashlib.sha256(payload_zip).hexdigest() == 'baa6674a55d68e1f42c70ac1ad4a2468bee4b6bc636e9a491e0ace9ab3e82ee0'
import io
expected = {'eval_gsd_tta_v2.py', 'tta_gsd_v2.py', 'graph_spectral_v2.py',
            'graph_spectral_v2_offdiag.py', 'legacy_gsd_artifacts.py'}
with zipfile.ZipFile(io.BytesIO(payload_zip)) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == expected | {'manifest.json'}
    manifest = json.loads(z.read('manifest.json'))
    assert set(manifest['files']) == expected
    payload = {name: z.read(name) for name in expected}
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip() == manifest['base_commit']
assert subprocess.check_output(['git','-C',str(repo),'branch','--show-current'], text=True).strip() == 'gsd-smooth-spectrum'
subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD','--','*.py'], check=True)
backup = {}
for name, data in payload.items():
    info, target = manifest['files'][name], repo / name
    assert hashlib.sha256(data).hexdigest() == info['sha256']
    assert len(data) == info['bytes'] and not target.is_symlink()
    if target.exists() and target.read_bytes() != data:
        prior = target.read_bytes()
        assert hashlib.sha256(prior).hexdigest() in info['accepted_previous_sha256'], f'Unknown edit: {name}'
        backup[name] = prior
if backup:
    backup_dir = repo / 'result/preparation' / ('isolate_source_backup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name, data in backup.items():
        (backup_dir / name).write_bytes(data)
    print('Previous overlay backed up:', backup_dir)
for name, data in payload.items():
    target = repo / name
    if not target.exists() or name in backup:
        target.write_bytes(data)
    assert target.read_bytes() == data
print('Source verified. No inference started.')
```

## 2 — Locked parameters and preview (CPU)

```python
# Run once per session. Do not change the common parameters between arms.
assert 'session' not in globals(), 'Session already initialized; keep existing paths.'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
session = repo / 'result/modelnet40_c/legacy_isolate_control' / stamp
session.mkdir(parents=True, exist_ok=False)
(session / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
common = ['conda','run','--no-capture-output','-n','3dd_tta_env',
          'python','eval_gsd_tta_v2.py',
          '--batch_size','70','--weight_spectral','1.17','--weight_chamfer','1.0',
          '--M_max','800','--beta','2.0','--gamma','0.01','--eta','0.01',
          '--lambdaa','0.95','--denoising_step_bg','30','--denoising_step_normal','10',
          '--dataset_root','./data/modelnet40_c','--label_path','./data/modelnet40_c/label.npy',
          '--csv_name','eval_results.csv','--seed','0']
policies = ('legacy', 'offdiag')
def launch(policy, tag, *extra):
    for name, info in manifest['files'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == info['sha256']
    output = session / (policy + '_' + tag + '_seed0')
    subprocess.run(common + ['--isolate_policy',policy,'--output_dir',str(output), *extra], cwd=repo, check=True)
    return output
def completed(path, n, status):
    record = json.loads((path/'status.json').read_text())
    assert record['execution_status'] == 'complete' and record['status'] == status, record
    assert record['completed_examples'] == n, record
for policy in policies:
    subprocess.run(common + ['--isolate_policy',policy,'--preview'], cwd=repo, check=True)
print('SESSION:', session)
```

## 3 — Gaussian + Background smoke (GPU)

```python
# GPU: 4 short runs, 2 batches each. All must finish before full evaluation.
for policy in policies:
    for corruption in ('gaussian', 'background'):
        path = launch(policy, 'smoke_' + corruption,
                      '--corruption',corruption,'--max_batches','2')
        completed(path, 140, 'partial')
print('Both policies passed Gaussian and Background integration checks.')
```

## 4 — Legacy all15 (GPU)

```python
# GPU: original behavior, all15, 2468 examples/corruption, seed0.
for policy in policies:
    for corruption in ('gaussian', 'background'):
        completed(session/(policy+'_smoke_'+corruption+'_seed0'), 140, 'partial')
legacy_dir = launch('legacy', 'all15')
completed(legacy_dir, 37020, 'complete')
```

## 5 — Corrected isolate all15 (GPU)

```python
# GPU: only isolate detection changes; keep the same runtime and parameters.
completed(session/'legacy_all15_seed0', 37020, 'complete')
fixed_dir = launch('offdiag', 'all15')
completed(fixed_dir, 37020, 'complete')
```

## 6 — Preliminary comparison (CPU)

```python
# CPU: preliminary count-based summary, not a full independent archive audit.
import csv
tables, configs = {}, {}
for policy in policies:
    path = session / (policy + '_all15_seed0')
    completed(path, 37020, 'complete')
    configs[policy] = json.loads((path/'config.json').read_text())
    with (path/'per_corruption.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15 and len({r['corruption'] for r in rows}) == 15
    assert set(r['corruption'] for r in rows) == set(configs[policy]['corruptions'])
    for r in rows:
        assert r['status'] == 'complete' and int(r['n_examples']) == 2468
        assert 0 <= int(r['n_correct']) <= 2468
    tables[policy] = {r['corruption']:100*int(r['n_correct'])/2468 for r in rows}
# Reject changed assets, native builds, common source identities or method settings.
a, b = configs['legacy'], configs['offdiag']
for key in ('assets','native_extensions','external_python_sources','model_modes','scheduler'):
    assert a[key] == b[key], 'Provenance mismatch: ' + key
assert a['source_manifest'] == b['source_manifest'], 'Source identity mismatch'
for key in a['cli']:
    if key not in {'isolate_policy','output_dir'}:
        assert a['cli'][key] == b['cli'][key], key
print(f'{"corruption":20s} {"legacy %":>10s} {"offdiag %":>10s} {"delta pp":>10s}')
lines = ['corruption,legacy_accuracy_pct,offdiag_accuracy_pct,delta_pp']
for name in tables['legacy']:
    x, y = tables['legacy'][name], tables['offdiag'][name]
    print(f'{name:20s} {x:10.4f} {y:10.4f} {y-x:+10.4f}')
    lines.append(f'{name},{x:.8f},{y:.8f},{y-x:.8f}')
for label, names in [('macro15', list(tables['legacy'])),
                     ('macro14', [n for n in tables['legacy'] if n != 'background'])]:
    x, y = (sum(tables[p][n] for n in names)/len(names) for p in policies)
    print(f'{label:20s} {x:10.4f} {y:10.4f} {y-x:+10.4f}')
    lines.append(f'{label},{x:.8f},{y:.8f},{y-x:.8f}')
with (session/'comparison_seed0.csv').open('x') as f:
    f.write('\n'.join(lines)+'\n')
print('One seed; uncertainty and independent prediction audit remain pending.')
```

## 7 — Download full evidence (CPU)

```python
# CPU: can also run after a failure; includes any sealed partial/failed ZIPs.
bundles = sorted(session.glob('*.zip'))
assert bundles, 'No sealed run archive exists yet; keep the run directory and traceback.'
handoff = session.parent / (session.name + '_handoff.zip')
with zipfile.ZipFile(handoff, 'x', zipfile.ZIP_STORED) as z:
    for path in bundles + sorted(session.glob('*.json')) + sorted(session.glob('*.csv')):
        z.write(path, path.name)
print('Archive:', handoff, 'bytes:', handoff.stat().st_size)
files.download(str(handoff))
# Return this complete ZIP for independent analysis, including failed runs.
```
