# Git / conda continuation: legacy isolate control — 2026-10-10

Run the first **six code cells** of `gsd_guidance_ablations_colab.ipynb` first.
Then run these cells in the same Colab GPU runtime. Cell7 switches from dev to
gsd-smooth-spectrum. No source ZIP upload is needed after the new commit is
pushed. All previews, experiment subprocesses, analysis and packaging execute
inside **3dd_tta_env**; the notebook kernel only dispatches and downloads.

Two policies: unchanged legacy versus only nonself isolate detection corrected.
Fixed B70/M800/beta2/spectral1.17/SCD1/gamma=eta=.01/lambda=.95/steps10-30,
updated final style, same raw-eigenvalue weighting convention and DDIM boundary.
The correction changes the isolated-node eigenvalues; it does not change beta.
Full seed0 evaluations: 37020 examples per policy; 74040 total, plus560 smoke.

[Open] These are seed-controlled separate runs, not prepared-input hash-proven
pairs. Each evaluation command is launched as `conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta_v2.py ...`.
The Cell4 PointNet2 build changes the tracked generated helper
`Pointnet2_PyTorch/pointnet2_ops_lib/build/lib.linux-x86_64-cpython-38/pointnet2_ops/pointnet2_utils.py`.
Cell7 and run guards ignore that exact file while rejecting other changed
Python sources. The run guard permits a newer descendant commit when all five
experiment source hashes remain identical. Unknown historical seed/build
prevents bitwise historical reproduction.
No new accuracy evidence or +1–2pp guarantee exists. Do not lower batch70 after
OOM without declaring a different protocol. Keep failed ZIPs; Cell13 also works
after failed runs. Retain the SESSION path on disconnect. Full runs never
overwrite an existing directory; do not rerun Cell8 to conceal a failed attempt.

Return the complete handoff ZIP, including logs and predictions, for independent
analysis. No seed1/2 campaign or parameter sweep is automatically scheduled.

## Cell 7 - Branch and conda verification

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git fetch origin
git switch gsd-smooth-spectrum
git merge --ff-only origin/gsd-smooth-spectrum
git merge-base --is-ancestor c18926ff7751ed9023313869229f5d4dffe218be HEAD
git diff --quiet HEAD -- '*.py' ':(exclude,glob)Pointnet2_PyTorch/pointnet2_ops_lib/build/lib.linux-x86_64-cpython-38/pointnet2_ops/pointnet2_utils.py'
git ls-files --error-unmatch eval_gsd_tta_v2.py graph_spectral_v2_offdiag.py legacy_gsd_artifacts.py
conda run --no-capture-output -n 3dd_tta_env python -c "import sys, torch; print(sys.executable); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Cell 8 - Session and previews (conda)

```python
# Notebook kernel only dispatches commands; all experiment/analysis code runs in conda.
from pathlib import Path
from datetime import datetime, timezone
import subprocess

assert 'SESSION' not in globals(), 'Session already initialized; keep the existing paths.'
REPO = Path('/content/3DD-TTA')
SESSION = REPO / 'result/modelnet40_c/legacy_isolate_control' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
CONDA = ['conda','run','--no-capture-output','-n','3dd_tta_env']
CONTEXT = r"""
from pathlib import Path
import hashlib, json, subprocess, sys
repo = Path('/content/3DD-TTA')
session = Path(SESSION_PATH)
policies = ('legacy', 'offdiag')
common = ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env',
          'python', 'eval_gsd_tta_v2.py',
          '--batch_size','70','--weight_spectral','1.17','--weight_chamfer','1.0',
          '--M_max','800','--beta','2.0','--gamma','0.01','--eta','0.01',
          '--lambdaa','0.95','--denoising_step_bg','30','--denoising_step_normal','10',
          '--dataset_root','./data/modelnet40_c','--label_path','./data/modelnet40_c/label.npy',
          '--csv_name','eval_results.csv','--seed','0']
def launch(policy, tag, *extra):
    manifest = json.loads((session/'source_manifest.json').read_text())
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',manifest['commit'],head], cwd=repo, check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--','*.py',
                    ':(exclude,glob)Pointnet2_PyTorch/pointnet2_ops_lib/build/lib.linux-x86_64-cpython-38/pointnet2_ops/pointnet2_utils.py'],
                   cwd=repo, check=True)
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest, name
    output = session / (policy + '_' + tag + '_seed0')
    subprocess.run(common + ['--isolate_policy',policy,'--output_dir',str(output), *extra], cwd=repo, check=True)
    return output
def completed(path, n, status):
    record = json.loads((path/'status.json').read_text())
    assert record['execution_status'] == 'complete' and record['status'] == status, record
    assert record['completed_examples'] == n, record
""".replace('SESSION_PATH', repr(str(SESSION)))

def conda_python(source):
    subprocess.run(CONDA + ['python','-c',CONTEXT + '\n' + source], cwd=REPO, check=True)

conda_python(r"""
session.mkdir(parents=True, exist_ok=False)
names = ('eval_gsd_tta_v2.py', 'tta_gsd_v2.py', 'graph_spectral_v2.py',
         'graph_spectral_v2_offdiag.py', 'legacy_gsd_artifacts.py')
manifest = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
                files={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names})
(session/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
for policy in policies:
    subprocess.run(common + ['--isolate_policy',policy,'--preview'], cwd=repo, check=True)
print('SESSION:', session)
""")
```

## Cell 9 - Gaussian/Background smoke (conda/GPU)

```python
conda_python(r"""
# GPU: 4 short runs, 2 batches each. All must finish before full evaluation.
for policy in policies:
    for corruption in ('gaussian', 'background'):
        path = launch(policy, 'smoke_' + corruption,
                      '--corruption',corruption,'--max_batches','2')
        completed(path, 140, 'partial')
print('Both policies passed Gaussian and Background integration checks.')
""")
```

## Cell 10 - Legacy all15 seed0 (conda/GPU)

```python
conda_python(r"""
# GPU: original behavior, all15, 2468 examples/corruption, seed0.
for policy in policies:
    for corruption in ('gaussian', 'background'):
        completed(session/(policy+'_smoke_'+corruption+'_seed0'), 140, 'partial')
legacy_dir = launch('legacy', 'all15')
completed(legacy_dir, 37020, 'complete')
""")
```

## Cell 11 - Offdiag all15 seed0 (conda/GPU)

```python
conda_python(r"""
# GPU: only isolate detection changes; keep the same runtime and parameters.
completed(session/'legacy_all15_seed0', 37020, 'complete')
fixed_dir = launch('offdiag', 'all15')
completed(fixed_dir, 37020, 'complete')
""")
```

## Cell 12 - Count/provenance comparison (conda/CPU)

```python
conda_python(r"""
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
""")
```

## Cell 13 - Full evidence download

```python
conda_python(r"""
import zipfile
from datetime import datetime, timezone
bundles = sorted(session.glob('*.zip'))
assert bundles, 'No sealed run ZIP yet; retain the run directory and traceback.'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
handoff = session.parent / (session.name + '_handoff_' + stamp + '.zip')
with zipfile.ZipFile(handoff, 'x', zipfile.ZIP_STORED) as z:
    for path in bundles + sorted(session.glob('*.json')) + sorted(session.glob('*.csv')):
        z.write(path, path.name)
(session/'last_handoff.txt').write_text(str(handoff))
print('ZIP:', handoff, 'bytes:', handoff.stat().st_size)
""")

from google.colab import files
files.download((SESSION/'last_handoff.txt').read_text().strip())
```
