# Phase 6: frozen P_PC versus C_SCD on all15 - 2026-10-06

[User report/Decision] The user explicitly authorized preparing the full15
comparison after Phase5 passed. This is two arms,15 corruptions,2468 indices
and seeds0/1/2:45 paired blocks,90 arm ZIPs,222120 classifications. GPU
execution remains user-operated in Colab. No algorithm or inference source
change is needed; `all15` and strict replication-reference checks already exist.

[Run/Code] Phase5 ZIP SHA-256 is
`028d31bec666bf0bd0e4427e7a64799ade78f2b3f3eb2e283b0285a6ed1e3a29`.
The new selection's source is `replicate`; canonical P_PC and only C_SCD
are selected. Selection SHA-256:
`33e39ba3f601451122581fe335fd0810704c87f6abae42b4b3c7e75e857e29b7`.
The old Phase5 selection remains unchanged. The helper makes a separate
reference manifest with relocated ZIP paths, because the existing runner
requires readable `bundle_path` entries and the downloaded archive contains
arm ZIPs instead of extracted arm directories. Original manifests are unchanged.

[Run/Inference] Existing three full-screen SCD archives have no matching
composition prepared-input/component identities. No full all15 composition
attempt exists in the current result inventory. New paired C_SCD is required
to compare with new P_PC. Historical controls remain descriptive evidence;
do not rerun other algorithms, pilots or failed Background100/1000 scale arms.

## Frozen reporting scope

- Primary: equal-weight macro15 P_PC-C_SCD, each seed and mean/sample SD.
- Report macro14 without Background, Background and every corruption;
  paired corrected/broken/disagreement counts and per-corruption uncertainty.
- Cluster seed repeats within original indices; no pooled cross-corruption
  bootstrap until object correspondence is established. Timestep rows are
  not accuracy examples. Include raw counts, runtime and peak memory.
- Fixed model, graph, DDIM5/35, B32, gamma/eta.01, lambda.95 and original
  decode. No tuning after inspecting outcomes. This is full-test-set
  development evaluation, not independent confirmation. Full-scope mechanism
  claims require full-scope P_SUM/P_NORM, which are outside this two-arm run.
- No new numeric promotion threshold is invented here. Report magnitude,
  seed directions and uncertainty; a nonpositive macro15 contrast or mixed
  seed directions would contradict a broadly stable benefit interpretation.

## Cell 1 - create an isolated Phase6 workspace and inspect the environment

The earlier cell stopped on tracked edits. The follow-up also showed that the
old checkout has no `scripts/` directory. This cell leaves the existing
checkout untouched and creates a detached worktree from the pushed Phase6
commit. It links the required data, checkpoints, compiled Chamfer extension
and result directory. The nine inference files are checked against the
completed Phase5 commit. A mismatch is printed and stops execution.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git fetch origin gsd-smooth-spectrum
PHASE_COMMIT=b7d095a68cd9ab95294bce86cbd1a1f94cb87cc4
PHASE_REPO=/content/3DD-TTA-phase6
git cat-file -e "$PHASE_COMMIT^{commit}"

if test -e "$PHASE_REPO"; then
  test "$(git -C "$PHASE_REPO" rev-parse HEAD)" = "$PHASE_COMMIT"
else
  git worktree add --detach "$PHASE_REPO" "$PHASE_COMMIT"
fi

echo "Original checkout retained: $(git branch --show-current) / $(git rev-parse --short HEAD)"
git status --short

conda run --no-capture-output -n 3dd_tta_env python - <<'PY'
import hashlib, os, subprocess, sys, numpy, torch, diffusers, scipy
from pathlib import Path

original = Path('/content/3DD-TTA')
workspace = Path('/content/3DD-TTA-phase6')
os.chdir(workspace)
from research_artifacts import CORRUPTIONS
links = [
    ('data/modelnet40_c', 'data/modelnet40_c'),
    ('lion_ckpts', 'lion_ckpts'),
    ('pointnet_ckpts', 'pointnet_ckpts'),
    ('result/modelnet40_c/gsd_guidance_composition_v1',
     'result/modelnet40_c/gsd_guidance_composition_v1'),
    ('third_party/ChamferDistancePytorch/tmp/chamfer_3D.so',
     'third_party/ChamferDistancePytorch/tmp/chamfer_3D.so'),
]
for source_name, target_name in links:
    source, target = original / source_name, workspace / target_name
    assert source.exists(), 'Missing Colab asset: ' + str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source.resolve(), 'Unexpected link: ' + str(target)
    elif target.exists():
        if target.is_dir():
            assert not any(target.iterdir()), 'Preserving nonempty path: ' + str(target)
            target.rmdir()
        else:
            assert target.read_bytes() == source.read_bytes(), 'Unexpected file: ' + str(target)
            continue
        target.symlink_to(source, target_is_directory=source.is_dir())
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
    print('Linked:', target_name)

run_commit = '86fb15544f1a82a31afccd99427bb39f3a3c53e4'
names = ['scripts/run_gsd_composition.py', 'gsd_composition_protocol.py',
         'gsd_composition.py', 'gsd_paired_inputs.py', 'tta_gsd.py', 'tta.py',
         'graph_spectral.py', 'run_baseline.py', 'main_3dd_tta.py']
changed = []
for name in names:
    expected = subprocess.check_output(['git', 'show', run_commit + ':' + name], cwd=workspace)
    path = workspace / name
    if not path.is_file() or path.read_bytes() != expected:
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'MISSING'
        changed.append((name, hashlib.sha256(expected).hexdigest(), actual))
if changed:
    for name, wanted, actual in changed:
        print('Inference source differs:', name, 'expected', wanted, 'found', actual)
    raise SystemExit('Phase5 source check failed; original checkout remains unchanged.')
print('All nine inference files match the Phase5 run commit.')

assert '3dd_tta_env' in sys.executable
assert torch.cuda.is_available(), 'Select a Colab GPU runtime first'
print('Python:', sys.executable)
print('NumPy:', numpy.__version__, 'Torch:', torch.__version__)
print('Diffusers:', diffusers.__version__, 'SciPy:', scipy.__version__)
print('GPU:', torch.cuda.get_device_name())
for name in ['data/modelnet40_c/label.npy', 'cfgs/tta_modelnet.yaml',
             'pointnet_ckpts/modelnet_jt.pth',
             'lion_ckpts/unconditional_all55_cfg.yml',
             'lion_ckpts/epoch_10999_iters_2100999.pt']:
    assert Path(name).is_file(), 'Missing asset: ' + name
for corruption in CORRUPTIONS:
    assert Path('data/modelnet40_c/data_' + corruption + '_5.npy').is_file(), corruption
print('Environment and required asset paths: OK')
PY
```

## Cell 2 - reference recovery, frozen selection and CPU preview

Normal notebook Python cell, without `%%bash`. It reuses the original Phase5
directory/ZIP if present; otherwise upload the accepted `attempt-0001.zip`.
The hash must match the accepted artifact. This cell creates preparation
files only, no `all15/attempt-*` directory and no inference.

```python
import os, json, subprocess, hashlib
from pathlib import Path

repo = Path('/content/3DD-TTA-phase6')
os.chdir(repo)
root = repo / 'result/modelnet40_c/gsd_guidance_composition_v1'
reference = root / 'replicate/attempt-0001'
if not ((reference / 'phase_manifest.json').is_file()
        and len(list((reference / 'arms').glob('*.zip'))) == 48):
    reference = root / 'replicate/attempt-0001.zip'
    if not reference.is_file():
        from google.colab import files
        print('Upload the validated Phase 5 attempt-0001.zip')
        uploaded = files.upload()
        assert len(uploaded) == 1, 'Upload exactly one Phase 5 ZIP'
        data = next(iter(uploaded.values()))
        expected = '028d31bec666bf0bd0e4427e7a64799ade78f2b3f3eb2e283b0285a6ed1e3a29'
        assert hashlib.sha256(data).hexdigest() == expected, 'Wrong Phase 5 ZIP'
        reference = Path('/content/phase5_' + expected + '.zip')
        if reference.exists():
            assert reference.read_bytes() == data
        else:
            with reference.open('xb') as out:
                out.write(data)
        del data, uploaded
subprocess.run(['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env',
                'python', 'scripts/gsd_all15_colab.py', 'prepare',
                '--reference', str(reference)], cwd=repo, check=True)
preview = json.loads((root / 'all15/preparation/preview.json').read_text())
assert preview['arms'] == ['C_SCD', 'P_PC']
assert (preview['paired_blocks'], preview['arm_archives'],
        preview['classifications']) == (45, 90, 222120)
assert not preview['inference_executed']
print('READY: 45 paired blocks / 90 arm ZIPs / 222,120 classifications')
print('Only the next cell starts GPU inference.')
```

## Cell 3 - user-started GPU execution

Run only after reviewing Cell2. Existing attempt/log blocks accidental
re-execution. A failed/interrupted run is still packaged using Cell4.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA-phase6
ROOT=result/modelnet40_c/gsd_guidance_composition_v1
ATTEMPT="$ROOT/all15/attempt-0001"
LOG="$ROOT/all15/attempt-0001_colab_execution.log"
test -f "$ROOT/all15/preparation/preview.json"
test ! -e "$ATTEMPT" || { echo 'Attempt already exists: do not rerun; use Cell 4.'; exit 2; }
test ! -e "$LOG" || { echo 'Execution log already exists: inspect before launching.'; exit 2; }
conda run --no-capture-output -n 3dd_tta_env python - <<'PY' 2>&1 | tee "$LOG"
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path('result/modelnet40_c/gsd_guidance_composition_v1/all15/preparation/preview.json')
preview = json.loads(p.read_text())
assert preview['classifications'] == 222120
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == preview['resolved_ref']['commit']
for name, identity in preview['resolved_ref']['source_manifest'].items():
    data = Path(name).read_bytes()
    assert len(data) == identity['bytes']
    assert hashlib.sha256(data).hexdigest() == identity['sha256'], name
raise SystemExit(subprocess.call([sys.executable, 'scripts/run_gsd_composition.py',
                                  *preview['runner_argv'], '--execute']))
PY
```

## Cell 4 - audit, analyze, package and download all available output

**Normal notebook Python cell. Do not add `%%bash`.** The audit/analyzer
run in `3dd_tta_env`. ZIP creation is independent of audit success. The kernel
calls `files.download` only after package CRC/SHA validation. A failed result
is clearly named `UNVALIDATED`; it is never reported as complete accuracy.
Each invocation has unique receipt/archive names, so an old download cannot
be selected silently. Byte-identical extracted arm copies are omitted;
partial or divergent raw files are retained. Raw Colab files are unchanged.

```python
import hashlib, json, subprocess, uuid
from pathlib import Path
from google.colab import files

repo = Path('/content/3DD-TTA-phase6')
root = repo / 'result/modelnet40_c/gsd_guidance_composition_v1'
attempt = root / 'all15/attempt-0001'
log = root / 'all15/attempt-0001_colab_execution.log'
assert attempt.is_dir(), 'No attempt directory exists; inspect Cell 3 output.'
receipt = Path('/content/gsd_all15_download_' + uuid.uuid4().hex + '.json')
command = ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env',
           'python', 'scripts/gsd_all15_colab.py', 'finalize',
           '--attempt', str(attempt), '--receipt', str(receipt), '--log', str(log)]
result = subprocess.run(command, cwd=repo)
if result.returncode != 0 or not receipt.is_file():
    raise RuntimeError('Packaging did not finish. Raw files remain intact; inspect the output above.')
info = json.loads(receipt.read_text())
archive = Path(info['archive'])
assert archive.is_file() and archive.stat().st_size == info['bytes']
digest = hashlib.sha256()
with archive.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == info['sha256'], 'Package hash mismatch'
print('Package:', archive)
print('Size: %.1f MiB' % (info['bytes'] / 1024**2))
print('SHA-256:', info['sha256'])
if info['validated_complete'] and info['analysis_complete']:
    print('VALIDATED: 90 complete arms, 45 paired blocks, 222,120 classifications.')
else:
    print('UNVALIDATED / PARTIAL: raw output packaged; no complete accuracy claim.')
    print('Details:', info['errors'])
try:
    files.download(str(archive))
    print('Browser download requested. Confirm completion in your browser.')
except Exception as error:
    print('Browser download could not start:', error)
    print('Package is intact. In Colab Files, locate the path above and choose Download.')
```

Browser delivery requires an active Colab notebook session and a browser
that permits downloads. A transport failure does not erase the verified ZIP;
download the printed file from Colab's Files panel without rerunning inference.
Return the entire `_all15_*.zip` to the repository `all15/` directory for
offline ingestion, including logs and any failure/partial artifacts. Never
upload credentials or only a screenshot of accuracy.

## Local preparation evidence

[Code/Verification] CPU preview on the accepted48 ZIPs confirms45 blocks,
90 arms and222120 classifications;78 batches per block, last batch4 examples.
Python3.8 syntax parses. The packaging function was exercised on the existing
Phase5 artifact: all51 original entries, including48 arm ZIPs, were preserved
byte-for-byte, and the output CRC passed. Evidence lives in
`result/preparation/gsd_all15_20261006/packaging_verification.json`.
No local GPU/model inference or live Colab browser download was performed.
