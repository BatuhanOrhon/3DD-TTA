# Legacy smooth-v2 restoration on gsd-smooth-spectrum - 2026-10-08

## User evidence and scope

[User report] The64.532684% CSV was produced using
`origin/gsd-tta-smooth-integration:eval_gsd_tta_v2.py`, batch70,
spectral1.17, SCD1, M_max800, beta2, gamma/eta.01, lambda.95,
normal10/Background30, output `gsd_smooth_optimal_800`.
This resolves the previously missing command and explains the launcher/output
name association. It does not supply the historical exact commit, seed,
package/build hashes or checkpoint/data hashes. The command is user evidence,
not recovered execution metadata. Preserve the earlier dated uncertainty.

[User decision] Port this structure into gsd-smooth-spectrum and test it.
This supersedes the October7 normal10/Background35-only proposal as the
immediate action. No parameter sweep is included in this restoration.

[Code] Main base:c18926ff7751ed9023313869229f5d4dffe218be.
Local origin/gsd-tta-smooth-integration resolves to
5c3ed88314f1c3d50ab77d17f7c6ba22da59c441, matching the other worktree.
No network fetch was used; this is the locally available remote-tracking ref.
The reference commit is the port source, not a claimed historical run commit.

## What differed

These are distinct implementation choices, not all demonstrated bugs.

| Item | Current established paths | Restored legacy smooth-v2 |
|---|---|---|
| Entry | run_baseline/gsd_protocol; Phase6 composition runner | eval_gsd_tta_v2.py |
| Batch |32|70|
| Normal/Background reverse steps |5/35|10/30|
| Final DDIM alpha |True in existing protocol|False explicitly|
| Decode style |original|updated by default|
| Graph neighbors |explicit nonself cdist|KNN_CUDA self-query, unchanged|
| Outlier cutoff |.6*sum(directed A)/(N*k)|.6*mean symmetric degree|
| Isolates |removed from active eigensystem|diagonal penalty1000 plus jitter1e-5|
| Current smooth filter |exp(-beta*lambda/mean_active_degree), all active modes|exp(-beta*raw lambda), zero indices>=M_max|
| Hard Phase6 filter |M100 with boundary expansion|not used in this restored path|
| Spectral loss |smooth: original-point normalization; hard-v1: rank normalization|weighted squared GFT error sum/(3*2048)|
| Composition |Phase6 local SCD/style projection|SCD+spectral total loss differentiated into local and style|
| Update parameter names |gamma local, eta style|eta local, gamma style; equal at supplied settings|

[Code] Shared LION constructor does not globally enable eval in this branch;
legacy constructor did. The restored entry explicitly sets VAE/prior eval
after raw checkpoint loading, keeping existing runners intact. EMA remains off.
Point-MAE is loaded using the legacy builder.load_model call and is eval.
PVConv eval-mode backward support is already present: only comments/format
differ between branches. No PVConv or checkpoint-loading rewrite is necessary.
Active ModelNet preprocessing helpers are equivalent; current labels use
as_tensor instead of from_numpy and default severity5. Current device-safe
unnormalize differs only on the unused ScanObjectNN path.

[Open] PointNet2 CUDA architecture build defaults differ across branches.
The existing caller-selectable build setup is retained; native binary identity
is logged during the new run. CUDA KNN/Chamfer/FPS/eigh parity is not established
by CPU tests. A different old software/hardware build can change results.

## Implemented files and limits

- `tta_gsd_v2.py`, `graph_spectral_v2.py`: exact bytes restored from5c3ed883.
  Graph/gradient equations are preserved, including legacy self-query and raw
  eigenvalue conventions. No clean-up that changes their behavior was applied.
- `eval_gsd_tta_v2.py`: original model/batch path, explicit LION eval,
  deferred imports, preview/seed/smoke options and a prediction observer.
  AST comparison proves process_batches is unchanged after removing observer
  instrumentation. The supplied CLI parameter names remain accepted.
- `legacy_gsd_artifacts.py`: fresh-output execution, assets/source/native hashes,
  runtime/package/mode records, count-based summaries, indexed prediction/logit
  NPZs, stdout/stderr and ZIP sealing, including partial failures.
- `tests/test_legacy_gsd_restore.py`, `tests/test_legacy_gsd_numerics.py`:
  CPU contracts, synthetic math and artifact lifecycle checks.

The active restoration supports static XYZ ModelNet40-C, invariant weight0.
Dynamic graph/4D/invariant experiments are rejected, not silently approximated.
`--resume` is rejected. Existing output directories/ZIPs are never overwritten.
Original decoder style remains an explicit optional flag; the supplied command
does not select it. Original CLI M_max default1200 is preserved, so always pass
`--M_max 800` for this reproduction. Omitted seed preserves unseeded behavior;
the recommended new evaluation explicitly selects seed0 and records it.
All15 equals37020 classifications per seed; a subset/smoke is marked partial.
Artifact runtime includes observation/prediction-writing overhead and does not
equal a separately measured pure-inference benchmark.

## Verification

[Code/CPU tests] 51 tests passed under local Python3.13, with no trained model
or GPU inference. Command:

```powershell
python -m unittest tests.test_legacy_gsd_restore tests.test_legacy_gsd_numerics tests.test_gsd_protocol tests.test_graph_spectral_smooth tests.test_gsd_trajectory -q
```

Checks cover exact core source parity, observer-stripped inference AST parity,
reported CLI/preview, restored eval modes, graph cutoff/penalty/raw spectrum,
one-step analytic spectral+SCD gradients and both update rates, full/failed
artifact lifecycle using CPU dummy data, and existing GSD regressions.
Synthetic all15 counts are fixture checks, not accuracy evidence.
The initial contract run failed because the restored files did not yet exist;
the first artifact check exposed a Windows temp ACL issue. Tests now use unique
workspace fixture directories with verified cleanup paths.

[Code/review] Independent review found one smoke-status boundary issue:
a max_batches cap covering the full file could be labeled complete all15.
Both row and overall completion now require max_batches=0. The CPU artifact
test covers that full-size smoke and requires partial status. No other
critical/important review finding was reported. Final verification log and
hash receipt are in `result/modelnet40_c/diagnostics/legacy_restore_20261008/`.

[Open] Local diffusers/CUDA extensions/environment are not the Colab stack.
Real CUDA model integration and reproduction accuracy remain untested.
51 CPU passes do not establish bitwise native-kernel or historical-score parity.
No original algorithm files, user notebook, raw results, M1800 notes or failed
Background100/1000 records were edited. No stage/commit/push.

## Colab handoff (user-operated)

Use the existing3dd_tta_env and checkout at main basec18926f above. There is no
push in this task. Transfer the prepared source overlay ZIP and apply the four
new Python files to that checkout before running. The overlay manifest pins
the base and hashes; do not assume a git pull contains these uncommitted files.

Upload `legacy_smooth_source_20261008.zip` from the diagnostics directory using
Colab's files panel, then use this **CPU-only transfer cell** in the repo root:

```python
from pathlib import Path
import hashlib, json, subprocess, zipfile

repo = Path('/content/3DD-TTA')
archive = Path('/content/legacy_smooth_source_20261008.zip')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read('manifest.json'))
    expected = {'eval_gsd_tta_v2.py', 'tta_gsd_v2.py',
                'graph_spectral_v2.py', 'legacy_gsd_artifacts.py'}
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == expected | {'manifest.json'}
    assert set(manifest['files']) == expected
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == manifest['base_commit']
    subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD','--','*.py'], check=True)
    # Check the complete overlay before writing any file. Repeat application
    # is accepted only when the existing bytes already match.
    payload = {name: z.read(name) for name in expected}
    for name, data in payload.items():
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
        target = repo / name
        assert not target.exists() or target.read_bytes() == data, f'Existing different file: {target}'
    for name, data in payload.items():
        target = repo / name
        if not target.exists():
            with target.open('xb') as f:
                f.write(data)
print('Overlay verified and applied; no model execution.')
```

### Preview, smoke, then the supplied configuration

This cell builds one command with exactly the supplied method parameters.
Preview runs first and prints the resolved protocol without importing models.

```python
from pathlib import Path
from datetime import datetime, timezone
import subprocess

repo = Path('/content/3DD-TTA')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
run_root = repo / 'result/modelnet40_c/gsd_legacy_smooth_v2'
common = ['conda','run','--no-capture-output','-n','3dd_tta_env',
          'python','eval_gsd_tta_v2.py',
          '--batch_size','70','--weight_spectral','1.17',
          '--weight_chamfer','1.0','--M_max','800','--beta','2.0',
          '--gamma','0.01','--eta','0.01','--lambdaa','0.95',
          '--denoising_step_bg','30','--denoising_step_normal','10',
          '--dataset_root','./data/modelnet40_c',
          '--label_path','./data/modelnet40_c/label.npy',
          '--csv_name','eval_results.csv','--seed','0']
subprocess.run(common + ['--preview'], cwd=repo, check=True)
```

Separate **GPU smoke cell** (Gaussian, two batches at70, no accuracy conclusion):

```python
smoke_dir = run_root / (stamp + '_legacy-m800-smoke-seed0')
subprocess.run(common + ['--corruption','gaussian','--max_batches','2',
                        '--output_dir',str(smoke_dir)], cwd=repo, check=True)
```

After reviewing the completed smoke, separate **all15 GPU cell**:

```python
full_dir = run_root / (stamp + '_legacy-m800-all15-seed0')
subprocess.run(common + ['--output_dir',str(full_dir)], cwd=repo, check=True)
```

Seed0 is an explicitly documented addition to the old unseeded command.
It enables repeatable new evaluation; it does not recreate an unknown old RNG
state. No automatic three-seed campaign or search is started. Keep batch70;
an OOM must be reported as such before considering a separately labeled batch
change. Do not silently lower batch size and claim the supplied protocol.

Download the complete ZIP after the run (also retain any failed smoke ZIP):

```python
from google.colab import files
files.download(str(full_dir.with_suffix('.zip')))
```

Provide that complete ZIP for independent count/provenance comparison with
the old64.532684% CSV. The package contains command/config/environment/logs,
per-corruption/summary CSVs and indexed logits/predictions; see result/README.md.
Do not include credentials. Reproduction is assessed before any parameter
search; even a matching score will not isolate the spectral contribution from
the other legacy host differences.

## Result ingestion - 2026-10-10

[Run/Code/Inference/Open] The complete seed0 isolate A/B result and source
validation are recorded in
[`validation.md`](../result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_ingested/validation.md).
Legacy macro15 is 64.5705%; offdiag is 64.5543% (-0.0162 pp). This confirms the
restored path executes all 15 successfully and its archived counts are
reproducible from predictions. It does not reproduce the unarchived historical
64.5327% bit-for-bit or prove any +1 to 2 pp accuracy gain. Single-seed, separate
stochastic streams, no prepared-input hashes; no seed variance/CI. Historical
unknown seed/build limitations above remain.

