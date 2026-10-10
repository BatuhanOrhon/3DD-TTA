"""Auditable execution shell for the restored legacy smooth-v2 algorithm.

Importing this module does not import torch or launch inference. The legacy
graph and trajectory stay separate from current Phase6/calibrated methods.
"""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
import zipfile

from research_artifacts import CORRUPTIONS, RunBundle, corruption_row
from run_baseline import file_identity, module_inventory

REFERENCE_COMMIT = '5c3ed88314f1c3d50ab77d17f7c6ba22da59c441'
REPO = Path(__file__).resolve().parent
METHOD = 'gsd_legacy_smooth_v2'


def validate_arguments(args, parser) -> None:
    """Bound this restoration to the reported static XYZ ModelNet40-C path."""
    if args.resume:
        parser.error('Resume is disabled: use a fresh output directory; existing evidence is never merged.')
    if args.dataset_name != 'modelnet-c' or args.corruption not in (None, *CORRUPTIONS):
        parser.error('Legacy restoration supports ModelNet40-C severity5 and canonical corruptions.')
    if args.weight_invariant != 0 or args.dynamic_graph or args.use_4d_gft:
        parser.error('This restoration covers static XYZ guidance with weight_invariant=0 only.')
    if args.batch_size < 1 or args.max_batches < 0 or not 1 <= args.M_max <= 2048:
        parser.error('Require batch_size>0, max_batches>=0 and 1<=M_max<=2048.')
    for key in ('denoising_step_normal', 'denoising_step_bg', 'denoising_step'):
        value = getattr(args, key)
        if value is not None and not 1 <= value <= 100:
            parser.error(key + ' must be in [1,100].')
    for key in ('beta', 'gamma', 'eta', 'lambdaa', 'weight_spectral', 'weight_chamfer'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            parser.error(key + ' must be finite and nonnegative.')
    if args.beta <= 0 or not 0 < args.lambdaa <= 1:
        parser.error('Require beta>0 and 0<lambdaa<=1.')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('Seed must be in [0,2**32).')
    if (Path(args.csv_name).name != args.csv_name or '/' in args.csv_name or '\\' in args.csv_name
            or not args.csv_name.endswith('.csv') or args.csv_name in {'summary.csv', 'per_corruption.csv'}):
        parser.error('csv_name must be a CSV basename distinct from summary.csv/per_corruption.csv.')


def protocol(args) -> dict:
    names = [args.corruption] if args.corruption else list(CORRUPTIONS)
    n = min(2468, args.batch_size * args.max_batches) if args.max_batches else 2468
    return dict(
        schema_version=1, reference_commit=REFERENCE_COMMIT, method=METHOD,
        dataset='modelnet40_c', severity=5, seed=args.seed, cli=vars(args).copy(),
        corruptions=names, expected_classifications=len(names)*n,
        stage='smoke' if args.max_batches else 'full_file_development',
        decoder_style='original' if args.use_static_style else 'updated',
        scheduler=dict(beta_end=.02, beta_start=.0001, beta_schedule='linear',
                       clip_sample=False, num_train_timesteps=1000,
                       prediction_type='epsilon', set_alpha_to_one=False, inference_steps=100),
        graph=dict(k=10, delta=.1, gamma=.6, knn='knn_cuda self-query',
                   cutoff='.6 * mean symmetric degree', isolate_penalty=1000, jitter=1e-5,
                   isolate_policy=args.isolate_policy,
                   isolate_rule=('full row sum == 0' if args.isolate_policy == 'legacy'
                                 else 'count_nonzero(row) - nonzero(diagonal) == 0')),
        spectral_loss='sum(exp(-beta*raw_eigenvalues)[:M_max] * squared_GFT_residual) / (3*2048)',
        guidance='SCD + spectral scalar loss; backward to local and style; no projection',
        update_rates=dict(local=args.eta, style=args.gamma),
        random_protocol='one continuous RNG stream; no prepared-input pairing claim',
        legacy_execution_identity='user-reported command; historical seed/build/commit unrecorded',
    )


def create_output(path: Path) -> Path:
    path = path.resolve()
    if path.with_suffix('.zip').exists():
        raise FileExistsError('Output ZIP already exists: ' + str(path.with_suffix('.zip')))
    path.mkdir(parents=True, exist_ok=False)
    return path


def _git(*args) -> str:
    result = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'UNAVAILABLE: ' + result.stderr.strip()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_inventory() -> dict:
    paths = {Path(__file__).resolve(), REPO / 'eval_gsd_tta_v2.py',
             REPO / 'tta_gsd_v2.py', REPO / 'graph_spectral_v2.py',
             REPO / 'graph_spectral_v2_offdiag.py'}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            try:
                path.relative_to(REPO)
            except ValueError:
                continue
            if path.suffix == '.py':
                paths.add(path)
    return {str(p.relative_to(REPO)): file_identity(p) for p in sorted(paths) if p.is_file()}


def native_inventory() -> dict:
    result = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).suffix in {'.pyd', '.so', '.dll'} and Path(filename).is_file():
            result[name] = file_identity(Path(filename))
    # Some CUDA extensions are loaded by torch.ops rather than Python imports.
    torch = sys.modules.get('torch')
    for filename in sorted(getattr(getattr(torch, 'ops', None), 'loaded_libraries', set())):
        if Path(filename).is_file():
            result['torch.ops:' + filename] = file_identity(Path(filename))
    return result


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run(args, load_runtime, configure_model, process_batches) -> None:
    """Run a fresh invocation, persisting completed batches even on failure."""
    path = create_output(Path(args.output_dir))
    config = protocol(args)
    config.update(run_id=path.name, execution_status='running', status='running',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  resolved_commit=_git('rev-parse', 'HEAD'), git_status=_git('status', '--short'),
                  source_manifest=source_inventory())
    bundle = RunBundle(path)
    bundle.write_config(config)
    bundle.write_results([], 'running')
    (path / 'command.txt').write_text(shlex.join([sys.executable, *sys.argv])+'\n', encoding='utf-8')
    (path / 'environment.txt').write_text(
        'Runtime initialization pending; inspect stdout.log if initialization fails.\n', encoding='utf-8')
    (path / 'notes.md').write_text(
        '# Legacy smooth restoration\n\nDevelopment reproduction, not independent confirmation. '
        'Seed-controlled does not mean common-draw paired. CPU tests do not establish CUDA parity.\n', encoding='utf-8')
    (path / 'predictions').mkdir()
    rows = []
    with (path / 'stdout.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                _execute(args, config, bundle, rows, load_runtime, configure_model, process_batches)
                config['execution_status'] = 'complete'
                config['status'] = ('complete' if not args.max_batches and len(rows) == 15 and
                                    all(r['status'] == 'complete' for r in rows) else 'partial')
                bundle.write_results(rows, config['status'])
                if config['status'] == 'complete':
                    print('All15 macro accuracy: %.6f%%' % (100*sum(r['accuracy'] for r in rows)/15))
                else:
                    print('Partial scope completed; not an all15 result.')
            except BaseException as error:
                config.update(execution_status='failed', status='failed',
                              error=type(error).__name__ + ': ' + str(error))
                bundle.write_results(rows, 'failed')
                traceback.print_exc()
                raise
            finally:
                config['source_manifest'] = source_inventory()
                config['completed_examples'] = sum(r['n_examples'] for r in rows)
                config['finished_utc'] = datetime.now(timezone.utc).isoformat()
                bundle.write_config(config)
                _write_json(path / 'status.json', {k: config[k] for k in
                            ('execution_status', 'status', 'completed_examples', 'finished_utc')})
                log.flush()
                with zipfile.ZipFile(path.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
                    for member in sorted(path.rglob('*')):
                        if member.is_file():
                            archive.write(member, str(member.relative_to(path)))


def _execute(args, config, bundle, rows, load_runtime, configure_model, process_batches):
    load_runtime()
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from utilities_3dd_tta import PointDataset

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       cudnn_deterministic=torch.backends.cudnn.deterministic,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                       packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
    (bundle.path / 'environment.txt').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires Colab/CUDA; use --preview for CPU protocol checks.')
    environment['gpu'] = torch.cuda.get_device_name()
    (bundle.path / 'environment.txt').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    asset_paths = [Path(getattr(args, name)) for name in
                   ('pointmae_config', 'pointmae_ckpt', 'diff_config', 'diff_ckpt', 'label_path')]
    asset_paths += [Path(args.dataset_root) / ('data_' + c + '_5.npy') for c in config['corruptions']]
    config['assets'] = {str(p): file_identity(p) for p in asset_paths}
    base, lion, graph = configure_model(args)
    config['model_modes'] = dict(classifier=module_inventory(base), vae=module_inventory(lion.vae), priors=module_inventory(lion.priors))
    config['native_extensions'] = native_inventory()
    config['external_python_sources'] = {
        name: file_identity(Path(module.__file__))
        for name in ('diffusers', 'knn_cuda', 'pointnet2_ops', 'scipy')
        for module in [sys.modules.get(name)]
        if module is not None and getattr(module, '__file__', None)
    }
    bundle.write_config(config)
    with (bundle.path / args.csv_name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Dataset','Corruption','M_max','Beta','Weight_Spectral','Weight_Invariant','Weight_Chamfer','Accuracy'])
        for corruption in config['corruptions']:
            dataset = PointDataset(args.dataset_root, args.label_path, corruption)
            if len(dataset) != 2468 or len(dataset.labels) != 2468:
                raise ValueError('Expected 2468 ModelNet40-C examples and labels: ' + corruption)
            n = min(len(dataset), args.max_batches*args.batch_size) if args.max_batches else len(dataset)
            selected = dataset if n == len(dataset) else Subset(dataset, range(n))
            loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False)
            steps = args.denoising_step or (args.denoising_step_bg if corruption == 'background' else args.denoising_step_normal)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            seen, correct, batch_number = 0, 0, 0
            row_index = len(rows)

            def observe(target, pred, logits, metrics):
                nonlocal seen, correct, batch_number
                if not bool(torch.isfinite(logits).all()) or not all(math.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError('Nonfinite logits/loss diagnostics in ' + corruption)
                target_np, pred_np = target.detach().cpu().numpy(), pred.detach().cpu().numpy()
                count = len(target_np)
                if count == 0 or len(pred_np) != count or seen+count > n:
                    raise ValueError('Prediction count mismatch')
                if not np.array_equal(target_np, np.asarray(dataset.labels[seen:seen+count]).reshape(-1)):
                    raise ValueError('Label/index mismatch')
                np.savez_compressed(bundle.path / 'predictions' / (corruption + '_%04d.npz' % batch_number),
                                    indices=np.arange(seen, seen+count), targets=target_np, predictions=pred_np,
                                    logits=logits.detach().cpu().numpy())
                seen += count
                correct += int((target_np == pred_np).sum())
                batch_number += 1
                row = corruption_row(config['run_id'], args.seed, corruption, seen, correct,
                                     time.perf_counter()-start, torch.cuda.max_memory_allocated()/1024**2,
                                     'partial', method=METHOD)
                if len(rows) == row_index:
                    rows.append(row)
                else:
                    rows[row_index] = row
                bundle.write_results(rows, 'running')

            targets, preds = process_batches(loader, base, lion, graph, args, steps, batch_observer=observe)
            torch.cuda.synchronize()
            if seen != n or len(targets) != n or int((targets == preds).sum().item()) != correct:
                raise ValueError('Final counts disagree with persisted predictions')
            rows[row_index].update(status='complete' if n == 2468 and not args.max_batches else 'partial',
                                   runtime_seconds=time.perf_counter()-start)
            bundle.write_results(rows, 'running')
            acc = (preds == targets).float().mean().item()  # preserve legacy CSV rounding
            writer.writerow([args.dataset_name,corruption,args.M_max,args.beta,args.weight_spectral,
                             args.weight_invariant,args.weight_chamfer,acc])
            stream.flush()
            print('%s: %d/%d (%.6f%%)' % (corruption, correct, seen, correct/seen*100))
