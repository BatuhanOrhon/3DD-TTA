"""Offline entropy-weighted PM on the accepted CloudFixer N8/N12 traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import shlex
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_cloudfixer_exploration import main as validate_archive
from scripts.analyze_legacy_trajectory_macro14 import entropy_weighted_probabilities
from scripts.run_cloudfixer_exploration import unpack

BETAS = (0, .5, 1, 2, 4, 8)
INPUT_SHA = dict(nstep='2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7',
                 entropy='509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df')
LABEL_SHA = '97b6e660820103074ab192f10b6d9c33aa0446f1547ddb6ad2ca35a029a63b31'


def entropy_readouts(probability: np.ndarray, entropy: np.ndarray) -> dict:
    """Label-free per-example weights; entropy is in nats, not normalized."""
    p, h = np.asarray(probability, dtype=np.float64), np.asarray(entropy, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] < 2:
        raise ValueError('Expected examples x candidates x classes probabilities')
    linear, linear_weights = entropy_weighted_probabilities(p, h / np.log(p.shape[-1]))
    equal_weights = np.full(p.shape[:2], 1 / p.shape[1])
    equal = p.mean(1)
    terminal_weights = np.zeros_like(equal_weights)
    terminal_weights[:, -1] = 1
    result = dict(terminal=(p[:, -1], terminal_weights), equal_pm=(equal, equal_weights),
                  linear_entropy=(linear, linear_weights))
    for beta in BETAS:
        weight = np.exp(-beta * (h - h.min(1, keepdims=True)))
        weight /= weight.sum(1, keepdims=True)
        mean = equal.copy() if beta == 0 else np.einsum('nk,nkc->nc', weight, p)
        result['exp_beta_' + str(beta)] = (mean, weight)
    return result


def load_records(path: Path, role: str, output: Path) -> list:
    if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_SHA[role]:
        raise ValueError('Pinned input archive SHA mismatch: ' + role)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Corrupt or ambiguous input ZIP')
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe input ZIP path')
    # Validate full 420/60-unit coverage, registration, immutable source and model receipts.
    validate_archive(['--input', str(path), '--output', str(output / (role + '_validated_analysis.json'))])
    with zipfile.ZipFile(path) as archive:
        records = []
        for name in archive.namelist():
            if '/pending_unit_' in name and name.endswith('.npz'):
                payload = unpack(archive.read(name))
                unit = json.loads(str(payload['unit']))
                if unit['config']['steps'] in (8, 12):
                    records.append((unit, payload))
    return sorted(records, key=lambda pair: (pair[0]['corruption'], pair[0]['ids']))


def make_view(records: list, slots: list, labels: np.ndarray) -> dict:
    indices = [json.loads(str(p['sampler']))['candidate_indices'] for _, p in records]
    if not records or any(index != indices[0] for index in indices):
        raise ValueError('Mixed or missing candidate times')
    for _, p in records:
        if not np.array_equal(p['labels'], labels[p['ids']]):
            raise ValueError('Original label identity mismatch')
    return dict(probability=np.concatenate([p['probabilities'][:, slots] for _, p in records]),
                entropy=np.concatenate([p['entropy'][:, slots] for _, p in records]),
                labels=np.concatenate([p['labels'] for _, p in records]),
                ids=np.concatenate([p['ids'] for _, p in records]),
                corruptions=np.array([u['corruption'] for u, p in records for _ in p['ids']]),
                candidate_indices=[indices[0][i] for i in slots])


def metrics(correct: np.ndarray, pm: np.ndarray, terminal: np.ndarray) -> dict:
    return dict(n_examples=len(correct), n_correct=int(correct.sum()), accuracy=float(correct.mean()),
                delta_vs_equal_pm_pp=100 * float(correct.mean() - pm.mean()),
                delta_vs_terminal_pp=100 * float(correct.mean() - terminal.mean()),
                pm_wrong_weighted_correct=int((correct & ~pm).sum()),
                weighted_wrong_pm_correct=int((~correct & pm).sum()),
                terminal_wrong_weighted_correct=int((correct & ~terminal).sum()),
                weighted_wrong_terminal_correct=int((~correct & terminal).sum()))


def write_csv(path: Path, rows: list) -> None:
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--nstep-archive', type=Path, required=True)
    parser.add_argument('--entropy-archive', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log = (args.output / 'stdout.log').open('x')
    def report(message):
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != LABEL_SHA:
        raise ValueError('Pinned original-label SHA mismatch')
    labels = np.load(args.labels, allow_pickle=False).reshape(-1)
    (args.output / 'command.txt').write_text(shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + '\n')
    config = dict(run_id=args.output.name, status='running', evaluation='reused development subset',
                  input_sha256=INPUT_SHA, labels_sha256=LABEL_SHA, beta_grid=BETAS,
                  linear_rule='w proportional to 1-H/log(40), equal fallback at all-maximal entropy',
                  exponential_rule='w proportional to exp(-beta*H), H in nats',
                  labels_used_for_weights=False, labels_used_for_exploratory_reporting=True,
                  new_model_inference=False, seed=0, original_objects=128, corruptions=15, severity=5,
                  diffusion_start=80, scd_tau=.1, retained=.95, classifier='Point-MAE')
    sources = ['scripts/analyze_cloudfixer_weighted_pm.py', 'scripts/analyze_legacy_trajectory_macro14.py',
               'scripts/analyze_cloudfixer_exploration.py', 'scripts/run_cloudfixer_exploration.py']
    config['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config['source_files'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.output / 'sources').mkdir()
    for name in sources:
        (args.output / 'sources' / Path(name).name).write_bytes((ROOT / name).read_bytes())
    (args.output / 'environment.txt').write_text(json.dumps(dict(python=sys.version, numpy=np.__version__,
        platform=platform.platform(), execution='local CPU saved-logit analysis; Colab CPU for Drive transfer'), indent=2) + '\n')
    nstep = load_records(args.nstep_archive, 'nstep', args.output)
    dense = load_records(args.entropy_archive, 'entropy', args.output)
    views = {f'n{n}_four': make_view([r for r in nstep if r[0]['config']['steps'] == n], [0, 1, 2, 3], labels)
             for n in (8, 12)}
    views['n8_nine'] = make_view(dense, list(range(9)), labels)
    if not np.array_equal(views['n8_four']['probability'], views['n8_nine']['probability'][:, [1, 3, 5, 8]]):
        raise ValueError('N8 dense/sparse canonical-four parity failed')
    summaries, corruptions, saved = [], [], {}
    expected = dict(n8_four=(1250, 1269), n12_four=(1257, 1235), n8_nine=(1256, 1269))
    for name, view in views.items():
        p, h, target, names = (view[k] for k in ('probability', 'entropy', 'labels', 'corruptions'))
        if len(target) != 1920 or len(set(zip(names.tolist(), view['ids'].tolist()))) != 1920:
            raise ValueError('Incomplete or repeated original-object coverage')
        if len(set(names)) != 15 or any((names == c).sum() != 128 for c in set(names)):
            raise ValueError('Expected all15 x128 coverage')
        readouts = entropy_readouts(p, h)
        pm = readouts['equal_pm'][0].argmax(-1) == target
        terminal = p[:, -1].argmax(-1) == target
        if (int(pm.sum()), int(terminal.sum())) != expected[name]:
            raise ValueError('Accepted equal-PM/terminal baseline recount differs')
        for key in ('ids', 'labels', 'corruptions'):
            saved[name + '__' + key] = view[key]
        for policy, (probability, weights) in readouts.items():
            predicted = probability.argmax(-1)
            correct = predicted == target
            if policy == 'exp_beta_0' and not np.array_equal(probability, p.mean(1)):
                raise ValueError('Zero-strength equal-PM control differs')
            summaries.append(dict(run_id=args.output.name, dataset='modelnet40_c', severity=5, seed=0,
                view=name, policy=policy, status='complete', **metrics(correct, pm, terminal),
                candidate_indices=json.dumps(view['candidate_indices']), mean_weights=json.dumps(weights.mean(0).tolist())))
            for corruption in sorted(set(names)):
                mask = names == corruption
                corruptions.append(dict(run_id=args.output.name, dataset='modelnet40_c', severity=5, seed=0,
                    view=name, policy=policy, corruption=corruption, status='complete',
                    **metrics(correct[mask], pm[mask], terminal[mask])))
            saved[name + '__' + policy + '__prediction'] = predicted
            saved[name + '__' + policy + '__weights'] = weights
            report(f"{name} {policy}: {int(correct.sum())}/1920 accuracy={100*correct.mean():.6f}%")
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'per_corruption.csv', corruptions)
    np.savez_compressed(args.output / 'predictions_weights.npz', **saved)
    config.update(status='complete', analysis_wall_seconds=time.perf_counter() - started,
                  views={name: view['candidate_indices'] for name, view in views.items()}, summary_rows=len(summaries),
                  per_corruption_rows=len(corruptions))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.output / 'notes.md').write_text('Fixed predeclared entropy readout screen on reused128-object/all15/s5/seed0 traces. '
        'Primary N8/N12 use same four-candidate policy; their actual times differ. N8nine is a separate secondary view. '
        'No classifier/diffusion rerun, per-corruption weights, label-dependent weights or automatic baseline promotion. '
        'Best observed coefficient is development selection, not independent confirmation. Original inference timing is unchanged; '
        'reported wall time is CPU analysis only. Complete pinned source analyses are included.\n')
    report('CF_WEIGHTED_PM_ANALYSIS_DONE')
    log.close()


if __name__ == '__main__':
    main()
