#!/usr/bin/env python3
"""Verify a full frozen Point2Vec run and recount its registered policies from raw logits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import canonical, sha, unit_key
from research_artifacts import CORRUPTIONS

ARMS = ('source', 'proximal_pm', 'proximal_terminal', 'cloudfixer')
PROBES = {'proximal_pm': [20, 10, 0], 'proximal_terminal': [0]}
OBJECTS, BLOCK_SIZE = 2468, 32
# Locked before A inference, from the complete released registration and author file map.
REGISTRATION_SHA = 'f1a80f7c7f7b324d268de37525ae442e6f824f121f52edd7067c768362d54ca1'
AUTHOR_FILES_SHA = 'c02dbcf8282d7208101ba17886f5e50ebc259c29f209f84787df02dc32ac2e8a'
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261118, 10000
HEX = re.compile(r'[0-9a-f]{64}')


def members(archive):
    files, dirs = set(), set()
    for info in archive.infolist():
        name = info.orig_filename[:-1] if info.is_dir() else info.orig_filename
        parts = name.split('/')
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (not name or any(p in ('', '.', '..') for p in parts) or re.search(r'[:\\\x00]', name)
                or mode not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG)
                or name in files or name in dirs):
            raise ValueError(f'Unsafe or duplicate archive path: {name!r}')
        (dirs if info.is_dir() else files).add(name)
    if any(str(p) in files for name in files | dirs for p in PurePosixPath(name).parents):
        raise ValueError('Archive file used as a parent directory')
    return files


def validate_config(config, env):
    spec = config['manifest']
    campaign = sha(canonical(dict(manifest=spec, phase='full')))
    assets = json.loads((ROOT / 'cfgs/cloudfixer_transfer_assets.json').read_text())
    head = json.loads((ROOT / 'cfgs/cloudfixer_point2vec_assets.json').read_text())
    if (config['phase'] != 'full' or config['execution_status'] != 'complete' or config['campaign'] != campaign
            or spec['schema'] != 2 or spec['classifier'] != head or spec['classifier_sha256'] != head['classifier_sha256']
            or any(spec[k] != assets[k] for k in ('diffusion_sha256', 'author_commit', 'dataset_files'))
            or sha(json.dumps(spec['official_files'], sort_keys=True).encode()) != AUTHOR_FILES_SHA
            or sha(canonical(spec['registration'])) != REGISTRATION_SHA):
        raise ValueError('Full Point2Vec campaign, assets or registration mismatch')
    setup = env['supplemental_setup']
    if (len(env['model_hashes']) != 2 or not all(HEX.fullmatch(v) for v in env['model_hashes'])
            or setup['official'] != AUTHOR_FILES_SHA or setup['classifier_sha256'] != head['classifier_sha256']
            or setup['script_sha256'] != spec['overlay_files']['scripts/setup_cloudfixer_point2vec_colab.py']
            or not HEX.fullmatch(setup['pip_freeze_sha256']) or not setup['native_extensions']
            or not all(HEX.fullmatch(v) for v in setup['native_extensions'].values())
            or not (setup['toolchain'] and setup['package_versions'] and setup['native_sources'])):
        raise ValueError('Incomplete or inconsistent frozen model or supplemental setup evidence')
    options = dict(spec['registration']['author_options'], classifier='point2vec', method=['cloudfixer'],
                   random_seed=2, batch_size=1)
    if any(env['options'].get(k) != v for k, v in options.items()):
        raise ValueError('Archived author options differ from registration')
    return campaign, assets['dataset_files']['label.npy']


def units_for(order, corr, campaign):
    if (len(order) != OBJECTS or any(type(i) is not int for i in order)
            or sorted(order) != list(range(OBJECTS))):
        raise ValueError(f'Invalid original-ID permutation for {corr}')
    digest = sha(canonical(order))
    for offset in range(0, OBJECTS, BLOCK_SIZE):
        yield dict(campaign=campaign, corruption=corr, seed=2, batch_size=1,
                   offset=offset, order_sha256=digest, ids=order[offset:offset+BLOCK_SIZE])


def read_receipt(archive, raw, unit):
    key = unit_key(unit)
    receipt = json.loads(archive.read(raw + key))
    digest = receipt['sha256']
    payload = f'data_{digest}.npz'
    if (not HEX.fullmatch(digest) or receipt['schema'] != 1 or receipt['payload'] != payload
            or canonical(receipt['unit']) != canonical(unit)):
        raise ValueError(f'Invalid unit receipt: {key}')
    data = archive.read(raw + payload)
    if type(receipt['bytes']) is not int or len(data) != receipt['bytes'] or sha(data) != digest:
        raise ValueError(f'Incorrect payload length or SHA: {key}')
    return data, {raw + key, raw + payload}


def recount_block(data, unit, labels, previous, logsnr):
    n = len(unit['ids'])
    fields = {'unit', 'ids', 'labels', 'rng_before_sha256', 'rng_after', 'vote_log_probs',
              'input_sha256', 'seconds', 'peak_mb'}
    fields.update(f'{arm}_{field}' for arm in ARMS for field in ('logits', 'probs', 'entropy'))
    fields.update(f'{arm}_{field}' for arm in PROBES for field in ('indices', 'logsnr'))
    with np.load(io.BytesIO(data), allow_pickle=False) as p:
        if len(p.files) != len(fields) or set(p.files) != fields:
            raise ValueError('Invalid full prediction payload schema')
        ids, truth = p['ids'], p['labels']
        if (p['unit'].shape != () or canonical(json.loads(str(p['unit']))) != canonical(unit)
                or ids.dtype.kind not in 'iu' or not np.array_equal(ids, unit['ids'])
                or truth.dtype.kind not in 'iu' or not np.array_equal(truth, labels[ids])):
            raise ValueError('Block identity or original-ID label mismatch')
        before, after = str(p['rng_before_sha256']), json.loads(str(p['rng_after']))
        if (not HEX.fullmatch(before) or (previous is not None and before != previous)
                or set(after) != {'python', 'numpy', 'cpu', 'cuda'} or not all(isinstance(v, list) and v for v in after.values())):
            raise ValueError('Invalid RNG evidence or broken block chain')
        inputs = p['input_sha256']
        if inputs.shape != (n,) or not all(HEX.fullmatch(s) for s in inputs.tolist()):
            raise ValueError('Invalid per-object input hashes')
        seconds, peak, vote = p['seconds'], p['peak_mb'], p['vote_log_probs']
        for array in (seconds, peak):
            if array.shape != (n, 4) or not np.isfinite(array).all() or not (array > 0).all():
                raise ValueError('Invalid per-object time or peak memory')
        if (vote.shape != (n, 40) or not np.isfinite(vote).all()
                or not np.allclose(np.exp(vote.astype(np.float64)).sum(-1), 1, atol=1e-6, rtol=0)):
            raise ValueError('Invalid diagnostic vote1 log probabilities')

        correct = np.empty((n, len(ARMS)), dtype=bool)
        for col, arm in enumerate(ARMS):
            probes = PROBES.get(arm)
            probes_n = len(probes) if probes else 1
            if probes:
                indices, snr = p[arm+'_indices'], p[arm+'_logsnr']
                if (indices.dtype.kind not in 'iu' or not np.array_equal(indices, probes)
                        or snr.shape != (probes_n,) or not np.isfinite(snr).all()):
                    raise ValueError(f'Invalid {arm} native probe evidence')
                for t, value in zip(probes, snr.tolist()):
                    if value != logsnr.setdefault(t, value):
                        raise ValueError(f'Inconsistent logSNR for native index {t}')
            logits = p[arm+'_logits'].astype(np.float64)
            if logits.shape != (n, probes_n, 40) or not np.isfinite(logits).all():
                raise ValueError(f'Invalid {arm} logits')
            e = np.exp(logits - logits.max(-1, keepdims=True))
            prob = e / e.sum(-1, keepdims=True)
            entropy = -(prob * np.log(prob.clip(min=np.finfo(float).tiny))).sum(-1)
            for field, expected in (('probs', prob), ('entropy', entropy)):
                saved = p[arm+'_'+field]
                if saved.shape != expected.shape or not np.allclose(saved, expected, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Incorrect {arm} {field}')
            correct[:, col] = (prob.mean(axis=1).argmax(-1) == truth)
        vote_correct = int(np.count_nonzero(vote.argmax(-1) == truth))
        return correct, seconds.sum(0), peak.max(0), vote_correct, sha(canonical(after))


def paired(candidate, reference):
    """Inputs are aligned [original object, corruption]. Resample whole original objects."""
    delta = candidate.astype(np.int8) - reference.astype(np.int8)
    per_object = delta.mean(1)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    means = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 128):
        chunk = means[start:start+128]
        ids = rng.integers(len(per_object), size=(len(chunk), len(per_object)))
        chunk[:] = 100 * per_object[ids].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return dict(delta_pp=float(delta.mean()*100), ci95_low_pp=float(low), ci95_high_pp=float(high),
                repair=int((delta == 1).sum()), harm=int((delta == -1).sum()))


def compare_csv(data, expected, keys):
    reader = csv.DictReader(io.StringIO(data.decode()))
    if sorted(reader.fieldnames or []) != sorted(expected[0]):
        raise ValueError('Incorrect archived result CSV fields')
    wanted = {tuple(r[k] for k in keys): r for r in expected}
    for row in reader:
        key = tuple(row[k] for k in keys)
        if key not in wanted or None in row:
            raise ValueError(f'Unexpected or duplicate result row: {key}')
        for field, val in wanted.pop(key).items():
            if isinstance(val, float):
                equal = math.isclose(float(row[field]), val, abs_tol=1e-12, rel_tol=1e-12)
            else:
                equal = row[field] == str(val)
            if not equal:
                raise ValueError(f'Archived {key} {field} disagrees with raw recount')
    if wanted:
        raise ValueError('Missing archived result rows')


def analyze(path, expected_sha, output, label_file=None):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Fresh output required: {output}')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise ValueError('Raw archive SHA256 mismatch')
        with zipfile.ZipFile(stream) as archive:
            names = members(archive)
            if archive.testzip() is not None:
                raise ValueError('Raw archive CRC mismatch')
            configs = [name for name in names if PurePosixPath(name).name == 'config.json']
            if len(configs) != 1:
                raise ValueError('Expected one full-phase config.json')
            prefix = configs[0][:-len('config.json')]
            if not all(name.startswith(prefix) for name in names):
                raise ValueError('Archive entries outside run directory')
            def read(name):
                return archive.read(prefix + name)
            for name in ('command.txt', 'notes.md'):
                if not read(name).strip():
                    raise ValueError(f'Empty required evidence: {name}')
            if b'CF_POINT2VEC_PHASE_DONE full' not in read('stdout.log').splitlines():
                raise ValueError('Missing full-phase completion marker in stdout.log')
            env_bytes = read('environment.txt')
            env = json.loads(env_bytes)
            campaign, label_sha = validate_config(json.loads(read('config.json')), env)
            if prefix+'verified_label.npy' in names:
                label_data, label_source = read('verified_label.npy'), 'verified_label.npy'
            elif label_file is not None:
                label_data, label_source = label_file.read_bytes(), 'external'
            else:
                raise ValueError('Supply --labels when verified_label.npy is absent')
            if sha(label_data) != label_sha:
                raise ValueError('Original label file SHA mismatch')
            labels = np.load(io.BytesIO(label_data), allow_pickle=False).reshape(-1)
            if len(labels) != OBJECTS or labels.dtype.kind not in 'iu' or set(labels.tolist()) != set(range(40)):
                raise ValueError('Invalid original labels')

            used, orders, logsnr, vote_counts = set(), {}, {}, {}
            corruptions = sorted(CORRUPTIONS)
            if {name for name in names if name.endswith('_order.json')} != {prefix+c+'_order.json' for c in corruptions}:
                raise ValueError('Expected one registered order per corruption')
            hits = np.empty((OBJECTS, len(corruptions), len(ARMS)), dtype=bool)
            per = []
            raw = prefix+'raw/'
            for j, corr in enumerate(corruptions):
                order = json.loads(read(corr+'_order.json'))
                orders[corr] = sha(canonical(order))
                previous, vote_counts[corr] = None, 0
                runtime, peak = np.zeros(len(ARMS)), np.zeros(len(ARMS))
                for unit in units_for(order, corr, campaign):
                    data, paths = read_receipt(archive, raw, unit)
                    used.update(paths)
                    correct, seconds, memory, vote, previous = recount_block(data, unit, labels, previous, logsnr)
                    hits[unit['ids'], j, :] = correct
                    runtime += seconds
                    peak = np.maximum(peak, memory)
                    vote_counts[corr] += vote
                for k, arm in enumerate(ARMS):
                    n_correct = int(hits[:, j, k].sum())
                    per.append(dict(run_id=campaign, dataset='modelnet40_c', severity=5, method=arm, seed=2,
                                    corruption=corr, n_examples=OBJECTS, n_correct=n_correct, accuracy=n_correct/OBJECTS,
                                    runtime_seconds=float(runtime[k]), peak_gpu_memory_mb=float(peak[k]), status='complete'))
            if used != {name for name in names if name.startswith(raw)}:
                raise ValueError('Unexpected raw file or receipt')
            summary = []
            for arm in ARMS:
                rows = [r for r in per if r['method'] == arm]
                n, good = sum(r['n_examples'] for r in rows), sum(r['n_correct'] for r in rows)
                summary.append(dict(run_id=campaign, dataset='modelnet40_c', severity=5, method=arm, seed=2,
                                    n_corruptions=len(rows), macro_accuracy=sum(r['accuracy'] for r in rows)/len(rows),
                                    total_examples=n, total_correct=good, micro_accuracy=good/n,
                                    total_runtime_seconds=sum(r['runtime_seconds'] for r in rows),
                                    peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in rows), status='complete'))
            compare_csv(read('per_corruption.csv'), per, ['method', 'corruption'])
            compare_csv(read('summary.csv'), summary, ['method'])

    comparisons = [dict(method=arm, reference='cloudfixer',
                        **paired(hits[:, :, ARMS.index(arm)], hits[:, :, ARMS.index('cloudfixer')])) for arm in PROBES]
    validation = dict(campaign=campaign, archive_sha256=expected_sha, crc_verified=True, labels_sha256=label_sha,
                      labels_source=label_source, objects=OBJECTS, corruptions=corruptions, severity=5, seed=2,
                      batch_size=1, blocks=len(used)//2, order_sha256=orders, probe_logsnr=logsnr,
                      environment_sha256=sha(env_bytes), frozen_model_hashes=env['model_hashes'],
                      supplemental_setup=env['supplemental_setup'], diagnostic_vote1_correct_by_corruption=vote_counts,
                      inference='Single-seed whole-policy differences versus plain CloudFixer. N, K, tau and readout vary.',
                      bootstrap=dict(unit='original object across all 15 corruptions', seed=BOOTSTRAP_SEED,
                                     draws=BOOTSTRAP_DRAWS, interval='paired percentile 95%', rng='PCG64'))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('summary.csv', summary), ('per_corruption.csv', per), ('paired_policy_differences.csv', comparisons)]:
        with (output/name).open('x', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--sha256', required=True, help='expected SHA256 of the untouched raw archive')
    parser.add_argument('--output', required=True, type=Path, help='fresh directory for derived files')
    parser.add_argument('--labels', type=Path, help='pinned label.npy, if absent from archive')
    args = parser.parse_args()
    analyze(args.archive, args.sha256, args.output, args.labels)


if __name__ == '__main__':
    main()
