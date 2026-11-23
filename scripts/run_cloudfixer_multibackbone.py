#!/usr/bin/env python3
"""CloudFixer XYZ geometry shared by three accepted frozen heads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import signal
import threading
import time
import zipfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, sha, unit_key
from research_artifacts import CORRUPTIONS
from scripts.run_cloudfixer_point2vec import CF_OPTIONS, POLICIES, proximal_policy, rng_state, restore_rng
from scripts import eval_multibackbone_source as source

HEADS = ('dgcnn', 'curvenet', 'pointnext')
METHODS = ('source_raw', 'source_cf_frame', 'cloudfixer_plain')


ORIGINAL_PROFILE = 'original_cloudfixer'
BASELINE_PROFILE = 'cloudfixer_xyz_proximal_pm_n8_k80_late_v1'
BASELINE_TASK = 'cloudfixer_baseline_heads_20261123'
ORIGINAL_BATCH_SIZES = (32, 64)
BASELINE_BATCH_SIZES = (64, 96, 128)
BASELINE_METHODS = ('source_raw', 'proximal_n8_terminal', 'proximal_n8_pm')
BASELINE_POLICY = dict(steps=8, start=80, tau=.1, retained=.95, probe_indices=[20, 10, 0])
ORIGINAL_CF_CAMPAIGN = '3596befa489bc8f05cd5f139a976c7a00b7429d390a19fec850595814cd3f224'
ORIGINAL_CF_SOURCE = '228555fe952669bf4e1e9a1ee2e6f347d61b066c'
ORIGINAL_CF_REFERENCE = dict(bytes=33008, sha256='9629732356683ef5b34d5089e5ca07a96e6fdfde36cba6bc0e44cf73ba51106f')
ORIGINAL_CF_ARCHIVE_INDEX = dict(bytes=9721, sha256='8d8a9cd3bccf4268b1b5ce101ea6548530080444ee01834763d85672405d4497')
ORIGINAL_CF_ASSET_SHA256 = '5b6f7b1dbf2e9149dd76bbec02fc87aaa1713f520d33f829c997636243f9d235'
ASSET_FIELDS = ('head_configs', 'head_state_hashes', 'dataset_files', 'files',
                'diffusion_sha256', 'author_commit', 'official_files')
ORIGINAL_FIELDS = ('source_logits', 'frame_logits', 'cf_logits')
BASELINE_FIELDS = ('source_logits', 'terminal_logits', 'pm_probabilities')
WORKSPACE_CONFIG = ':4096:8'
OVERLAY_FILES = (
    'scripts/run_cloudfixer_multibackbone.py', 'scripts/setup_cloudfixer_multibackbone_colab.py',
    'scripts/eval_multibackbone_source.py', 'scripts/run_cloudfixer_point2vec.py',
    'scripts/run_cloudfixer_transfer.py', 'classifier_backbones.py', 'research_artifacts.py',
    'cloudfixer_store.py', 'cloudfixer_diffusion.py', 'pointcloud_proximal.py',
)
BASELINE_OVERLAY_FILES = OVERLAY_FILES + (
    'scripts/analyze_cloudfixer_multibackbone.py', 'scripts/prepare_cloudfixer_multibackbone.py',
    'knowledge/evidence/cloudfixer_multibackbone_20261122/a100_full_recount.json',
    'knowledge/evidence/cloudfixer_multibackbone_20261122/a100_original_archive_index.json',
)


def baseline_protocol(batch_size: int = 64) -> dict:
    validate_batch_size(batch_size, BASELINE_PROFILE)
    return dict(dataset='modelnet40_c', objects=2468, severity=5, seed=0, batch_size=batch_size,
        heads=list(HEADS), corruptions=list(CORRUPTIONS), methods=list(BASELINE_METHODS),
        geometry='authorrotateTrue;unitSTD;inverseaxis/unitsphereoutput',
        seed_policy='sha256(corruption,seed,ids) first 32 bits; unit-fork; index-addressed classifier RNG',
        readout='pre-correction 20/10; extra-denoiser true t0; equal float64 probability mean')


def methods_for(profile: str) -> tuple:
    if profile == ORIGINAL_PROFILE:
        return METHODS
    if profile == BASELINE_PROFILE:
        return BASELINE_METHODS
    raise ValueError('Unknown multibackbone profile')


def validate_batch_size(batch_size: int, profile: str) -> None:
    methods_for(profile)
    choices = BASELINE_BATCH_SIZES if profile == BASELINE_PROFILE else ORIGINAL_BATCH_SIZES
    if type(batch_size) is not int or batch_size not in choices:
        raise ValueError(f'{profile} batch size must be one of {choices}')


def manifest_profile(spec: dict) -> str:
    profile = spec.get('profile', ORIGINAL_PROFILE)
    methods_for(profile)
    protocol = spec.get('scientific_protocol', {})
    if not isinstance(protocol, dict):
        raise ValueError('Scientific protocol must be an object')
    if profile == ORIGINAL_PROFILE:
        if spec.get('task') == BASELINE_TASK or 'policy' in spec or 'baseline_archive_index' in spec:
            raise ValueError('Baseline manifest must declare its profile')
        validate_batch_size(protocol.get('batch_size', 32), profile)
        return profile
    overlay = spec.get('overlay_files', {})
    if (set(overlay) != set(BASELINE_OVERLAY_FILES)
            or any(not is_digest(value.get('sha256')) or type(value.get('bytes')) is not int
                   or value['bytes'] <= 0 for value in overlay.values())
            or spec.get('schema') != 1 or spec.get('task') != BASELINE_TASK
            or canonical(spec.get('policy')) != canonical(BASELINE_POLICY)
            or canonical(protocol) != canonical(baseline_protocol(protocol.get('batch_size')))
            or spec.get('baseline_source') != ORIGINAL_CF_SOURCE
            or spec.get('baseline_reference') != ORIGINAL_CF_REFERENCE
            or spec.get('baseline_archive_index') != ORIGINAL_CF_ARCHIVE_INDEX
            or not is_digest(spec.get('source_commit'), 40)
            or sha(canonical({key: spec.get(key) for key in ASSET_FIELDS})) != ORIGINAL_CF_ASSET_SHA256):
        raise ValueError('Baseline manifest policy, scope, source or accepted assets changed')
    return profile


def is_digest(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def require_workspace_config() -> None:
    if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', WORKSPACE_CONFIG) != WORKSPACE_CONFIG:
        raise ValueError('Baseline requires CUBLAS_WORKSPACE_CONFIG='+WORKSPACE_CONFIG)


@contextmanager
def fork_readout_rng(x):
    import numpy as np
    import torch

    python = random.getstate()
    numpy = np.random.get_state()
    devices = [x.device.index] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python)
            np.random.set_state(numpy)


def array_sha(tensor):
    return sha(tensor.detach().cpu().contiguous().numpy().tobytes())


def trajectory_sha(unit, values):
    return sha(canonical(dict(unit=unit, profile=BASELINE_PROFILE, policy=BASELINE_POLICY,
        geometry_seed=int(values['geometry_seed']), input_sha256=str(values['input_sha256']),
        mask_sha256=str(values['mask_sha256']), candidate_cloud_sha256=list(values['candidate_cloud_sha256']))))


def proximal_readout(model, classifier, x, mask, unit, *, geometry, timed) -> dict:
    """Score three observations of the exact shared N8 trajectory."""
    import numpy as np
    import torch

    candidate_seconds = []
    candidate_head_seconds = []

    def classify(cloud):
        with fork_readout_rng(cloud):
            (logits, head_seconds), seconds = timed(lambda: classifier(geometry(cloud)))
        candidate_seconds.append(seconds)
        candidate_head_seconds.append(head_seconds)
        return logits

    if canonical(POLICIES['proximal_pm']) != canonical(BASELINE_POLICY):
        raise ValueError('Exact Point2Vec proximal PM registration changed')
    seed = seed_for(unit)
    with fork_readout_rng(x):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed(seed)
        (logits, pm, clouds, indices, calls), wall = timed(
            lambda: proximal_policy(model, classify, x, mask, POLICIES['proximal_pm']))
    candidate_hashes = [array_sha(cloud) for cloud in clouds.unbind(1)]
    values = dict(profile=BASELINE_PROFILE, policy=canonical(BASELINE_POLICY).decode(), head_order=HEADS,
        candidate_indices=indices, denoiser_calls=calls, geometry_seed=seed,
        candidate_logits=logits.cpu().numpy(), terminal_logits=logits[:, -1].cpu().numpy(),
        candidate_probabilities=logits.double().softmax(dim=-1).cpu().numpy(), pm_probabilities=pm.cpu().numpy(),
        input_sha256=array_sha(x), mask_sha256=array_sha(mask), output_sha256=candidate_hashes[-1],
        candidate_cloud_sha256=candidate_hashes, candidate_seconds=candidate_seconds,
        candidate_head_seconds=candidate_head_seconds, policy_seconds=wall,
        reconstruction_seconds_shared=wall-sum(candidate_seconds))
    values['trajectory_sha256'] = trajectory_sha(unit, values)
    return values


def units(campaign, phase, batch_size=32, *, profile=ORIGINAL_PROFILE):
    validate_batch_size(batch_size, profile)
    if phase not in ('full', 'smoke', 'profile'):
        raise ValueError('Unknown registered phase')
    corruptions = list(CORRUPTIONS) if phase == 'full' else ['gaussian'] if phase == 'profile' else ['background', 'rotation', 'gaussian']
    ids = list(range(2468)) if phase == 'full' else list(range(batch_size if phase == 'profile' else 8))
    size=4 if phase=='smoke' else batch_size
    return [dict(campaign=campaign, phase=phase, corruption=c, seed=0,
                 ids=ids[i:i+size], batch_size=size)
            for c in corruptions for i in range(0, len(ids), size)]


def seed_for(unit):
    # Classifier-only substream; native geometry uses the continuous corruption seed0.
    return int(sha(canonical({k: unit[k] for k in ('corruption', 'seed', 'ids')}))[:8], 16)


def require_drive(folder, *, profile=ORIGINAL_PROFILE):
    from scripts.setup_cloudfixer_multibackbone_colab import require_task_drive
    return require_task_drive(folder, profile=profile)


def validate_full_gate(gate, campaign, profile_sha, *, now=None):
    now = time.time() if now is None else now
    fields = ('account_read_epoch','balance_cu','all_assignments_cu_per_h','remaining_eta_seconds','execution_window_seconds','reserve_cu')
    if any(type(gate.get(k)) not in (int,float) or not math.isfinite(gate[k]) for k in fields):
        raise ValueError('Finite current resource receipt required')
    if (gate.get('status')!='passed' or gate.get('campaign')!=campaign
            or gate.get('profile_payload_sha256')!=profile_sha
            or not 0 <= now-gate['account_read_epoch'] <= 600
            or gate['reserve_cu']<5 or gate['all_assignments_cu_per_h']<=0
            or gate['remaining_eta_seconds']<=0 or gate['execution_window_seconds']<120
            or gate['execution_window_seconds']-(now-gate['account_read_epoch'])<120
            or gate['execution_window_seconds']>gate['remaining_eta_seconds']
            or gate['balance_cu'] < gate['reserve_cu'] + gate['all_assignments_cu_per_h']*gate['execution_window_seconds']/3600):
        raise ValueError('Profile/current shared-balance gate failed')


def read_full_gate(store, key, campaign, profile_sha, *, now=None):
    if not isinstance(key,str) or not re.fullmatch('full_gate_[0-9a-f]{64}.json',key):
        raise ValueError('Attempt-specific immutable resource receipt key required')
    data=store.get(key)
    if key!='full_gate_'+sha(data)+'.json':
        raise ValueError('Resource receipt key/content SHA mismatch')
    gate=json.loads(data)
    validate_full_gate(gate,campaign,profile_sha,now=now)
    return sha(data)


def candidate_probabilities(logits):
    """Independent float64 recount of the saved per-candidate logit vectors."""
    import numpy as np

    logits = logits.astype(np.float64)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def frozen_receipt(environment, spec):
    prior = environment.get('frozen_prior_hash')
    heads = environment.get('frozen_head_hashes')
    if (environment.get('profile') != BASELINE_PROFILE or environment.get('policy') != BASELINE_POLICY
            or environment.get('cublas_workspace_config') != WORKSPACE_CONFIG
            or not is_digest(prior) or heads != spec['head_state_hashes']
            or set(heads) != set(HEADS) or not all(map(is_digest, heads.values()))):
        raise ValueError('Frozen prior/head-state receipt mismatch')
    return dict(prior=prior, heads=heads)


def seed0_digest_from_runtime(environment):
    seed0 = environment.get('seed0_rng_sha256')
    if not is_digest(seed0):
        raise ValueError('Baseline runtime receipt is missing the registered seed 0 RNG digest')
    return seed0


def reset_corruption_rng():
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def validate_baseline(values, unit, labels, frozen, seed0_rng_sha256):
    import numpy as np

    n = len(unit['ids'])
    arrays = (
        ('source_logits', (n, 3, 40)), ('terminal_logits', (n, 3, 40)),
        ('candidate_logits', (n, 3, 3, 40)), ('candidate_probabilities', (n, 3, 3, 40)),
        ('pm_probabilities', (n, 3, 40)), ('source_head_seconds', (3,)),
        ('candidate_head_seconds', (3, 3)), ('candidate_seconds', (3,)),
        ('source_seconds', ()), ('policy_seconds', ()), ('reconstruction_seconds_shared', ()), ('peak_mib', ()),
    )
    for field, shape in arrays:
        value = values[field]
        if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError('Invalid baseline array: ' + field)
    if (json.loads(values['unit'].item()) != unit
            or values['ids'].dtype.kind not in 'iu' or not np.array_equal(values['ids'], unit['ids'])
            or values['labels'].dtype.kind not in 'iu' or not np.array_equal(values['labels'], labels[unit['ids']])
            or values['profile'].item() != BASELINE_PROFILE
            or values['policy'].item() != canonical(BASELINE_POLICY).decode()
            or not np.array_equal(values['head_order'], HEADS)
            or values['candidate_indices'].dtype.kind not in 'iu'
            or not np.array_equal(values['candidate_indices'], [20, 10, 0])
            or values['denoiser_calls'].dtype.kind not in 'iu' or values['denoiser_calls'].item() != 9
            or values['geometry_seed'].dtype.kind not in 'iu' or values['geometry_seed'].item() != seed_for(unit)):
        raise ValueError('Baseline unit/policy/head/probe identity mismatch')
    if frozen is None or json.loads(values['frozen_state'].item()) != frozen:
        raise ValueError('Frozen unit state differs from runtime receipt')
    probabilities = candidate_probabilities(values['candidate_logits'])
    expected_pm = probabilities.mean(axis=1)
    pm = values['pm_probabilities']
    if (values['candidate_probabilities'].dtype != np.float64 or pm.dtype != np.float64
            or not np.array_equal(values['terminal_logits'], values['candidate_logits'][:, -1])
            or not np.allclose(values['candidate_probabilities'], probabilities, rtol=1e-12, atol=1e-14)
            or not np.allclose(pm, expected_pm, rtol=1e-12, atol=1e-14)
            or not np.array_equal(pm.argmax(axis=-1), expected_pm.argmax(axis=-1))):
        raise ValueError('Terminal or PM probability does not follow the saved candidates')
    for field, _ in arrays[5:]:
        if (values[field] < 0).any():
            raise ValueError('Negative measured timing/memory: ' + field)
    if (not (values['candidate_seconds'] > 0).all() or values['source_seconds'] <= 0
            or values['reconstruction_seconds_shared'] <= 0
            or not np.isclose(values['policy_seconds'],
                values['reconstruction_seconds_shared'] + values['candidate_seconds'].sum(), rtol=1e-12, atol=1e-14)):
        raise ValueError('Inconsistent baseline timings')
    if (values['source_head_seconds'].sum() > values['source_seconds'] + 1e-9
            or (values['candidate_head_seconds'].sum(axis=1) > values['candidate_seconds'] + 1e-9).any()):
        raise ValueError('Sequential head timings exceed enclosing wall time')
    hashes = values['candidate_cloud_sha256']
    if (hashes.shape != (3,) or not all(map(is_digest, hashes))
            or not is_digest(values['input_sha256'].item()) or not is_digest(values['mask_sha256'].item())
            or values['output_sha256'].item() != hashes[-1]
            or values['trajectory_sha256'].item() != trajectory_sha(unit, values)):
        raise ValueError('Baseline trajectory identity mismatch')
    if not is_digest(seed0_rng_sha256) or values['rng_before_sha256'].item() != seed0_rng_sha256:
        raise ValueError('Baseline RNG boundary differs from registered seed 0')
    state = json.loads(values['rng_after'].item())
    if (set(state) != {'python', 'numpy', 'cpu', 'cuda'} or len(state['python']) != 3
            or len(state['python'][1]) != 625 or state['numpy'][0] != 'MT19937'
            or len(state['numpy']) != 5 or len(state['numpy'][1]) != 624
            or any(not isinstance(state[k], list) or not state[k] or
                   any(type(v) is not int or not 0 <= v <= 255 for v in state[k]) for k in ('cpu', 'cuda'))
            or seed0_rng_sha256 != sha(canonical(state))):
        raise ValueError('Full RNG receipt does not prove unit-fork restoration')
    return values


def validate(data, unit, labels, *, profile=ORIGINAL_PROFILE, frozen=None, seed0_rng_sha256=None):
    import numpy as np
    methods_for(profile)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if len(z.namelist()) != len(set(z.namelist())) or sum(v.file_size for v in z.infolist()) > 100_000_000:
            raise ValueError('Unsafe prediction payload')
    with np.load(io.BytesIO(data), allow_pickle=False) as p:
        values = {k: p[k].copy() for k in p.files}
    if profile == BASELINE_PROFILE:
        try:
            return validate_baseline(values, unit, labels, frozen, seed0_rng_sha256)
        except (IndexError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Incomplete or invalid baseline payload') from exc
    if 'profile' in values:
        raise ValueError('Legacy CloudFixer cache cannot contain a profiled payload')
    n = len(unit['ids'])
    if (json.loads(str(values['unit'])) != unit or not np.array_equal(values['ids'], unit['ids'])
            or values['labels'].shape != (n,) or values['labels'].dtype.kind not in 'iu'
            or not np.array_equal(values['labels'], labels[unit['ids']])
            or int(values['denoiser_calls']) != 30):
        raise ValueError('Prediction identity/labels/native call count mismatch')
    for name in ('source_logits', 'frame_logits', 'cf_logits'):
        logits = values[name]
        if logits.shape != (n, 3, 40) or not np.isfinite(logits).all():
            raise ValueError('Invalid classifier predictions: '+name)
    if (values['seconds'].shape != (3,) or not np.isfinite(values['seconds']).all()
            or not (values['seconds'] > 0).all()
            or not math.isfinite(float(values['peak_mib'])) or float(values['peak_mib']) < 0):
        raise ValueError('Invalid measured timing/memory')
    if ('head_seconds' in values and (values['head_seconds'].shape != (3, 3)
            or not np.isfinite(values['head_seconds']).all() or (values['head_seconds'] < 0).any())):
        raise ValueError('Invalid head timing')
    for key in ('input_sha256', 'output_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(values[key])):
            raise ValueError('Invalid geometry identity')
    if (not re.fullmatch('[0-9a-f]{64}',str(values['rng_before_sha256']))
            or set(json.loads(str(values['rng_after']))) != {'python','numpy','cpu','cuda'}):
        raise ValueError('Full native RNG continuation required')
    return values


def recover(store, campaign, wanted, labels, ignored=(), *, profile=ORIGINAL_PROFILE,
            frozen=None, seed0_rng_sha256=None):
    class Remaining:
        def keys(self,prefix):return [k for k in store.keys(prefix) if k not in ignored]
        def get(self,key):return store.get(key)
    records = load_units(Remaining(), campaign)
    expected = {unit_key(u): u for u in wanted}
    if not set(records) <= set(expected):
        raise ValueError('Unexpected recovered work')
    for key, record in records.items():
        record['values'] = validate(record['data'], expected[key], labels, profile=profile, frozen=frozen,
                                    seed0_rng_sha256=seed0_rng_sha256)
    for corruption in {u['corruption'] for u in wanted}:
        keys = [unit_key(u) for u in wanted if u['corruption']==corruption]
        recovered = [k for k in keys if k in records]
        if recovered != keys[:len(recovered)]:
            raise ValueError('Recovered native stream must be a contiguous prefix')
    return records


def completed_corruptions(store, campaign, *, profile=ORIGINAL_PROFILE,
                          labels=None, frozen=None, seed0_rng_sha256=None):
    result={}
    for key in store.keys('completed_'):
        receipt=json.loads(store.get(key))
        corr=receipt['corruption']
        data=store.get(receipt['archive'])
        if (key!='completed_'+corr+'.json' or corr not in CORRUPTIONS
                or receipt['campaign']!=campaign or receipt['objects']!=2468
                or receipt['heads']!=list(HEADS) or sha(data)!=receipt['archive_sha256']):
            raise ValueError('Completed corruption archive/cursor mismatch')
        if profile == BASELINE_PROFILE:
            if receipt.get('profile') != profile or labels is None or frozen is None:
                raise ValueError('Baseline completion receipt is missing its profile or scope')
            validate_corruption_archive(data, receipt, store, labels, frozen, seed0_rng_sha256)
        result[corr]=receipt
    return result


def validate_corruption_archive(data, receipt, store, labels, frozen, seed0_rng_sha256):
    campaign = receipt['campaign']
    corruption = receipt['corruption']
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or sum(info.file_size for info in archive.infolist()) > 100_000_000
                or archive.testzip() is not None):
            raise ValueError('Baseline completed archive has invalid members or CRC')
        if not {'config.json', 'summary.csv', 'per_corruption.csv', 'environment.txt',
                'notes.md', 'command.txt'} <= set(names):
            raise ValueError('Baseline completed archive is missing reports or configuration')
        config = json.loads(archive.read('config.json'))
        spec = config['manifest']
        environment = json.loads(archive.read('environment.txt'))
        if (config.get('campaign') != campaign or sha(canonical(spec)) != campaign
                or manifest_profile(spec) != BASELINE_PROFILE or config.get('phase') != 'full'
                or config.get('artifact_scope') != 'single_corruption' or config.get('corruption') != corruption
                or config.get('expected_objects') != 2468
                or frozen_receipt(environment, spec) != frozen):
            raise ValueError('Baseline completed archive belongs to a different scope or frozen runtime')
        if seed0_digest_from_runtime(environment) != seed0_rng_sha256:
            raise ValueError('Completed archive RNG receipt differs from registered seed 0')

        class ArchivedRaw:
            def keys(self, prefix):
                return [name[4:] for name in names if name.startswith('raw/'+prefix)]

            def get(self, key):
                if 'raw/'+key not in names:
                    raise ValueError('Baseline completed archive is missing a raw member')
                return archive.read('raw/'+key)

        wanted = [u for u in units(campaign, 'full', spec['scientific_protocol']['batch_size'],
                                  profile=BASELINE_PROFILE) if u['corruption'] == corruption]
        records = recover(ArchivedRaw(), campaign, wanted, labels, profile=BASELINE_PROFILE, frozen=frozen,
                          seed0_rng_sha256=seed0_rng_sha256)
        if len(records) != len(wanted):
            raise ValueError('Baseline completed archive is missing committed units')
        raw_names = {name for name in names if name.startswith('raw/')}
        expected_raw = {'raw/'+key for key in records} | {'raw/'+r['payload'] for r in records.values()}
        if raw_names != expected_raw:
            raise ValueError('Baseline completed archive contains undeclared raw members')
        rows, summary = report_tables((records[unit_key(unit)] for unit in wanted), BASELINE_PROFILE)
        for name, table in (('per_corruption.csv', rows), ('summary.csv', summary)):
            if archive.read(name) != csv_bytes(table, campaign):
                raise ValueError('Baseline completed ZIP reports differ from raw records')
        if set(receipt.get('head_receipts', {})) != set(HEADS):
            raise ValueError('Baseline completion cursor is missing per-head receipts')
        for head in HEADS:
            saved = json.loads(store.get(receipt['head_receipts'][head]))
            expected = dict(campaign=campaign, profile=BASELINE_PROFILE, backbone=head, corruption=corruption,
                status='complete', archive=receipt['archive'], archive_sha256=receipt['archive_sha256'],
                rows=[r for r in rows if r['backbone']==head])
            if saved != expected:
                raise ValueError('Baseline completion head receipt does not match raw archive')


def report_rows(records, *, profile=ORIGINAL_PROFILE):
    import numpy as np
    groups = {}
    fields = BASELINE_FIELDS if profile == BASELINE_PROFILE else ORIGINAL_FIELDS
    methods = methods_for(profile)
    for r in records:
        u, p = r['unit'], r['values']
        for mode, (method, key) in enumerate(zip(methods, fields)):
            for head, name in enumerate(HEADS):
                g = groups.setdefault((method, name, u['corruption']),
                    dict(method=method, backbone=name, corruption=u['corruption'], n_examples=0,
                         n_correct=0, runtime_seconds=0., reconstruction_seconds_shared=0., peak_gpu_memory_mb=0.))
                g['n_examples'] += len(u['ids'])
                g['n_correct'] += int((p[key][:, head].argmax(-1) == p['labels']).sum())
                if profile == BASELINE_PROFILE:
                    native = float(p['reconstruction_seconds_shared']) if mode else 0.
                    head_cost = (p['source_head_seconds'][head] if mode == 0
                        else p['candidate_head_seconds'][-1, head] if mode == 1
                        else p['candidate_head_seconds'][:, head].sum())
                else:
                    native = float(p['seconds'][2]) if mode == 2 else 0.
                    head_cost = p.get('head_seconds', np.zeros((3, 3)))[mode, head]
                g['reconstruction_seconds_shared'] += native
                g['runtime_seconds'] += native + float(head_cost)
                g['peak_gpu_memory_mb'] = max(g['peak_gpu_memory_mb'], float(p['peak_mib']))
    return [dict(v, accuracy=v['n_correct']/v['n_examples'], seed=0, severity=5,
                 dataset='modelnet40_c', status='complete' if v['n_examples']==2468 else 'partial')
            for v in groups.values()]


def report_tables(records, profile):
    rows = report_rows(records, profile=profile)
    if not rows:
        return [], []
    summary = []
    for method in methods_for(profile):
        for head in HEADS:
            group = [r for r in rows if r['method']==method and r['backbone']==head]
            count = sum(r['n_examples'] for r in group); correct = sum(r['n_correct'] for r in group)
            summary.append(dict(dataset='modelnet40_c', severity=5, method=method, backbone=head,
                seed=0, n_corruptions=len(group), total_examples=count, total_correct=correct,
                micro_accuracy=correct/count, macro_accuracy=sum(r['accuracy'] for r in group)/len(group),
                total_runtime_seconds=sum(r['runtime_seconds'] for r in group),
                status='complete' if len(group)==15 and count==37020 else 'partial'))
    return rows, summary


def csv_bytes(rows, campaign):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['run_id', *rows[0]])
    writer.writeheader()
    writer.writerows(dict(run_id=campaign, **row) for row in rows)
    return stream.getvalue().encode()


def write_reports(output, records, campaign, *, profile=ORIGINAL_PROFILE):
    rows, summary = report_tables(records, profile)
    if rows:
        (output/'per_corruption.csv').write_bytes(csv_bytes(rows, campaign))
        (output/'summary.csv').write_bytes(csv_bytes(summary, campaign))


def checked_runtime(args, spec):
    import numpy as np
    import torch
    from scripts.setup_cloudfixer_multibackbone_colab import gpu_architecture,preload_head_native
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Single registered CUDA device required')
    accelerator=spec.get('accelerator','A100')
    gpu_architecture(accelerator,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
    setup = json.loads(Path('/content/cli_setup_verification.json').read_text())
    if setup.get('status') != 'passed' or sys.version_info[:2] != (3, 8):
        raise ValueError('Canonical10 setup and pinned Python3.8 required')
    for name, identity in spec['files'].items():
        source.verify_identity(source.asset_path(args.asset_root, name), identity)
    for name, identity in spec['official_files'].items():
        source.verify_identity(source.asset_path(args.official_root, name), identity)
    for name, identity in spec['overlay_files'].items():
        source.verify_identity(source.asset_path(ROOT, name), identity)
    for filename, digest in spec['dataset_files'].items():
        path = args.dataset_root/filename
        if source.file_identity(path)['sha256'] != digest:
            raise ValueError('Dataset identity mismatch: '+filename)
    if source.file_identity(args.diffusion)['sha256'] != spec['diffusion_sha256']:
        raise ValueError('Diffusion checkpoint identity mismatch')
    os.chdir(args.official_root); sys.path.insert(0, str(args.official_root))
    import adapt
    from cloudfixer_diffusion import verify_gamma
    from utils.pc_utils import scale_to_unit_cube_torch, rotate_shape_tensor
    from pointnet2_ops import pointnet2_utils
    from types import SimpleNamespace
    model_args = SimpleNamespace(model='transformer', probabilistic_model='diffusion', diffusion_steps=500,
        diffusion_noise_schedule='polynomial_2', diffusion_noise_precision=1e-5, diffusion_loss_type='l2')
    model = adapt.get_model(model_args, torch.device('cuda'))
    intended = model.gamma.gamma.detach().clone()
    model.load_state_dict(torch.load(args.diffusion, map_location='cpu'), strict=True)
    gamma = verify_gamma(model.gamma.gamma, intended)
    model = model.cuda().eval().requires_grad_(False)
    preload_head_native(args.asset_root.parent,accelerator)
    heads = {}
    for name in HEADS:
        config = dict(spec['head_configs'][name], asset_root=str(args.asset_root))
        heads[name] = source.load_classifier(config, spec)
        if source.state_hash(heads[name]) != spec['head_state_hashes'][name]:
            raise ValueError('Loaded head differs from accepted source: '+name)
    options = SimpleNamespace(**dict(CF_OPTIONS, rotate=True, dataset='modelnet40c_gaussian_5',
        dataset_dir=str(args.dataset_root), random_seed=0, batch_size=spec['scientific_protocol']['batch_size']))
    def fps(points, number):
        points = points.contiguous()
        idx = pointnet2_utils.furthest_point_sample(points, number)
        return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    def geometry(points):
        return rotate_shape_tensor(scale_to_unit_cube_torch(points), 'x', np.pi/2).contiguous()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    # Force lazy native loads before sealing the runtime signature.
    from pytorch3d.ops import knn_points
    probe=(torch.arange(96,device='cuda',dtype=torch.float32).reshape(1,32,3)/96).requires_grad_(True)
    d1,d2,_,_=adapt.chamfer_dist_fn(probe,probe.detach()+.01)
    (d1.mean()+d2.mean()).backward()
    if not torch.isfinite(probe.grad).all() or not torch.isfinite(knn_points(probe.detach(),probe.detach(),K=5).dists).all():
        raise ValueError('Official CF native forward/backward probe failed')
    if fps(probe.detach(),8).shape!=(1,8,3):
        raise ValueError('Matched classifier FPS native probe failed')
    torch.cuda.synchronize()
    return adapt, model, heads, options, fps, geometry, gamma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','asset-root','official-root','diffusion','dataset-root','drive-folder','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke','profile','full'), required=True)
    parser.add_argument('--full-gate-key')
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text()); campaign = sha(canonical(spec))
    profile = manifest_profile(spec)
    baseline = profile == BASELINE_PROFILE
    if baseline:
        require_workspace_config()
    batch_size=spec['scientific_protocol']['batch_size']
    args.drive_folder=require_drive(args.drive_folder, profile=profile)
    store = MountStore(args.drive_folder)
    store.put('task.json', canonical(spec))
    if args.phase != 'smoke':
        gate = json.loads(store.get('smoke_gate.json'))
        if gate.get('campaign') != campaign or gate.get('status') != 'passed':
            raise ValueError('Current campaign smoke gate required')
    full_gate_sha=None
    if args.phase=='full':
        profile_records=load_units(MountStore(store.folder/'profile'),campaign)
        profile_unit = units(campaign, 'profile', batch_size, profile=profile)[0]
        profile_key=unit_key(profile_unit)
        if set(profile_records)!={profile_key}:
            raise ValueError('Exactly one current registered-batch profile required')
        import numpy as np
        profile_runtime = json.loads(store.get('runtime_signature.json')) if baseline else None
        validate(profile_records[profile_key]['data'], profile_unit,
                 np.load(args.dataset_root/'label.npy',allow_pickle=False).reshape(-1),
                 profile=profile,
                 frozen=frozen_receipt(profile_runtime, spec) if baseline else None,
                 seed0_rng_sha256=seed0_digest_from_runtime(profile_runtime) if baseline else None)
        full_gate_sha=read_full_gate(store,args.full_gate_key,campaign,profile_records[profile_key]['sha256'])
    if args.output.exists():
        raise FileExistsError('Fresh output required; recovery uses immutable Drive units')
    args.output.mkdir(parents=True); (args.output/'raw').mkdir()
    for name, data in [('command.txt', canonical(sys.argv)), ('config.json', canonical(dict(manifest=spec, phase=args.phase, campaign=campaign,full_gate_key=args.full_gate_key,full_gate_sha256=full_gate_sha)))]:
        (args.output/name).write_bytes(data)
    baseline_notes = ('Frozen accepted ModelNet40 heads with shared author-frame XYZ proximal N8/K80.\n'
        'PM averages float64 candidate softmax vectors at pre-correction 20/10 and true extra-call t0.\n'
        'Terminal is the same observed trajectory at t0. Each head sees the author inverse-axis/unit-sphere output, then FPS1024.\n'
        'Per-head costs include shared reconstruction and one (terminal) or three (PM) forwards. The terminal-only wall time was not measured.\n'
        'Geometry uses unit-addressed RNG and is saved once for three heads. Comparisons describe one seed.\n')
    (args.output/'notes.md').write_text(baseline_notes if baseline else
        'Frozen accepted alternative heads. Original author geometry; plainCF/vote1.\n'
        'Geometry input: author rotateTrue/unitSTD/mask/ind. Output: author unit-sphere/inverse-axis, then accepted headFPS1024.\n'
        'Source_raw uses original arrays; source_cf_frame is separate. Different LION prior/output preprocessing and RNG: whole pipelines only.\n'
        'Native optimizer is nondeterministic. Shared geometry/head order is fixed; classifier RNG is forked.\n'
        'Peak field is MiB. Per-head cost includes the shared reconstruction; do not sum3heads as3independent CF runs.\n')
    import numpy as np
    import torch
    budget = Budget(store, cap=spec['budget_seconds'])
    if args.phase=='full':
        gate=json.loads(store.get(args.full_gate_key))
        store.put('used_'+full_gate_sha+'.json',canonical(dict(output=str(args.output.resolve()))))
        remaining_window=gate['execution_window_seconds']-(time.time()-gate['account_read_epoch'])
        budget.cap=min(budget.cap,budget.base+remaining_window)
    deadline=threading.Timer(max(.001,budget.remaining()),lambda:os.kill(os.getpid(),signal.SIGKILL))
    deadline.daemon=True;deadline.start()
    adapt, model, heads, options, fps, geometry, gamma = checked_runtime(args, spec)
    original_model = source.state_hash(model)
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0), gamma=gamma,
        frozen_head_hashes={n:source.state_hash(h) for n,h in heads.items()},
        native_optimizer='non-deterministic KNN backward; no exact cross-VM equality claim')
    seed0_rng_sha256 = None
    if baseline:
        reset_corruption_rng()
        seed0_rng_sha256 = sha(canonical(rng_state()))
        del environment['native_optimizer']
        environment.update(profile=profile, policy=BASELINE_POLICY, frozen_prior_hash=original_model,
                           cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                           seed0_rng_sha256=seed0_rng_sha256)
    validation = dict(profile=profile, frozen=frozen_receipt(environment, spec) if baseline else None,
                      seed0_rng_sha256=seed0_rng_sha256)
    native = {};native_paths={}
    for name, module in sorted(sys.modules.copy().items()):
        path = str(getattr(module,'__file__','') or '')
        if path.endswith('.so') and any(s in path for s in ('pytorch3d','knn_cuda','pointnet2','chamfer','torch_extensions')):
            native[name]=source.file_identity(Path(path))
            native_paths[name]=path
    if not native:
        raise ValueError('Loaded native ELF identities required')
    environment['native_elf_identities']=native
    from scripts.setup_cloudfixer_multibackbone_colab import native_snapshot
    native_snapshot(store,native_paths)
    store.put('runtime_signature.json',canonical(environment))
    (args.output/'environment.txt').write_bytes(canonical(environment))
    labels = np.load(args.dataset_root/'label.npy', allow_pickle=False).reshape(-1)
    all_wanted = units(campaign, args.phase, batch_size, profile=profile); cache = MountStore(store.folder/args.phase)
    completed = completed_corruptions(store, campaign, labels=labels, **validation) if args.phase=='full' else {}
    wanted=[u for u in all_wanted if u['corruption'] not in completed]
    ignored={unit_key(u) for u in all_wanted if u['corruption'] in completed}
    accepted = recover(cache, campaign, wanted, labels, ignored, **validation)
    recovered_at_start=len(accepted)
    current = None
    from torch.utils.data._utils.collate import default_collate
    def timed(fn):
        torch.cuda.synchronize(); start=time.perf_counter(); value=fn(); torch.cuda.synchronize()
        return value, time.perf_counter()-start
    def classify(points, seed=None):
        logits, seconds = [], []
        state=rng_state() if seed is not None else None
        try:
            with torch.no_grad():
                for i, name in enumerate(HEADS):
                    if seed is not None:
                        torch.manual_seed(seed+i); torch.cuda.manual_seed_all(seed+i)
                    def forward():
                        x, features = source.preprocess(points, spec['head_configs'][name], fps)
                        if args.phase!='smoke':
                            return heads[name](x, features=features)
                        reference_rng=rng_state()
                        wrapped=heads[name](x, features=features)
                        wrapped_rng=sha(canonical(rng_state()))
                        restore_rng(reference_rng)
                        direct=heads[name].model({'pos':x,'x':features}) if name=='pointnext' else heads[name].model(x)
                        if not torch.equal(wrapped,direct) or sha(canonical(rng_state()))!=wrapped_rng:
                            raise ValueError('Frozen source-wrapper/direct model or RNG parity failed: '+name)
                        return wrapped
                    values, elapsed = timed(forward); logits.append(values if baseline else values.cpu().numpy())
                    seconds.append(elapsed)
        finally:
            if state is not None:
                restore_rng(state)
        return (torch.stack(logits, dim=1) if baseline else np.stack(logits, axis=1)), seconds

    def check_frozen():
        prior = source.state_hash(model)
        head_hashes = {name: source.state_hash(head) for name, head in heads.items()}
        if prior != original_model or head_hashes != spec['head_state_hashes']:
            raise ValueError('Frozen prior/head state changed')
        return dict(prior=prior, heads=head_hashes)
    for unit in wanted:
        key = unit_key(unit)
        if key not in accepted and args.phase=='full' and budget.remaining()<90:
            write_reports(args.output,list(accepted.values()),campaign, profile=profile)
            print('CF_MULTIBACKBONE_PARTIAL',len(accepted),len(wanted),'funded window ended; Drive receipts retained',flush=True)
            return
        if current != unit['corruption']:
            reset_corruption_rng()
            options.dataset='modelnet40c_'+unit['corruption']+'_5'
            dataset=adapt.ModelNet40C(options, partition='test'); current=unit['corruption']
            raw=np.load(args.dataset_root/('data_'+current+'_5.npy'), mmap_mode='r', allow_pickle=False)
        before_sha=sha(canonical(rng_state()))
        if baseline and before_sha != seed0_rng_sha256:
            raise ValueError('Live baseline RNG boundary differs from registered seed 0')
        if key in accepted:
            if str(accepted[key]['values']['rng_before_sha256'])!=before_sha:
                raise ValueError('Recovered native input RNG differs from continuous seed0 stream')
            restore_rng(json.loads(str(accepted[key]['values']['rng_after'])))
        if key not in accepted:
            with budget.operation(1800):
                seed = seed_for(unit)
                batch=default_collate([dataset[i] for i in unit['ids']])
                x, mask, indices = [batch[i].cuda() for i in (0,2,3)]
                if not np.array_equal(batch[1].numpy().reshape(-1), labels[unit['ids']]):
                    raise ValueError('Official loader original-ID labels disagree')
                raw_points=torch.as_tensor(np.asarray(raw[unit['ids']]).copy(), device='cuda', dtype=torch.float32)
                if args.phase=='smoke':
                    reference=raw_points-raw_points.mean(1,keepdim=True)
                    reference=reference/reference.norm(dim=-1,keepdim=True).amax(1,keepdim=True)
                    if not torch.allclose(geometry(x),reference,atol=2e-5,rtol=2e-5):
                        raise ValueError('Author/input/classifier coordinate round-trip failed')
                torch.cuda.reset_peak_memory_stats()
                classification_rng=sha(canonical(rng_state()))
                (src, src_s), source_wall = timed(lambda: classify(raw_points, seed+1000))
                if baseline:
                    prediction = proximal_readout(model, classify, x, mask, unit, geometry=geometry, timed=timed)
                    if sha(canonical(rng_state())) != classification_rng:
                        raise ValueError('Baseline source or proximal readout shifted the outer RNG')
                    prediction.update(source_logits=src.cpu().numpy(), source_head_seconds=src_s,
                                      source_seconds=source_wall, frozen_state=canonical(check_frozen()).decode(),
                                      peak_mib=torch.cuda.max_memory_allocated()/2**20)
                else:
                    (frame, frame_s), frame_wall = timed(lambda: classify(geometry(x), seed+1000))
                    if sha(canonical(rng_state()))!=classification_rng:
                        raise ValueError('Classifier controls shifted geometry RNG')
                    repaired, native_s=timed(lambda:adapt.cloudfixer(options, torch.nn.DataParallel(model), x, mask, indices).detach())
                    classification_rng=sha(canonical(rng_state()))
                    (cf, cf_s), _=timed(lambda: classify(geometry(repaired), seed+1000))
                    if sha(canonical(rng_state()))!=classification_rng:
                        raise ValueError('Classifier after adaptation shifted native RNG')
                    check_frozen()
                    prediction = dict(source_logits=src, frame_logits=frame, cf_logits=cf,
                        seconds=[source_wall,frame_wall,native_s], head_seconds=[src_s,frame_s,cf_s],
                        peak_mib=torch.cuda.max_memory_allocated()/2**20, denoiser_calls=30,
                        input_sha256=array_sha(x), output_sha256=array_sha(repaired))
                stream=io.BytesIO()
                np.savez_compressed(stream, unit=json.dumps(unit,sort_keys=True), ids=unit['ids'], labels=labels[unit['ids']],
                    **prediction, rng_before_sha256=before_sha, rng_after=canonical(rng_state()).decode())
                data=stream.getvalue()
                values = validate(data, unit, labels, **validation)
                commit_unit(cache,unit,data)
                accepted[key]=dict(unit=unit,data=data,values=values)
        record=accepted[key];receipt=json.loads(cache.get(key))
        (args.output/'raw'/key).write_bytes(canonical(receipt))
        (args.output/'raw'/receipt['payload']).write_bytes(record['data'])
        print('CF_MULTIBACKBONE_BLOCK',unit['corruption'],unit['ids'][0],len(accepted),len(wanted),flush=True)
        if unit==wanted[-1] or unit['ids'][-1]==2467:
            check_frozen()
            write_reports(args.output,list(accepted.values()),campaign, profile=profile)
            for name in ('summary.csv','per_corruption.csv'):
                store.put('report_'+args.output.name+'_'+str(len(accepted))+'_'+name,(args.output/name).read_bytes())
            if args.phase=='full':
                scoped=[r for r in accepted.values() if r['unit']['corruption']==unit['corruption']]
                if baseline:
                    scoped.sort(key=lambda r: r['unit']['ids'][0])
                    scoped_wanted = [u for u in all_wanted if u['corruption'] == unit['corruption']]
                    if [r['unit'] for r in scoped] != scoped_wanted:
                        raise ValueError('Full baseline archive requires all registered original-ID blocks')
                scoped_dir=args.output/('scope_'+unit['corruption']);scoped_dir.mkdir()
                write_reports(scoped_dir,scoped,campaign, profile=profile)
                archive=args.output/(unit['corruption']+'.zip')
                with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                    for name in ('command.txt','environment.txt','notes.md'):
                        z.write(args.output/name,name)
                    z.writestr('config.json',canonical(dict(manifest=spec,campaign=campaign,phase=args.phase,
                        artifact_scope='single_corruption',corruption=unit['corruption'],expected_objects=2468,
                        campaign_complete=False,full_gate_key=args.full_gate_key,full_gate_sha256=full_gate_sha)))
                    for name in ('summary.csv','per_corruption.csv'):
                        z.write(scoped_dir/name,name)
                    for r in scoped:
                        receipt=cache.get(unit_key(r['unit']))
                        z.writestr('raw/'+unit_key(r['unit']),receipt)
                        z.writestr('raw/'+json.loads(receipt)['payload'],r['data'])
                data=archive.read_bytes()
                destination='corruption_'+args.output.name+'_'+unit['corruption']+'.zip'
                store.put(destination,data)
                print('CF_MULTIBACKBONE_CORRUPTION_ARCHIVE',json.dumps(dict(path=str(archive),drive=str(store.folder/destination),sha256=sha(data),bytes=len(data))),flush=True)
                head_receipts = {}
                scoped_rows = report_rows(scoped, profile=profile)
                for head in HEADS:
                    result=dict(campaign=campaign,backbone=head,corruption=unit['corruption'],
                        status='complete',archive=destination,archive_sha256=sha(data),
                        rows=[r for r in scoped_rows if r['backbone']==head])
                    if baseline:
                        result['profile'] = profile
                    name='head_'+args.output.name+'_'+unit['corruption']+'_'+head+'.json'
                    store.put(name,canonical(result))
                    head_receipts[head] = name
                    print('CF_MULTIBACKBONE_HEAD_COMPLETE',str(store.folder/name),flush=True)
                cursor=dict(campaign=campaign,corruption=unit['corruption'],objects=2468,heads=list(HEADS),
                    archive=destination,archive_sha256=sha(data))
                if baseline:
                    cursor.update(profile=profile, head_receipts=head_receipts)
                    validate_corruption_archive(data, cursor, store, labels, validation['frozen'], seed0_rng_sha256)
                store.put('completed_'+unit['corruption']+'.json',canonical(cursor))
                completed[unit['corruption']]=cursor
    if args.phase=='smoke':
        key='smoke_gate.json' if recovered_at_start==len(wanted) else 'smoke_produced.json'
        store.put(key,canonical(dict(campaign=campaign,status='passed',units=len(accepted),environment=environment)))
    if args.phase=='full':
        print('CF_MULTIBACKBONE_DONE',args.phase,'completed_corruptions',len(completed),15,flush=True)
    else:
        print('CF_MULTIBACKBONE_DONE',args.phase,len(accepted),flush=True)
    deadline.cancel()


if __name__=='__main__':
    main()
