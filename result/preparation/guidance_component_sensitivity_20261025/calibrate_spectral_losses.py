"""Label-free first-step spectral calibration; run only inside the approved GPU phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import sys
import time
import traceback
import zipfile

ROOT = Path('/content/studies/guidance-components')
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
SESSION = 'tta-guidance-components-a100-20261025-r1'
OLD_MANIFEST = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
OLD_SOURCE = 'f5bca11a543fd5463ef5a09b26cf216b2bd049523d61a97da24e184e5576d0c6'
ACCEPTANCE_SHA = 'e978c558f4ec70244c5ad14030fc09191bf78435541466866744366fb73a078c'
PHASE = dict(bytes=405066186, sha256='8abd0ec00c71996407d0fabc3933a7984c2e9860c8844e2d2940bff1db15901d')
KINDS = ('legacy_l2', 'huber', 'power')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
TINY = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(path, expected):
    observed = identity(path)
    require(observed == {key: expected[key] for key in ('bytes', 'sha256')}, 'Identity mismatch: ' + str(path))
    return observed


def owned(path, root=None):
    root = ROOT if root is None else root
    path, root = Path(path).absolute(), Path(root).absolute()
    require(root in path.parents and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents),
            'Path must be inside the owned root without symlinks: ' + str(path))
    require(root.resolve() in path.resolve().parents, 'Resolved path escaped owned root')
    return path


def json_from_zip(path, name):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate evidence ZIP members')
        return json.loads(archive.read(name))


def coefficient_report(rows):
    require(len(rows) == 2100 and all(sum(row['dataset'] == dataset for row in rows) == 1050 for dataset in DATASETS),
            'Calibration requires2100 events with equal dataset representation')
    keys = {(row['dataset'], row['corruption'], row['index']) for row in rows}
    require(len(keys) == 2100, 'Duplicate original-ID/corruption events')
    groups = {(row['dataset'], row['corruption']) for row in rows}
    require(len(groups) == 30 and all(sum((row['dataset'], row['corruption']) == group for row in rows) == 70 for group in groups),
            'Calibration requires30 equally represented70-object batches')
    result = {}
    for kind in KINDS:
        ratios, tiny_c, tiny_s = [], 0, 0
        for row in rows:
            require(set(row['norms']) == {'chamfer', *KINDS}, 'Missing calibration loss norms')
            for loss in ('chamfer', kind):
                require(set(row['norms'][loss]) == {'local', 'style'} and
                        all(math.isfinite(value) and value >= 0 for value in row['norms'][loss].values()),
                        'Nonfinite/negative gradient norm')
            c, s = row['norms']['chamfer']['local'], row['norms'][kind]['local']
            tiny_c += c <= TINY
            tiny_s += s <= TINY
            if c > TINY and s > TINY:
                ratio = c / s
                require(math.isfinite(ratio), 'Nonfinite per-object calibration ratio')
                ratios.append(ratio)
        require(len(ratios) >= 1890, 'Fewer than90% valid local-gradient pairs: ' + kind)
        scalar = statistics.median(ratios)
        require(math.isfinite(scalar) and 1e-12 <= scalar <= 1e12, 'Calibration scalar outside declared bounds: ' + kind)
        result[kind] = dict(coefficient=scalar, valid_pairs=len(ratios), total_pairs=2100,
                            tiny_chamfer_local=tiny_c, tiny_spectral_local=tiny_s,
                            ratio_min=min(ratios), ratio_max=max(ratios))
    return result


def validate_native_parity(relative, repeat):
    require(all(math.isfinite(value) and value >= 0 for value in (relative, repeat)), 'Nonfinite/negative native parity difference')
    require(repeat <= 1e-3 and relative <= 1e-3 and relative <= max(1e-5, 3 * repeat),
            'Native helper/inline gradient parity exceeded its predeclared repeatability floor/tolerance')


def accepted_caches(receipt_path):
    """Bind metadata to the accepted physical phase, without reading target labels."""
    accepted_path = ROOT / 'accepted_development.json'
    require(identity(accepted_path)['sha256'] == ACCEPTANCE_SHA, 'Unrecognized independent development acceptance')
    accepted = json.loads(accepted_path.read_text())
    acceptance_receipt = json.loads((ROOT / 'accepted_development_receipt.json').read_text())
    require(accepted['status'] == 'accepted' and accepted['manifest_sha256'] == OLD_MANIFEST
            and accepted['source_snapshot_sha256'] == OLD_SOURCE and acceptance_receipt['status'] == 'accepted',
            'Accepted development/source proof missing')
    verify(accepted_path, acceptance_receipt['acceptance'])
    acceptance_zip = owned(acceptance_receipt['archive']['drive_path'], DRIVE)
    verify(acceptance_zip, acceptance_receipt['archive'])
    require(json_from_zip(acceptance_zip, 'accepted_development.json') == accepted, 'Physical acceptance bytes differ')
    phase = accepted['phase_archive']
    require({key: phase[key] for key in PHASE} == PHASE, 'Accepted development phase changed')
    phase_path = owned(phase['drive_path'], DRIVE)
    verify(phase_path, PHASE)
    receipt = json.loads(Path(receipt_path).read_text())
    require({key: receipt['archive'][key] for key in PHASE} == PHASE
            and receipt['state'] == json_from_zip(phase_path, 'state.json'), 'Cache receipt differs from accepted archived phase')
    state = receipt['state']
    require(state['status'] == 'complete' and state['mode'] == 'development' and state['session'] == SESSION
            and state['manifest_sha256'] == OLD_MANIFEST and state['source_snapshot_sha256'] == OLD_SOURCE,
            'Foreign/incomplete cache-record phase')
    jobs = state['jobs']
    require(len(jobs) == 8 and {(job['dataset'], job['arm']) for job in jobs} ==
            {(dataset, arm) for dataset in DATASETS for arm in ('combined', 'chamfer', 'spectral', 'unguided')},
            'Development coverage mismatch')
    records, traces = {}, {}
    for job in jobs:
        require(job['status'] == 'complete', 'Incomplete cached-state producer')
        for record in job['cache']:
            path = str(owned(record['local_path']))
            owned(record['drive_path'], DRIVE)
            if path in records:
                require(records[path] == record, 'Cached state changed across arms')
            records[path] = record
        for corruption, trace in job['traces'].items():
            key = (job['dataset'], corruption)
            preparation = dict(initial_hashes=trace['initial_hashes'], paired_state=trace['paired_state'])
            if key in traces:
                require(traces[key] == preparation, 'Accepted arms did not share preparation')
            traces[key] = preparation
    require(len(records) == len(traces) == 30, 'Require30 accepted development preparations')
    return records, traces, dict(acceptance=identity(accepted_path), phase_archive=phase, cache_receipt=identity(receipt_path))


def cpu_proof(manifest):
    prepared = ROOT / 'prepared_v1'
    require((prepared / '.complete').read_text().strip() == OLD_MANIFEST, 'Accepted CPU proof marker changed')
    receipt = json.loads((prepared / 'accepted_cpu_preflight_receipt.json').read_text())
    require(receipt['state']['status'] == 'complete' and receipt['state']['strict_loads'] == 8
            and receipt['state']['cuda_initialized'] is False, 'Eight-load no-CUDA CPU proof missing')
    verify(owned(receipt['archive']['local_path']), receipt['archive'])
    verify(owned(receipt['archive']['drive_path'], DRIVE), receipt['archive'])
    path = prepared / 'cpu_preflight.json'; verify(path, receipt['report_identity'])
    proof = json.loads(path.read_text())
    require(json_from_zip(receipt['archive']['drive_path'], 'cpu_preflight.json') == proof, 'Physical CPU proof/report mismatch')
    require(proof['status'] == 'complete' and proof['manifest_sha256'] == OLD_MANIFEST
            and proof['source_snapshot_sha256'] == OLD_SOURCE and proof['session'] == SESSION
            and proof['cuda_initialized'] is False and proof['torch'] == '2.1.2+cu121'
            and proof['native']['no_build'] is True and len(proof['models']) == 8,
            'CPU proof source/environment mismatch')
    require({(row['dataset'], row['backbone']) for row in proof['models']} ==
            {(dataset, head) for dataset in DATASETS for head in ('dgcnn', 'curvenet', 'pointnext', 'pointmae')}, 'CPU proof case mismatch')
    for row in proof['models']:
        load = row['checkpoint_load']
        contract = (2468, 40, 'official_modelnet40_test') if row['dataset'] == DATASETS[0] else (581, 15, 'main_split')
        require((row['n_examples'], row['num_classes'], row['split']) == contract, 'CPU proof dataset contract mismatch')
        require(load['strict'] is True and load['missing_keys'] == [] and load['unexpected_keys'] == []
                and row['cpu_frozen_eval'] is True and row['forward_executed'] is False, 'CPU checkpoint/frozen proof failed')
    old_path = prepared / 'assets/manifest.json'
    require(identity(old_path)['sha256'] == OLD_MANIFEST, 'Original accepted asset manifest changed')
    old = json.loads(old_path.read_text())
    fixed = [name for name in old['files'] if name.startswith('source_snapshot/models/') or
             name in ('source_snapshot/graph_spectral_v2.py', 'source_snapshot/default_config.py', 'source_snapshot/utilities_3dd_tta.py')]
    require(fixed and all(manifest['files'].get(name) == old['files'][name] for name in fixed), 'Frozen LION/prior/graph source changed')
    require(manifest['guidance']['lion_checkpoint'] == old['guidance']['lion_checkpoint']
            and manifest['guidance']['lion_config'] == old['guidance']['lion_config'], 'Pinned LION assets changed')
    return proof, dict(report=identity(path), archive=receipt['archive'], frozen_source_files=fixed)


def load_cache(record, initial, device):
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state_hash, tensor_hash
    path = owned(record['local_path']); verify(path, record)
    verify(owned(record['drive_path'], DRIVE), record)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        require(metadata['schema_version'] == 1 and metadata['input_sha256'] == initial['input_points']
                and re.fullmatch('[0-9a-f]{64}', metadata['rng_before_sha256']) is not None, 'Cache input/RNG schema mismatch')
        count = metadata['cuda_devices']
        require(type(count) is int and count == torch.cuda.device_count() == 1, 'Cache CUDA device topology changed')
        expected = {'metadata', 'numpy_rng_keys', 'torch_rng_cpu', 'shape_latent', 'local_latent', 'noise'} | {'torch_rng_cuda_%d' % i for i in range(count)}
        require(set(archive.files) == expected and set(metadata['tensor_hashes']) == {'shape_latent', 'local_latent', 'noise'}, 'Cache array schema mismatch')
        tensors = {}
        for name in ('shape_latent', 'local_latent', 'noise'):
            array = archive[name].copy()
            require(array.dtype == np.float32 and array.ndim == 4 and array.shape[0] == 70
                    and array.shape[1] > 0 and array.shape[2:] == (1, 1) and np.isfinite(array).all(), 'Invalid finite B70 cached latent/noise')
            value = torch.from_numpy(array)
            require(tensor_hash(value) == metadata['tensor_hashes'][name] == initial[name], 'Cache tensor hash mismatch: ' + name)
            tensors[name] = value.to(device)
        require(tensors['local_latent'].shape == tensors['noise'].shape == (70, 8192, 1, 1), 'Invalid local latent/noise dimensions')
        py = metadata['python_rng']; nr = metadata['numpy_rng']
        state = ((py[0], tuple(py[1]), py[2]), (nr[0], archive['numpy_rng_keys'].copy(), *nr[1:]),
                 torch.from_numpy(archive['torch_rng_cpu'].copy()),
                 [torch.from_numpy(archive['torch_rng_cuda_%d' % i].copy()) for i in range(count)])
        require(_rng_state_hash(state) == metadata['rng_after_sha256'], 'Cached post-preparation RNG hash mismatch')
    return tensors, state, metadata


def measure_first_step(lion, graph, tensors, rng, initial, horizon, chamfer):
    import torch
    from diffusers import DDIMScheduler
    from legacy_trajectory import _restore_rng, rng_hash, tensor_hash
    from spectral_guidance_losses import spectral_guidance_loss
    h0, z0, noise = (tensors[name] for name in ('local_latent', 'shape_latent', 'noise'))
    scheduler = DDIMScheduler(beta_end=.02, beta_schedule='linear', beta_start=.0001,
        clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=False)
    scheduler.set_timesteps(100, device=h0.device)
    timesteps = scheduler.timesteps[-horizon:]; t = timesteps[0]
    alpha = scheduler.alphas_cumprod[t]
    with torch.no_grad():
        reference, basis, eigenvalues = graph(h0.view(70, 2048, 4))
        style = lion.vae.global2style(z0).detach().clone()
        local = torch.sqrt(alpha) * h0 + noise * torch.sqrt(1 - alpha)
    require(not graph.use_4d_gft, 'Calibration requires unchanged XYZ legacy graph')
    for name, value in dict(style=style, noisy_local=local, timesteps=timesteps, alpha_bar=alpha).items():
        require(tensor_hash(value) == initial[name], 'Recreated initial state differs: ' + name)
    _restore_rng(rng)
    before_rng = rng_hash()
    local.requires_grad_(True); style.requires_grad_(True)
    time_batch = torch.ones(70, dtype=torch.int64, device=h0.device) * (t + 1)
    prediction = lion.priors[1](x=local, t=time_batch.float(), condition_input=style, clip_feat=None)
    clean = scheduler.step(prediction, t, local).pred_original_sample.view(70, 2048, 4)
    predicted = torch.bmm(basis.transpose(1, 2), clean[:, :, :3])
    distances = chamfer(clean[:, :, :3], h0.view(70, 2048, 4)[:, :, :3])
    loss_c = sum(torch.sort(distance, dim=1).values[:, :int(2048 * .95)].sum() for distance in distances[:2])
    losses = dict(chamfer=loss_c)
    losses.update({kind: spectral_guidance_loss(predicted, reference, eigenvalues, 2., 800, kind) for kind in KINDS})
    weights = torch.exp(-2. * eigenvalues); weights[:, 800:] = 0.
    inline = (weights.unsqueeze(-1) * (predicted - reference)**2).sum() / (3 * 2048)
    require(torch.equal(inline, losses['legacy_l2']), 'Legacy inline/helper scalar differs')
    inline_direct = torch.autograd.grad(inline, predicted, retain_graph=True)[0]
    helper_direct = torch.autograd.grad(losses['legacy_l2'], predicted, retain_graph=True)[0]
    require(torch.equal(inline_direct, helper_direct), 'Legacy inline/helper direct coefficient gradient differs')
    inline_grad = torch.autograd.grad(inline, (local, style), retain_graph=True)
    repeated_grad = torch.autograd.grad(inline, (local, style), retain_graph=True)
    require(all(torch.isfinite(value).all() for value in (*inline_grad, *repeated_grad)), 'Nonfinite old-inline parity gradient')
    norms, legacy_grad = {}, None
    for number, (kind, loss) in enumerate(losses.items()):
        require(torch.isfinite(loss).all(), 'Nonfinite objective: ' + kind)
        gradients = torch.autograd.grad(loss, (local, style), retain_graph=number < len(losses) - 1)
        require(all(torch.isfinite(value).all() for value in gradients), 'Nonfinite local/style gradient: ' + kind)
        if kind == 'legacy_l2':
            legacy_grad = tuple(value.detach().clone() for value in gradients)
        norms[kind] = dict(local=(.01 * gradients[0]).reshape(70, -1).square().mean(1).sqrt().detach().cpu().tolist(),
                           style=(.01 * gradients[1]).reshape(70, -1).square().mean(1).sqrt().detach().cpu().tolist())
    differences = {}
    for block, original, repeated, helper in zip(('local', 'style'), inline_grad, repeated_grad, legacy_grad):
        denominator = original.double().norm().clamp_min(1e-30)
        floor = float((repeated.double() - original.double()).norm() / denominator)
        relative = float((helper.double() - original.double()).norm() / denominator)
        validate_native_parity(relative, floor)
        differences[block] = dict(inline_repeat_relative_l2=floor, helper_vs_inline_relative_l2=relative,
            allowed_relative_l2=min(1e-3, max(1e-5, 3 * floor)),
            helper_vs_inline_exact=torch.equal(helper, original), inline_repeat_exact=torch.equal(repeated, original))
    after_rng = rng_hash()
    require(after_rng == before_rng, 'Loss/gradient calibration changed post-preparation RNG')
    parity = dict(scalar_exact=True, direct_coefficient_gradient_exact=True, loss_sha256=tensor_hash(inline),
        direct_coefficient_gradient_sha256=tensor_hash(inline_direct), native_gradient_differences=differences,
        absolute_relative_l2_cap=1e-3, minimum_relative_l2_tolerance=1e-5, null_floor_multiplier=3,
        rng_before_sha256=before_rng, rng_after_sha256=after_rng)
    return norms, parity


def execute(args):
    require(re.fullmatch('[0-9a-f]{64}', args.manifest_sha256) is not None, 'Require exact asset manifest SHA256')
    assets, output, receipt_path = owned(args.asset_root), owned(args.output), owned(args.cache_records)
    require(not output.exists(), 'Calibration output exists; inspect it instead of overwriting')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='running', session=SESSION, manifest_sha256=args.manifest_sha256,
                 labels_used=False, classifier_forwards=0, seed=0, completed_batches=0)
    lion, source, before = None, None, None
    try:
        shutil.copyfile(Path(__file__), output / 'calibrate_spectral_losses.py')
        records, traces, acceptance = accepted_caches(receipt_path)
        sys.path.insert(0, str(assets / 'source_snapshot'))
        from scripts import eval_multibackbone_source as source
        from scripts import eval_multibackbone_guidance as runner
        require(source.REPO == assets / 'source_snapshot', 'Foreign project source imported')
        manifest = source.read_manifest(assets, args.manifest_sha256)
        for name in ('tta_gsd_v2.py', 'spectral_guidance_losses.py', 'scripts/eval_multibackbone_guidance.py', 'graph_spectral_v2.py'):
            verify(assets / 'source_snapshot' / name, manifest['files']['source_snapshot/' + name])
        proof, proof_refs = cpu_proof(manifest)
        from research_artifacts import CORRUPTIONS
        require(len(CORRUPTIONS) == 15 and set(traces) == {(dataset, corruption) for dataset in DATASETS for corruption in CORRUPTIONS},
                'Cached preparation corruption scope mismatch')
        selections = {}
        for dataset in DATASETS:
            name = 'study/' + dataset + '_development_ids.json'; verify(assets / name, manifest['files'][name])
            selection = json.loads((assets / name).read_text()); ids = selection['indices']
            require(selection['dataset'] == dataset and selection['stage'] == 'development' and len(ids) == 70
                    and all(type(value) is int and 0 <= value < (2468 if dataset == DATASETS[0] else 581) for value in ids)
                    and ids == sorted(set(ids)), 'Invalid label-free development selection')
            old_ids = ROOT / 'prepared_v1/assets' / name
            require((assets / name).read_bytes() == old_ids.read_bytes(), 'Development objects differ from accepted cache source')
            selections[dataset] = ids
        import numpy as np
        import torch
        require(sys.version_info[:2] == (3, 8) and torch.__version__ == '2.1.2+cu121'
                and not torch.cuda.is_initialized(), 'Calibration requires pinned fresh-process environment')
        os.chdir(str(assets / 'source_snapshot'))
        config = dict(asset_root=str(assets), method=runner.METHOD, component_study=True,
                      guidance_protocol=runner.PROTOCOL.copy(), seed=0)
        lion, graph, _, _, native, resolved = runner.load_guidance_runtime(config, manifest)
        require(native['no_build'] is True and native['bindings'] == proof['native']['bindings'], 'Pinned no-build native proof differs')
        require(torch.cuda.is_available() and torch.cuda.is_initialized()
                and all(not layer.training for model in (lion.vae, lion.priors) for layer in model.modules())
                and all(not parameter.requires_grad for model in (lion.vae, lion.priors) for parameter in model.parameters()),
                'Runtime must be existing frozen eval CUDA LION/prior')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        before = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        source.snapshot_sources(type('CalibrationBundle', (), {'path': output})(), config, manifest)
        state.update(acceptance=acceptance, cpu_preflight=proof_refs, source_snapshot_sha256=manifest['source_snapshot_sha256'],
            executed_source_files=config['executed_source_files'], native_extensions=config['native_extensions'], native=native,
            lion_state_before=before, resolved_lion_config=resolved,
            environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                             cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                             cudnn_benchmark=False, cudnn_deterministic=False, deterministic_algorithms=False))
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        chamfer = chamfer_3DDist()
        rows, parity_rows, cache_rows = [], [], []
        state.update(per_batch_parity=parity_rows, cache_records=cache_rows)
        with (output / 'per_object_norms.jsonl').open('x') as stream:
            for dataset in DATASETS:
                ids = selections[dataset]
                name = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
                for corruption in CORRUPTIONS:
                    path = ROOT / 'paired_state_v1_seed0' / dataset / corruption / name
                    record = records.get(str(path)); trace = traces[(dataset, corruption)]
                    require(record is not None and trace['paired_state']['path'] == str(path)
                            and {key: record[key] for key in PHASE} == {key: trace['paired_state'][key] for key in PHASE},
                            'Cache path/selection identity differs from accepted trace')
                    tensors, rng, metadata = load_cache(record, trace['initial_hashes'], 'cuda')
                    horizon = 30 if corruption == 'background' else 10
                    norms, parity = measure_first_step(lion, graph, tensors, rng, trace['initial_hashes'], horizon, chamfer)
                    for number, index in enumerate(ids):
                        row = dict(dataset=dataset, corruption=corruption, index=index, horizon=horizon,
                            timestep=290 if horizon == 30 else 90,
                            norms={kind: {block: norms[kind][block][number] for block in ('local', 'style')} for kind in ('chamfer', *KINDS)})
                        row['ratios'] = {kind: row['norms']['chamfer']['local'] / row['norms'][kind]['local']
                                         if min(row['norms']['chamfer']['local'], row['norms'][kind]['local']) > TINY else None for kind in KINDS}
                        stream.write(json.dumps(row, allow_nan=False) + '\n'); rows.append(row)
                    stream.flush()
                    parity_rows.append(dict(dataset=dataset, corruption=corruption, horizon=horizon, **parity))
                    cache_rows.append(dict(record, input_sha256=metadata['input_sha256'], rng_before_sha256=metadata['rng_before_sha256'],
                                           rng_after_sha256=metadata['rng_after_sha256'], tensor_hashes=metadata['tensor_hashes']))
                    state['completed_batches'] += 1
                    (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
                    print('AGENT_B_CALIBRATION_BATCH', dataset, corruption, horizon, state['completed_batches'], flush=True)
                    del tensors, rng
        diagnostics = coefficient_report(rows)
        after = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        require(before == after, 'Calibration changed frozen LION/prior weights or buffers')
        state.update(status='verified_label_free_gpu', elapsed_seconds=time.monotonic() - started, lion_state_after=after, legacy_inline_parity=True,
            coefficients={kind: diagnostics[kind]['coefficient'] for kind in KINDS}, coefficient_diagnostics=diagnostics,
            cache_records=cache_rows, per_batch_parity=parity_rows, n_input_events=2100,
            per_object_norms=identity(output / 'per_object_norms.jsonl'),
            scope='seed0; first actual DDIM clean estimate of10/30 steps; B70/M800/beta2/p.95; no labels or classifier scores',
            calibration='One pooled median of per-object C/S local applied-update RMS ratios per loss; eta=gamma=.01; spectral coefficient1; equal1050 events/dataset; scalar frozen globally',
            tiny_norm_threshold=TINY, scalar_bounds=[1e-12, 1e12], minimum_valid_fraction=.9,
            limitations='Input-point identity is joined to accepted preparation tensor hashes; input clouds are not reprocessed. Style norms logged separately, not jointly calibrated. Same cached states, not a claim of cross-process eigenspace bit identity.')
        (output / 'calibration_report.json').write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + '\n')
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        if lion is not None and source is not None and 'lion_state_after' not in state:
            try:
                state['lion_state_after'] = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
            except Exception as error:
                state['lion_state_after_error'] = repr(error)
        state['elapsed_seconds'] = time.monotonic() - started
        (output / 'state.json').write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('AGENT_B_SPECTRAL_CALIBRATION_VERIFIED', json.dumps(state['coefficients']), flush=True)


def self_check():
    rows = [dict(dataset=dataset, corruption='c%d' % corruption, index=index,
                 norms={kind: dict(local=(8. if dataset == DATASETS[0] else 12.) if kind == 'chamfer'
                                         else (2. if dataset == DATASETS[0] else 6.), style=0.)
                        for kind in ('chamfer', *KINDS)})
            for dataset in DATASETS for corruption in range(15) for index in range(70)]
    result = coefficient_report(rows)
    require(all(result[kind]['coefficient'] == 3. for kind in KINDS), 'Must median per-object ratios, not ratio of medians')
    for row in rows[:210]:
        row['norms']['power']['local'] = 0.
    require(coefficient_report(rows)['power']['valid_pairs'] == 1890, 'Exactly90% valid must pass; zero norms must be recorded/excluded')
    rows[210]['norms']['power']['local'] = 0.
    try:
        coefficient_report(rows)
    except ValueError:
        pass
    else:
        raise AssertionError('Insufficient nonzero pairs accepted')
    rows[210]['norms']['power']['local'] = 2.
    for bad in (float('nan'), float('inf'), -1.):
        previous = rows[-1]['norms']['legacy_l2']['style']
        rows[-1]['norms']['legacy_l2']['style'] = bad
        try:
            coefficient_report(rows)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite/negative gradients accepted')
        rows[-1]['norms']['legacy_l2']['style'] = previous
    for row in rows:
        row['norms']['legacy_l2']['local'] = 1e-11
        row['norms']['chamfer']['local'] = 1e3
    try:
        coefficient_report(rows)
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-range coefficient was silently accepted/clipped')
    for relative, repeat in ((2e-4, 5e-5), (2e-3, 1e-3), (1e-6, 2e-3), (float('nan'), 0.)):
        try:
            validate_native_parity(relative, repeat)
        except ValueError:
            pass
        else:
            raise AssertionError('Native parity exceeded its predeclared floor/tolerance')
    validate_native_parity(2e-6, 0.)
    validate_native_parity(1e-4, 5e-5)
    require('torch' not in sys.modules, 'Aggregation self-check imported Torch')
    print('AGENT_B_SPECTRAL_CALIBRATION_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt('Calibration terminated; preserve partial evidence')
    signal.signal(signal.SIGTERM, terminate)
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--asset-root', required=True, type=Path)
        parser.add_argument('--manifest-sha256', required=True)
        parser.add_argument('--cache-records', required=True, type=Path)
        parser.add_argument('--output', required=True, type=Path)
        execute(parser.parse_args())
