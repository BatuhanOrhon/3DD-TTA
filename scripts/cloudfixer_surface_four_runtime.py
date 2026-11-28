"""Shared CloudFixer surface trajectory and four frozen ModelNet40 readouts."""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
from hashlib import sha256
import inspect
import json
from pathlib import Path
import random
import sys
import textwrap
import time
from unittest.mock import patch

import numpy as np
import torch

from cloudfixer_store import canonical
from cloudfixer_diffusion import sample
from cloudfixer_rotation import OutputPose, SURFACE_ARM, rotation_policy
from scripts import eval_multibackbone_source as source
from scripts.run_cloudfixer_multibackbone import checked_runtime, require_workspace_config
from scripts.run_cloudfixer_point2vec import POLICIES, deterministic_proximal, model_hash, rng_state

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
PROBES = (20, 10, 0)
MANIFEST_SHA = '08e99de7de377d3a94ec76071df79e51ea27df81743d819ec9a28407097193e7'
AUTHOR_STATES = {'prior': 'b06cec8f25456e182c469304d9bfac98ea673db693fe3f92e7a3ed2251aa5574',
                 'pointmae': 'd61ed6f54fe15c3dbcfa180b6a5c02ff462562a153b6872d5457356e1e2ced91'}
CANDIDATE_SOURCES = {
    'cloudfixer_diffusion.py': 'b62adb9a1ddc01ec989c31b744592dff2a233c4d4d1c77a94af70967d01021f7',
    'cloudfixer_rotation.py': '1a07fbc641b1f90cdc6893feb1dfbd8f4a77f22be878b94d199b7358159dbca8',
    'knowledge/evidence/cloudfixer_rotation_refocus_20261122/frame_check.py':
        '7b660cc5989040af9ad3010ea8198a63999fb5e5e0a62552b89cc763a0c5d4e9',
}
# The first two are a historical seed-0 Rotation accept/veto pair. No labels used.
PROFILE_IDS = (321, 1146, 531, 1410, 262, 426, 395, 1169)
TOLERANCE = 1e-4


def tensor_hash(*tensors):
    digest = sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def host_rng():
    py, numpy = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)


def devices(x):
    return [x.device.index] if x.is_cuda else []


@torch.no_grad()
def shared_trajectory(model, items):
    """Object-addressed B1 noise and readout draws, also when phi is batched."""
    x = torch.cat([item[1] for item in items])
    mask = torch.cat([item[2] for item in items])
    draws, seeds, poses = [], [], []
    with deterministic_proximal(), torch.random.fork_rng(devices=devices(x)):
        for rng, xyz, valid, indices in items:
            torch.manual_seed(rng)
            draws.append(torch.randn_like(xyz))
            seeds.append(torch.randint(2**31, (501,), device='cpu').tolist())
            poses.append(OutputPose(xyz, valid, indices, SURFACE_ARM))

        def update(index, clean, reference):
            if index != 50:
                return None
            changes = [pose.update(index, clean[i:i+1], reference[i:i+1]) for i, pose in enumerate(poses)]
            if len(changes) == 1:
                return changes[0]
            if all(change is None for change in changes):
                return None
            identity = torch.eye(3, dtype=x.dtype, device=x.device)
            return torch.stack([identity if change is None else change for change in changes])

        clouds, meta = sample(model, x, mask, torch.cat(draws), output_rotation=update, **POLICIES['proximal_pm'])
    if (meta['denoiser_calls'] != 9 or tuple(meta['candidate_indices']) != PROBES
            or any(len(p.updates) != 1 or p.updates[0]['index'] != 50 for p in poses)):
        raise ValueError('Unexpected denoiser, pose update or readout schedule')
    return clouds, seeds, [pose.updates[0] for pose in poses]


@torch.no_grad()
def score_clouds(forward, clouds, seeds):
    rows = []
    with deterministic_proximal(), host_rng(), torch.random.fork_rng(devices=devices(clouds)):
        for i, cloud_views in enumerate(clouds):
            probes = []
            for cloud, index in zip(cloud_views, PROBES):
                seed = seeds[i][index]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                probes.append(forward(cloud[None].contiguous())[0])
            rows.append(torch.stack(probes))
        logits = torch.stack(rows)
    if logits.shape != (len(clouds), 3, 40) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite 40-class logits at each readout')
    return logits


def pm_decision(logits):
    scores = logits.astype(np.float64)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)).mean(axis=1).argmax(axis=-1)


def parity(actual, reference):
    close = dict(atol=TOLERANCE, rtol=TOLERANCE)
    receipt = dict(input_equal=np.array_equal(actual['input_sha256'], reference['input_sha256']),
                   decisions_equal=np.array_equal(actual['reason'], reference['reason']),
                   pose_close=bool(np.allclose(actual['pose'], reference['pose'], **close)),
                   max_abs_pose_error=float(np.abs(actual['pose'] - reference['pose']).max()),
                   angle_close=bool(np.allclose(actual['angle_degrees'], reference['angle_degrees'], **close)))
    heads = {}
    for i, model in enumerate(MODELS):
        a, b = actual['logits'][:, i], reference['logits'][:, i]
        heads[model] = dict(logits_close=bool(np.allclose(a, b, **close)),
                            max_abs_logit_error_per_probe=np.abs(a - b).max(axis=(0, 2)).tolist(),
                            probe_decisions_equal=np.array_equal(a.argmax(-1), b.argmax(-1)),
                            pm_decisions_equal=np.array_equal(pm_decision(a), pm_decision(b)))
    equivalent = (all(receipt[k] for k in ('input_equal', 'decisions_equal', 'pose_close', 'angle_close'))
                  and all(v['logits_close'] and v['probe_decisions_equal'] and v['pm_decisions_equal']
                          for v in heads.values()))
    return dict(receipt, heads=heads, equivalent=equivalent)


def source_identity(args, spec):
    """Authenticate the old package separately from the executing surface sampler."""
    old = args.asset_root.parent / 'code'
    adapted = {'scripts/run_cloudfixer_multibackbone.py': ('checked_runtime',),
               'scripts/setup_cloudfixer_multibackbone_colab.py': ('gpu_architecture', 'preload_head_native')}
    current = {}
    for name, ident in spec['overlay_files'].items():
        historic = source.asset_path(old, name)
        source.verify_identity(historic, ident)
        live = source.asset_path(ROOT, name)
        if name in adapted:
            def definitions(path):
                return {n.name: ast.dump(n) for n in ast.parse(path.read_bytes()).body
                        if isinstance(n, ast.FunctionDef)}
            prior, active = definitions(historic), definitions(live)
            if any(prior[n] != active[n] for n in adapted[name]):
                raise ValueError('Historical loading adapter changed: ' + name)
        elif name not in CANDIDATE_SOURCES:
            source.verify_identity(live, ident)
        current[name] = source.file_identity(live)
    for name, expected in CANDIDATE_SOURCES.items():
        current[name] = source.file_identity(source.asset_path(ROOT, name))
        if current[name]['sha256'] != expected:
            raise ValueError('Registered candidate source changed: ' + name)
    for name in ('scripts/cloudfixer_surface_four_runtime.py', 'cloudfixer_background.py',
                 'cfgs/cloudfixer_transfer_assets.json', 'cfgs/cloudfixer_point2vec_assets.json'):
        current[name] = source.file_identity(ROOT / name)
    return current


def assert_origin(module, expected):
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError('Unexpected executed source for ' + module.__name__)


CURVE_B1_EDITS = {
    'CurveGrouping': ('curvenet_util', 'start_index.squeeze().unsqueeze(2)', 'start_index.squeeze(1).unsqueeze(2)'),
    'Walk': ('walk', 'cur_feature.squeeze(), pre_feature.squeeze()', 'cur_feature.squeeze(-1), pre_feature.squeeze(-1)'),
}


def repaired_curve_forward(original, layer):
    """Bind only the documented singleton axis in the authenticated method body."""
    _, before, after = CURVE_B1_EDITS[layer]
    method = textwrap.dedent(inspect.getsource(original))
    if method.count(before) != 1:
        raise ValueError('Unexpected original CurveNet forward method: ' + layer)
    fixed = method.replace(before, after)
    namespace = {}
    exec(compile(fixed, '<surface CurveNet ' + layer + '.forward>', 'exec'), original.__globals__, namespace)
    receipt = dict(before_sha256=sha256(method.encode()).hexdigest(), after_sha256=sha256(fixed.encode()).hexdigest(),
                   original=before, replacement=after)
    return namespace['forward'], receipt


def repair_curvenet(head, root, manifest):
    layers, receipts = [], {}
    for name, (filename, _, _) in CURVE_B1_EDITS.items():
        module = sys.modules['study_svwa.' + filename]
        rel = 'model_probe_sources/study_svwa/' + filename + '.py'
        asset = source.asset_path(root, rel)
        assert_origin(module, asset)
        identity = source.verify_identity(asset, manifest['files'][rel])
        cls = getattr(module, name)
        forward, receipt = repaired_curve_forward(cls.forward, name)
        instances = [layer for layer in head.modules() if type(layer) is cls]
        if not instances:
            raise ValueError('Registered CurveNet layer missing: ' + name)
        for layer in instances:
            layer.forward = forward.__get__(layer)
        layers.extend(instances)
        receipts[name] = dict(receipt, asset=rel, asset_sha256=identity['sha256'], instances=len(instances))
    return layers, receipts


def load_pointmae(args):
    from classifier import pointMAE
    from utils.config import cfg_from_yaml_file

    assert_origin(pointMAE, args.official_root / 'classifier/pointMAE.py')
    config = cfg_from_yaml_file('cfgs/cfgs_mate/tta/tta_modelnet.yaml').model
    head = pointMAE.build_model_from_cfg(config)
    state = torch.load(args.classifier, map_location='cpu')['base_model']
    stripped = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    if len(stripped) != len(state):
        raise ValueError('Point-MAE state has duplicate unprefixed keys')
    head.load_state_dict(stripped, strict=True)
    head.rotate = True
    if (head.group_divider.num_group, head.group_divider.group_size, config.cls_dim) != (32, 32, 40):
        raise ValueError('Unexpected author grouping or number of classes')
    return head.cuda().eval().requires_grad_(False)


def pointmae_options(adapt, args):
    # Match the separate transfer runner's parser and audit its actual dataset output.
    argv = ['adapt.py', '--mode', 'eval', '--method', 'cloudfixer', '--classifier', 'pointMAE',
            '--classifier_dir', str(args.classifier), '--dataset_dir', str(args.dataset_root), '--input_lr', '.2',
            '--rotation', '.02', '--lam_h', '10', '--lam_l', '1', '--n_update', '30', '--vote', '1',
            '--random_seed', '0', '--subsample', '2048']
    with patch.object(sys, 'argv', argv):
        return adapt.parse_arguments()


class SurfaceRuntime:
    def __init__(self, args):
        require_workspace_config()
        if source.file_identity(args.manifest)['sha256'] != MANIFEST_SHA:
            raise ValueError('Unexpected historical A100 model manifest')
        spec = json.loads(args.manifest.read_bytes())
        current = source_identity(args, spec)
        assets = json.loads((ROOT / 'cfgs/cloudfixer_transfer_assets.json').read_text())
        if (spec['author_commit'] != assets['author_commit'] or spec['dataset_files'] != assets['dataset_files']
                or spec['diffusion_sha256'] != assets['diffusion_sha256']
                or source.file_identity(args.classifier)['sha256'] != assets['classifier_sha256']):
            raise ValueError('Asset identities disagree with the author Point-MAE and three-head archives')
        self.adapt, self.model, self.heads, self.options, self.fps, self.geometry, gamma = checked_runtime(
            args, dict(spec, overlay_files=current))
        assert_origin(self.adapt, args.official_root / 'adapt.py')
        assert_origin(sys.modules[self.adapt.ModelNet40C.__module__], args.official_root / 'datasets/dataloader.py')
        for name in MODELS[:-1]:
            module = sys.modules[type(self.heads[name].model).__module__]
            path = Path(module.__file__).resolve()
            rel = str(path.relative_to(args.asset_root.resolve()))
            source.verify_identity(path, spec['files'][rel])
        self.mae_options = pointmae_options(self.adapt, args)
        self.heads['pointmae'] = load_pointmae(args)
        from classifier_backbones import FrozenClassifier
        from knowledge.evidence.cloudfixer_rotation_refocus_20261122 import frame_check
        from utils import pc_utils
        assert_origin(pc_utils, args.official_root / 'utils/pc_utils.py')
        assert_origin(frame_check, ROOT / 'knowledge/evidence/cloudfixer_rotation_refocus_20261122/frame_check.py')
        for name in ('cloudfixer_diffusion','cloudfixer_rotation','scripts.eval_multibackbone_source',
                     'scripts.run_cloudfixer_multibackbone','scripts.run_cloudfixer_point2vec',
                     'pointcloud_proximal','scripts.setup_cloudfixer_multibackbone_colab'):
            assert_origin(sys.modules[name], ROOT / (name.replace('.', '/') + '.py'))
        assert_origin(sys.modules[FrozenClassifier.__module__], ROOT / 'classifier_backbones.py')
        self.head_configs = spec['head_configs']
        self.dataset_root = args.dataset_root
        self.labels = np.load(args.dataset_root / 'label.npy', allow_pickle=False).reshape(-1)
        self.class_order = json.loads((ROOT / 'cfgs/cloudfixer_point2vec_assets.json').read_text())['class_order']
        if self.labels.shape != (2468,) or set(self.labels.tolist()) != set(range(40)) or len(self.class_order) != 40:
            raise ValueError('Wrong ModelNet40-C test split or label ordering')
        self.expected_states = dict(spec['head_state_hashes'], **AUTHOR_STATES)
        self.curve_layers, curve_repair = repair_curvenet(self.heads['curvenet'], args.asset_root, spec)
        self.dataset = None
        self.condition = None
        self.native = {}
        self.identity = dict(seed=0, protocol=dict(arm=SURFACE_ARM, **POLICIES['proximal_pm']),
            historical_manifest_sha256=MANIFEST_SHA, author_commit=spec['author_commit'],
            checkpoint_sha256={**{k: spec['files'][v['checkpoint']]['sha256'] for k,v in self.head_configs.items()},
                               'pointmae': assets['classifier_sha256'], 'prior': assets['diffusion_sha256']},
            dataset_files=spec['dataset_files'], source_files={k: v['sha256'] for k,v in current.items()},
            class_order=self.class_order, frozen_states=self.verify_frozen(), gamma=gamma,
            curvenet_b1_axis_repair=curve_repair,
            heads={k: ('raw author-frame probe -> author rotate=True forward, unit sphere, '
                       '32 groups of 32, all tokens, no external FPS' if k == 'pointmae' else
                       'author unit sphere then inverse x-axis rotation, external native FPS1024, '
                       + ('BN3 positions + B3N features' if k == 'pointnext' else 'BN3 positions')) for k in MODELS},
            rng='seed=0; NumPy input and Torch B1 noise use 100003*seed+original_ID; 501 CPU randint '
                'draws identify t20/t10/t0 seeds shared across heads; each B1 head resets Python/NumPy/Torch',
            timing='synchronized shared geometry incl. RNG/PCA/SCD; synchronized per-head three B1 adapters, '
                   'softmax/mean; batch inference wall excludes input loading, host serialization and Drive',
            hardware=dict(python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), total_bytes=torch.cuda.get_device_properties(0).total_memory,
                          torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__),
            numeric=dict(cublas_workspace_config=':4096:8', tf32=False, cudnn_benchmark=False,
                         cudnn_deterministic=True, deterministic_algorithms=True))

    def verify_frozen(self):
        states = {'prior': model_hash(self.model), 'pointmae': model_hash(self.heads['pointmae'])}
        states.update({k: source.state_hash(self.heads[k]) for k in MODELS[:-1]})
        for obj in [self.model, *self.heads.values()]:
            if any(m.training for m in obj.modules()) or any(p.requires_grad for p in obj.parameters()):
                raise ValueError('Unfrozen model state detected')
        if states != self.expected_states:
            raise ValueError('Frozen tensors differ from archived model states')
        return states

    def _dataset(self, options, condition):
        options.dataset = 'modelnet40c_' + condition + '_5'
        ds = self.adapt.ModelNet40C(options, partition='test')
        if (not ds.rotate or ds.subsample != 2048 or ds.scenario != 'normal' or ds.severity != '5'
                or ds.corruption != condition or not np.array_equal(ds.label_list.reshape(-1), self.labels)
                or [ds.idx_to_label[i] for i in range(40)] != self.class_order):
            raise ValueError('Author dataset frame, severity, label or class order mismatch')
        return ds

    def _items(self, condition, ids):
        if ('data_' + condition + '_5.npy' not in self.identity['dataset_files'] or not len(ids)
                or any(not isinstance(i, (int, np.integer)) or not 0 <= i < 2468 for i in ids)):
            raise ValueError('Unexpected corruption or object ID')
        if self.condition != condition:
            self.dataset = self._dataset(self.options, condition)
            self.condition = condition
        return [self._item(self.dataset, int(i)) for i in ids]

    def _item(self, dataset, ident):
        with host_rng():
            np.random.seed(ident)
            x, label, mask, indices = dataset[ident]
        if np.asarray(label).item() != self.labels[ident]:
            raise ValueError('Original ID label changed')
        return (ident, torch.from_numpy(x)[None].cuda().float(), torch.from_numpy(mask).reshape(1,-1).cuda().bool(),
                torch.from_numpy(indices).reshape(1,-1).cuda())

    def _forward(self, name, cloud, *, direct=False):
        head = self.heads[name]
        if name == 'pointmae':
            if not direct:
                return head(cloud)
            from utils.pc_utils import rotate_shape_tensor, scale_to_unit_cube_torch
            framed = scale_to_unit_cube_torch(rotate_shape_tensor(cloud, 'x', np.pi/2))
            return head.classification_only(framed, only_unmasked=False)
        points, features = source.preprocess(self.geometry(cloud), self.head_configs[name], self.fps)
        if not direct:
            return head(points, features=features)
        if points.shape != (len(cloud), 1024, 3):
            raise ValueError('External FPS1024 contract failed')
        if name == 'pointnext':
            if not torch.equal(points.transpose(1,2), features):
                raise ValueError('PointNeXt position/features mismatch')
            return head.model({'pos': points, 'x': features})
        return head.model(points)

    def predict(self, condition, ids):
        return self._predict_items(self._items(condition, ids))

    @torch.no_grad()
    def _predict_items(self, items):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        clouds, seeds, updates = shared_trajectory(self.model, items)
        torch.cuda.synchronize()
        denoise_seconds = time.perf_counter() - start
        logits, pm, terminal, head_seconds = [], [], [], []
        for name in MODELS:
            start_head = time.perf_counter()
            values = score_clouds(lambda cloud: self._forward(name, cloud), clouds, seeds)
            pm.append(values.double().softmax(-1).mean(1).argmax(-1))
            terminal.append(values[:, -1].argmax(-1))
            torch.cuda.synchronize()
            head_seconds.append(time.perf_counter() - start_head)
            logits.append(values)
        wall = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
        return dict(logits=torch.stack(logits, 1).cpu().numpy(),
                    pm=torch.stack(pm, 1).cpu().numpy(), terminal=torch.stack(terminal, 1).cpu().numpy(),
                    pose=np.array([row['pose'] for row in updates]), reason=np.asarray([r['reason'] for r in updates]),
                    angle_degrees=np.array([r['applied_degrees'] for r in updates]),
                    denoise_seconds=denoise_seconds, head_seconds=np.asarray(head_seconds), wall_seconds=wall,
                    peak_mib=peak, trajectory_sha256=np.asarray([tensor_hash(c) for c in clouds]),
                    input_sha256=np.asarray([tensor_hash(*item[1:]) for item in items]))

    @torch.no_grad()
    def _audit_independent(self, condition, items, shared):
        """Rerun the original B1 pipeline with each direct author/source model."""
        mae_data = self._dataset(self.mae_options, condition)
        raw_points = np.load(self.dataset_root / ('data_' + condition + '_5.npy'), mmap_mode='r', allow_pickle=False)
        receipts = {}
        for i, (ident, x, mask, indices) in enumerate(items):
            mae_item = self._item(mae_data, ident)
            if any(not torch.equal(a, b) for a, b in zip(mae_item[1:], (x, mask, indices))):
                raise ValueError('Point-MAE and multi-head geometry inputs diverged')
            raw = torch.tensor(raw_points[ident, :, :3], dtype=x.dtype, device=x.device)[None]
            centered = raw - raw.mean(1, keepdim=True)
            unit = centered / centered.norm(dim=-1, keepdim=True).amax(1, keepdim=True)
            frame_error = float((self.geometry(x) - unit).abs().max())
            torch.testing.assert_close(self.geometry(x), unit, atol=2e-5, rtol=2e-5)
            receipts[str(ident)] = record = dict(frame_max_abs=frame_error, point_count=x.shape[1],
                    valid_anchors=int(mask.sum()), input_sha256=tensor_hash(x, mask, indices), heads={})
            for h, name in enumerate(MODELS):
                # Each model independently reruns all nine phi calls and its own three classifiers.
                with host_rng(), torch.random.fork_rng(devices=devices(x)):
                    torch.manual_seed(ident)
                    def direct(cloud):
                        # rotation_policy fixes the Torch probe draw immediately before this callback.
                        seed = torch.initial_seed()
                        random.seed(seed)
                        np.random.seed(seed)
                        return self._forward(name, cloud, direct=True)
                    ref = rotation_policy(self.model, direct, *(mae_item[1:] if name == 'pointmae'
                                                                else (x, mask, indices)), SURFACE_ARM)
                if (ref.diagnostics['denoiser_calls'] != 9 or ref.diagnostics['classifier_calls'] != 3
                        or tensor_hash(ref.clouds[0]) != shared[i]['trajectory_sha256'][0]
                        or not np.array_equal(ref.pose.cpu().numpy(), shared[i]['pose'][0])
                        or ref.diagnostics['updates'][0]['reason'] != shared[i]['reason'][0]
                        or ref.diagnostics['updates'][0]['applied_degrees'] != shared[i]['angle_degrees'][0]
                        or not np.array_equal(ref.logits.cpu().numpy()[0], shared[i]['logits'][0, h])
                        or int(ref.probabilities.argmax()) != int(shared[i]['pm'][0, h])
                        or int(ref.logits[0, -1].argmax()) != int(shared[i]['terminal'][0, h])):
                    raise ValueError(f'Shared vs independent B1 mismatch for {condition}/{ident}/{name}')
                record['heads'][name] = dict(clouds_sha256=tensor_hash(ref.clouds[0]),
                    logits_sha256=tensor_hash(ref.logits), probe_predictions=ref.logits[0].argmax(-1).tolist(),
                    pm=int(ref.probabilities.argmax()), reason=ref.diagnostics['updates'][0]['reason'],
                    angle_degrees=ref.diagnostics['updates'][0]['applied_degrees'], exact=True)
        return receipts

    @torch.no_grad()
    def _audit_curve_repair(self, items):
        # Compare the exact same checkpoint and FPS/head path on distinct real objects in its historical B>=2 domain.
        receipts = {}
        for size in (2, 4):
            clouds = torch.cat([item[1] for item in items[:size]])
            def observe(original):
                with deterministic_proximal(), host_rng(), torch.random.fork_rng(devices=devices(clouds)), ExitStack() as undo:
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    if original:
                        for layer in self.curve_layers:
                            undo.enter_context(patch.object(layer, 'forward', type(layer).forward.__get__(layer)))
                    logits = self._forward('curvenet', clouds, direct=original)
                    state = sha256(canonical(rng_state())).hexdigest()
                return logits, state
            historical, prior_rng = observe(True)
            repaired, active_rng = observe(False)
            if not torch.equal(historical, repaired) or prior_rng != active_rng:
                raise ValueError(f'CurveNet B{size} logit or RNG parity failed after singleton-axis repair')
            receipts[str(size)] = dict(ids=[i[0] for i in items[:size]], exact=True, rng_after_sha256=active_rng,
                                       logits_sha256=tensor_hash(repaired), predictions=repaired.argmax(-1).tolist())
        return receipts

    def calibrate(self, condition):
        items = self._items(condition, PROFILE_IDS)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        curve_parity = self._audit_curve_repair(items)
        self._predict_items(items[:1])
        refs = [self._predict_items([item]) for item in items]
        audited = self._audit_independent(condition, items[:2], refs[:2])
        if not self.native:
            for name, module in list(sys.modules.items()):
                path = str(getattr(module, '__file__', '') or '')
                if path.endswith('.so') and any(part in path for part in (
                        'pointnet2', 'knn_cuda', 'pytorch3d', 'chamfer', 'torch_extensions')):
                    self.native[name] = source.file_identity(Path(path))
        if condition == 'rotation' and [str(r['reason'][0]) == 'accepted' for r in refs[:2]] != [True, False]:
            raise ValueError('Registered seed-0 Rotation acceptance/veto controls changed')
        fields = ('input_sha256', 'logits', 'pose', 'reason', 'angle_degrees')
        baseline = {k: np.concatenate([r[k] for r in refs]) for k in fields}
        best = dict(size=1, equivalent=True, objects_per_second=len(items)/sum(r['wall_seconds'] for r in refs),
                    peak_mib=max(r['peak_mib'] for r in refs), free_after_bytes=torch.cuda.mem_get_info()[0])
        trials = [best]
        for size in (2, 4, 8):
            try:
                self._predict_items(items[:size])
                batches = [self._predict_items(items[i:i+size]) for i in range(0, len(items), size)]
                actual = {k: np.concatenate([b[k] for b in batches]) for k in fields}
                outcome = parity(actual, baseline)
                mismatches = [ident for i, ident in enumerate(PROFILE_IDS)
                              if not parity({k: v[i:i+1] for k,v in actual.items()},
                                            {k: v[i:i+1] for k,v in baseline.items()})['equivalent']]
                trial = dict(outcome, size=size, mismatch_ids=mismatches,
                             objects_per_second=len(items)/sum(b['wall_seconds'] for b in batches),
                             peak_mib=max(b['peak_mib'] for b in batches), free_after_bytes=torch.cuda.mem_get_info()[0])
            except torch.cuda.OutOfMemoryError:
                trials.append(dict(size=size, equivalent=False, out_of_memory=True))
                break
            trials.append(trial)
            if trial['free_after_bytes'] < max(2 * 2**30, total // 10):
                break
            if trial['equivalent'] and trial['objects_per_second'] > best['objects_per_second']:
                best = trial
        torch.cuda.empty_cache()
        self.verify_frozen()
        return dict(condition=condition, selected_batch_size=best['size'], ids=list(PROFILE_IDS),
                    atol=TOLERANCE, rtol=TOLERANCE, total_bytes=total, free_before_bytes=free,
                    reason_b1=baseline['reason'].tolist(), independent_b1=audited, native=self.native, trials=trials,
                    curvenet_historical_parity=curve_parity)
