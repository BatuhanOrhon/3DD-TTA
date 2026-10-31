"""CPU depth-study contract checks; no native kernels or real model inference."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import _rng_state, _restore_rng, rng_hash, tensor_hash
from shared_trajectory import SharedTrajectoryObserver
from scripts.analyze_legacy_trajectory import policies
from scripts.eval_multibackbone_guidance import paired_preparation

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def installed_modules(modules):
    # Restore only our stubs; broad sys.modules rollback can re-register lazy Torch ops.
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_sampler(source=None):
    fake_chamfer = types.ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
    fake_chamfer.chamfer_3DDist = object
    fake_diffusers = types.ModuleType('diffusers')
    fake_diffusers.DDIMScheduler = object
    fake_ops = types.ModuleType('utilities_3dd_tta')
    fake_ops.grad_freeze = lambda model: None
    with installed_modules({fake_chamfer.__name__: fake_chamfer, 'diffusers': fake_diffusers,
                            'utilities_3dd_tta': fake_ops}):
        module = types.ModuleType('depth_test_sampler')
        with installed_modules({module.__name__: module}):
            exec(source or (ROOT / 'tta.py').read_text(), module.__dict__)
    return module


class CPU:
    def __getattr__(self, name):
        return getattr(torch, name)

    def ones(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return torch.ones(*args, **kwargs)


class Scheduler:
    def __init__(self, **kwargs):
        self.config = dict(kwargs, set_alpha_to_one=True)
        self.final_alpha_cumprod = torch.tensor(1.)

    def set_timesteps(self, total, device):
        self.timesteps = torch.arange(990, -1, -10)
        self.alphas_cumprod = torch.linspace(.999, .05, 1000)

    def step(self, noise, timestep, latent):
        return types.SimpleNamespace(pred_original_sample=latent-noise, prev_sample=latent-.1*noise)


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.zeros(1))
        self.styles = []
        self.encode_calls = 0

    def encode(self, points):
        self.encode_calls += 1
        return None, None, [[torch.randn(len(points), 128)], [torch.randn(len(points), 8192)]]

    def global2style(self, style):
        return 2 * style

    def decoder(self, unused, beta, context, style):
        self.styles.append(style.detach().clone())
        # Deliberately stochastic and buffer-mutating: probe isolation must restore both.
        self.counter.add_(1)
        return context.reshape(len(context), 2048, 4)[:, :, :3] + .01*style.mean() + .001*torch.rand(1)


class Prior(torch.nn.Module):
    def forward(self, x, t, condition_input, clip_feat):
        return .1*x + .01*condition_input.mean(dim=1, keepdim=True)


def chamfer(a, b):
    distances = (a-b).square().sum(-1)
    return distances, distances, None, None


def seeded(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(module, depth=5, gamma=.01, eta=.01, **kwargs):
    lion = types.SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
    with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                        chamfer_grad=lambda: chamfer, grad_freeze=lambda model: None):
        output = module.tta_reconstruct(torch.ones(2, 2048, 3), lion, depth, gamma, eta, .95, **kwargs)
    return output, lion


class SamplerTests(unittest.TestCase):
    def test_default_matches_frozen_prechange_math_and_rng(self):
        # Parent branch baseline is a fixed source oracle, not a second new-code invocation.
        old_source = subprocess.check_output(['git', 'show', '4da29b3:tta.py'], cwd=ROOT, text=True)
        old, new = load_sampler(old_source), load_sampler()
        for depth in (1, 2, 3, 5, 10, 35):
            seeded()
            expected, _ = run(old, depth)
            expected_rng = rng_hash()
            seeded()
            actual, _ = run(new, depth)
            self.assertTrue(torch.equal(actual, expected), depth)
            self.assertEqual(rng_hash(), expected_rng)

    def test_original_style_clean_probes_are_isolated_and_do_not_change_terminal(self):
        module = load_sampler()
        for depth in (1, 2, 3, 5, 10, 35):
            seeded()
            reference, _ = run(module, depth)
            expected_rng = rng_hash()
            seeded()
            lion = types.SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
            head = torch.nn.Linear(3, 40).eval()
            # Model construction is kept outside inference RNG boundary.
            seeded()
            checkpoints = sorted({(depth+3)//4, (depth+1)//2, (3*depth+3)//4})
            def classify(points):
                return {'pointmae': head(points.mean(1)) + torch.rand(1)}
            observer = SharedTrajectoryObserver(lion.vae, {'pointmae': head}, classify, depth,
                                                num_classes=40, checkpoint_steps=checkpoints)
            with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                chamfer_grad=lambda: chamfer, grad_freeze=lambda model: None):
                actual = module.tta_reconstruct(torch.ones(2, 2048, 3), lion, depth, .01, .01, .95,
                                                trajectory_observer=observer)
            self.assertTrue(torch.equal(actual, reference), depth)
            self.assertEqual(rng_hash(), expected_rng)
            self.assertEqual(lion.vae.counter.item(), 1)
            observer.terminal({'pointmae': torch.zeros(2, 40)})
            arrays = observer.arrays('pointmae')
            self.assertEqual(arrays['trajectory_logits'].shape, (2, len(checkpoints)+1, 40))
            self.assertEqual(arrays['candidate_loop_counts'].tolist(), checkpoints+[depth])
            self.assertEqual(len(lion.vae.styles), len(checkpoints)+1)
            self.assertTrue(all(torch.equal(s, lion.vae.styles[-1]) for s in lion.vae.styles))
            self.assertTrue(observer.diagnostics()['probe_rng_unchanged'])
            self.assertTrue(observer.diagnostics()['probe_buffers_unchanged'])

    def test_cache_pairs_encoded_states_noise_rng_but_different_depth_noisy_latents(self):
        module = load_sampler()
        records = []
        with tempfile.TemporaryDirectory() as name:
            cache = Path(name).resolve() / 'batch.npz'
            for depth in (5, 10):
                seeded()
                state, recorder = paired_preparation(cache, torch.ones(2, 2048, 3))
                capture = {}
                class Observer:
                    def mark(self, name): pass
                    def initial(self, **tensors): capture.update({k: tensor_hash(v) for k, v in tensors.items()})
                    def clean(self, *args, **kwargs): pass
                    def final_state(self, *args, **kwargs): pass
                _, lion = run(module, depth, prepared_state=state, state_observer=recorder,
                              trajectory_observer=Observer())
                self.assertEqual(lion.vae.encode_calls, 1 if depth == 5 else 0)
                records.append(capture)
            for key in ('input_points', 'shape_latent', 'local_latent', 'noise', 'style'):
                self.assertEqual(records[0][key], records[1][key], key)
            self.assertNotEqual(records[0]['noisy_local'], records[1]['noisy_local'])
            seeded(8)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                paired_preparation(cache, torch.ones(2, 2048, 3))

    def test_same_depth_cache_replay_preserves_output_and_full_rng(self):
        module = load_sampler()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / 'batch.npz'
            results = []
            for _ in range(2):
                seeded()
                state, recorder = paired_preparation(path, torch.ones(2, 2048, 3))
                output, _ = run(module, 5, prepared_state=state, state_observer=recorder)
                results.append((output, rng_hash()))
            self.assertTrue(torch.equal(results[0][0], results[1][0]))
            self.assertEqual(results[0][1], results[1][1])

    def test_gamma_updates_local_and_eta_updates_conditioning_style(self):
        module = load_sampler()
        values = []
        for gamma, eta in ((0., 0.), (.01, 0.), (0., .01)):
            seeded()
            trajectory, _ = run(module, 1, gamma=gamma, eta=eta, return_trajectory=True)
            values.append(trajectory)
        self.assertFalse(torch.equal(values[0].final_local_latent, values[1].final_local_latent))
        self.assertTrue(torch.equal(values[0].updated_style, values[1].updated_style))
        self.assertTrue(torch.equal(values[0].final_local_latent, values[2].final_local_latent))
        self.assertFalse(torch.equal(values[0].updated_style, values[2].updated_style))


class DepthConfigTests(unittest.TestCase):
    def args(self, root):
        manifest = root / 'ids.json'
        manifest.write_text(json.dumps(dict(dataset='modelnet40_c', n_examples=2468,
                                            stage='development', indices=[1, 8, 20])))
        return ['--preview', '--dataset-root', str(root), '--label-path', str(root/'label.npy'),
                '--asset-root', str(root), '--result-root', str(root/'result'),
                '--pointmae-config', 'model.yml', '--pointmae-checkpoint', 'model.pth',
                '--index-manifest', str(manifest), '--paired-state-root', str(root/'paired')]

    def test_depth_config_and_unique_checkpoint_counts(self):
        from scripts.eval_chamfer_depth import configurations, relative_checkpoints
        with tempfile.TemporaryDirectory() as name:
            args = self.args(Path(name))
            for depth, checkpoints in ((1,[1]), (2,[1,2]), (3,[1,2,3]), (5,[2,3,4]), (10,[3,5,8]), (35,[9,18,27])):
                self.assertEqual(relative_checkpoints(depth), checkpoints)
                if depth == 35: continue
                config, = configurations(args + ['--normal-reverse-steps', str(depth)])
                self.assertEqual(config['normal_reverse_steps'], depth)
                self.assertEqual(config['background_reverse_steps'], 35)
                self.assertEqual(config['batch_size'], 32)
                self.assertEqual(config['decoder_style'], 'original')
                self.assertTrue(config['scheduler_final_alpha_to_one'])
                self.assertEqual(config['guidance_rate_mapping'], {'gamma': 'local_latent', 'eta': 'conditioning_style'})
                self.assertEqual(config['trajectory_selection']['normal_loop_counts'], checkpoints+[depth])
                logits = np.random.default_rng(5).normal(size=(2,len(checkpoints)+1,40))
                result = policies(logits, len(checkpoints))
                expected = result['probabilities'].sum(1)/(len(checkpoints)+1)
                np.testing.assert_array_equal(result['probability_mean'], expected.argmax(1))

    def test_preview_needs_only_standard_library_and_records_replayable_entrypoint(self):
        from scripts import eval_chamfer_depth as depth
        from scripts import eval_multibackbone_guidance as shared
        with tempfile.TemporaryDirectory() as name:
            args = self.args(Path(name))
            result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/eval_chamfer_depth.py'),*args],
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)[0]['method'],depth.METHOD)
            args.remove('--preview')
            args += ['--asset-manifest-sha256','a'*64]
            with patch.object(shared.source.RunBundle,'create',side_effect=RuntimeError('captured')) as create:
                with self.assertRaisesRegex(RuntimeError,'captured'):
                    depth.main(args)
            self.assertIn(str(ROOT/'scripts/eval_chamfer_depth.py'),create.call_args.args[3])
            self.assertNotIn(str(ROOT/'scripts/eval_multibackbone_guidance.py'),create.call_args.args[3])

    def test_invalid_depth_parameters_fail(self):
        from scripts.eval_chamfer_depth import configurations
        with tempfile.TemporaryDirectory() as name:
            args = self.args(Path(name))
            for key, value in (('p','nan'), ('p','0'), ('p','1.01'), ('gamma','inf'),
                               ('gamma','-.01'), ('eta','nan'), ('eta','-1'),
                               ('normal-reverse-steps','0'), ('normal-reverse-steps','11')):
                with self.subTest(key=key, value=value), redirect_stderr(StringIO()), self.assertRaises((ValueError, SystemExit)):
                    configurations(args + ['--'+key, value])

    def test_technical_smoke_is_explicit_bounded_and_preserves_research_all15(self):
        from scripts.eval_chamfer_depth import configurations
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = self.args(root)
            smoke = ['--technical-smoke', '--corruptions', 'gaussian', 'background']
            with self.assertRaises(ValueError):
                configurations(args + smoke)
            (root/'ids.json').write_text(json.dumps(dict(dataset='modelnet40_c', n_examples=2468,
                                                        stage='smoke', indices=list(range(32)))))
            config, = configurations(args + smoke)
            self.assertEqual(config['corruptions'], ['gaussian', 'background'])
            self.assertTrue(config['technical_smoke'])
            for bad in (['--normal-reverse-steps','2'], ['--gamma','.03'], ['--seed','1']):
                with self.assertRaises(ValueError):
                    configurations(args + smoke + bad)
            with self.assertRaises(ValueError):
                configurations(args + ['--corruptions','gaussian','background'])

    def test_actual_scheduler_depth_records_and_original_guard_remain_distinct(self):
        from scripts.eval_chamfer_depth import scheduler_record
        from scripts.eval_multibackbone_guidance import original_scheduler_record
        scheduler = Scheduler(beta_end=.02, beta_start=.0001, beta_schedule='linear',
                              clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon')
        scheduler.set_timesteps(100, 'cpu')
        for depth in (1,2,3,5,10,35):
            record = scheduler_record(scheduler, depth)
            self.assertEqual(record['initial_timestep'], (depth-1)*10)
            self.assertEqual(len(record['active_timesteps']), depth)
            self.assertEqual(record['reverse_steps'], depth)
        with self.assertRaises(ValueError):
            original_scheduler_record(scheduler, 10)
        scheduler.config['set_alpha_to_one'] = False
        with self.assertRaises(ValueError):
            scheduler_record(scheduler, 10)

    def test_depth_geometry_forwards_rates_and_records_actual_horizon_without_fallback(self):
        from scripts import eval_chamfer_depth as depth
        from scripts import eval_multibackbone_guidance as shared
        ops = types.SimpleNamespace(normalize=lambda points:(points,None,None),
            upsample_all=lambda points,n:np.resize(points,(len(points),n,3)),
            rotate_pointcloud=lambda points:points, rotateback_pointcloud=lambda points:points)
        calls = []
        def reconstruct(**kwargs):
            calls.append(kwargs)
            scheduler = Scheduler(beta_end=.02,beta_start=.0001,beta_schedule='linear',
                clip_sample=False,num_train_timesteps=1000,prediction_type='epsilon')
            scheduler.set_timesteps(100,'cpu')
            kwargs['scheduler_observer'](scheduler)
            return kwargs['x'].clone()
        protocol = dict(gamma=.003,eta=.03,p=.96,total=100,scd_normalize=False)
        with tempfile.TemporaryDirectory() as temporary:
            for normal,corruption,expected in ((1,'gaussian',1),(5,'gaussian',5),(10,'gaussian',10),(1,'background',35)):
                record = {}
                path = Path(temporary).resolve() / ('%s-%s.npz'%(normal,corruption))
                geometry,metrics = depth.reconstruct_geometry(torch.ones(2,17,3),'modelnet40_c',corruption,
                    None,None,reconstruct,ops,depth_config={'normal_reverse_steps':normal},device='cpu',
                    scheduler_observer=record.update,observer_factory=lambda *a:object(),
                    protocol=protocol,paired_state_path=path)
                self.assertEqual(geometry.shape,(2,2048,3))
                self.assertEqual(metrics,{})
                self.assertEqual(calls[-1]['steps_back_local'],expected)
                self.assertEqual(record['initial_timestep'],(expected-1)*10)
                self.assertEqual({key:calls[-1][key] for key in protocol},protocol)
            before = len(calls)
            def failed(**kwargs):
                output = reconstruct(**kwargs)
                return torch.full_like(output,float('nan'))
            with self.assertRaisesRegex(ValueError,'finite decoded'):
                depth.reconstruct_geometry(torch.ones(2,17,3),'modelnet40_c','gaussian',None,None,failed,ops,
                    depth_config={'normal_reverse_steps':1},device='cpu',observer_factory=lambda *a:object(),
                    protocol=protocol,paired_state_path=Path(temporary).resolve()/'fail.npz')
            self.assertEqual(len(calls),before+1)

    def test_default_observer_horizon_contract_and_explicit_override_validation(self):
        vae, head = VAE().eval(), torch.nn.Linear(3,40).eval()
        for depth in (1,2,3,5,35):
            with self.assertRaises(ValueError):
                SharedTrajectoryObserver(vae, {'pointmae':head}, None, depth, num_classes=40)
        for steps in ([], [2,1], [1,1], [0], [6], [True], [1.5]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                SharedTrajectoryObserver(vae, {'pointmae':head}, None, 5, num_classes=40, checkpoint_steps=steps)


class TransferConfigTests(unittest.TestCase):
    args = DepthConfigTests.args
    def transfer_args(self, root, dataset='modelnet40_c', stage='development', count=3):
        args = self.args(root)
        (root/'ids.json').write_text(json.dumps(dict(dataset=dataset,
            n_examples=581 if dataset == 'scanobjectnn_c' else 2468,
            stage=stage, indices=list(range(count)))))
        return args + ['--transfer-study', '--dataset', dataset]

    def test_frozen_transfer_resolves_both_datasets_and_all_selected_heads(self):
        from scripts import eval_chamfer_depth as depth
        with tempfile.TemporaryDirectory() as name:
            for dataset, classes in (('modelnet40_c',40), ('scanobjectnn_c',15)):
                configs = depth.configurations(self.transfer_args(Path(name),dataset))
                self.assertEqual([c['backbone'] for c in configs], ['pointmae','dgcnn','curvenet','pointnext'])
                for config in configs:
                    self.assertEqual(config['num_classes'], classes)
                    self.assertEqual(config['normal_reverse_steps'],8)
                    self.assertEqual(config['guidance_protocol'],dict(gamma=.03,eta=.003,p=.95,total=100,scd_normalize=False))
                    self.assertEqual(config['trajectory_selection']['normal_loop_counts'],[2,4,6,8])
                    self.assertEqual(config['trajectory_selection']['background_loop_counts'],[9,18,27,35])
                    self.assertEqual(config['decoder_style'],'original')
                    self.assertTrue(config['scheduler_final_alpha_to_one'])
                    self.assertFalse(config['lion_ema'])
                    self.assertEqual(config['batch_size'],32)
                    self.assertEqual(len(config['corruptions']),15)

    def test_transfer_rejects_parameter_search_variants_and_legacy_scope_expansion(self):
        from scripts import eval_chamfer_depth as depth
        with tempfile.TemporaryDirectory() as name:
            args = self.transfer_args(Path(name))
            for extra in (['--normal-reverse-steps','5'], ['--eta','.03'], ['--gamma','.003'],
                          ['--p','.96'], ['--corruptions','gaussian'], ['--backbones','dgcnn','dgcnn'],
                          ['--chamfer-experiment','semantic'], ['--spectral-loss-kind','legacy']):
                with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises((ValueError,SystemExit)):
                    depth.configurations(args+extra)
            for extra in (['--dataset','scanobjectnn_c'], ['--backbones','dgcnn']):
                with self.assertRaises(ValueError):
                    depth.configurations(self.args(Path(name))+extra)

    def test_transfer_smoke_requires_exact32_smoke_ids_and_frozen_candidate(self):
        from scripts import eval_chamfer_depth as depth
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            smoke=['--technical-smoke','--corruptions','gaussian','background']
            for stage,count in (('development',32),('smoke',31)):
                with self.assertRaises(ValueError):
                    depth.configurations(self.transfer_args(root,'scanobjectnn_c',stage,count)+smoke)
            configs=depth.configurations(self.transfer_args(root,'scanobjectnn_c','smoke',32)+smoke)
            self.assertEqual(len(configs),4)
            self.assertTrue(all(c['technical_smoke'] and c['normal_reverse_steps']==8 for c in configs))
            with self.assertRaises(ValueError):
                depth.configurations(self.transfer_args(root,'scanobjectnn_c','smoke',32))

    def test_transfer_accepts_other_source_heads_without_pointmae_assets(self):
        from scripts import eval_chamfer_depth as depth
        with tempfile.TemporaryDirectory() as name:
            args=self.transfer_args(Path(name),'scanobjectnn_c')
            for key in ('--pointmae-config','--pointmae-checkpoint'):
                offset=args.index(key);del args[offset:offset+2]
            configs=depth.configurations(args+['--backbones','dgcnn','curvenet','pointnext'])
            self.assertEqual([c['backbone'] for c in configs],['dgcnn','curvenet','pointnext'])

    def test_scan_geometry_restores_input_frame_only_for_explicit_transfer(self):
        from scripts import eval_chamfer_depth as depth
        raw=torch.arange(102,dtype=torch.float32).reshape(2,17,3)
        center=raw.mean(1,keepdim=True)
        scale=torch.tensor([2.,4.]).reshape(2,1,1)
        ops=types.SimpleNamespace(normalize=lambda p:((p-center)/scale,center,scale),
            upsample_all=lambda p,n:np.stack([np.resize(row,(n,3)) for row in p]),
            rotate_pointcloud=lambda p:p,rotateback_pointcloud=lambda p:p,
            unnormalize_data=lambda p,s,c:p*s+c)
        def reconstruct(**kwargs):
            scheduler=Scheduler(beta_end=.02,beta_start=.0001,beta_schedule='linear',
                clip_sample=False,num_train_timesteps=1000,prediction_type='epsilon')
            scheduler.set_timesteps(100,'cpu');kwargs['scheduler_observer'](scheduler)
            return kwargs['x'].clone()
        with tempfile.TemporaryDirectory() as name:
            config=depth.configurations(self.transfer_args(Path(name),'scanobjectnn_c'))[0]
            kwargs=dict(depth_config=config,device='cpu',observer_factory=lambda *a:object(),
                protocol=config['guidance_protocol'],paired_state_path=Path(name).resolve()/'state.npz')
            geometry,_=depth.reconstruct_geometry(raw,'scanobjectnn_c','gaussian',None,None,reconstruct,ops,**kwargs)
            torch.testing.assert_close(geometry,torch.from_numpy(np.stack([np.resize(row,(2048,3)) for row in raw.numpy()])))
            config['transfer_study']=False
            with self.assertRaises(ValueError):
                depth.reconstruct_geometry(raw,'scanobjectnn_c','gaussian',None,None,reconstruct,ops,**kwargs)

    def test_worker_rejects_mutated_or_mixed_transfer_bundles_before_assets(self):
        from scripts import eval_chamfer_depth as depth
        from scripts import eval_multibackbone_guidance as shared
        import copy
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            configs=depth.configurations(self.transfer_args(root,'scanobjectnn_c'))
            cases=[]
            for update in (dict(guidance_protocol=dict(gamma=.003,eta=.03,p=.95,total=100,scd_normalize=False)),
                           dict(num_classes=40),dict(decoder_style='updated'),dict(lion_ema=True),
                           dict(scheduler_final_alpha_to_one=False),dict(graph={'k':10}),
                           dict(chamfer_experiment={'variant':'semantic'}),dict(normal_reverse_steps=5),
                           dict(primary_policy='minimum_entropy'),dict(evaluation_n_examples=32),
                           dict(selected_original_ids=[1,2,3])):
                cases.append([dict(copy.deepcopy(configs[0]),**update)])
            changed=copy.deepcopy(configs[0]);changed['trajectory_selection']['temperature']=2.;cases.append([changed])
            changed=copy.deepcopy(configs[0]);changed['trajectory_selection']['early_stopping']=True;cases.append([changed])
            mixed=copy.deepcopy(configs[:2]);mixed[1]['seed']=1;cases.append(mixed)
            mixed=copy.deepcopy(configs[:2]);mixed[1]['selected_original_ids']=[1,2,3];cases.append(mixed)
            cases.append([configs[0],configs[0]])
            for case in cases:
                directories=[]
                for i,config in enumerate(case):
                    directory=root/str(i);directory.mkdir(exist_ok=True)
                    (directory/'config.json').write_text(json.dumps(config));directories.append(str(directory))
                with self.subTest(configs=case),patch.object(shared.source,'read_manifest',side_effect=AssertionError('asset access')):
                    with self.assertRaises(ValueError):depth.worker(directories)


class WorkerCompositionTests(unittest.TestCase):
    args = DepthConfigTests.args
    def test_selected_all15_worker_shards_policies_hashes_and_completion(self):
        self.run_worker()

    def test_transfer_all_heads_share_geometry_and_write_class_correct_shards(self):
        for dataset in ('modelnet40_c','scanobjectnn_c'):
            with self.subTest(dataset=dataset):
                self.run_worker(dataset=dataset,transfer=True)

    def run_worker(self,dataset='modelnet40_c',transfer=False):
        from scripts import eval_chamfer_depth as depth
        from scripts import eval_multibackbone_guidance as shared
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args=TransferConfigTests().transfer_args(root,dataset) if transfer else self.args(root)+['--normal-reverse-steps','1']
            configs=depth.configurations(args)
            bundles=[]
            for config in configs:
                config['asset_manifest_sha256']='a'*64
                bundles.append(shared.source.RunBundle.create(root/'result',config['backbone'],config,'cpu mocked worker'))
            config=configs[0]
            labels = np.zeros(21, dtype=np.int64)
            arrays = np.zeros((21, 2048, 3), dtype=np.float32)
            dataset = {'files': {Path(config['label_path']).name: {}}}
            for name in config['corruptions']:
                dataset['files'][shared.source.selected_data_path(config,name).name] = {}
            lion = types.SimpleNamespace(vae=torch.nn.Linear(1,1).eval(), priors=torch.nn.Linear(1,1).eval())
            def initialize(configs, manifest):
                for i,c in enumerate(configs.values()):c['checkpoint_load']={'checkpoint_sha256':str(i)*64,'strict':True}
                return {name:torch.nn.Linear(3,c['num_classes']).eval() for name,c in configs.items()}, {'rng_unchanged':True}
            def snapshots(bundle, config, manifest):
                config['executed_source_files'] = {'repo/shared_trajectory.py': {'sha256':'c'*64,'bytes':1}}
            def classify(geometry, heads, configs, fps):
                return ({n:torch.zeros(len(geometry),configs[n]['num_classes']) for n in heads},
                        {n:{'rng_unchanged':True} for n in heads})
            geometry_calls=[]
            def geometry(raw, dataset, corruption, lion, graph, reconstruct, ops, **kwargs):
                geometry_calls.append(corruption)
                observer = kwargs['observer_factory'](None,None)
                latent = torch.ones(len(raw),3,1,1)
                observer.initial(input_points=raw, shape_latent=latent, local_latent=latent,
                                 noise=latent, noisy_local=latent, style=latent)
                for i in range(observer.horizon):
                    observer.clean(latent, latent, i, (observer.horizon-i-1)*10)
                observer.final_state(latent,latent,0)
                path = kwargs['paired_state_path']
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'cpu cache fixture')
                return torch.zeros(len(raw),2048,3), {}
            class Decoder(torch.nn.Module):
                def decoder(self, unused, beta, context, style):
                    return torch.zeros(len(context),2048,3)
            lion.vae = Decoder().eval()
            native = types.ModuleType('pointnet2_ops')
            native.pointnet2_utils = object()
            with ExitStack() as stack:
                stack.enter_context(installed_modules({'pointnet2_ops':native}))
                for name, value in dict(is_available=True, max_memory_allocated=0, get_device_name='CPU mock').items():
                    stack.enter_context(patch.object(torch.cuda,name,return_value=value))
                for name in ('manual_seed_all','synchronize','reset_peak_memory_stats'):
                    stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.multiple(shared.source, read_manifest=lambda *a:{'guidance':{}},
                    dataset_manifest=lambda *a:dataset,
                    load_array=lambda path,*a,**kw: labels if kw.get('labels') else arrays,
                    snapshot_sources=snapshots, save_drive_checkpoint=lambda *a,**kw:None))
                stack.enter_context(patch.object(shared,'initialize_heads',side_effect=initialize))
                stack.enter_context(patch.object(shared,'matched_source',return_value={'status':'CPU fixture'}))
                stack.enter_context(patch.object(shared,'load_guidance_runtime',return_value=(lion,None,None,None,{},'')))
                stack.enter_context(patch.object(depth,'reconstruct_geometry',side_effect=geometry))
                stack.enter_context(patch.object(shared,'classify_geometry',side_effect=classify))
                stack.enter_context(patch.object(shared,'decoded_to_geometry',side_effect=lambda decoded,*a:decoded))
                stack.enter_context(redirect_stdout(StringIO()))
                depth.worker([str(bundle.path) for bundle in bundles])
            self.assertEqual(len(geometry_calls),15)
            for bundle in bundles:
                self.check_bundle(bundle,config)

    def check_bundle(self,bundle,config):
        import hashlib
        selected=config['selected_original_ids']
        finished = json.loads((bundle.path/'config.json').read_text())
        self.assertEqual(finished['status'],'complete')
        self.assertEqual(len(finished['completed_batches']),15)
        source_hash = hashlib.sha256(json.dumps(finished['executed_source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(finished['executed_source_sha256'],source_hash)
        self.assertEqual(finished['lion_state_before'],finished['lion_state_after'])
        for corruption in config['corruptions']:
            path = bundle.path/'batches'/corruption/'batch0000'
            diagnostic = json.loads((path/'diagnostics.json').read_text())
            self.assertEqual(diagnostic['executed_source_sha256'],source_hash)
            self.assertTrue(diagnostic['trajectory']['probe_rng_unchanged'])
            self.assertEqual(diagnostic['checkpoint_load'],finished['checkpoint_load'])
            with np.load(path/'predictions.npz',allow_pickle=False) as payload:
                self.assertEqual(payload['index'].tolist(),selected)
                self.assertEqual(payload['trajectory_logits'].shape,(3,4 if corruption=='background' or config.get('transfer_study') else 2,config['num_classes']))
                self.assertEqual(payload['probability_mean'].shape,(3,))
                np.testing.assert_array_equal(payload['prediction'],payload['terminal'])


if __name__ == '__main__':
    unittest.main()
