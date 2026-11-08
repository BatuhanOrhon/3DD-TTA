#!/usr/bin/env python3
"""Repeated operational LION speed study; no changes to production defaults."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arm_order(arms, block):
    """Rotate positions, reverse each complete cycle; independent of outcomes."""
    order = list(arms) if (block // len(arms)) % 2 == 0 else list(reversed(arms))
    shift = block % len(arms)
    return order[shift:] + order[:shift]


def pm_readout(h, z, terminal, decode, classify, models):
    """Preserve production clean/no-grad versus terminal/grad decoder semantics."""
    from legacy_trajectory import isolated_probe
    if terminal:
        decoded = decode(h.detach().requires_grad_(True), z)
        with isolated_probe(models):
            return classify(decoded).detach()
    with isolated_probe(models):
        return classify(decode(h.detach().clone(), z.detach().clone())).detach()


class SpeedObserver:
    """Identical PM readout/capture for both samplers, without audit hashes."""
    def __init__(self, checkpoints, readout):
        self.steps, self.readout = list(checkpoints), readout
        self.latents, self.saved = [], []

    def mark(self, name):
        pass

    def initial(self, **values):
        pass

    def clean(self, latent, style, loop_index, timestep):
        if loop_index + 1 in self.steps:
            if loop_index + 1 != self.steps[len(self.saved)]:
                raise ValueError('Missing/reordered PM checkpoint')
            self.latents.append(latent.detach().clone())
            self.saved.append(self.readout(latent, style, False).detach())

    def final_state(self, latent, style, timestep):
        if len(self.latents) != len(self.steps):
            raise ValueError('Incomplete PM readout')
        self.latents.append(latent.detach().clone())

    def terminal(self, logits):
        if len(self.saved) != len(self.steps):
            raise ValueError('Duplicate/incomplete terminal')
        self.saved.append(logits.detach())

    def logits(self):
        import torch
        if len(self.saved) != len(self.steps) + 1:
            raise ValueError('Incomplete PM logits')
        return torch.stack(self.saved, 1)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def checkpoint_case(case, destination):
    """Preserve closed case bytes outside timing; never replace prior evidence."""
    case, destination = Path(case), Path(destination)
    if Path('/content/drive') in destination.parents and not os.path.ismount('/content/drive'):
        raise OSError('Drive disconnected before checkpoint')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(case.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence symlink forbidden')
        if not path.is_file():
            continue
        name = str(path.relative_to(case))
        data = path.read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise IOError('Checkpoint copy mismatch')
        manifest[name] = dict(bytes=len(data), sha256=digest)
    if Path('/content/drive') in destination.parents and not os.path.ismount('/content/drive'):
        raise OSError('Drive disconnected during checkpoint')
    write(destination / 'checkpoint_manifest.json', manifest)


def checkpoint_integrity(models, before, case, destination):
    """Only persist a closed shard after its frozen model state is verified."""
    from scripts.eval_multibackbone_source import state_hash
    after = [state_hash(model) for model in models]
    if after != before:
        raise ValueError('Model state mutated before integrity checkpoint')
    case = Path(case)
    case.mkdir(parents=True, exist_ok=False)
    write(case / 'integrity.json', dict(status='verified', before=before, after=after))
    checkpoint_case(case, destination)


def iter_batches(ids, size):
    for batch, start in enumerate(range(0, len(ids), size)):
        yield batch, ids[start:start + size]


def warm_batch_sizes(phase, ids, size):
    sizes = {len(group) for _, group in iter_batches(ids, size)}
    if phase == 'profile' and size > 1:
        sizes.add(2468 % size or size)
    return sorted(sizes)


def phase_layout(phase, selection, size):
    from research_artifacts import CORRUPTIONS
    if phase in ('full14', 'continue14'):
        return dict(ids=list(range(2468)), corruptions=[c for c in CORRUPTIONS if c != 'background'],
                    seeds=[0], repetitions=1)
    if phase == 'background':
        return dict(ids=list(range(2468)), corruptions=['background'], seeds=[0], repetitions=1)
    if phase == 'profile':
        return dict(ids=selection['calibration'] if size == 1 else selection['confirm'][:size], corruptions=['uniform', 'occlusion'],
                    seeds=[0], repetitions=3)
    return dict(ids=selection['calibration' if phase == 'repeat' else phase],
                corruptions=['gaussian','background','density','occlusion'] if phase == 'repeat' else list(CORRUPTIONS),
                seeds=[0,1,2] if phase == 'confirm' else [0],
                repetitions=3 if phase == 'repeat' else 2)


def arm_protocols(phase, arms):
    """Declare the horizon actually selected by the timed corruption policy."""
    if phase not in ('profile', 'full14', 'continue14', 'background'):
        return None
    return {arm: (dict(steps=35 if phase == 'background' else 5,
                       gamma=.01, eta=.01, p=.95, readouts=1,
                       method='repository original eval/raw with common controlled backend')
                  if arm == 'terminal' else
                  dict(steps=35 if phase == 'background' else 8,
                       gamma=.03, eta=.003, p=.95, readouts=4)) for arm in arms}


def corruption_schedule(corruptions, skipped, blocks_per_corruption):
    """Keep the original RNG indices and arm-order offsets after a shard skip."""
    if (len(skipped) != len(set(skipped)) or not set(skipped).issubset(corruptions)
            or len(skipped) >= len(corruptions) or blocks_per_corruption <= 0):
        raise ValueError('Invalid corruption skips or block count')
    return [(ci, corruption, ci * blocks_per_corruption)
            for ci, corruption in enumerate(corruptions) if corruption not in skipped]


def terminal_inference(raw_array, corruption, lion, original, ops, classify, models):
    """Repository original N5/N35 SCD (normal/Background), with one terminal readout."""
    import numpy as np
    import torch
    from legacy_trajectory import isolated_probe
    from scripts import eval_multibackbone_guidance as shared
    raw = torch.from_numpy(np.array(raw_array, dtype=np.float32, copy=True))
    geometry, _ = shared.reconstruct_geometry(raw, 'modelnet40_c', corruption, lion,
                                              None, original, ops, method=shared.ORIGINAL_METHOD)
    with isolated_probe(models):
        logits = classify(geometry).detach()
        probabilities = logits.softmax(-1)
    return logits[:, None, :], probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root', 'manifest-sha256', 'selection', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--phase', choices=('repeat', 'screen', 'confirm', 'profile', 'full14', 'continue14', 'background'), required=True)
    parser.add_argument('--skip-corruptions', nargs='*', default=[])
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=32)
    parser.add_argument('--progress-drive', required=True)
    parser.add_argument('--arms', nargs='+', choices=('original', 'original_audit', 'full', 'cheap', 'hybrid', 'sparse', 'unguided', 'terminal'),
                        default=['original', 'full', 'cheap', 'hybrid', 'sparse'])
    args = parser.parse_args()
    if len(args.arms) < 2 or len(args.arms) != len(set(args.arms)) or 'original' not in args.arms:
        parser.error('At least two unique arms including original are required')
    if 'original_audit' in args.arms and args.phase != 'repeat':
        parser.error('Audit observer is not a timed deployment comparator')
    if 'terminal' in args.arms and args.phase not in ('profile', 'full14', 'continue14', 'background'):
        parser.error('Original terminal requires a matched full14/profile/background protocol')
    if args.skip_corruptions and args.phase != 'continue14':
        parser.error('Corruption skips require the explicitly partial continue14 phase')
    if args.phase == 'continue14' and not args.skip_corruptions:
        parser.error('Continuation requires declared independently accepted corruption skips')
    if args.phase == 'repeat' and not {'original_audit', 'full'}.issubset(args.arms):
        parser.error('Repeat phase requires original_audit and full cross controls')
    output = Path(args.output)
    progress = Path(args.progress_drive) / output.name
    if not os.path.ismount('/content/drive') or Path('/content/drive/MyDrive') not in progress.parents:
        parser.error('Verified Drive progress path required')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'command.json', sys.argv)
    (output / 'command.txt').write_text(' '.join(sys.argv) + '\n')
    write(output / 'config.json', vars(args))
    started, rows = time.monotonic(), []
    try:
        progress.mkdir(parents=True, exist_ok=False)
        import numpy as np
        import torch
        from scripts import eval_multibackbone_source as source
        from scripts import eval_multibackbone_guidance as shared
        from scripts.diagnose_decoded_spectral import configuration, apply_backend_policy, verify_setup
        from legacy_trajectory import isolated_probe, _rng_state, _restore_rng, tensor_hash
        from lion_guidance_pilot import sample
        from pointnet2_ops import pointnet2_utils
        from research_artifacts import CORRUPTIONS
        from shared_trajectory import SharedTrajectoryObserver
        from scripts.analyze_lion_guidance_speed import paired_summary, cross_control_differences

        asset = Path(args.asset_root)
        manifest = source.read_manifest(asset, args.manifest_sha256)
        selection = json.loads(Path(args.selection).read_text())
        assert selection['dataset'] == 'modelnet40_c'
        groups = [selection[k] for k in ('calibration', 'screen', 'confirm')]
        assert list(map(len, groups)) == [4, 32, 256]
        flattened = sum(groups, [])
        assert len(flattened) == len(set(flattened)) and all(type(i) is int and 0 <= i < 2468 for i in flattened)
        layout = phase_layout(args.phase, selection, args.batch_size)
        ids, corruptions = layout['ids'], layout['corruptions']
        seeds, repetitions = layout['seeds'], layout['repetitions']
        batch_size = 2 if args.phase == 'repeat' else args.batch_size
        batches = list(iter_batches(ids, batch_size))
        schedule = corruption_schedule(corruptions, args.skip_corruptions,
                                       len(batches) * len(seeds) * repetitions)
        executed_corruptions = [corruption for _, corruption, _ in schedule]
        warm_sizes = warm_batch_sizes(args.phase, ids, batch_size)
        base = configuration(['--asset-root', str(asset), '--asset-manifest-sha256', args.manifest_sha256,
            '--result-root', str(output), '--paired-state-root', str(output / 'paired'), '--arm', 'control',
            '--count', '2', '--backend-policy', 'controlled', '--corruptions', *corruptions])
        base.update(batch_size=batch_size, selected_original_ids=ids, count=len(ids))
        backend = apply_backend_policy('controlled')
        setup = verify_setup(asset, manifest, base)
        dataset = source.dataset_manifest(manifest, base)
        base['dataset_root'] = dataset['root']
        arrays = {c: source.load_array(Path(dataset['root']) / ('data_' + c + '_5.npy'),
            dataset['files']['data_' + c + '_5.npy'], base) for c in executed_corruptions}
        labels = source.load_array(Path(dataset['root']) / 'label.npy', dataset['files']['label.npy'], base, labels=True)
        lion, _, original, ops, native, resolved = shared.load_guidance_runtime(dict(base, method=shared.ORIGINAL_METHOD), manifest)
        classifier = source.load_classifier(base, manifest)
        models = [lion.vae, lion.priors, classifier]
        initial_hashes = [source.state_hash(m) for m in models]
        assert all(not m.training for model in models for m in model.modules())
        assert all(not p.requires_grad for model in models for p in model.parameters())
        write(output / 'resolved_config.json', dict(base=base, ids=ids, seeds=seeds, repetitions=repetitions,
            arms=args.arms, tau=.1, refresh_modulo=3, warmups_per_arm_per_corruption=2,
            executed_corruptions=executed_corruptions, skipped_corruptions=args.skip_corruptions,
            warm_batch_sizes=warm_sizes,
            arm_protocols=arm_protocols(args.phase, args.arms),
            timing_boundary='CPU batch array ready through fresh preparation/encode/sampler/required readouts/GPU probabilities; synchronized',
            exclusions='model load, dataset disk I/O, RNG seeding, hashes, archive writes; same for all methods',
            repeat_phase='same prepared state and RNG, excludes encode; diagnostics only',
            stochastic_pairing='Same method-independent seed; fresh native encode per operational arm; not identical cached states',
            primary_gate='saving point>=15%, accuracy CI lower>-0.3pp, nondegenerate; conditional on seeds/runtime'))
        write(output / 'environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), backend=backend, native=native,
            gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
            setup=setup, dataset=dataset, manifest_sha256=args.manifest_sha256, lion_config=resolved,
            model_hashes=initial_hashes, selection=selection, source_overlay_receipt=manifest.get('source_overlay_receipt')))
        (output / 'environment.txt').write_text('See environment.json for source/native/data/checkpoint identities.\n')
        header = output / 'progress_header'
        header.mkdir()
        for name in ('command.txt', 'config.json', 'resolved_config.json', 'environment.json'):
            shutil.copyfile(output / name, header / name)
        checkpoint_case(header, progress / 'header')

        def prepare(raw_array):
            raw = torch.from_numpy(np.array(raw_array, dtype=np.float32, copy=True))
            normalized, center, scale = ops.normalize(raw)
            prepared = ops.rotate_pointcloud(torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().cuda() * 3.3885)
            with torch.no_grad():
                latent = lion.vae.encode(prepared)
                z = latent[2][0][0].unsqueeze(2).unsqueeze(3)
                h = latent[2][1][0].unsqueeze(2).unsqueeze(3)
                lion.vae.global2style(z)
                state = dict(shape_latent=z, local_latent=h, noise=torch.randn_like(h))
            state['rng_after_prepare'] = _rng_state()
            return prepared, center, scale, state

        def infer(raw_array, arm, steps, corruption, cached=None):
            count = len(raw_array)
            def fps(points, n):
                indices = pointnet2_utils.furthest_point_sample(points.contiguous(), n)
                return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
            def classify_geometry(geometry):
                points, features = source.preprocess(geometry, base, fps)
                return classifier(points, features=features)
            if arm == 'terminal':
                logits, pm = terminal_inference(raw_array, corruption, lion, original, ops,
                                               classify_geometry, models)
                return logits, pm, [], None
            prepared, center, scale, state = prepare(raw_array) if cached is None else cached
            def classify(decoded):
                geometry = shared.decoded_to_geometry(decoded, 'modelnet40_c', scale, center, ops)
                return classify_geometry(geometry)
            def decode(h, z):
                return lion.vae.decoder(None, beta=None, context=h.reshape(count, -1), style=z.reshape(count, -1))
            def readout(h, z, terminal):
                return pm_readout(h, z, terminal, decode, classify, models)
            checkpoints = [9, 18, 27] if steps == 35 else [2, 4, 6]
            observer = SpeedObserver(checkpoints, readout)
            _restore_rng(state['rng_after_prepare'])
            if arm == 'original_audit':
                class Audit(SharedTrajectoryObserver):
                    def clean(self, h, z, i, t):
                        if i + 1 in self.steps:
                            observer.latents.append(h.detach().clone())
                        return super().clean(h, z, i, t)
                    def final_state(self, h, z, t):
                        observer.latents.append(h.detach().clone())
                        return super().final_state(h, z, t)
                audit = Audit(lion.vae, {'pointmae': classifier}, lambda x: {'pointmae': classify(x)},
                              steps, 40, checkpoint_steps=checkpoints)
                decoded = original(prepared, lion, steps, .03, .003, .95, prepared_state=state, trajectory_observer=audit)
                with isolated_probe(models):
                    audit.terminal({'pointmae': classify(decoded)})
                logits = torch.as_tensor(audit.arrays('pointmae')['trajectory_logits'], device='cuda')
            elif arm == 'original':
                decoded = original(prepared, lion, steps, .03, .003, .95, prepared_state=state, trajectory_observer=observer)
                with isolated_probe(models):
                    observer.terminal(classify(decoded))
                logits = observer.logits()
            else:
                def callback(h, z, loop):
                    if loop == steps:
                        observer.terminal(readout(h, z, True))
                    else:
                        if loop != checkpoints[len(observer.saved)]:
                            raise ValueError('Reordered candidate readout')
                        observer.saved.append(readout(h, z, False))
                observer.latents, _ = sample(lion, state, steps, arm, tau=.1, callback=callback)
                assert len(observer.latents) == 4
                logits = observer.logits()
            pm = logits.softmax(-1).mean(1)
            return logits, pm, observer.latents, state

        numerical, cross_controls = [], []
        for ci, corruption, block in schedule:
            corruption_hashes = [source.state_hash(m) for m in models]
            assert corruption_hashes == initial_hashes, 'Model state mutated before corruption'
            steps = 35 if corruption == 'background' else 8
            # Calibration IDs only; warm the actual batch size by cycling fixed IDs.
            for warm_size in warm_sizes:
                warm_ids = (selection['calibration'] * ((warm_size + 3) // 4))[:warm_size]
                warm_raw = arrays[corruption][warm_ids]
                for warm in range(2):
                    for arm in arm_order(args.arms, ci * 2 + warm):
                        shared.seed_inference(90000 + ci)
                        with torch.enable_grad():
                            result = infer(warm_raw, arm, steps, corruption)
                        torch.cuda.synchronize()
                        del result
                    print('LION_SPEED_WARMUP', corruption, warm_size, warm + 1, flush=True)
            for seed in seeds:
                for batch, batch_ids in batches:
                    raw = arrays[corruption][batch_ids]
                    case = output / ('%s_s%d_b%d' % (corruption, seed, batch))
                    case.mkdir()
                    cached = None
                    if args.phase == 'repeat':
                        shared.seed_inference(0)
                        cached = prepare(raw)
                        prepared, _, _, state = cached
                        _restore_rng(state['rng_after_prepare'])
                        _, recorder = shared.paired_preparation(case / 'paired_state.npz', prepared)
                        recorder({k: state[k] for k in ('shape_latent', 'local_latent', 'noise')})
                    previous = {}
                    for repeat in range(repetitions):
                        draw_seed = seed * 100000 + ci * 1000 + batch * 10
                        # Restore identical pseudorandom draws across native repeats;
                        # every operational arm still recomputes its own native encode.
                        for position, arm in enumerate(arm_order(args.arms, block)):
                            shared.seed_inference(draw_seed)
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            tick = time.perf_counter()
                            logits, pm, latents, state = infer(raw, arm, steps, corruption, cached)
                            torch.cuda.synchronize()
                            seconds = time.perf_counter() - tick
                            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
                            reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
                            assert logits.shape == (len(batch_ids), 1 if arm == 'terminal' else 4, 40) and torch.isfinite(logits).all()
                            prediction = pm.argmax(-1).cpu().numpy()
                            logits_cpu = logits.detach().cpu().numpy()
                            stem = '%s_r%d' % (arm, repeat)
                            saved = dict(logits=logits_cpu, probability_mean=pm.detach().cpu().numpy(),
                                prediction=prediction, index=np.asarray(batch_ids), label=np.asarray(labels[batch_ids]).reshape(-1))
                            if args.phase == 'repeat':
                                saved['latents'] = torch.stack(latents, 1).cpu().numpy()
                                if arm in previous:
                                    old = previous[arm]
                                    numerical.append(dict(arm=arm, corruption=corruption, batch=batch, repeat=repeat,
                                        latent_rms=float(np.sqrt(np.mean((saved['latents'].astype(np.float64) - old['latents']) ** 2))),
                                        logit_rms=float(np.sqrt(np.mean((logits_cpu.astype(np.float64) - old['logits']) ** 2))),
                                        prediction_changes=int((prediction != old['prediction']).sum())))
                                else:
                                    previous[arm] = saved
                            np.savez_compressed(case / (stem + '.npz'), **saved)
                            write(case / (stem + '.json'), dict(seconds=seconds, peak_mb=peak, peak_reserved_mb=reserved, order=position,
                                draw_seed=draw_seed, ids=batch_ids, repeat=repeat,
                                state_hashes=None if state is None else {k: tensor_hash(state[k]) for k in ('shape_latent', 'local_latent', 'noise')}))
                            for i, identity in enumerate(batch_ids):
                                rows.append(dict(arm=arm, corruption=corruption, seed=seed, repeat=repeat, batch=batch,
                                    object_id=identity, label=int(labels[identity]), correct=int(prediction[i] == int(labels[identity])),
                                    seconds=seconds, batch_size=len(batch_ids), peak_mb=peak, peak_reserved_mb=reserved, order=position))
                            del logits, pm, latents, state, saved
                            print('LION_SPEED_PROGRESS', args.phase, corruption, seed, batch, repeat, arm, round(seconds, 3), flush=True)
                        block += 1
                    if args.phase == 'repeat':
                        for other in ('original_audit', 'full'):
                            differences = cross_control_differences(case, 'original', other, repetitions)
                            cross_controls.extend(dict(row, corruption=corruption, batch=batch) for row in differences)
                    write(case / 'complete.json', dict(rows=len(batch_ids) * repetitions * len(args.arms)))
                    checkpoint_case(case, progress / case.name)
                    print('LION_SPEED_CASE_DRIVE_VERIFIED', case.name, flush=True)
            checkpoint_integrity(models, corruption_hashes, output / 'integrity' / corruption,
                                 progress / 'integrity' / corruption)
            print('LION_SPEED_CORRUPTION_INTEGRITY_VERIFIED', corruption, flush=True)
        assert initial_hashes == [source.state_hash(m) for m in models], 'Model state mutated'
        with (output / 'per_object.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        expected = len(ids) * len(executed_corruptions) * len(seeds) * repetitions * len(args.arms)
        assert len(rows) == expected
        comparisons = [paired_summary(rows, 'original', arm) for arm in args.arms if arm != 'original']
        if 'terminal' in args.arms and 'cheap' in args.arms:
            comparisons.append(paired_summary(rows, 'terminal', 'cheap'))
        write(output / 'summary.json', dict(status='complete', rows=expected, comparisons=comparisons,
            executed_corruptions=executed_corruptions, skipped_corruptions=args.skip_corruptions,
            numerical_repeats=numerical, cross_controls=cross_controls, elapsed=time.monotonic() - started,
            limitation='Conditional on these official-test exposed IDs, seeds/repeats and runtime; not external replication'))
        for filename, values in (('summary.csv', comparisons), ('per_corruption.csv', [dict(reference=r['reference'], arm=r['candidate'], corruption=c, delta_pp=v)
                               for r in comparisons for c, v in r['per_corruption_delta_pp'].items()])):
            flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in values]
            with (output / filename).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        (output / 'notes.md').write_text('Registered repeated operational speed study. Repeat-phase timing excludes encode and audit-arm time is diagnostic only.\n'
            + ('Continuation contains only the declared executed corruptions; skipped-shard rows belong in a separate provenance-bound composite.\n'
               if args.phase == 'continue14' else ''))
        print('LION_SPEED_PHASE_COMPLETE', args.phase, flush=True)
    except BaseException:
        write(output / 'failure.json', dict(traceback=traceback.format_exc(), completed_rows=rows, elapsed=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
