"""Read-only tensor instrumentation around the frozen shared runner; no model edits."""
import json
import os
from pathlib import Path
import sys
import traceback


def instrumented_worker(directories, specification):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    from legacy_trajectory import rng_hash
    from scripts import eval_multibackbone_guidance as runner
    from shared_trajectory import SharedTrajectoryObserver
    from semantic_trajectory import SemanticTrajectoryObserver

    destination = Path(specification['sidecar_root'])
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(step=0, corruption=None, scope='outside', arrays={}, saved=set(), metadata=[])
    handles = []
    original_load, original_step = runner.load_guidance_runtime, DDIMScheduler.step

    def capture(name, value):
        if not isinstance(value, torch.Tensor):
            return
        before = rng_hash()
        key = '%s_%s' % (state['corruption'], name)
        if key in state['arrays']:
            raise ValueError('Duplicate diagnostic tensor: ' + key)
        state['arrays'][key] = value.detach().cpu().numpy().copy()
        state['metadata'].append(dict(name=key, shape=list(value.shape), strides=list(value.stride()),
                                     dtype=str(value.dtype), device=str(value.device),
                                     contiguous=value.is_contiguous(), requires_grad=value.requires_grad))
        if rng_hash() != before:
            raise RuntimeError('Diagnostic capture changed RNG')

    def save(corruption, partial=False):
        if corruption is None or (corruption, partial) in state['saved']:
            return
        values = {key: value for key, value in state['arrays'].items() if key.startswith(corruption + '_')}
        if not values:
            return
        name = corruption + ('_partial' if partial else '')
        with (destination / (name + '.npz')).open('xb') as stream:
            np.savez_compressed(stream, **values)
        with (destination / (name + '.json')).open('x') as stream:
            json.dump(dict(status='partial' if partial else 'complete', case=specification['case'],
                           capture='detached CPU copies; no RNG/model/gradient edits',
                           tensors=[row for row in state['metadata'] if row['name'].startswith(corruption + '_')]), stream, indent=2)
        state['saved'].add((corruption, partial))

    def prior_before(module, args, kwargs):
        state['step'] += 1
        step = state['step']
        local = kwargs.get('x', args[0] if args else None)
        style = kwargs['condition_input']
        capture('step%02d_prior_input' % step, local)
        capture('step%02d_prior_style' % step, style)
        capture('step%02d_timestep' % step, kwargs.get('t'))
        if local.requires_grad:
            local.register_hook(lambda gradient, number=step: capture('step%02d_grad_local' % number, gradient))
        if style.requires_grad:
            style.register_hook(lambda gradient, number=step: capture('step%02d_grad_style' % number, gradient))

    def prior_after(module, args, kwargs, output):
        capture('step%02d_prior_output' % state['step'], output)

    def decoder_after(module, args, kwargs, output):
        capture('%s_decoded' % state['scope'], output)

    def load(config, manifest):
        result = original_load(config, manifest)
        lion = result[0]
        handles.extend([lion.priors[1].register_forward_pre_hook(prior_before, with_kwargs=True),
                        lion.priors[1].register_forward_hook(prior_after, with_kwargs=True),
                        lion.vae.decoder.register_forward_hook(decoder_after, with_kwargs=True)])
        return result

    def scheduler_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        capture('step%02d_scheduler_prev' % state['step'], result.prev_sample)
        capture('step%02d_scheduler_pred' % state['step'], result.pred_original_sample)
        return result

    def observer_provider(**values):
        corruption = values.pop('corruption')
        values.pop('original_ids')
        state.update(step=0, corruption=corruption, scope='outside')
        cls = SemanticTrajectoryObserver if 'kind' in values else SharedTrajectoryObserver
        observer = cls(**values)
        initial, clean, final, terminal = observer.initial, observer.clean, observer.final_state, observer.terminal
        def observe_initial(**tensors):
            for name, value in tensors.items():
                capture('initial_' + name, value)
            return initial(**tensors)
        def observe_clean(latent, style, loop_index, timestep):
            capture('step%02d_clean_latent' % (loop_index + 1), latent)
            capture('step%02d_clean_style' % (loop_index + 1), style)
            state['scope'] = 'step%02d_probe' % (loop_index + 1)
            try:
                return clean(latent, style, loop_index, timestep)
            finally:
                state['scope'] = 'outside'
        def observe_final(latent, style, timestep):
            capture('final_local', latent); capture('final_style', style)
            state['scope'] = 'terminal'
            return final(latent, style, timestep)
        def observe_terminal(logits):
            result = terminal(logits)
            for name, value in logits.items():
                capture('terminal_logits_' + name, value)
            save(corruption)
            return result
        observer.initial, observer.clean = observe_initial, observe_clean
        observer.final_state, observer.terminal = observe_final, observe_terminal
        return observer

    runner.load_guidance_runtime, DDIMScheduler.step = load, scheduler_step
    try:
        runner.worker(directories, observer_provider=observer_provider)
    except BaseException:
        (destination / 'failure.txt').write_text(traceback.format_exc())
        save(state['corruption'], partial=True)
        raise
    finally:
        runner.load_guidance_runtime, DDIMScheduler.step = original_load, original_step
        for handle in handles:
            handle.remove()


def main():
    specification = json.loads(Path(os.environ['CHAMFER_NATIVE_DIAGNOSTIC_SPEC']).read_text())
    sys.path.insert(0, specification['source_root'])
    if sys.argv[1:2] == ['--worker-bundles']:
        instrumented_worker(json.loads(sys.argv[2]), specification)
        return
    from scripts import eval_multibackbone_guidance as runner
    arguments = specification['effective_runner_arguments']
    runner.main(arguments, worker_script=Path(__file__).resolve())


if __name__ == '__main__':
    main()
