"""Opt-in observation of the unchanged Legacy trajectory (lazy torch import)."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np


def tensor_hash(value) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def _rng_state():
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _restore_rng(state):
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def rng_hash() -> str:
    return _rng_state_hash(_rng_state())


def _rng_state_hash(state) -> str:
    py, np_state, cpu, cuda = state
    return hashlib.sha256(pickle.dumps((py, np_state, cpu.numpy(),
                                      [v.cpu().numpy() for v in cuda]), protocol=4)).hexdigest()


def model_state_hashes(models) -> dict:
    """Audit weights/BN/buffers without loading checkpoints or changing flags."""
    return {name: {key: tensor_hash(value) for key, value in model.state_dict().items()}
            for name, model in models.items()}


def model_buffer_hashes(models) -> dict:
    """Hash named buffers only; useful for checking an isolated probe boundary."""
    return {name: {key: tensor_hash(value) for key, value in model.named_buffers()}
            for name, model in models.items()}


@contextmanager
def isolated_probe(models):
    """Restore RNG and changed buffers even on failure; never toggle eval modes.

    Do not copy unchanged buffers in-place: that would increment version counters
    of buffers possibly saved by the still-live guidance autograd graph.
    """
    import torch
    if any(m.training for model in models for m in model.modules()):
        raise ValueError('Trajectory probes require the existing all-eval model contract')
    state = _rng_state()
    buffers = [(b, b.detach().clone()) for model in models for b in model.buffers()]
    try:
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                if not torch.equal(buffer, saved):
                    buffer.copy_(saved)
        _restore_rng(state)


def checkpoint_list(value: str, horizon: int) -> list:
    steps = [int(v) for v in value.split(',')]
    if not steps or steps != sorted(set(steps)) or any(s < 1 or s >= horizon for s in steps):
        raise ValueError('Checkpoints must be unique, increasing and strictly before terminal')
    return steps


def trajectory_protocol(args) -> dict:
    result = dict(mode=args.trajectory_mode,
                checkpoints={'10': checkpoint_list(args.trajectory_steps_normal, 10),
                             '30': checkpoint_list(args.trajectory_steps_bg, 30)},
                geometry_samples_per_batch=args.trajectory_geometry_samples,
                style='predicted_clean: pre-update forward style; terminal: original updated decode',
                tie_break='terminal first, then earliest checkpoint', temperature=1,
                timing='synchronized component wall seconds; includes observer/hash/I/O overhead; no speedup claim')
    if getattr(args, 'trajectory_sampling', 'off') == 'paired':
        from trajectory_sampling import sampling_protocol
        result['sampling'] = sampling_protocol()
    return result


def corruption_names(args) -> list:
    """Predeclared evaluation scope; never infer a corruption from input."""
    from research_artifacts import CORRUPTIONS
    if args.corruption:
        return [args.corruption]
    return [c for c in CORRUPTIONS
            if c != 'background' or getattr(args, 'corruption_scope', 'all15') == 'all15']


def selected_indices(args) -> dict:
    """Explicit ordered original IDs, independent of inference RNG and labels."""
    if not args.indices_manifest:
        return {}
    value = json.loads(Path(args.indices_manifest).read_text(encoding='utf-8'))
    from research_artifacts import CORRUPTIONS
    if value.get('schema_version') != 1 or value.get('dataset') != 'modelnet40_c':
        raise ValueError('Invalid index manifest schema/dataset')
    indices = value['indices']
    required = corruption_names(args)
    for name in required:
        ids = indices.get(name, [])
        if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < 2468 for i in ids):
            raise ValueError('Invalid/duplicate original indices for ' + name)
    return {name: indices[name] for name in required}


class TrajectoryObserver:
    """Own one corruption's disk artifacts; hold only the current batch on CPU."""
    def __init__(self, path, corruption, seed, indices, horizon, options):
        self.path = Path(path) / 'trajectory' / corruption
        self.path.mkdir(parents=True, exist_ok=False)
        self.corruption, self.seed, self.indices = corruption, seed, list(indices)
        self.horizon, self.options = horizon, options
        self.steps = options['checkpoints'][str(horizon)]
        self.offset, self.batch = 0, 0
        self.active = None

    def _sync(self):
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def mark(self, component):
        self._sync()
        now = time.perf_counter()
        if self.active is not None:
            self.times[self.active] = self.times.get(self.active, 0.) + now - self.clock
        self.clock, self.active = now, component

    def begin(self, count, device):
        self.device, self.count = device, count
        self.ids = self.indices[self.offset:self.offset + count]
        if len(self.ids) != count:
            raise ValueError('Trajectory index count mismatch')
        self.times, self.candidates, self.hashes = {}, [], {}
        self.encode_replay = None
        self.active = None
        self.start = time.perf_counter()
        self.rng_before = rng_hash()
        self.mark('input_preprocess')

    def bind(self, vae, classifier, preprocess, prepare_sampling_cloud=None):
        self.vae, self.classifier, self.preprocess = vae, classifier, preprocess
        self.prepare_sampling_cloud = prepare_sampling_cloud

    @property
    def sampling_enabled(self):
        return self.options.get('sampling', {}).get('mode') == 'paired'

    def encode_rng_start(self):
        return _rng_state() if self.options['mode'] == 'audit' else None

    def initial(self, **states):
        self.mark('state_hashing')
        self.hashes = {name: tensor_hash(value) for name, value in states.items()}

    def replay_encode(self, vae, points, original_latents, rng_before):
        """Repeat VAE encode with the exact pre-encode RNG state, then restore post-state."""
        if self.options['mode'] != 'audit':
            return
        import torch
        post_rng = _rng_state()
        post_rng_hash = _rng_state_hash(post_rng)
        buffers_before = model_buffer_hashes({'vae': vae})
        original = dict(shape=tensor_hash(original_latents[2][0][0]),
                        local=tensor_hash(original_latents[2][1][0]))
        with isolated_probe([vae]):
            _restore_rng(rng_before)
            with torch.no_grad():
                replay = vae.encode(points)
            repeated = dict(shape=tensor_hash(replay[2][0][0]),
                            local=tensor_hash(replay[2][1][0]))
        rng_restored_hash = rng_hash()
        buffers_after = model_buffer_hashes({'vae': vae})
        self.encode_replay = dict(rng_replayed=True, original_hashes=original,
                                  replay_hashes=repeated,
                                  exact_latent_match=original == repeated,
                                  post_rng_restored=(post_rng_hash == rng_restored_hash),
                                  buffers_unchanged=buffers_before == buffers_after)

    def clean(self, latent, style, loop_index, timestep):
        if self.options['mode'] != 'observe' or loop_index + 1 not in self.steps:
            return
        self.mark('probe_guard_and_hash')
        before = rng_hash()
        buffers_before = model_buffer_hashes({'vae': self.vae, 'classifier': self.classifier})
        with isolated_probe([self.vae, self.classifier]):
            latent, style = latent.detach().clone(), style.detach().clone()
            hashes = dict(clean_latent=tensor_hash(latent), style=tensor_hash(style))
            self.mark('probe_decode')
            points = self.vae.decoder(None, beta=None, context=latent.squeeze(3).squeeze(2),
                                      style=style.squeeze(3).squeeze(2))
            geometry = points[:self.options['geometry_samples_per_batch']].detach().cpu().numpy().copy()
            self.mark('probe_preprocess_fps')
            if self.sampling_enabled:
                from trajectory_sampling import native_fps
                cloud = self.prepare_sampling_cloud(points)
                points, indices_a = native_fps(cloud)
            else:
                points = self.preprocess(points)
            self.mark('probe_classifier')
            logits = self.classifier.classification_only(points, only_unmasked=False)
            extra = None
            if self.sampling_enabled:
                from trajectory_sampling import collect_views
                extra, hashes['sampling'] = collect_views(
                    cloud, indices_a, logits, self.classifier, self.ids,
                    self.seed, 2, self.mark)
            self.mark('probe_artifact')
            self.mark('probe_guard_and_hash')
        after = rng_hash()
        buffers_after = model_buffer_hashes({'vae': self.vae, 'classifier': self.classifier})
        hashes.update(probe_rng_before=before, probe_rng_after=after,
                      probe_buffers_before=buffers_before, probe_buffers_after=buffers_after)
        if after != before:
            raise RuntimeError('Probe changed main RNG')
        if buffers_after != buffers_before:
            raise RuntimeError('Probe changed model buffers after restoration')
        self._save(logits, 'clean_%02d' % (loop_index + 1), loop_index, int(timestep),
                   'predicted_clean', 'pre_update', hashes, geometry, extra)

    def final_state(self, latent, style, timestep):
        self.mark('terminal_state_hashing')
        self.final_hashes = dict(latent=tensor_hash(latent), style=tensor_hash(style))
        self.last_timestep = int(timestep)

    def _save(self, logits, name, loop_index, timestep, kind, style_timing, hashes, geometry=None,
              extra=None):
        array = logits.detach().cpu().numpy().copy()
        if array.shape != (self.count, 40) or not np.isfinite(array).all():
            raise ValueError('Expected finite B x 40 trajectory logits')
        shifted = array.astype(np.float64) - array.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        entropy = -(np.exp(logp) * logp).sum(1) / np.log(40)
        filename = '%04d_%s.npz' % (self.batch, name)
        data = dict(indices=np.asarray(self.ids), logits=array, predictions=array.argmax(1), entropy=entropy)
        if extra is not None:
            data.update(extra)
        if geometry is not None and len(geometry):
            data.update(decoded_geometry=geometry, geometry_indices=np.asarray(self.ids[:len(geometry)]))
        # Exclusive write; interrupted traces remain present but cannot pass analysis.
        with (self.path / filename).open('xb') as stream:
            np.savez_compressed(stream, **data)
        self.candidates.append(dict(name=name, file=filename, loop_index=loop_index,
                                    loop_count=loop_index+1, timestep=timestep, kind=kind,
                                    style_timing=style_timing, hashes=hashes))

    def terminal(self, logits, targets, sampling_context=None):
        extra = None
        if self.sampling_enabled:
            if sampling_context is None:
                raise ValueError('Sampling requires normalized terminal cloud and A indices')
            from trajectory_sampling import collect_views
            cloud, indices_a = sampling_context
            extra, self.final_hashes['sampling'] = collect_views(
                cloud, indices_a, logits, self.classifier, self.ids,
                self.seed, 8, self.mark)
        self.mark('terminal_artifact')
        self._save(logits, 'terminal', self.horizon-1, self.last_timestep,
                   'original_terminal', 'post_update', self.final_hashes, extra=extra)
        self.mark(None)
        record = dict(schema_version=1, complete=True, corruption=self.corruption, seed=self.seed,
                      indices=self.ids, targets=targets.detach().cpu().view(-1).tolist(),
                      candidates=self.candidates, initial_hashes=self.hashes,
                      rng_before=self.rng_before, rng_after=rng_hash(), times=self.times,
                      encode_replay=self.encode_replay,
                      wall_seconds=time.perf_counter()-self.start)
        with (self.path / ('%04d.json' % self.batch)).open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        self.offset += self.count
        self.batch += 1
