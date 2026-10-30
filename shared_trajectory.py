"""Observe fixed dataset-validated trajectories without advancing RNG."""
import time

import numpy as np

from legacy_trajectory import isolated_probe, model_buffer_hashes, rng_hash, tensor_hash


class SharedTrajectoryObserver:
    """One batch; callback owns dataset coordinates and frozen-head preprocessing."""

    def __init__(self, vae, heads: dict, classify_decoded, horizon: int, num_classes: int = 15):
        if horizon not in (10, 30) or not heads or 'vae' in heads:
            raise ValueError('Expected horizon 10/30 and nonempty classifier heads')
        self.vae, self.heads = vae, dict(heads)
        if type(num_classes) is not int or num_classes not in (15, 40):
            raise ValueError('Expected 15 Scan or 40 ModelNet classes')
        self.num_classes = num_classes
        self.classify_decoded, self.horizon = classify_decoded, horizon
        self.steps = [3, 5, 8] if horizon == 10 else [8, 15, 23]
        self.models = dict(vae=vae, **heads)
        self.names = ['clean_%02d' % step for step in self.steps] + ['terminal']
        self.saved = {name: [] for name in heads}
        self.probes, self.initial_hashes, self.final_hashes = [], {}, {}
        self.times, self.active, self.clock = {}, None, time.perf_counter()
        self.count, self.complete = None, False

    def mark(self, name) -> None:
        # Wall time includes asynchronous dispatch; the runner measures GPU wall time.
        now = time.perf_counter()
        if self.active is not None:
            self.times[self.active] = self.times.get(self.active, 0.) + now - self.clock
        self.active, self.clock = name, now

    def encode_rng_start(self):
        return None

    def initial(self, **tensors) -> None:
        if self.initial_hashes or not tensors:
            raise ValueError('Expected exactly one nonempty initial state')
        self.initial_hashes = {name: tensor_hash(value) for name, value in tensors.items()}

    def _logits(self, logits: dict) -> dict:
        import torch
        if not isinstance(logits, dict) or set(logits) != set(self.heads):
            raise ValueError('Trajectory classifier heads do not match')
        saved = {}
        for name, values in logits.items():
            if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != self.num_classes:
                raise ValueError('Expected finite B x %d trajectory logits' % self.num_classes)
            count = values.shape[0]
            if count == 0 or (self.count is not None and count != self.count) or not torch.isfinite(values).all():
                raise ValueError('Expected consistent finite B x %d trajectory logits' % self.num_classes)
            self.count = count
            saved[name] = values.detach().cpu().numpy().copy()
        return saved

    def clean(self, latent, style, loop_index, timestep) -> None:
        count = loop_index + 1
        if count not in self.steps:
            return
        if self.complete or len(self.probes) >= 3 or count != self.steps[len(self.probes)]:
            raise ValueError('Missing, duplicate or reordered clean checkpoint')
        self.mark('trajectory_probe')
        before = rng_hash()
        buffers_before = model_buffer_hashes(self.models)
        latent_hash, style_hash = tensor_hash(latent), tensor_hash(style)
        with isolated_probe(list(self.models.values())):
            decoded = self.vae.decoder(None, beta=None,
                                       context=latent.detach().clone().squeeze(3).squeeze(2),
                                       style=style.detach().clone().squeeze(3).squeeze(2))
            decoded_hash = tensor_hash(decoded)
            logits = self._logits(self.classify_decoded(decoded))
            if tensor_hash(decoded) != decoded_hash:
                raise RuntimeError('Classifier callback mutated decoded geometry')
        after = rng_hash()
        buffers_after = model_buffer_hashes(self.models)
        inputs_unchanged = latent_hash == tensor_hash(latent) and style_hash == tensor_hash(style)
        if before != after or buffers_before != buffers_after or not inputs_unchanged:
            raise RuntimeError('Trajectory probe changed RNG, model buffers or live latent')
        for name, values in logits.items():
            self.saved[name].append(values)
        self.probes.append(dict(name=self.names[len(self.probes)], loop_count=count,
                                timestep=int(timestep), clean_latent_sha256=latent_hash,
                                style_sha256=style_hash, decoded_geometry_sha256=decoded_hash,
                                rng_before_sha256=before, rng_after_sha256=after,
                                rng_unchanged=True, buffers_unchanged=True,
                                live_inputs_unchanged=True,
                                buffers_before=buffers_before, buffers_after=buffers_after))

    def final_state(self, latent, style, timestep) -> None:
        if self.final_hashes:
            raise ValueError('Duplicate final state')
        self.final_hashes = dict(latent=tensor_hash(latent), style=tensor_hash(style), timestep=int(timestep))

    def terminal(self, logits: dict) -> None:
        if self.complete or len(self.probes) != 3 or not self.initial_hashes or not self.final_hashes:
            raise ValueError('Incomplete or duplicate trajectory terminal')
        saved = self._logits(logits)
        for name, values in saved.items():
            self.saved[name].append(values)
        self.complete = True
        self.mark(None)

    def arrays(self, backbone: str) -> dict:
        if not self.complete:
            raise ValueError('Trajectory is incomplete')
        return dict(trajectory_logits=np.stack(self.saved[backbone], axis=1),
                    candidate_names=np.asarray(self.names),
                    candidate_loop_counts=np.asarray(self.steps + [self.horizon], dtype=np.int64))

    def diagnostics(self) -> dict:
        return dict(complete=self.complete, horizon=self.horizon, candidate_names=self.names,
                    candidate_loop_counts=self.steps + [self.horizon], probes=self.probes,
                    initial_hashes=self.initial_hashes, final_hashes=self.final_hashes,
                    probe_rng_unchanged=all(p['rng_unchanged'] for p in self.probes),
                    probe_buffers_unchanged=all(p['buffers_unchanged'] for p in self.probes),
                    times=dict(self.times), timing='unsynchronized component dispatch wall seconds')
