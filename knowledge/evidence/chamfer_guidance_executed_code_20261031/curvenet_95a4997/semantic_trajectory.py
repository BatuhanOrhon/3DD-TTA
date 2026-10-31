"""Frozen-head semantic guidance reusing the existing probability-mean probes."""
from contextlib import contextmanager
import time

import torch

from legacy_trajectory import (_restore_rng, _rng_state, model_buffer_hashes,
                               rng_hash, tensor_hash)
from shared_trajectory import SharedTrajectoryObserver


@contextmanager
def gradient_probe(models):
    """Preserve probe RNG/buffers while retaining gradients to decoder inputs."""
    models = list(models)
    if any(module.training for model in models for module in model.modules()):
        raise ValueError('Semantic probes require all-eval models')
    if any(parameter.requires_grad for model in models for parameter in model.parameters()):
        raise ValueError('Semantic probes require frozen model parameters')
    state = _rng_state()
    buffers = [(value, value.detach().clone()) for model in models for value in model.buffers()]
    try:
        with torch.enable_grad():
            yield
    finally:
        with torch.no_grad():
            for value, saved in buffers:
                if not torch.equal(value, saved):
                    value.copy_(saved)
        _restore_rng(state)


def semantic_loss(baseline: torch.Tensor, reference: torch.Tensor, kind: str) -> torch.Tensor:
    """Symmetric KL or per-object marginal H against a detached corrupted-X view."""
    if baseline.shape != reference.shape or baseline.ndim != 2:
        raise ValueError('Semantic logits must have matching B x classes shapes')
    if reference.requires_grad:
        raise ValueError('Corrupted-X semantic reference must be detached')
    if not torch.isfinite(baseline).all() or not torch.isfinite(reference).all():
        raise ValueError('Semantic logits must be finite')
    log_p, log_q = baseline.log_softmax(-1), reference.log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()
    if kind == 'kl':
        # The fixed-X branch is detached; only decoded inputs receive gradients.
        return .5 * ((p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1)).sum()
    if kind == 'marginal_entropy':
        marginal = .5 * (p + q)
        return -(marginal * marginal.clamp_min(torch.finfo(marginal.dtype).tiny).log()).sum()
    raise ValueError('Expected kl or marginal_entropy semantic loss')


class SemanticTrajectoryObserver(SharedTrajectoryObserver):
    """Only the last two clean probes build a guidance autograd graph.

    The callback produces one baseline forward per selected head. The detached
    corrupted-X reference is computed once per batch. Only baseline logits vote.
    The returned scalar is consumed by the sampler's existing joint backward.
    """

    def __init__(self, vae, heads, classify_decoded, horizon, num_classes=15, *,
                 classify_semantic_decoded, kind, reference_logits, reference_record,
                 guidance_head='pointmae'):
        super().__init__(vae, heads, classify_decoded, horizon, num_classes)
        if kind not in ('kl', 'marginal_entropy') or guidance_head not in ('pointmae', 'curvenet') or guidance_head not in heads:
            raise ValueError('Semantic guidance requires KL/marginal entropy and selected Point-MAE head')
        if guidance_head == 'curvenet' and (kind != 'marginal_entropy' or num_classes != 15):
            raise ValueError('CurveNet guidance is restricted to Scan marginal entropy')
        self.classify_semantic_decoded = classify_semantic_decoded
        self.kind, self.guidance_head = kind, guidance_head
        if (not isinstance(reference_logits, torch.Tensor) or reference_logits.ndim != 2
                or reference_logits.shape[1] != num_classes or reference_logits.shape[0] < 1
                or reference_logits.requires_grad or not torch.isfinite(reference_logits).all()):
            raise ValueError('Expected finite detached B x classes corrupted-X reference logits')
        self.reference_logits = reference_logits.detach().clone()
        self.reference_record = dict(reference_record, logits_sha256=tensor_hash(self.reference_logits))
        self.semantic_records = []

    def clean(self, latent, style, loop_index, timestep):
        count = loop_index + 1
        if count not in self.steps[1:]:
            return super().clean(latent, style, loop_index, timestep)
        if self.complete or len(self.probes) >= 3 or count != self.steps[len(self.probes)]:
            raise ValueError('Missing, duplicate or reordered clean checkpoint')
        self.mark('trajectory_semantic_probe')
        started = time.perf_counter()
        before, buffers_before = rng_hash(), model_buffer_hashes(self.models)
        latent_hash, style_hash = tensor_hash(latent), tensor_hash(style)
        with gradient_probe(self.models.values()):
            decoded = self.vae.decoder(None, beta=None,
                                       context=latent.clone().squeeze(3).squeeze(2),
                                       style=style.clone().squeeze(3).squeeze(2))
            decoded_hash = tensor_hash(decoded)
            baseline = self.classify_semantic_decoded(decoded)
            saved = self._logits(baseline)
            loss = semantic_loss(baseline[self.guidance_head], self.reference_logits, self.kind)
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError('Semantic guidance must be finite and connected to live latents')
            if tensor_hash(decoded) != decoded_hash:
                raise RuntimeError('Classifier callback mutated decoded geometry')
        after, buffers_after = rng_hash(), model_buffer_hashes(self.models)
        inputs_unchanged = latent_hash == tensor_hash(latent) and style_hash == tensor_hash(style)
        if before != after or buffers_before != buffers_after or not inputs_unchanged:
            raise RuntimeError('Semantic probe changed RNG, model buffers or live latent')
        for name, values in saved.items():
            self.saved[name].append(values)
        self.probes.append(dict(name=self.names[len(self.probes)], loop_count=count,
                                timestep=int(timestep), clean_latent_sha256=latent_hash,
                                style_sha256=style_hash, decoded_geometry_sha256=decoded_hash,
                                rng_before_sha256=before, rng_after_sha256=after,
                                rng_unchanged=True, buffers_unchanged=True, live_inputs_unchanged=True,
                                buffers_before=buffers_before, buffers_after=buffers_after))
        self.semantic_records.append(dict(loop_count=count, kind=self.kind, head=self.guidance_head,
                                          raw_loss=float(loss.detach()),
                                          baseline_logits_sha256=tensor_hash(baseline[self.guidance_head]),
                                          reference_logits_sha256=self.reference_record['logits_sha256'],
                                          baseline_logits_reused=True, decoded_geometry_reused=True,
                                          probe_dispatch_seconds=time.perf_counter() - started))
        return loss

    def diagnostics(self):
        result = super().diagnostics()
        result.update(semantic_guidance=dict(kind=self.kind, head=self.guidance_head,
                                            loss_reduction='sum over objects; per-object class/view loss',
                                            checkpoint_loop_counts=self.steps[1:], records=self.semantic_records,
                                            reference=self.reference_record,
                                            reference_votes=False, head_dependent_geometry=True),
                      operation_counts=dict(probe_decode_forward=len(self.probes),
                                            probe_classifier_forward=len(self.probes) * len(self.heads),
                                            semantic_reference_forward=self.reference_record['forward_calls']))
        return result
