"""Fixed physical-X graph guidance reusing probability-mean clean decodes."""
import math
import statistics
import time

import torch

from graph_spectral import SmoothSpectralTarget
from legacy_trajectory import model_buffer_hashes, rng_hash, tensor_hash
from semantic_trajectory import gradient_probe
from shared_trajectory import SharedTrajectoryObserver


def update_norm_diagnostics(scd, spectral, local, style, gamma, eta, scd_weight, spectral_weight) -> dict:
    """Optional two-VJP probe of the actual per-object local/style update."""
    gradients_c = torch.autograd.grad(scd, (local, style), retain_graph=True)
    gradients_g = torch.autograd.grad(spectral, (local, style), retain_graph=True)
    dc, dg = [torch.cat([rate * grad.detach().flatten(1).double()
                        for rate, grad in zip((gamma, eta), gradients)], dim=1)
              for gradients in (gradients_c, gradients_g)]
    if not torch.isfinite(dc).all() or not torch.isfinite(dg).all():
        raise FloatingPointError('Nonfinite decoded component update')
    cn, gn = dc.norm(dim=1), dg.norm(dim=1)
    active = (cn > 1e-12) & (gn > 1e-12)
    mixed = (scd_weight * dc + spectral_weight * dg).norm(dim=1)
    denominator = scd_weight * cn + spectral_weight * gn
    if not all(torch.isfinite(value).all() for value in (cn, gn, mixed, denominator)):
        raise FloatingPointError('Nonfinite decoded update norm')
    return dict(scd_update_norm=cn.cpu().tolist(), spectral_update_norm=gn.cpu().tolist(),
        scd_over_spectral_update_ratio=[float(c/g) if ok else None for c,g,ok in zip(cn,gn,active)],
        cosine=[float(dot/(c*g)) if ok else None for dot,c,g,ok in zip((dc*dg).sum(1),cn,gn,active)],
        mixed_update_norm=mixed.cpu().tolist(),
        cancellation_fraction=[float(1-m/d) if d > 1e-12 else None for m,d in zip(mixed,denominator)],
        status=['active' if ok else 'inactive' for ok in active], norm_floor=1e-12,
        local_step=gamma, style_step=eta, scd_weight=scd_weight, spectral_weight=spectral_weight,
        extra_component_vjp_calls=2)


def freeze_update_alpha(records: list) -> float:
    """Freeze the median ratio; every predeclared calibration row must be active."""
    ratios = [ratio for record in records for ratio in record['scd_over_spectral_update_ratio']]
    if not ratios or any(r is None or not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('Calibration requires finite positive active update ratios for every row')
    return float(statistics.median(ratios))


class DecodedSpectralObserver(SharedTrajectoryObserver):
    """Compare decoded clean XYZ with the exact prepared LION input.

    Build ``target`` once with ``build_smooth_spectral_target(prepared_x, ...)``.
    Its frozen graph and vertex order stay fixed throughout the trajectory.
    Classification uses a detached view; only physical geometry supplies loss.
    The sampler passes original encoded z0 as decoder style. Conditioning-style
    gradients reach the sampler through the live predicted-clean local latent.
    Opt-in frozen residual guidance preserves decoded values but uses only the
    identity local-XYZ derivative, bypassing decoder backward.
    """

    def __init__(self, vae, heads: dict, classify_decoded, horizon: int, num_classes: int = 40,
                 *, target: SmoothSpectralTarget, checkpoint_steps=None,
                 record_gradients: bool = False, freeze_decoder_residual: bool = False,
                 record_updates: bool = False):
        if type(freeze_decoder_residual) is not bool:
            raise ValueError('freeze_decoder_residual must be a bool')
        super().__init__(vae, heads, classify_decoded, horizon, num_classes,
                         checkpoint_steps=checkpoint_steps)
        if not isinstance(target, SmoothSpectralTarget):
            raise ValueError('Decoded spectral guidance requires a SmoothSpectralTarget')
        reference = target.reference_xyz
        if (not isinstance(reference, torch.Tensor) or reference.ndim != 3
                or reference.shape[0] < 1 or reference.shape[1:] != (2048, 3)
                or reference.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(reference).all()):
            raise ValueError('Expected finite physical reference B x 2048 x 3 XYZ')
        if reference.requires_grad:
            raise ValueError('Physical spectral reference must be detached')
        if len(target.filters) != len(reference):
            raise ValueError('Expected one frozen spectral filter per reference object')
        for spectral_filter in target.filters:
            if (not isinstance(spectral_filter, torch.Tensor) or spectral_filter.shape != (2048, 2048)
                    or spectral_filter.device != reference.device or spectral_filter.dtype != reference.dtype
                    or spectral_filter.requires_grad or not torch.isfinite(spectral_filter).all()):
                raise ValueError('Expected finite detached reference-domain spectral filters')
        self.target, self.record_gradients = target, record_gradients
        self.freeze_decoder_residual = freeze_decoder_residual
        self.loss_gradient_mode = 'frozen_residual_surrogate' if freeze_decoder_residual else 'full_decoder_gradient'
        self.spectral_records = []
        self.record_updates = record_updates

    def record_update_norms(self, scd, spectral, local, style, gamma, eta, scd_weight, spectral_weight):
        if local.is_cuda:
            torch.cuda.synchronize(local.device)
        started = time.perf_counter()
        record = update_norm_diagnostics(scd, spectral, local, style, gamma, eta, scd_weight, spectral_weight)
        if local.is_cuda:
            torch.cuda.synchronize(local.device)
        record['extra_component_vjp_seconds'] = time.perf_counter() - started
        self.spectral_records[-1]['update_norms'] = record

    def initial(self, **tensors) -> None:
        prepared = tensors.get('input_points')
        reference = self.target.reference_xyz
        if (not isinstance(prepared, torch.Tensor) or prepared.shape != reference.shape
                or prepared.device != reference.device or prepared.dtype != reference.dtype
                or not torch.equal(prepared, reference)):
            raise ValueError('Spectral reference must equal the exact prepared LION input_points')
        super().initial(**tensors)

    def clean(self, latent, style, loop_index, timestep):
        count = loop_index + 1
        if count not in self.steps:
            return None
        if self.complete or len(self.probes) >= len(self.steps) or count != self.steps[len(self.probes)]:
            raise ValueError('Missing, duplicate or reordered clean checkpoint')
        self.mark('trajectory_decoded_spectral_probe')
        before, buffers_before = rng_hash(), model_buffer_hashes(self.models)
        latent_hash, style_hash = tensor_hash(latent), tensor_hash(style)
        with gradient_probe(self.models.values()):
            decoder_started = time.perf_counter()
            decoded = self.vae.decoder(None, beta=None,
                                       context=latent.clone().squeeze(3).squeeze(2),
                                       style=style.clone().squeeze(3).squeeze(2))
            decoded_hash = tensor_hash(decoded)
            decoder_seconds = time.perf_counter() - decoder_started
            prediction = decoded
            if self.freeze_decoder_residual:
                latent_xyz = latent.reshape(latent.shape[0], 2048, 4)[:, :, :3]
                # Equivalent frozen residual form retains bit-exact decoded values.
                prediction = decoded.detach() + (latent_xyz - latent_xyz.detach())
            loss_started = time.perf_counter()
            loss = self.target.loss(prediction)
            if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError('Decoded spectral loss must be finite and connected to live latents')
            loss_seconds = time.perf_counter() - loss_started
            # A separate detached copy keeps head preprocessing out of autograd.
            with torch.no_grad():
                classifier_points = decoded.detach().clone()
                saved = self._logits(self.classify_decoded(classifier_points))
                if tensor_hash(classifier_points) != decoded_hash:
                    raise RuntimeError('Classifier callback mutated decoded geometry')
            record = dict(loop_count=count, raw_loss=float(loss.detach()),
                          decoder_and_hash_seconds=decoder_seconds, spectral_loss_seconds=loss_seconds,
                          decoded_geometry_reused=True, classifier_gradients=False,
                          loss_gradient_mode=self.loss_gradient_mode,
                          decoder_jacobian_in_loss=not self.freeze_decoder_residual)
            if self.record_gradients:
                live = [value for value in (decoded, latent, style) if value.requires_grad]
                gradients = torch.autograd.grad(loss, live, retain_graph=True, allow_unused=True)
                by_id = {id(value): grad for value, grad in zip(live, gradients)}
                for name, value in (('decoded', decoded), ('clean_latent', latent), ('decoder_style', style)):
                    gradient = by_id.get(id(value))
                    record[name + '_connected'] = gradient is not None
                    record[name + '_gradient_norm'] = float(gradient.detach().norm()) if gradient is not None else None
        after, buffers_after = rng_hash(), model_buffer_hashes(self.models)
        inputs_unchanged = latent_hash == tensor_hash(latent) and style_hash == tensor_hash(style)
        if before != after or buffers_before != buffers_after or not inputs_unchanged:
            raise RuntimeError('Decoded spectral probe changed RNG, model buffers or live latent')
        for name, values in saved.items():
            self.saved[name].append(values)
        self.probes.append(dict(name=self.names[len(self.probes)], loop_count=count,
                                timestep=int(timestep), clean_latent_sha256=latent_hash,
                                style_sha256=style_hash, decoded_geometry_sha256=decoded_hash,
                                rng_before_sha256=before, rng_after_sha256=after,
                                rng_unchanged=True, buffers_unchanged=True, live_inputs_unchanged=True,
                                buffers_before=buffers_before, buffers_after=buffers_after))
        self.spectral_records.append(record)
        return loss

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result['decoded_spectral'] = dict(
            reference_xyz_sha256=tensor_hash(self.target.reference_xyz),
            reference_domain='exact prepared B x 2048 x 3 LION input; fixed vertex order',
            loss_reduction='sum over objects of <decoded-X, F(decoded-X)> / (3*2048)',
            decoder_style='original encoded z0; conditioning-style gradient flows through clean local',
            loss_gradient_mode=self.loss_gradient_mode, decoder_jacobian_in_loss=not self.freeze_decoder_residual,
            checkpoint_loop_counts=self.steps, graph=self.target.diagnostics,
            records=self.spectral_records)
        return result
