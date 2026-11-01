"""One frozen LION-frame posterior roundtrip and separate two-pass readouts."""
import numpy as np
import torch

from legacy_trajectory import _rng_state, isolated_probe
from shape_latent_refresh import validate_input
from shared_trajectory import SharedTrajectoryObserver


@torch.no_grad()
def reencode(decoded, base, initial, vae):
    """Refresh z and conditional h, preserving explicit posterior/noise draws."""
    validate_input(decoded, vae)
    if len(decoded) != len(base['shape_latent']):
        raise ValueError('Bridge batch differs from the original preparation')
    with isolated_probe([vae]):
        qz = vae.encode_global(decoded)
        z = qz.sample_given_rho(initial['eps_z'])
        qh = vae.encode_local(decoded, vae.global2style(z))
        h = qh.sample_given_rho(initial['eps_h'])
        cycle = vae.decoder(None, beta=None, context=h, style=z)
        validate_input(cycle, vae)
        proof = dict(initial, bridge_points=decoded, cycle_points=cycle,
                     new_z_mu=qz.mu, new_z_sigma=qz.sigma,
                     new_h_mu=qh.mu, new_h_sigma=qh.sigma,
                     selected_z=z, selected_h=h, diffusion_noise=base['noise'])
        if any(not torch.isfinite(value).all() for value in proof.values()):
            raise ValueError('Nonfinite reencoded posterior')
        proof = {key: value.detach().clone() for key, value in proof.items()}
    selected = dict(shape_latent=z.detach()[:, :, None, None],
                    local_latent=h.detach()[:, :, None, None],
                    noise=base['noise'].detach().clone(), rng_after_prepare=_rng_state())
    return selected, proof


def readouts(first, second):
    """Temperature1 equal probability averages, four candidates per pass."""
    first, second = np.asarray(first), np.asarray(second)
    if (first.ndim != 3 or first.shape != second.shape or first.shape[1] != 4
            or first.shape[0] == 0 or first.shape[2] < 2
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError('Need paired finite Bx4xC logits')
    result = {}
    for name, logits in [('first', first), ('second', second)]:
        exponent = np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True))
        result[name + '_probabilities'] = (exponent / exponent.sum(-1, keepdims=True)).mean(1)
    result['combined_probabilities'] = .5 * (result['first_probabilities'] + result['second_probabilities'])
    for name in ('first', 'second', 'combined'):
        result[name + '_prediction'] = result[name + '_probabilities'].argmax(-1)
    return result


class TwoPassObserver:
    """Keep the baseline trace separate while the common worker reads pass2."""
    def __init__(self, *, vae, heads, classify_decoded, horizon, num_classes,
                 checkpoint_steps, **unused):
        self.first = SharedTrajectoryObserver(vae, heads, classify_decoded, horizon,
            num_classes, checkpoint_steps=checkpoint_steps)
        self.second = SharedTrajectoryObserver(vae, heads, classify_decoded, 3,
            num_classes, checkpoint_steps=[1, 2, 3])
        self.names = self.second.names
        self.cycle_logits = {}

    def terminal(self, logits):
        self.second.terminal(logits)

    def cycle(self, logits):
        self.cycle_logits = self.first._logits(logits)

    def arrays(self, backbone):
        first = self.first.arrays(backbone)
        second = self.second.arrays(backbone)
        return dict(second, first_trajectory_logits=first['trajectory_logits'],
                    first_candidate_names=first['candidate_names'],
                    first_candidate_loop_counts=first['candidate_loop_counts'],
                    cycle_logits=self.cycle_logits[backbone],
                    **readouts(first['trajectory_logits'], second['trajectory_logits']))

    def diagnostics(self):
        return dict(first=self.first.diagnostics(), second=self.second.diagnostics(),
                    second_reverse_steps=3, global_diffusion=False,
                    combined_weighting='equal8 candidates, equal4 per pass')
