"""Explicit fixed-basis spectral objectives; coefficient calibration stays external."""

SPECTRAL_LOSS_KINDS = ('legacy_l2', 'huber', 'power')


def spectral_guidance_loss(predicted, reference, eigenvalues, beta, M_max, kind='legacy_l2'):
    """Return a weighted scalar with the historical3N reduction and mode cutoff."""
    import torch
    import torch.nn.functional as F

    if kind not in SPECTRAL_LOSS_KINDS:
        raise ValueError('Unknown spectral loss kind: ' + str(kind))
    if predicted.ndim != 3 or predicted.shape != reference.shape or predicted.shape[1] < 1 or eigenvalues.shape != predicted.shape[:2]:
        raise ValueError('Expected matching BxNxC spectra and BxN eigenvalues')
    smooth_weights = torch.exp(-beta * eigenvalues)
    smooth_weights[:, M_max:] = 0.0
    smooth_weights = smooth_weights.unsqueeze(-1)
    if kind == 'legacy_l2':
        squared_diff = (predicted - reference)**2
        weighted_sq_diff = smooth_weights * squared_diff
        return weighted_sq_diff.sum() / (3 * predicted.shape[1])
    if kind == 'huber':
        difference = F.smooth_l1_loss(predicted, reference, beta=1.0, reduction='none')
        return (smooth_weights * difference).sum() / (3 * predicted.shape[1])
    energy_predicted = predicted.square().sum(-1)
    energy_reference = reference.square().sum(-1)
    return (smooth_weights.squeeze(-1) * (energy_predicted - energy_reference).square()).sum() / (3 * predicted.shape[1])
