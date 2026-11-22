"""Registered B1 geometry alternatives for the Point2Vec Background study."""
import torch

EXPERIMENT_ARMS = ('bg_weighted', 'bg_scalar', 'bg_high', 'bg_gate',
                   'bg_view', 'bg_random_view', 'bg_weighted_pose', 'bg_mid', 'bg_start_only')


def anchor_weights(x, mask, k):
    """Relative neighbor spacing, computed on distinct valid reference coordinates."""
    valid = mask.reshape(-1).bool()
    points, inverse = torch.unique(x[0, valid], dim=0, return_inverse=True)
    weights = torch.zeros_like(x[..., 0])
    if len(points) <= k:
        weights[0, valid] = 1
        return weights
    distance = torch.cdist(points, points, compute_mode='donot_use_mm_for_euclid_dist')
    distance.fill_diagonal_(float('inf'))
    nearest, neighbors = distance.topk(k, largest=False)
    spacing = nearest.median(1).values
    typical = spacing[neighbors].median(1).values
    ratio = (typical / spacing.clamp_min(torch.finfo(spacing.dtype).tiny)).clamp_max(1.)
    weights[0, valid] = ratio[inverse]
    return weights


class BackgroundGeometry:
    def __init__(self, x, mask, arm, config):
        self.arm = arm
        self.threshold = config['filter_below']
        self.weights = None if arm in ('bg_high', 'bg_mid', 'bg_start_only') else anchor_weights(x, mask, config['k'])
        self.weak_fraction = (float((self.weights[mask.reshape_as(self.weights).bool()] < self.threshold).float().mean())
                              if self.weights is not None else None)
        self.high = arm == 'bg_high' or arm == 'bg_gate' and self.weak_fraction > config['gate_fraction']
        self.schedule = config['high_grid'] if self.high else None
        if arm in ('bg_mid', 'bg_start_only'):
            self.schedule = config['mid_grid' if arm == 'bg_mid' else 'start_only_grid']
        self.keep = None

    def pairs(self, index, forward, reverse, keep_forward, keep_reverse):
        fw = self.weights.gather(1, forward)
        if self.arm in ('bg_view', 'bg_random_view'):
            if index == 10:
                self.keep = fw[0] >= self.threshold
            return forward, reverse, keep_forward, keep_reverse
        weighted_fw = fw * keep_forward
        weighted_rv = self.weights * keep_reverse
        if self.arm == 'bg_scalar':
            scale = (weighted_fw.sum() + weighted_rv.sum()) / (keep_forward.sum() + keep_reverse.sum())
            weighted_fw, weighted_rv = scale * keep_forward, scale * keep_reverse
        return forward, reverse, weighted_fw, weighted_rv

    def view(self, cloud, seed):
        if self.keep is None:
            return cloud
        keep = self.keep.nonzero().flatten()
        if len(keep) < 64 or len(keep) == cloud.shape[1]:
            return cloud
        if self.arm == 'bg_random_view':
            rng = torch.Generator(device='cpu').manual_seed(seed)
            keep = torch.randperm(cloud.shape[1], generator=rng)[:len(keep)].sort().values.to(cloud.device)
        return cloud.index_select(1, keep)
