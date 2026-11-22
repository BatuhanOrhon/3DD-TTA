"""Opt-in, gradient-free pose updates for the CloudFixer XYZ pilot."""
from dataclasses import dataclass
import math

import torch

from cloudfixer_background import BackgroundGeometry, EXPERIMENT_ARMS
from cloudfixer_diffusion import sample, trimmed_neighbors, trimmed_scd


KABSCH_STEPS = {
    'identity': {},
    'kabsch_single': {80: 1},
    'kabsch_fixed': {80: 3},
    'kabsch_progressive': {80: 1, 50: 1, 20: 1},
}
SPECTRAL_STEPS = {
    'kabsch_spectral_single': KABSCH_STEPS['kabsch_single'],
    'kabsch_spectral_fixed': KABSCH_STEPS['kabsch_fixed'],
    'kabsch_spectral_progressive': KABSCH_STEPS['kabsch_progressive'],
}
SCD_SEARCH_STEPS = (15., 10.)
GRAPH_K = 12
GRAPH_STEPS = 3
SEARCH_ARMS = ('output', 'feedback', 'random', 'view_mean')
ARMS = (*KABSCH_STEPS, *SEARCH_ARMS)


def axis_rotation(axis, angle, like):
    rotation = torch.eye(3, device=like.device, dtype=like.dtype)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    rotation[a, a] = rotation[b, b] = math.cos(angle)
    rotation[a, b], rotation[b, a] = -math.sin(angle), math.sin(angle)
    return rotation


def rotation_angle(rotation):
    skew = (rotation - rotation.T) * .5
    vector = torch.stack((skew[2, 1], skew[0, 2], skew[1, 0]))
    return math.atan2(float(torch.linalg.vector_norm(vector)), float((rotation.trace() - 1) * .5))


def clip_rotation(rotation, angle, limit):
    if angle <= limit:
        return rotation
    skew = rotation - rotation.T
    axis = torch.stack((skew[2, 1], skew[0, 2], skew[1, 0]))
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    if angle > math.pi - 1e-4:
        symmetric = (rotation + rotation.T) * .5 + identity
        direction = symmetric[symmetric.diagonal().argmax()]
        axis = torch.where(direction @ axis < 0, -direction, direction)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return identity + math.sin(limit) * cross + (1 - math.cos(limit)) * (cross @ cross)


def deduplicated_mask(mask, indices):
    mask = mask.reshape(1, -1).bool()
    if indices is None:
        return mask
    valid = mask[0].nonzero().flatten()
    ids = indices.reshape(-1)[valid]
    order = ids.argsort(stable=True)
    sorted_ids = ids[order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    result = torch.zeros_like(mask)
    result[0, valid[order[first]]] = True
    return result


def has_plane(points):
    points = points - points.mean(0)
    squared = points.square().sum(1)
    axis = points[squared.argmax()].expand_as(points)
    area = torch.linalg.cross(points, axis).square().sum(1).amax()
    return bool(area > 1e-8 * squared.amax().square() and squared.amax() > 0)


def unmeasured(_, function, *args, **kwargs):
    return function(*args, **kwargs)


def mutual_graph(points):
    distances = torch.cdist(points, points).square()
    distances.fill_diagonal_(float('inf'))
    neighbors = distances.topk(min(GRAPH_K, len(points) - 1), largest=False, sorted=True).indices
    offsets = points[:, None] - points[neighbors]
    edge_squared = offsets.square().sum(-1)
    bandwidth = edge_squared[:, -1].sqrt()
    vertex = torch.arange(len(points), device=points.device)
    mutual = (neighbors[neighbors] == vertex[:, None, None]).any(-1)
    scale = (2 * bandwidth[:, None] * bandwidth[neighbors]).clamp_min(torch.finfo(points.dtype).eps)
    return neighbors, (-edge_squared / scale).exp() * mutual


def spectral_sweeps(points, neighbors, weights):
    degree = weights.sum(1)
    step_weights = weights / (2 * degree.max().clamp_min(torch.finfo(points.dtype).eps))
    for _ in range(GRAPH_STEPS):
        offsets = points[neighbors] - points[:, None]
        points = points + (step_weights[..., None] * offsets).sum(1)
    return points


@torch.no_grad()
def graph_lowpass(x, mask=None, *, measure=unmeasured):
    """Apply a bounded Laplacian polynomial to valid physical XYZ coordinates."""
    points = x[0] if mask is None else x[0, mask[0]]
    if len(points) < 2:
        return x
    neighbors, weights = measure('spectral_graph', mutual_graph, points)
    filtered = measure('spectral_filter', spectral_sweeps, points, neighbors, weights)
    if mask is None:
        return filtered.unsqueeze(0)
    result = x.clone()
    result[0, mask[0]] = filtered
    return result


class KabschPose:
    def __init__(self, x, mask, indices, schedule, *, max_degrees=25., measure=unmeasured, filter_points=None):
        self.x = x
        self.mask = deduplicated_mask(mask, indices) if schedule else None
        self.schedule = schedule
        self.pose = torch.eye(3, device=x.device, dtype=x.dtype)
        self.anchors = x
        self.filter_points = filter_points
        self.match_x = filter_points(x, self.mask) if filter_points is not None else x
        self.match_anchors = self.match_x
        self.remaining = math.radians(max_degrees)
        self.travel = 0.
        self.measure = measure
        self.updates = []

    def pairs(self, target):
        fw, rv, keep_fw, keep_rv = trimmed_neighbors(target, self.match_anchors, self.mask, .95)
        q_ids = keep_fw[0].nonzero().flatten()
        x_ids = keep_rv[0].nonzero().flatten()
        pairs_x = torch.cat((fw[0, q_ids], x_ids))
        pairs_q = torch.cat((q_ids, rv[0, x_ids]))
        support = (pairs_x.unique().numel(), pairs_q.unique().numel())
        return self.match_anchors[0, pairs_x], target[0, pairs_q], support

    def reference(self, index, target):
        iterations = self.schedule.get(index, 0)
        if iterations and self.filter_points is not None:
            target = self.filter_points(target, None)
        for _ in range(iterations):
            row = dict(index=index, anchors=0, targets=0, pairs=0, matching=0, svd=0,
                       reason='support', proposed_degrees=0., applied_degrees=0.)
            if int(self.mask.sum()) >= 4:
                src, dst, support = self.measure('matching', self.pairs, target)
                row.update(anchors=support[0], targets=support[1], pairs=len(src), matching=1)
                if min(support) >= 3:
                    self.fit(src, dst, row)
            row.update(total_degrees=math.degrees(rotation_angle(self.pose)),
                       travel_degrees=math.degrees(self.travel), pose=self.pose.tolist())
            self.updates.append(row)
        return self.anchors

    def fit(self, source, target, row):
        row['reason'] = 'rank'
        if not has_plane(source) or not has_plane(target):
            return
        covariance = source.T @ target / len(source)
        u, singular, vh = self.measure('svd', torch.linalg.svd, covariance)
        row['svd'] = 1
        if singular[1] <= singular[0] * 1e-4:
            return
        sign = torch.linalg.det(u @ vh).sign()
        oriented = torch.cat((u[:, :2], u[:, 2:] * sign), 1) @ vh
        angle = rotation_angle(oriented)
        row['proposed_degrees'] = math.degrees(angle)
        delta = min(angle, self.remaining)
        if delta <= 1e-7:
            row['reason'] = 'budget' if self.remaining <= 1e-7 else 'identity'
            return
        self.pose = self.pose @ clip_rotation(oriented, angle, delta)
        self.anchors = self.x @ self.pose
        self.match_anchors = self.anchors if self.filter_points is None else self.match_x @ self.pose
        self.remaining -= delta
        self.travel += delta
        row.update(reason='accepted', applied_degrees=math.degrees(delta))


class SCDSearchPose:
    def __init__(self, x, mask, indices, *, max_degrees=25., measure=unmeasured):
        self.x = self.anchors = x
        self.mask = deduplicated_mask(mask, indices)
        self.pose = torch.eye(3, device=x.device, dtype=x.dtype)
        self.max_degrees = max_degrees
        self.measure = measure
        self.updates = []

    def reference(self, index, target):
        if index == 80 and int(self.mask.sum()) >= 4:
            self.measure('scd_search', self.search, target)
        return self.anchors

    def search(self, target):
        score = float(trimmed_scd(target, self.x, self.mask, .95)[0])
        support = int(self.mask.sum())
        travel = 0.
        for round_index, step in enumerate(SCD_SEARCH_STEPS):
            angle = min(step, self.max_degrees - travel)
            if angle <= 0:
                break
            rotations = torch.stack([axis_rotation(axis, math.radians(sign * angle), self.x)
                                     for axis in range(3) for sign in (-1, 1)])
            poses = self.pose @ rotations
            scores = trimmed_scd(target.expand(6, -1, -1), self.x @ poses, self.mask.expand(6, -1), .95)
            best = int(scores.argmin())
            candidate = float(scores[best])
            before = score
            accepted = candidate < before
            if accepted:
                self.pose = poses[best]
                self.anchors = self.x @ self.pose
                score = candidate
                travel += angle
            self.updates.append(dict(index=80, matching=7 if round_index == 0 else 6, svd=0,
                                     anchors=support, applied_degrees=angle if accepted else 0.,
                                     travel_degrees=travel, total_degrees=math.degrees(rotation_angle(self.pose)),
                                     pose=self.pose.tolist(), score_before=before, score_after=score,
                                     reason='accepted' if accepted else 'identity'))


@dataclass
class RotationResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    clouds: torch.Tensor
    pose: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def rotation_policy(model, classifier, x, mask, point_indices, arm, *, theta=15., penalty=.1,
                    threshold=.015, max_degrees=25., measure=unmeasured, noise_rotation=None, background_config=None):
    """Run an arm with the baseline RNG contract and fixed, index-addressed FPS draws."""
    from scripts.run_cloudfixer_point2vec import POLICIES, deterministic_proximal

    if (arm not in ARMS and arm not in SPECTRAL_STEPS and arm != 'scd_search' and arm not in EXPERIMENT_ARMS) or x.shape[0] != 1:
        raise ValueError('Expected a registered B1 rotation arm')
    config = POLICIES['proximal_pm']
    devices = [x.device.index] if x.is_cuda else []
    with deterministic_proximal(), torch.random.fork_rng(devices=devices):
        noise = torch.randn_like(x)
        if noise_rotation is not None:
            noise = noise @ noise_rotation
        seeds = torch.randint(2**31, (501,), device='cpu').tolist()
        diagnostics = dict(classifier_calls=0)
        pose = torch.eye(3, device=x.device, dtype=x.dtype)

        def classify(index, cloud):
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seeds[index])
                if x.is_cuda:
                    torch.cuda.manual_seed(seeds[index])
                diagnostics['classifier_calls'] += 1
                return measure('classifier', classifier, cloud.contiguous())

        choice = 0
        selected_logits = None
        view_probabilities = None

        def select(probe):
            nonlocal selected_logits, pose, choice, view_probabilities
            angle = math.radians(theta)
            rotations = [pose] + [axis_rotation(axis, direction * angle, x)
                                  for axis in range(3) for direction in (-1, 1)]
            options = torch.stack([classify(20, probe if i == 0 else probe @ rotation)
                                   for i, rotation in enumerate(rotations)])
            probabilities = options.double().softmax(-1)
            scores = probabilities.amax(-1).flatten()
            scores[1:] -= penalty * angle**2
            best = int(scores.argmax())
            if arm == 'random':
                choice = seeds[0] % len(rotations)
            elif arm != 'view_mean' and float(scores[best] - scores[0]) > threshold:
                choice = best
            pose, selected_logits = rotations[choice], options[choice]
            view_probabilities = probabilities.mean(0)
            diagnostics.update(choice=choice, option_logits=options.squeeze(1).tolist(),
                               scores=scores.tolist(), angle_degrees=theta if choice else 0.)
            return pose if choice else None

        schedule = KABSCH_STEPS.get(arm, SPECTRAL_STEPS.get(arm))
        geometry = None
        sampler_options = {}
        if arm in EXPERIMENT_ARMS:
            geometry = BackgroundGeometry(x, mask, arm, background_config)
            diagnostics['weak_fraction'] = geometry.weak_fraction
            if geometry.schedule is not None:
                config = dict(config, start=geometry.schedule[0])
                sampler_options['schedule'] = geometry.schedule
            if arm not in ('bg_high', 'bg_gate', 'bg_mid', 'bg_start_only'):
                sampler_options['pair_transform'] = geometry.pairs
            if arm == 'bg_weighted_pose':
                schedule = KABSCH_STEPS['kabsch_progressive']
        filter_points = None
        if arm in SPECTRAL_STEPS:
            from functools import partial
            filter_points = partial(graph_lowpass, measure=measure)
        estimator = (KabschPose(x, mask, point_indices, schedule, max_degrees=max_degrees,
                             measure=measure, filter_points=filter_points)
                  if schedule is not None else None)
        if arm == 'scd_search':
            estimator = SCDSearchPose(x, mask, point_indices, max_degrees=max_degrees, measure=measure)
        clouds, metadata = sample(model, x, mask, noise, **config, **sampler_options, measure=measure,
                                  pose_reference=estimator.reference if estimator is not None else None,
                                  feedback_rotation=select if arm == 'feedback' else None)
        if arm in SEARCH_ARMS and arm != 'feedback':
            select(clouds[:, 0])
        if estimator is not None:
            pose = estimator.pose
            diagnostics['updates'] = estimator.updates
        logits = []
        for index, cloud in zip(metadata['candidate_indices'], clouds.unbind(1)):
            if index == 20 and selected_logits is not None:
                logits.append(selected_logits)
            else:
                if choice and arm in ('output', 'random'):
                    cloud = cloud @ pose
                if geometry is not None and geometry.keep is not None and index == 10:
                    cloud = geometry.view(cloud, seeds[index])
                    diagnostics['t10_view_points'] = cloud.shape[1]
                logits.append(classify(index, cloud))
        logits = torch.stack(logits, 1)
        probabilities = logits.double().softmax(-1)
        if arm == 'view_mean':
            probabilities[:, 0] = view_probabilities
        if geometry is not None:
            diagnostics['grid'] = metadata['indices']
        diagnostics['denoiser_calls'] = metadata['denoiser_calls']
        return RotationResult(logits, probabilities.mean(1), clouds, pose, diagnostics)
