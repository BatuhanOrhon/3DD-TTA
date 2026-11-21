"""Opt-in, gradient-free pose updates for the CloudFixer XYZ pilot."""
from dataclasses import dataclass
import math

import torch

from cloudfixer_diffusion import sample, trimmed_neighbors


KABSCH_STEPS = {
    'identity': {},
    'kabsch_single': {80: 1},
    'kabsch_fixed': {80: 3},
    'kabsch_progressive': {80: 1, 50: 1, 20: 1},
}
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


class KabschPose:
    def __init__(self, x, mask, indices, schedule, *, max_degrees=25., measure=unmeasured):
        self.x = x
        self.mask = deduplicated_mask(mask, indices) if schedule else None
        self.schedule = schedule
        self.pose = torch.eye(3, device=x.device, dtype=x.dtype)
        self.anchors = x
        self.remaining = math.radians(max_degrees)
        self.travel = 0.
        self.measure = measure
        self.updates = []

    def pairs(self, target):
        fw, rv, keep_fw, keep_rv = trimmed_neighbors(target, self.anchors, self.mask, .95)
        q_ids = keep_fw[0].nonzero().flatten()
        x_ids = keep_rv[0].nonzero().flatten()
        pairs_x = torch.cat((fw[0, q_ids], x_ids))
        pairs_q = torch.cat((q_ids, rv[0, x_ids]))
        support = (pairs_x.unique().numel(), pairs_q.unique().numel())
        return self.anchors[0, pairs_x], target[0, pairs_q], support

    def reference(self, index, target):
        for _ in range(self.schedule.get(index, 0)):
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
        self.remaining -= delta
        self.travel += delta
        row.update(reason='accepted', applied_degrees=math.degrees(delta))


@dataclass
class RotationResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    clouds: torch.Tensor
    pose: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def rotation_policy(model, classifier, x, mask, point_indices, arm, *, theta=15., penalty=.1,
                    threshold=.015, max_degrees=25., measure=unmeasured, noise_rotation=None):
    """Run an arm with the baseline RNG contract and fixed, index-addressed FPS draws."""
    from scripts.run_cloudfixer_point2vec import POLICIES, deterministic_proximal

    if arm not in ARMS or x.shape[0] != 1:
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

        kabsch = (KabschPose(x, mask, point_indices, KABSCH_STEPS[arm], max_degrees=max_degrees,
                             measure=measure) if arm in KABSCH_STEPS else None)
        clouds, metadata = sample(model, x, mask, noise, **config, measure=measure,
                                  pose_reference=kabsch.reference if kabsch is not None else None,
                                  feedback_rotation=select if arm == 'feedback' else None)
        if arm in SEARCH_ARMS and arm != 'feedback':
            select(clouds[:, 0])
        if kabsch is not None:
            pose = kabsch.pose
            diagnostics['updates'] = kabsch.updates
        logits = []
        for index, cloud in zip(metadata['candidate_indices'], clouds.unbind(1)):
            if index == 20 and selected_logits is not None:
                logits.append(selected_logits)
            else:
                if choice and arm in ('output', 'random'):
                    cloud = cloud @ pose
                logits.append(classify(index, cloud))
        logits = torch.stack(logits, 1)
        probabilities = logits.double().softmax(-1)
        if arm == 'view_mean':
            probabilities[:, 0] = view_probabilities
        diagnostics['denoiser_calls'] = metadata['denoiser_calls']
        return RotationResult(logits, probabilities.mean(1), clouds, pose, diagnostics)
