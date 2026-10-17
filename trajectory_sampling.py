"""External FPS views and frozen-classifier policies; no model imports at load.

Sampling inspiration: Bahri et al., WACV 2025, Sampling Variation with Weight
Averaging. Confidence/consistency inspiration: MEMO (NeurIPS 2022); JS training
precedent: AugMix (ICLR 2020). H+JS trajectory selection is an unvalidated
synthesis, not an implementation of those methods. See the knowledge plan.
"""
import hashlib
import random

import numpy as np


def sampling_protocol() -> dict:
    return dict(schema_version=1, mode='paired', points_per_view=1024,
                candidate_views=2, terminal_views=8,
                start_rule='sha256(seed:original_id), local Random.shuffle nonzero indices',
                reordering='swap chosen start with index0; map FPS indices back',
                normalization='once on whole cloud before any sampling',
                score='H(mean)/log40 + JS(A,B)/log2',
                tie_break='terminal then clean03, clean05, clean08',
                acquisition_classifier_forwards_per_example=14,
                policy_classifier_forwards_per_example=8)


def view_starts(seed: int, ids: list, cloud_size: int, count: int) -> np.ndarray:
    """Deterministic distinct starts; no corruption, labels or global RNG input."""
    if cloud_size < 2 or not 1 <= count < cloud_size:
        raise ValueError('Need 1 <= alternate views < cloud size')
    rows = []
    for index in ids:
        digest = hashlib.sha256(('%d:%d' % (seed, index)).encode('ascii')).digest()
        choices = list(range(1, cloud_size))
        random.Random(int.from_bytes(digest, 'big')).shuffle(choices)
        rows.append(choices[:count])
    return np.asarray(rows, dtype=np.int64)


def native_fps(points, count: int = 1024) -> tuple:
    """The exact native FPS/gather sequence used by utils_mate.misc.fps."""
    from pointnet2_ops import pointnet2_utils
    indices = pointnet2_utils.furthest_point_sample(points, count)
    sampled = pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
    return sampled, indices


def alternate_fps(points, starts, count: int = 1024, sampler=None) -> tuple:
    """Swap explicit starts to index0, use native FPS, return ORIGINAL indices."""
    import torch
    sampler = native_fps if sampler is None else sampler
    batch, n, dims = points.shape
    starts = torch.as_tensor(starts, device=points.device, dtype=torch.long)
    if (dims != 3 or not 1 <= count <= n or starts.shape != (batch,)
            or bool(((starts <= 0) | (starts >= n)).any())):
        raise ValueError('Invalid cloud/sample count/alternate start')
    order = torch.arange(n, device=points.device).expand(batch, n).clone()
    rows = torch.arange(batch, device=points.device)
    order[:, 0] = starts
    order[rows, starts] = 0
    reordered = points.gather(1, order[..., None].expand(-1, -1, 3)).contiguous()
    sampled, selected = sampler(reordered, count)
    if selected.shape != (batch, count) or not bool((selected[:, 0] == 0).all()):
        raise ValueError('Native FPS must start at input index0')
    return sampled, order.gather(1, selected.long())


def collect_views(cloud, indices_a, logits_a, classifier, ids: list, seed: int,
                  views: int, mark, sampler=None) -> tuple:
    """Reuse original A; isolate each additional view from main RNG/buffers."""
    from legacy_trajectory import isolated_probe, model_buffer_hashes, rng_hash
    sampler = native_fps if sampler is None else sampler
    count = indices_a.shape[1]
    if (indices_a.shape != (len(ids), count) or cloud.shape[0] != len(ids)
            or not bool((indices_a[:, 0] == 0).all())):
        raise ValueError('A FPS identity/start mismatch')
    starts = view_starts(seed, ids, cloud.shape[1], views - 1)
    before = rng_hash()
    buffers_before = model_buffer_hashes({'classifier': classifier})
    predictions = [logits_a.detach().cpu().numpy().copy()]
    indices = [indices_a.detach().cpu().numpy().astype(np.int64)]
    for view in range(views - 1):
        with isolated_probe([classifier]):
            mark('sampling_fps')
            sampled, selected = alternate_fps(cloud, starts[:, view], count, sampler)
            mark('sampling_classifier')
            logits = classifier.classification_only(sampled, only_unmasked=False)
            predictions.append(logits.detach().cpu().numpy().copy())
            indices.append(selected.detach().cpu().numpy().astype(np.int64))
    mark('sampling_artifact')
    after = rng_hash()
    buffers_after = model_buffer_hashes({'classifier': classifier})
    if before != after or buffers_before != buffers_after:
        raise RuntimeError('Sampling changed main RNG or model buffers')
    values = np.stack(predictions, axis=1)
    if values.shape != (len(ids), views, 40) or not np.isfinite(values).all():
        raise ValueError('Expected finite B x views x 40 sampling logits')
    return dict(sampling_logits=values, sampling_indices=np.stack(indices, axis=1),
                sampling_starts=np.column_stack((np.zeros(len(ids), dtype=np.int64), starts)),
                sampling_cloud_size=np.asarray(cloud.shape[1])), dict(
                    rng_before=before, rng_after=after,
                    buffers_before=buffers_before, buffers_after=buffers_after,
                    classifier_forwards=views - 1)


def view_policies(logits: np.ndarray, terminal_votes: np.ndarray) -> dict:
    """Label-free N x four candidates x two views x 40 policy computation."""
    from scripts.analyze_legacy_trajectory import probabilities
    x, votes = np.asarray(logits), np.asarray(terminal_votes)
    if (x.ndim != 4 or x.shape[1:] != (4, 2, 40)
            or votes.shape != (x.shape[0], 8, 40)
            or not np.array_equal(x[:, -1], votes[:, :2])):
        raise ValueError('Expected four paired candidates and matching eight terminal votes')
    p, _ = probabilities(x.reshape(len(x), 8, 40))
    p = p.reshape(x.shape)
    q, _ = probabilities(votes)
    mean = p.mean(axis=2)

    def entropy(prob):
        return -(prob * np.log(np.maximum(prob, np.finfo(np.float64).tiny))).sum(-1)

    h_raw = entropy(mean)
    js = np.clip((h_raw - entropy(p).mean(axis=2)) / np.log(2), 0, 1)
    h = np.clip(h_raw / np.log(40), 0, 1)
    order = np.array([3, 0, 1, 2])
    selected_h = order[h[:, order].argmin(1)]
    selected_js = order[(h + js)[:, order].argmin(1)]
    rows = np.arange(len(x))
    return dict(terminal=p[:, -1, 0].argmax(1),
                four_a_mean=p[:, :, 0].mean(1).argmax(1),
                eight_view_mean=mean.mean(1).argmax(1),
                view_entropy=mean[rows, selected_h].argmax(1),
                view_entropy_js=mean[rows, selected_js].argmax(1),
                terminal_eight_mean=q.mean(1).argmax(1),
                entropy=h, js=js, selected_entropy=selected_h,
                selected_entropy_js=selected_js)
