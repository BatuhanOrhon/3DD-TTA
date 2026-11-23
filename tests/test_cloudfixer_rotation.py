"""CPU contracts for the experimental rotation policies."""
import math

import torch

from test_cloudfixer_diffusion import ConstantEpsilon


def cloud():
    return torch.randn((1, 32, 3), generator=torch.Generator().manual_seed(7), dtype=torch.float64)


def test_kabsch_recovers_inverse_for_small_angle_and_masks_duplicate_anchors():
    from cloudfixer_rotation import KabschPose, axis_rotation

    reference = cloud()
    true_pose = axis_rotation(2, math.radians(1), reference)
    corrupted = reference @ true_pose
    extended = torch.cat((corrupted, corrupted[:, :3], corrupted[:, :1] * 100), dim=1)
    indices = torch.cat((torch.arange(32), torch.arange(3), torch.tensor([99])))[None]
    mask = torch.ones_like(indices, dtype=torch.bool)
    mask[:, -1] = False
    pose = KabschPose(extended, mask, indices, {80: 1})
    posed = pose.reference(80, reference)
    torch.testing.assert_close(posed[:, :32], reference)
    torch.testing.assert_close(pose.pose, true_pose.T)
    torch.testing.assert_close(torch.det(pose.pose), reference.new_tensor(1.))
    assert pose.updates[0]['anchors'] <= 32


def test_kabsch_rejects_line_support():
    from cloudfixer_rotation import KabschPose

    line = torch.zeros(1, 32, 3, dtype=torch.float64)
    line[0, :, 0] = torch.arange(32)
    mask = torch.ones(1, 32, dtype=torch.bool)
    pose = KabschPose(line, mask, None, {80: 1})
    assert pose.reference(80, line) is line
    assert pose.updates[0]['reason'] == 'rank'


def test_explicit_identity_matches_legacy_policy_exactly():
    from cloudfixer_rotation import rotation_policy
    from scripts.run_cloudfixer_point2vec import POLICIES, proximal_policy

    x = cloud()
    mask = torch.ones(x.shape[:2], dtype=torch.bool)

    def head(cloud):
        return cloud.mean(1) + torch.rand(3)

    before = torch.get_rng_state()
    logits, probabilities, clouds, _, _ = proximal_policy(ConstantEpsilon(), head, x, mask, POLICIES['proximal_pm'])
    result = rotation_policy(ConstantEpsilon(), head, x, mask, None, 'identity')
    assert torch.equal(logits, result.logits)
    assert torch.equal(probabilities, result.probabilities)
    assert torch.equal(clouds, result.clouds)
    assert torch.equal(torch.get_rng_state(), before)


def test_feedback_transports_corrected_cloud_and_noise_and_reuses_t20_selection():
    from cloudfixer_rotation import rotation_policy

    x = cloud() + torch.tensor([1., 1., 0.])
    mask = torch.ones(x.shape[:2], dtype=torch.bool)

    class Epsilon(ConstantEpsilon):
        def __init__(self):
            super().__init__()
            self.states = {}

        def phi(self, state, t, node_mask):
            value = super().phi(state, t, node_mask)
            self.states[self.calls[-1]] = state.clone()
            return value

    def head(value):
        score = value[..., 0].mean(1)
        return torch.stack((score, -score), 1)

    output_model, feedback_model = Epsilon(), Epsilon()
    kwargs = dict(theta=20, penalty=0, threshold=0)
    output = rotation_policy(output_model, head, x, mask, None, 'output', **kwargs)
    feedback = rotation_policy(feedback_model, head, x, mask, None, 'feedback', **kwargs)
    assert output.diagnostics['choice'] == feedback.diagnostics['choice'] != 0
    assert output.diagnostics['classifier_calls'] == feedback.diagnostics['classifier_calls'] == 9
    assert len(output_model.calls) == len(feedback_model.calls) == 9
    assert torch.equal(output.logits[:, 0], feedback.logits[:, 0])
    torch.testing.assert_close(feedback_model.states[10], output_model.states[10] @ feedback.pose)
    assert not torch.equal(feedback.clouds[:, 2], output.clouds[:, 2])


def test_rotation_clip_preserves_axis_and_orientation_at_half_turn():
    from cloudfixer_rotation import axis_rotation, clip_rotation, rotation_angle

    reference = cloud()
    half_turn = axis_rotation(1, math.pi, reference)
    limited = clip_rotation(half_turn, rotation_angle(half_turn), math.pi / 8)
    assert math.isclose(rotation_angle(limited), math.pi / 8)
    torch.testing.assert_close(limited, axis_rotation(1, math.pi / 8, reference))
    torch.testing.assert_close(torch.det(limited), reference.new_tensor(1.))


def test_simultaneous_bootstrap_keeps_object_corruption_and_seed_groups():
    import numpy as np
    from scripts.analyze_cloudfixer_rotation import joint_radius, paired_counts

    left = np.array([[True, False], [False, True]])
    right = ~left
    assert paired_counts(left, right)['repairs'] == paired_counts(left, right)['harms'] == 2
    opposing = np.array([[100., -100.], [-100., 100.]])
    assert joint_radius(opposing, draws=1000, seed=0) == 100.
    assert joint_radius(opposing.mean(1, keepdims=True), draws=1000, seed=0) == 0.


def test_validation_decision_serializes_numpy_pose_recovery():
    import json
    import numpy as np
    from scripts.analyze_cloudfixer_rotation import decide
    from scripts.run_cloudfixer_rotation import REGISTRY

    fit_arms = ('kabsch_single', 'kabsch_fixed', 'kabsch_progressive', 'output', 'feedback')
    conditions = ['clean', 'rotation', 'background']
    config = dict(campaign='test', conditions=conditions, freeze={'development_passed': True},
                  spec={'registration': REGISTRY})
    metrics = [dict(arm=arm, reference=ref, group=group, delta_pp=0., simultaneous_lower_pp=0.)
               for arm in fit_arms for ref in ('identity', 'random', 'view_mean')
               for group in (*conditions, 'all15', 'macro14')]
    entries = {arm: [{'seconds': 1.}] for arm in ('identity', *fit_arms)}
    synthetic = [dict(arm='kabsch_single', condition='clean', scenario='compound',
                      mean_degree_recovery=np.float64(.2))]
    assert json.loads(json.dumps(decide(config, metrics, entries, synthetic)))['arms']['kabsch_single']['gates']['pose']


def test_graph_lowpass_ignores_padding_and_preserves_rigid_equivariance():
    from cloudfixer_rotation import axis_rotation, graph_lowpass

    points = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [100., 100., 100.]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    filtered = graph_lowpass(points, mask)
    expected = points.clone()
    expected[0, :2, 0] = 1.
    torch.testing.assert_close(filtered, expected)
    rotation = axis_rotation(1, .4, points)
    torch.testing.assert_close(graph_lowpass(points @ rotation, mask), expected @ rotation)
    torch.testing.assert_close(graph_lowpass(cloud() @ rotation), graph_lowpass(cloud()) @ rotation)


def test_spectral_proxy_is_cached_by_target_and_returns_rotated_raw_anchors():
    from cloudfixer_rotation import KabschPose, axis_rotation

    x = cloud()
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    desired = axis_rotation(1, .02, x)
    calls = []

    def smoother(points, valid):
        calls.append(points)
        return points * .5

    fit = KabschPose(x, mask, None, {80: 3}, filter_points=smoother)
    corrected = fit.reference(80, x @ desired)
    torch.testing.assert_close(fit.pose, desired)
    torch.testing.assert_close(corrected, x @ desired)
    assert len(calls) == 2


def test_direct_scd_search_recovers_known_pose_and_keeps_identity_on_ties():
    from cloudfixer_rotation import SCDSearchPose, axis_rotation

    target = cloud()
    applied = axis_rotation(1, math.radians(15), target)
    observed = target @ applied
    x = torch.cat((observed, observed[:, :2], observed[:, :1] + 100.), dim=1)
    ids = torch.cat((torch.arange(32), torch.tensor([0, 1, 99])))[None]
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, -1] = False
    fit = SCDSearchPose(x, mask, ids)
    assert fit.reference(70, target) is x
    torch.testing.assert_close(fit.reference(80, target), x @ applied.T)
    assert sum(row['matching'] for row in fit.updates) == 13
    assert fit.updates[-1]['score_after'] < fit.updates[0]['score_before']
    assert fit.updates[-1]['travel_degrees'] <= 25
    assert fit.reference(50, target) is fit.anchors
    aligned = SCDSearchPose(target, mask[:, :32], None)
    assert aligned.reference(80, target) is target
    assert all(row['reason'] == 'identity' for row in aligned.updates)


def test_scd_phase_pairs_three_methods_on_disjoint_cohort():
    from scripts.run_cloudfixer_rotation import phase_scope, REGISTRY, SPECTRAL_REGISTRY

    conditions, ids, seeds, arms = phase_scope('scd', {}, {})
    assert len(conditions) == 16 and seeds == [0, 1, 2] and len(set(ids)) == 12
    assert not set(ids).intersection(REGISTRY['development_ids'] + REGISTRY['validation_ids'] + SPECTRAL_REGISTRY['ids'])
    assert set(arms) == {'identity', 'kabsch_fixed', 'scd_search'}
    assert phase_scope('scd_smoke', {}, {}) == (('background', 'rotation'), ids[:1], [0], arms)


def test_output_scd_rotates_prediction_toward_reference_and_transports_increment():
    from cloudfixer_diffusion import trimmed_neighbors
    from cloudfixer_rotation import OutputPose, axis_rotation

    x = cloud()
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    target = x @ axis_rotation(2, math.radians(1), x)
    pose = OutputPose(x, mask, None, 'pose_scd_t50')
    pose.capture_pairs(50, *trimmed_neighbors(target, x, mask, .95))
    assert pose.update(60, target, x) is None
    delta = pose.update(50, target, x)
    torch.testing.assert_close(target @ delta, x)
    assert pose.updates[0]['score_after'] < pose.updates[0]['score_before']


def test_output_pca_rejects_isotropic_cloud_and_handles_rotation_on_real_grid():
    import itertools
    from cloudfixer_rotation import OUTPUT_ARMS, OutputPose, axis_rotation, rotation_policy

    x = torch.tensor(list(itertools.product([-3., 3.], [-2., 2.], [-1., 1.])), dtype=torch.float64)
    # Repeat balanced geometry in both fixed halves for a deterministic recovery check.
    x = x.repeat(128, 1)[None]
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    rotation = axis_rotation(2, math.radians(16), x)
    pose = OutputPose(x, mask, None, 'pose_pca_t50')
    torch.testing.assert_close(pose.update(50, x @ rotation, x), rotation.T)
    assert pose.update(50, x * 0, x) is None
    for arm in OUTPUT_ARMS:
        probe = cloud()
        result = rotation_policy(ConstantEpsilon(), lambda z: z.mean(1), probe,
                                 mask[:, :probe.shape[1]], None, arm)
        expected = [50] if arm.endswith('t50') else list(range(80, 0, -10))
        updates = result.diagnostics['updates']
        assert [u['index'] for u in updates] == expected
        assert updates[-1]['travel_degrees'] <= 30.00001
        assert result.diagnostics['classifier_calls'] == 3 and result.diagnostics['denoiser_calls'] == 9


def test_output_feedback_eligibility_and_reserved_ids():
    from scripts.run_cloudfixer_rotation import OUTPUT_REGISTRY, phase_scope
    import pytest

    for phase in ('feedback_focus','feedback_smoke'):
        conditions, ids, seeds, arms = phase_scope(phase, {}, {})
        assert conditions == ('rotation', 'clean') and len(arms) == 5
    with pytest.raises(ValueError):
        phase_scope('feedback_other', {}, {'eligible': []})
    conditions, ids, seeds, arms = phase_scope('feedback_other', {}, {'eligible':['pose_pca_t50']})
    assert len(conditions) == 15 and seeds == [0] and list(arms) == ['identity','pose_pca_t50']
    assert len(set(ids + OUTPUT_REGISTRY['ids'] + OUTPUT_REGISTRY['smoke_ids'])) == 257


def test_separate_all15_request_keeps_original_expansion_gate_and_uses_reserved_panel():
    from scripts.run_cloudfixer_rotation import CONDITIONS, OUTPUT_REGISTRY, phase_scope

    conditions, ids, seeds, arms = phase_scope('feedback_all15', {}, {})
    assert set(conditions) == set(CONDITIONS) - {'clean'} and len(conditions) == 15
    assert len(set(ids)) == 128 and ids == OUTPUT_REGISTRY['other_ids'] and seeds == [0]
    assert list(arms) == ['identity', 'pose_pca_t50', 'pose_pca_all']
    smoke = phase_scope('feedback_all15_smoke', {}, {})
    assert smoke == (('rotation', 'clean'), OUTPUT_REGISTRY['smoke_ids'], [0], arms)
    assert not set(smoke[1]) & set(ids)
