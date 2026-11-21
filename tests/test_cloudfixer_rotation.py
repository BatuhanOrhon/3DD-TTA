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
