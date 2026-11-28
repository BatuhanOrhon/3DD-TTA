import numpy as np
import torch

from test_cloudfixer_diffusion import ConstantEpsilon


def test_shared_trajectory_matches_independent_surface_policy_and_object_rng():
    from cloudfixer_rotation import rotation_policy, SURFACE_ARM
    from scripts.cloudfixer_surface_four_runtime import host_rng, shared_trajectory, score_clouds

    shape = (1, 32, 3)
    points = torch.randn(shape, generator=torch.Generator().manual_seed(512))
    valid = torch.ones(shape[:2], dtype=torch.bool)
    indices = torch.arange(32)[None]
    objects = [(19, points, valid, indices), (3, points + 0.4, valid, indices)]
    def head(cloud):
        return cloud.mean((1, 2))[:, None] + torch.rand(1, 40) + np.random.rand()

    shared_model = ConstantEpsilon()
    before = torch.get_rng_state().clone()
    clouds, seeds, updates = shared_trajectory(shared_model, objects)
    assert shared_model.calls == list(range(80, -1, -10))
    logits = score_clouds(head, clouds, seeds)
    assert logits.shape == (2, 3, 40) and torch.equal(torch.get_rng_state(), before)
    for j, (rng, x, mask, ids) in enumerate(objects):
        step = iter((20, 10, 0))
        def independently_seeded(cloud):
            current = next(step)
            np.random.seed(seeds[j][current])
            return head(cloud)
        with host_rng(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng)
            independent = rotation_policy(ConstantEpsilon(), independently_seeded, x, mask, ids, SURFACE_ARM)
        torch.testing.assert_close(independent.clouds[0], clouds[j], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(independent.logits[0], logits[j], atol=1e-5, rtol=1e-5)
        assert updates[j]['reason'] == independent.diagnostics['updates'][0]['reason']


def test_calibration_rejects_changed_veto_pose_logits_and_decisions():
    from scripts.cloudfixer_surface_four_runtime import parity

    base = dict(input_sha256=np.array(['abc']), pose=np.eye(3)[None],
                reason=np.array(['unstable']), angle_degrees=np.zeros(1),
                logits=np.zeros((1, 4, 3, 40), dtype=np.float32))
    assert parity(base, base)['equivalent']
    for key,value in [('reason', ['surface_veto']), ('pose', np.zeros((1,3,3))),
                      ('input_sha256', ['xyz']), ('angle_degrees', [4.0])]:
        assert not parity(dict(base, **{key: np.asarray(value)}), base)['equivalent']
    for delta in (1e-6, 1.0):
        logits = base['logits'].copy()
        logits[:, 3, 2, 1] = delta
        result = parity(dict(base, logits=logits), base)
        assert not result['equivalent']
        assert result['heads']['pointmae']['probe_decisions_equal'] is False


def test_curvenet_singleton_shape_fix_preserves_batch_math_and_rng():
    from scripts.cloudfixer_surface_four_runtime import repaired_curve_forward

    class Group(torch.nn.Module):
        def forward(self, start_index):
            return start_index.squeeze().unsqueeze(2)

    class Walk(torch.nn.Module):
        def forward(self, cur_feature, pre_feature):
            return torch.cat((cur_feature.squeeze(), pre_feature.squeeze()),dim=1)

    for cls, layer in ((Group, 'CurveGrouping'), (Walk, 'Walk')):
        original = cls()
        repaired = cls()
        method, receipt = repaired_curve_forward(cls.forward, layer)
        repaired.forward = method.__get__(repaired)
        assert receipt['before_sha256'] != receipt['after_sha256']
        for size in (1, 2, 4):
            values = ([torch.ones(size, 1, 4)] if cls is Group else
                      [torch.ones(size, 3, 4, 1), torch.ones(size, 3, 4, 1)])
            result = repaired(*values)
            assert result.shape == ((size, 4, 1) if cls is Group else (size, 6, 4))
            if size > 1:
                assert torch.equal(original(*values), result)
