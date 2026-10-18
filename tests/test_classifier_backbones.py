"""CPU boundary checks; native backbone/checkpoint parity needs separate evidence."""

import copy
import unittest

import torch
from torch import nn

from classifier_backbones import FrozenClassifier


class PointMAEContract(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.head = nn.Linear(3, 4)
        self.dropout = nn.Dropout(0.5)

    def classification_only(self, points, *, only_unmasked):
        assert only_unmasked is False
        self.received = points
        features = self.bn(points.transpose(1, 2)).mean(-1)
        return self.head(self.dropout(features)) + torch.rand((), device=points.device)

    def forward(self, points):
        raise AssertionError("Point-MAE reconstruction forward must not be called")


class LayoutContract(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, points):
        self.received = points
        xyz = points if self.layout == "bnc" else points.transpose(1, 2)
        assert xyz.shape[-1] == 3
        return torch.cat((xyz.mean(1), xyz.mean((1, 2))[:, None]), dim=1) * self.weight


class OutputContract(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, points):
        return self.output


class PointNeXtContract(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, data):
        assert set(data) == {"pos", "x"}
        self.received = data
        return torch.cat((data["pos"].mean(1),
                          data["x"].mean((1, 2))[:, None]), dim=1) * self.weight


class FrozenClassifierTests(unittest.TestCase):
    def test_pointmae_matches_direct_logits_input_rng_and_buffers(self):
        model = PointMAEContract().eval().requires_grad_(False)
        direct_model = copy.deepcopy(model)
        points = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        original_points = points.clone()
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        rng_before = torch.random.get_rng_state().clone()
        classifier = FrozenClassifier(model, "pointmae", 4, input_layout="bnc")
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

        expected = direct_model.classification_only(points, only_unmasked=False)
        rng_after_direct = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng_before)
        actual = classifier(points)

        self.assertIs(model.received, points)
        self.assertTrue(torch.equal(points, original_points))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng_after_direct, torch.random.get_rng_state()))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]), name)

    def test_declared_layout_controls_dispatch_for_both_backbones_and_batch_one(self):
        for backbone in ("dgcnn", "curvenet"):
            for layout in ("bnc", "bcn"):
                with self.subTest(backbone=backbone, layout=layout):
                    points = torch.tensor([[[1., 2., 3.], [5., 8., 11.]]], requires_grad=True)
                    model = LayoutContract(layout)
                    classifier = FrozenClassifier(model, backbone, 4, input_layout=layout)
                    actual = classifier(points)
                    torch.testing.assert_close(actual, torch.tensor([[6., 10., 14., 10.]]))
                    expected_input = points if layout == "bnc" else points.transpose(1, 2)
                    self.assertTrue(torch.equal(model.received, expected_input))
                    self.assertTrue(model.received.is_contiguous())
                    actual.sum().backward()
                    torch.testing.assert_close(points.grad, torch.full_like(points, 4. / 3.))
                    self.assertIsNone(model.weight.grad)

    def test_eval_freezes_parameters_and_bn_while_preserving_input_gradients(self):
        model = PointMAEContract()
        classifier = FrozenClassifier(model, "pointmae", 4, input_layout="bnc")
        classifier.train()
        self.assertFalse(classifier.training)
        self.assertTrue(all(not module.training for module in model.modules()))
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        points = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3).requires_grad_()

        classifier(points).square().sum().backward()

        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0.)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]), name)

    def test_rejects_malformed_or_nonfinite_xyz_before_model_call(self):
        classifier = FrozenClassifier(LayoutContract("bnc"), "dgcnn", 4, input_layout="bnc")
        bad_inputs = [None, torch.zeros(5, 3), torch.zeros(1, 3, 5),
                      torch.zeros(1, 5, 4), torch.zeros(0, 5, 3),
                      torch.zeros(1, 0, 3), torch.ones(1, 5, 3, dtype=torch.int64),
                      torch.full((1, 5, 3), float("nan")),
                      torch.full((1, 5, 3), float("inf"))]
        for points in bad_inputs:
            with self.subTest(points=points):
                with self.assertRaises((TypeError, ValueError)):
                    classifier(points)
        self.assertFalse(hasattr(classifier.model, "received"))

    def test_rejects_wrong_shape_type_and_nonfinite_logits(self):
        bad_outputs = [None, (torch.ones(1, 4),), torch.ones(4), torch.ones(2, 4),
                       torch.ones(1, 3), torch.ones(1, 4, dtype=torch.int64),
                       torch.full((1, 4), float("nan")),
                       torch.full((1, 4), float("inf"))]
        for output in bad_outputs:
            with self.subTest(output=output):
                classifier = FrozenClassifier(OutputContract(output), "dgcnn", 4,
                                              input_layout="bnc")
                with self.assertRaises((TypeError, ValueError)):
                    classifier(torch.ones(1, 5, 3))

    def test_rejects_unknown_backbone_layout_and_invalid_class_count(self):
        for kwargs in ({"backbone": "unknown"}, {"input_layout": "other"},
                       {"backbone": "pointmae", "input_layout": "bcn"},
                       {"backbone": "pointnext", "input_layout": "bcn"},
                       {"num_classes": 0}, {"num_classes": True},
                       {"num_classes": 4.5}):
            with self.subTest(kwargs=kwargs):
                arguments = dict(backbone="dgcnn", num_classes=4, input_layout="bnc")
                arguments.update(kwargs)
                with self.assertRaises((TypeError, ValueError)):
                    FrozenClassifier(LayoutContract("bnc"), **arguments)

    def test_pointnext_default_features_preserve_xyz_layout_and_gradients(self):
        model = PointNeXtContract()
        classifier = FrozenClassifier(model, "pointnext", 4, input_layout="bnc")
        points = torch.tensor([[[1., 2., 3.], [5., 8., 11.]]], requires_grad=True)

        logits = classifier(points)

        self.assertIs(model.received["pos"], points)
        torch.testing.assert_close(model.received["x"],
                                   torch.tensor([[[1., 5.], [2., 8.], [3., 11.]]]))
        self.assertTrue(model.received["x"].is_contiguous())
        torch.testing.assert_close(logits, torch.tensor([[6., 10., 14., 10.]]))
        logits.sum().backward()
        torch.testing.assert_close(points.grad, torch.full_like(points, 4. / 3.))
        self.assertIsNone(model.weight.grad)

    def test_pointnext_explicit_four_features_are_unchanged_and_differentiable(self):
        model = PointNeXtContract()
        classifier = FrozenClassifier(model, "pointnext", 4, input_layout="bnc")
        points = torch.tensor([[[1., 2., 3.], [5., 8., 11.]]], requires_grad=True)
        features = torch.tensor([[[2., 4.], [6., 8.], [10., 12.], [14., 16.]]],
                                requires_grad=True)
        original_features = features.detach().clone()

        logits = classifier(points, features=features)

        self.assertIs(model.received["pos"], points)
        self.assertIs(model.received["x"], features)
        self.assertTrue(torch.equal(features, original_features))
        torch.testing.assert_close(logits, torch.tensor([[6., 10., 14., 18.]]))
        logits.sum().backward()
        torch.testing.assert_close(points.grad, torch.ones_like(points))
        torch.testing.assert_close(features.grad, torch.full_like(features, .25))
        self.assertIsNone(model.weight.grad)

    def test_pointnext_rejects_malformed_features_before_model_call(self):
        model = PointNeXtContract()
        classifier = FrozenClassifier(model, "pointnext", 4, input_layout="bnc")
        bad_features = ["xyz", torch.ones(1, 5), torch.ones(1, 5, 4),
                        torch.ones(2, 4, 5), torch.ones(1, 0, 5),
                        torch.ones(1, 4, 5, dtype=torch.int64),
                        torch.ones(1, 4, 5, dtype=torch.float64),
                        torch.ones(1, 4, 5, device="meta"),
                        torch.full((1, 4, 5), float("nan")),
                        torch.full((1, 4, 5), float("inf"))]
        for features in bad_features:
            with self.subTest(features=features):
                with self.assertRaises((TypeError, ValueError)):
                    classifier(torch.ones(1, 5, 3), features=features)
        self.assertFalse(hasattr(model, "received"))

    def test_other_backbones_reject_features_instead_of_ignoring_them(self):
        for backbone in ("pointmae", "dgcnn", "curvenet"):
            with self.subTest(backbone=backbone):
                model = PointMAEContract() if backbone == "pointmae" else LayoutContract("bnc")
                classifier = FrozenClassifier(model, backbone, 4, input_layout="bnc")
                with self.assertRaises(ValueError):
                    classifier(torch.ones(1, 5, 3), features=torch.ones(1, 4, 5))
                self.assertFalse(hasattr(model, "received"))


if __name__ == "__main__":
    unittest.main()
