import unittest

import numpy as np

from scripts.analyze_legacy_trajectory_macro14 import (
    entropy_weighted_probabilities,
    trajectory_policy_predictions,
)


class Macro14PolicyTests(unittest.TestCase):
    def test_entropy_weights_are_mild_and_fall_back_to_equal_at_max_entropy(self):
        probabilities = np.array([[[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]])
        entropy = np.array([[0.0, 0.5, 1.0]])

        mean, weights = entropy_weighted_probabilities(probabilities, entropy)

        np.testing.assert_allclose(weights, [[2 / 3, 1 / 3, 0.0]])
        np.testing.assert_allclose(mean, [[0.6, 0.4]])

        uniform_entropy = np.ones((1, 3))
        fallback, fallback_weights = entropy_weighted_probabilities(probabilities, uniform_entropy)
        np.testing.assert_allclose(fallback_weights, [[1 / 3, 1 / 3, 1 / 3]])
        np.testing.assert_allclose(fallback, [[0.5, 0.5]])

    def test_trace_policies_include_entropy_mean_and_late_clean_terminal_mean(self):
        names = ['clean_03', 'clean_05', 'clean_08', 'terminal']
        logits = np.array([
            [[0.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        ])

        result = trajectory_policy_predictions(logits, names)

        self.assertEqual(result['terminal'].tolist(), [1, 1])
        self.assertEqual(result['latest_clean_terminal_mean'].tolist(), [1, 0])
        self.assertEqual(result['entropy_weighted_mean'].shape, (2,))
        self.assertEqual(result['probability_mean'].shape, (2,))

    def test_entropy_weighting_rejects_malformed_probability_and_entropy_arrays(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            entropy_weighted_probabilities(np.ones((2, 3)), np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
            probabilities = np.array([[[0.5, 0.5], [0.5, 0.5]]])
            entropy_weighted_probabilities(probabilities, np.array([[0.0, 1.1]]))


if __name__ == '__main__':
    unittest.main()
