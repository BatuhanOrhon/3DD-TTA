import numpy as np
import pytest

from scripts.analyze_cloudfixer_weighted_pm import entropy_readouts


def entropy(p):
    return -(p * np.log(p)).sum(-1)


def test_low_entropy_can_change_an_equal_pm_prediction():
    p = np.array([[[.8, .2], [.8, .2], [.01, .99]]])
    result = entropy_readouts(p, entropy(p))
    assert result['equal_pm'][0].argmax(-1).tolist() == [0]
    assert result['linear_entropy'][0].argmax(-1).tolist() == [1]
    assert result['exp_beta_8'][0].argmax(-1).tolist() == [1]
    assert result['linear_entropy'][1][0, 2] > result['linear_entropy'][1][0, 0]


def test_zero_strength_exactly_preserves_equal_pm_and_terminal():
    p = np.array([[[.6, .4], [.2, .8], [.7, .3]]])
    result = entropy_readouts(p, entropy(p))
    np.testing.assert_array_equal(result['exp_beta_0'][0], p.mean(1))
    np.testing.assert_array_equal(result['terminal'][0], p[:, -1])
    for probabilities, weights in result.values():
        np.testing.assert_allclose(probabilities.sum(-1), 1)
        np.testing.assert_allclose(weights.sum(-1), 1)
        assert np.isfinite(weights).all() and (weights >= 0).all()


def test_uniform_predictions_do_not_make_zero_or_nan_weights():
    p = np.full((2, 4, 40), 1 / 40)
    result = entropy_readouts(p, np.full((2, 4), np.log(40)))
    np.testing.assert_allclose(result['linear_entropy'][1], .25)
    for probabilities, _ in result.values():
        np.testing.assert_allclose(probabilities, 1 / 40)


def test_invalid_probability_or_entropy_is_rejected():
    with pytest.raises(ValueError):
        entropy_readouts(np.array([[[1.2, -.2]]]), np.array([[.1]]))
    with pytest.raises(ValueError):
        entropy_readouts(np.array([[[.5, .5]]]), np.array([[np.nan]]))
