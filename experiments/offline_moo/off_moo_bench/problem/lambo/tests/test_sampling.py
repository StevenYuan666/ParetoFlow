import numpy as np
from lambo.utils import weighted_resampling
from scipy.special import softmax


def test_weighted_resampling():
    np.random.seed(1)
    k = 1.0
    scores = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ]
    )
    true_ranks = np.array([2, 2, 3, 4])
    true_weights = softmax(-np.log(true_ranks) / k)

    ranks, weights, resampled_idxs = weighted_resampling(scores, k=k)

    assert np.all(ranks == true_ranks), f"{true_ranks} != {ranks}"
    assert np.all(weights == true_weights), f"{true_weights} != {weights}"
    assert np.all(resampled_idxs == np.array([1, 2, 0, 0]))
