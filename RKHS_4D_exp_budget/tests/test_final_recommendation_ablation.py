from __future__ import annotations

import numpy as np

from analyze_final_recommendation_ablation import (
    RULE_CURRENT,
    RULE_MEAN,
    RULE_MINUS_TEN_SD,
    _coefficient,
    terminal_posterior_on_sampled_points,
)
from sequential_gp import SequentialFixedKernelGP


def test_final_recommendation_ablation_signed_coefficients() -> None:
    assert _coefficient(RULE_CURRENT, 20.0) == 4.0
    assert _coefficient(RULE_MEAN, 20.0) == 0.0
    assert _coefficient(RULE_MINUS_TEN_SD, 20.0) == -10.0


def test_direct_terminal_posterior_matches_sequential_gp() -> None:
    rng = np.random.default_rng(20260915)
    universe = rng.uniform(size=(12, 4))
    selected = np.asarray([0, 3, 3, 8, 2, 8], dtype=int)
    outputs = rng.normal(size=selected.size)
    noise = rng.uniform(0.02, 0.15, size=selected.size)
    process_variance = 1.4
    constant_mean = -0.3
    lengthscale = 0.25

    model = SequentialFixedKernelGP(
        universe=universe,
        kernel="matern52",
        lengthscale=lengthscale,
        process_variance=process_variance,
        constant_mean=constant_mean,
        maximum_observations=selected.size,
    )
    for index, output, variance in zip(selected, outputs, noise):
        model.add_observation(int(index), float(output), float(variance))

    sampled, means, sds = terminal_posterior_on_sampled_points(
        universe=universe,
        selected_indices=selected,
        observed_means=outputs,
        mean_noise_variances=noise,
        kernel="matern52",
        lengthscale=lengthscale,
        process_variance=process_variance,
        constant_mean=constant_mean,
    )
    expected_sampled = np.unique(selected)
    np.testing.assert_array_equal(sampled, expected_sampled)
    np.testing.assert_allclose(means, model.posterior_mean[expected_sampled])
    np.testing.assert_allclose(
        sds,
        np.sqrt(np.maximum(model.posterior_variance[expected_sampled], 0.0)),
    )
