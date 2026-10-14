from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from comparison_runner import COMPARISON_ALGORITHMS
from continuous_acquisition import polish_sko
from run_experiments import comparison_specs, lqei_specs
from sequential_gp import SequentialFixedKernelGP
from sko import (
    SKO_INCUMBENT_SD_MULTIPLIER,
    stochastic_kriging_optimization_acquisition,
)


def test_sko_is_the_registered_four_dimensional_comparator() -> None:
    assert COMPARISON_ALGORITHMS == ("SKO", "EQI", "TSSO")
    assert "AEI" not in COMPARISON_ALGORITHMS


def test_sko_matches_the_source_augmented_ei_formula() -> None:
    sampled_means = np.array([0.30, 0.10])
    sampled_sds = np.array([0.10, 0.50])
    candidate_means = np.array([0.15, 0.35, 0.25])
    candidate_variances = np.array([0.04, 0.09, 0.01])
    candidate_noise_variances = np.array([0.01, 0.04, 0.00])
    actual, incumbent = stochastic_kriging_optimization_acquisition(
        sampled_means,
        sampled_sds,
        candidate_means,
        candidate_variances,
        candidate_noise_variances,
    )
    assert incumbent == 0.30
    deviations = np.sqrt(candidate_variances)
    improvement = incumbent - candidate_means
    standardized = improvement / deviations
    density = np.exp(-0.5 * standardized**2) / np.sqrt(2.0 * np.pi)
    expected_improvement = improvement * ndtr(standardized) + deviations * density
    penalty = 1.0 - np.sqrt(candidate_noise_variances) / np.sqrt(
        candidate_variances + candidate_noise_variances
    )
    np.testing.assert_allclose(actual, expected_improvement * penalty)


def test_sko_uses_the_exact_lqei_batch_schedule() -> None:
    main = {
        (spec.function_name, spec.total_budget): spec.schedule for spec in lqei_specs()
    }
    seen = set()
    for spec in comparison_specs():
        np.testing.assert_array_equal(
            spec.schedule, main[(spec.function_name, spec.total_budget)]
        )
        assert int(np.sum(spec.schedule)) == spec.total_budget
        seen.add(spec.algorithm)
    assert seen == {"SKO", "EQI", "TSSO"}


def test_sko_four_dimensional_continuous_polishing_is_finite() -> None:
    rng = np.random.default_rng(20260914)
    universe = rng.uniform(size=(32, 4))
    response = SequentialFixedKernelGP(
        universe=universe,
        kernel="matern52",
        lengthscale=0.2,
        process_variance=1.0,
        constant_mean=0.0,
        maximum_observations=8,
    )
    noise = SequentialFixedKernelGP(
        universe=universe,
        kernel="matern52",
        lengthscale=0.2,
        process_variance=0.4,
        constant_mean=0.8,
        maximum_observations=8,
    )
    for index in range(4):
        response.add_observation(index, float(index) / 10.0, 0.02)
        noise.add_observation(index, 0.7 + 0.05 * index, 0.01)
    sampled = np.arange(4)
    coarse, incumbent = stochastic_kriging_optimization_acquisition(
        response.posterior_mean[sampled],
        np.sqrt(response.posterior_variance[sampled]),
        response.posterior_mean,
        response.posterior_variance,
        np.clip(noise.posterior_mean, 0.5, 1.2) ** 2,
    )
    result = polish_sko(response, noise, incumbent, 0.5, 1.2, coarse)
    assert result.point.shape == (4,)
    assert np.all((0.0 <= result.point) & (result.point <= 1.0))
    assert np.isfinite(result.acquisition)
    assert result.acquisition + 1e-12 >= result.coarse_acquisition


def test_sko_uses_one_deviation_for_incumbent_and_recommendation() -> None:
    assert SKO_INCUMBENT_SD_MULTIPLIER == 1.0
