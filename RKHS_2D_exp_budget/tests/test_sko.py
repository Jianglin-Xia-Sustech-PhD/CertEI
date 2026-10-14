from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from comparison_runner import COMPARISON_ALGORITHMS
from run_experiments import comparison_specs, lqei_specs
from sko import (
    SKO_INCUMBENT_SD_MULTIPLIER,
    stochastic_kriging_optimization_acquisition,
)


def test_sko_is_a_formal_comparison_name():
    assert COMPARISON_ALGORITHMS == ("SKO", "EQI", "TSSO")
    assert "AEI" not in COMPARISON_ALGORITHMS


def test_sko_matches_the_source_augmented_ei_formula():
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

    # mean + sd is [0.40, 0.60], so the source rule uses mean 0.30.
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


def test_sko_uses_the_same_exact_batch_schedule_as_every_comparator():
    main = {
        (spec.function_name, spec.total_budget): spec.schedule
        for spec in lqei_specs()
    }
    seen = set()
    for spec in comparison_specs():
        key = (spec.function_name, spec.total_budget)
        np.testing.assert_array_equal(spec.schedule, main[key])
        assert int(np.sum(spec.schedule)) == spec.total_budget
        seen.add(spec.algorithm)
    assert seen == {"SKO", "EQI", "TSSO"}


def test_sko_uses_one_deviation_for_incumbent_and_recommendation():
    assert SKO_INCUMBENT_SD_MULTIPLIER == 1.0
