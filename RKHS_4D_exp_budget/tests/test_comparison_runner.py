from __future__ import annotations

import numpy as np

from comparison_runner import (
    ComparisonSpec,
    _TSSOMeanPosterior,
    _allocate_ocba,
    _integer_allocation,
    _tsso_stage_split,
)
from study_config import N_INITIAL


def test_tsso_stage_count_comes_from_the_common_batch_schedule() -> None:
    spec = ComparisonSpec(
        algorithm="TSSO",
        phase_code=1,
        function_name="rkhs_se_4d",
        total_budget=10000,
        replication_multiplier=6.0,
        anchor_budget=10000,
        macro_count=1,
        candidate_count=64,
    )
    assert spec.sequential_stage_count == spec.schedule.size - N_INITIAL
    assert spec.sequential_stage_count == 126


def test_tsso_stage_split_consumes_one_complete_common_batch() -> None:
    first_search, first_allocation = _tsso_stage_split(60, 1, 100)
    last_search, last_allocation = _tsso_stage_split(60, 100, 100)
    assert first_search + first_allocation == 60
    assert last_search + last_allocation == 60
    assert first_search > last_search >= 2
    assert first_allocation < last_allocation


def test_integer_allocation_uses_exact_budget() -> None:
    allocation = _integer_allocation(np.asarray([1.0, 2.0, 3.0]), 17)
    np.testing.assert_array_equal(allocation, [3, 6, 8])
    assert int(np.sum(allocation)) == 17


def test_ocba_adds_exact_budget() -> None:
    current = np.asarray([10, 10, 10])
    targets = _allocate_ocba(
        current,
        sample_means=np.asarray([0.0, 0.4, 1.0]),
        mean_variances=np.asarray([0.2, 0.3, 0.5]),
        additional_budget=19,
    )
    assert np.all(targets >= current)
    assert int(np.sum(targets - current)) == 19


def test_tsso_aggregated_mean_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(20260901)
    posterior = _TSSOMeanPosterior(
        observed_points=rng.uniform(size=(7, 4)),
        weights=rng.normal(size=7),
        kernel="matern52",
        lengthscale=0.2,
        process_variance=1.3,
        constant_mean=-0.4,
    )
    point = rng.uniform(0.2, 0.8, size=4)
    value, gradient = posterior.mean_with_gradient(point)
    assert np.isfinite(value)
    step = 1e-6
    finite_difference = np.empty(4)
    for coordinate in range(4):
        plus = point.copy()
        minus = point.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        finite_difference[coordinate] = (
            posterior.mean_with_gradient(plus)[0]
            - posterior.mean_with_gradient(minus)[0]
        ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-6)
