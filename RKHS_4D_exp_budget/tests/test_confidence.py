from __future__ import annotations

import numpy as np

from confidence import (
    build_confidence_schedule,
    chi_square_correction,
    finite_batch_information_gain,
    kernel_order_information_gain,
)


def test_chi_square_correction_is_finite_and_at_least_one() -> None:
    value = chi_square_correction(batch_size=40, batch_count=40, delta=0.05)
    assert np.isfinite(value)
    assert value >= 1.0


def test_four_dimensional_kernel_order_information_gain() -> None:
    u = np.asarray([0.0, 2.0, 10.0])
    logarithm = np.log1p(u)
    np.testing.assert_allclose(
        kernel_order_information_gain("squared_exponential", u),
        logarithm**5,
    )
    np.testing.assert_allclose(
        kernel_order_information_gain("matern52", u),
        u ** (4.0 / 9.0) * logarithm ** (5.0 / 9.0),
    )


def test_finite_batch_information_gain() -> None:
    indices = np.asarray([0.0, 1.0, 25.0])
    expected = 0.5 * indices * np.log1p(80.0 / 20.0)
    np.testing.assert_allclose(
        finite_batch_information_gain(
            indices,
            information_gain_batch_size=80,
            variance_min=20.0,
        ),
        expected,
    )


def test_lambda_schedule_uses_combined_minimum_and_equation_eight() -> None:
    schedule = build_confidence_schedule(
        kernel="matern52",
        information_gain_batch_size=80,
        correction_batch_size=80,
        batch_count=25,
        variance_min=20.0,
        rkhs_norm=7.0,
        delta=0.05,
    )
    expected_u = np.arange(26) * 80.0 / 20.0
    expected_gamma = (
        expected_u ** (4.0 / 9.0)
        * np.log1p(expected_u) ** (5.0 / 9.0)
    )
    expected_finite = 0.5 * np.arange(26) * np.log1p(80.0 / 20.0)
    expected_combined = np.minimum(expected_gamma, expected_finite)
    expected_lambda = 7.0 + np.sqrt(
        2.0
        * schedule.kappa_t
        * (expected_combined + np.log(3.0 / (2.0 * 0.05)))
    )
    np.testing.assert_allclose(schedule.effective_information_budget, expected_u)
    np.testing.assert_allclose(schedule.kernel_order_upper_bound, expected_gamma)
    np.testing.assert_allclose(
        schedule.finite_batch_upper_bound,
        expected_finite,
    )
    np.testing.assert_allclose(
        schedule.information_gain_upper_bound,
        expected_combined,
    )
    np.testing.assert_allclose(schedule.lambda_values, expected_lambda)
    assert np.all(np.diff(schedule.lambda_values) >= 0.0)


def test_se_schedule_uses_finite_batch_bound_when_it_is_smaller() -> None:
    schedule = build_confidence_schedule(
        kernel="squared_exponential",
        information_gain_batch_size=70,
        correction_batch_size=70,
        batch_count=1142,
        variance_min=53.0,
        rkhs_norm=7.0,
        delta=0.05,
    )
    np.testing.assert_allclose(
        schedule.information_gain_upper_bound,
        np.minimum(
            schedule.kernel_order_upper_bound,
            schedule.finite_batch_upper_bound,
        ),
    )
    assert schedule.finite_batch_upper_bound[-1] < schedule.kernel_order_upper_bound[-1]
