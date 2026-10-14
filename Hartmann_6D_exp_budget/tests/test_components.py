from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_gp import (
    approximate_rkhs_norm,
    fit_dynamic_gp,
    matern52_correlation,
    posterior_mean_rkhs_norm,
)
from prepare_inputs import _build_faure_candidates
from study_config import (
    BUDGETS,
    DIMENSION,
    MIN_SEQUENTIAL_STAGES,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    make_batch_schedule,
)
from test_functions import hartmann6
from test_functions import function_without_rkhs_metadata


def test_function_optima_and_noise_envelope() -> None:
    function = function_without_rkhs_metadata("hartmann6")
    values = function.evaluate(function.optimum_points)
    np.testing.assert_allclose(values, function.optimum_value, atol=2e-9)
    np.testing.assert_allclose(function.optimum_value, -3.3223680114, atol=2e-9)
    sds = function.noise_standard_deviation(function.optimum_points)
    np.testing.assert_allclose(
        sds, NOISE_A_MULTIPLIER * function.response_range, atol=1e-10
    )
    standardized = function.standardize_outputs(
        [function.optimum_value, function.maximum_value]
    )
    np.testing.assert_allclose(standardized, [0.0, 1.0])
    np.testing.assert_allclose(
        function.standardized_noise_sd_from_standardized_response(standardized),
        [NOISE_A_MULTIPLIER, NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER],
    )
    np.testing.assert_allclose(
        function.unstandardize_outputs(standardized),
        [function.optimum_value, function.maximum_value],
    )


def test_balanced_schedules_exhaust_budget() -> None:
    for budget in BUDGETS:
        schedule = make_batch_schedule(budget)
        assert int(np.sum(schedule)) == budget
        assert schedule.size >= N_INITIAL + MIN_SEQUENTIAL_STAGES
        assert int(np.max(schedule) - np.min(schedule)) <= 1


def test_faure_candidates_reproduce_jalali_discrete_optimum() -> None:
    candidates = _build_faure_candidates()
    assert candidates.shape == (N_CANDIDATES, DIMENSION)
    values = hartmann6(candidates)
    best = int(np.argmin(values))
    assert best == 1940
    np.testing.assert_allclose(
        candidates[best],
        [
            0.23823406913785922,
            0.13910870470637235,
            0.36651395251978336,
            0.3286130778842149,
            0.35193669304456476,
            0.7017909204498125,
        ],
        atol=1e-14,
    )
    np.testing.assert_allclose(values[best], -3.019973969971238, atol=1e-13)


def test_matern_correlation_is_symmetric() -> None:
    points = np.asarray(
        [np.zeros(DIMENSION), np.linspace(0.1, 0.6, DIMENSION), np.ones(DIMENSION)]
    )
    correlation = matern52_correlation(points, points, np.linspace(0.3, 0.8, DIMENSION))
    np.testing.assert_allclose(correlation, correlation.T)
    np.testing.assert_allclose(np.diag(correlation), 1.0)


def test_rkhs_norm_accounts_for_kernel_variance() -> None:
    points = np.random.default_rng(123).random((12, DIMENSION))
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    lengthscales = np.full(DIMENSION, 0.3)
    unit_norm, _ = approximate_rkhs_norm(points, outputs, lengthscales, 1.0)
    scaled_norm, _ = approximate_rkhs_norm(points, outputs, lengthscales, 4.0)
    np.testing.assert_allclose(scaled_norm, unit_norm / 2.0, rtol=1e-10)


def test_dynamic_gp_estimates_finite_parameters() -> None:
    points = np.random.default_rng(124).random((14, DIMENSION))
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    fit = fit_dynamic_gp(
        points,
        outputs,
        np.full(points.shape[0], 0.01),
        points,
        max_iterations=20,
    )
    assert np.all(np.isfinite(fit.posterior_mean))
    assert np.all(fit.posterior_variance >= 0.0)
    assert fit.hyperparameters.process_variance > 0.0
    assert np.all(fit.hyperparameters.lengthscales > 0.0)


def test_posterior_mean_rkhs_norm_is_noise_regularized() -> None:
    points = np.random.default_rng(127).random((16, DIMENSION))
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    low_noise = np.full(points.shape[0], 0.001)
    fit = fit_dynamic_gp(points, outputs, low_noise, points, max_iterations=20)
    low_noise_norm = posterior_mean_rkhs_norm(
        points, outputs, low_noise, fit.hyperparameters
    )
    high_noise_norm = posterior_mean_rkhs_norm(
        points, outputs, np.full(points.shape[0], 0.2), fit.hyperparameters
    )
    assert np.isfinite(low_noise_norm) and low_noise_norm > 0.0
    assert 0.0 <= high_noise_norm < low_noise_norm


def test_weak_kernel_parameter_priors_are_recorded() -> None:
    points = np.random.default_rng(125).random((14, DIMENSION))
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    fit = fit_dynamic_gp(
        points,
        outputs,
        np.full(points.shape[0], 0.01),
        points,
        process_variance_log_prior_mean=np.log(0.1),
        process_variance_log_prior_sd=1.0,
        lengthscale_log_prior_mean=np.log(np.full(DIMENSION, 0.25)),
        lengthscale_log_prior_sd=1.0,
        max_iterations=20,
    )
    assert fit.hyperparameters.regularization_penalty >= 0.0
    np.testing.assert_allclose(
        fit.hyperparameters.penalized_objective,
        fit.hyperparameters.negative_log_likelihood
        + fit.hyperparameters.regularization_penalty,
    )


def test_process_variance_prior_works_without_lengthscale_prior() -> None:
    points = np.random.default_rng(126).random((14, DIMENSION))
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    fit = fit_dynamic_gp(
        points,
        outputs,
        np.full(points.shape[0], 0.01),
        points,
        process_variance_log_prior_mean=np.log(0.1),
        process_variance_log_prior_sd=1.0,
        max_iterations=20,
    )
    assert fit.hyperparameters.regularization_penalty >= 0.0
    np.testing.assert_allclose(
        fit.hyperparameters.penalized_objective,
        fit.hyperparameters.negative_log_likelihood
        + fit.hyperparameters.regularization_penalty,
    )
