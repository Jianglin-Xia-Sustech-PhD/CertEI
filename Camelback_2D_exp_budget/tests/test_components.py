from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_gp import approximate_rkhs_norm, fit_dynamic_gp, matern52_correlation
from prepare_inputs import load_candidates, prepare_inputs
from study_config import (
    BUDGETS,
    CANDIDATE_MEAN_NOISE_SD_MULTIPLIER,
    FUNCTION_NAMES,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    make_batch_schedule,
)
from test_functions import function_without_rkhs_metadata


def test_function_optima_and_linear_noise() -> None:
    for name in FUNCTION_NAMES:
        function = function_without_rkhs_metadata(name)
        values = function.evaluate(function.optimum_points)
        np.testing.assert_allclose(values, function.optimum_value, atol=2e-9)
        standardized = np.asarray([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            function.standardized_noise_sd_from_standardized_response(standardized),
            NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER * standardized,
        )
        sds = function.standardized_noise_sd_from_standardized_response(standardized)
        assert sds[0] > sds[1] > sds[2] > 0.0


def test_candidate_set_and_mean_noise_constraint() -> None:
    prepare_inputs()
    function = function_without_rkhs_metadata("six_hump_camelback")
    candidates = load_candidates("six_hump_camelback")
    assert candidates.shape == (N_CANDIDATES, 2)
    distances = np.sqrt(
        np.sum(
            (candidates[:, np.newaxis, :] - function.optimum_points[np.newaxis, :, :])
            ** 2,
            axis=2,
        )
    )
    assert float(np.min(distances)) > 1e-12
    standardized = function.standardize_outputs(function.evaluate(candidates))
    standardized_sds = function.standardized_noise_sd_from_standardized_response(
        standardized
    )
    np.testing.assert_allclose(
        np.mean(standardized_sds),
        CANDIDATE_MEAN_NOISE_SD_MULTIPLIER,
        atol=2e-12,
    )


def test_balanced_schedules_exhaust_budget() -> None:
    for budget in BUDGETS:
        schedule = make_batch_schedule(budget)
        assert int(np.sum(schedule)) == budget
        assert schedule.size > N_INITIAL
        assert int(np.max(schedule) - np.min(schedule)) <= 1


def test_matern_correlation_is_symmetric() -> None:
    points = np.asarray([[0.0, 0.0], [0.25, 0.7], [1.0, 1.0]])
    correlation = matern52_correlation(points, points, [0.3, 0.5])
    np.testing.assert_allclose(correlation, correlation.T)
    np.testing.assert_allclose(np.diag(correlation), 1.0)


def test_rkhs_norm_accounts_for_kernel_variance() -> None:
    points = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    outputs = np.asarray([0.0, 1.0, 1.0, 0.0])
    unit_norm, _ = approximate_rkhs_norm(points, outputs, [0.3, 0.3], 1.0)
    scaled_norm, _ = approximate_rkhs_norm(points, outputs, [0.3, 0.3], 4.0)
    np.testing.assert_allclose(scaled_norm, unit_norm / 2.0, rtol=1e-10)


def test_dynamic_gp_estimates_finite_parameters() -> None:
    points = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
    )
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


def test_weak_kernel_parameter_priors_are_recorded() -> None:
    points = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
    )
    outputs = np.sum((points - 0.4) ** 2, axis=1)
    fit = fit_dynamic_gp(
        points,
        outputs,
        np.full(points.shape[0], 0.01),
        points,
        process_variance_log_prior_mean=np.log(0.1),
        process_variance_log_prior_sd=1.0,
        lengthscale_log_prior_mean=np.log([0.25, 0.25]),
        lengthscale_log_prior_sd=1.0,
        max_iterations=20,
    )
    assert fit.hyperparameters.regularization_penalty >= 0.0
    np.testing.assert_allclose(
        fit.hyperparameters.penalized_objective,
        fit.hyperparameters.negative_log_likelihood
        + fit.hyperparameters.regularization_penalty,
    )
