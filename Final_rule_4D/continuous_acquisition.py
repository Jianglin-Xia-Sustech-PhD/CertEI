"""Hybrid Sobol screening and multi-start continuous acquisition polishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from sequential_gp import SequentialFixedKernelGP
from study_config import (
    CONTINUOUS_POLISH_FTOL,
    CONTINUOUS_POLISH_GTOL,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_MINIMUM_GAIN,
    CONTINUOUS_POLISH_STARTS,
    DIMENSION,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolishResult:
    point: FloatArray
    acquisition: float
    coarse_acquisition: float
    improved: bool
    function_evaluations: int
    successful_starts: int


def _normal_density(value: float) -> float:
    return float(np.exp(-0.5 * value * value) / np.sqrt(2.0 * np.pi))


def _positive_part_with_gradient(
    improvement: float,
    deviation: float,
    improvement_gradient: FloatArray,
    deviation_gradient: FloatArray,
) -> tuple[float, FloatArray]:
    """Expected positive part and its gradient."""

    if deviation <= 1e-14:
        if improvement > 0.0:
            return float(improvement), improvement_gradient.copy()
        return 0.0, np.zeros_like(improvement_gradient)
    standardized = float(improvement / deviation)
    distribution = float(ndtr(standardized))
    density = _normal_density(standardized)
    value = float(improvement * distribution + deviation * density)
    gradient = distribution * improvement_gradient + density * deviation_gradient
    return max(value, 0.0), np.asarray(gradient, dtype=float)


def _log_positive_part_with_gradient(
    improvement: float,
    deviation: float,
    improvement_gradient: FloatArray,
    deviation_gradient: FloatArray,
) -> tuple[float, FloatArray]:
    """Log expected positive part and its gradient without tail underflow."""

    if deviation <= 1e-14:
        if improvement > 0.0:
            return (
                float(np.log(improvement)),
                np.asarray(improvement_gradient / improvement, dtype=float),
            )
        return -np.inf, np.zeros_like(improvement_gradient)
    z = float(improvement / deviation)
    log_cdf = float(log_ndtr(z))
    if z >= -8.0:
        density = _normal_density(z)
        scaled = max(z * float(ndtr(z)) + density, np.finfo(float).tiny)
        log_scaled = float(np.log(scaled))
    else:
        a = -z
        if a < 1.0e4:
            log_density = -0.5 * z * z - 0.5 * np.log(2.0 * np.pi)
            difference = np.exp(log_density - log_cdf) + z
            log_scaled = float(
                log_cdf + np.log(max(difference, np.finfo(float).tiny))
            )
        else:
            inverse_square = 1.0 / (a * a)
            correction = max(
                1.0 - 3.0 * inverse_square + 15.0 * inverse_square**2,
                np.finfo(float).tiny,
            )
            log_density = -0.5 * a * a - 0.5 * np.log(2.0 * np.pi)
            log_scaled = float(
                log_density - 2.0 * np.log(a) + np.log(correction)
            )
    log_value = float(np.log(deviation) + log_scaled)
    cdf_over_scaled_ei = float(np.exp(log_cdf - log_scaled))
    z_gradient = (
        improvement_gradient / deviation
        - z * deviation_gradient / deviation
    )
    gradient = (
        deviation_gradient / deviation + cdf_over_scaled_ei * z_gradient
    )
    return log_value, np.asarray(gradient, dtype=float)


def _top_start_indices(values: FloatArray, count: int) -> NDArray[np.int_]:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise RuntimeError("acquisition contains no finite value")
    count = min(int(count), finite.size)
    local = np.argpartition(values[finite], -count)[-count:]
    selected = finite[local]
    return selected[np.argsort(-values[selected], kind="stable")]


def _polish(
    universe: FloatArray,
    coarse_acquisition: FloatArray,
    objective,
    accept_candidate: Callable[[FloatArray], bool] | None = None,
    batched_objective=None,
) -> PolishResult:
    starts = _top_start_indices(coarse_acquisition, CONTINUOUS_POLISH_STARTS)
    coarse_index = int(starts[0])
    best_point = np.asarray(universe[coarse_index], dtype=float).copy()
    best_value = float(coarse_acquisition[coarse_index])
    evaluations = 0
    successful = 0
    bounds = [(0.0, 1.0)] * DIMENSION
    initial_values = None
    initial_gradients = None
    if batched_objective is not None:
        initial_values, initial_gradients = batched_objective(
            universe[starts], starts
        )
        initial_values = np.asarray(initial_values, dtype=float)
        initial_gradients = np.asarray(initial_gradients, dtype=float)
        if initial_values.shape != (starts.size,) or initial_gradients.shape != (
            starts.size,
            DIMENSION,
        ):
            raise ValueError("batched acquisition objective has invalid shape")
    for start_position, index in enumerate(starts):
        start = np.asarray(universe[int(index)], dtype=float)
        used_cached_start = False

        def cached_objective(point: FloatArray):
            nonlocal used_cached_start
            if (
                initial_values is not None
                and not used_cached_start
                and np.array_equal(point, start)
            ):
                used_cached_start = True
                return (
                    float(initial_values[start_position]),
                    initial_gradients[start_position].copy(),
                )
            return objective(point)

        result = minimize(
            cached_objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": CONTINUOUS_POLISH_MAXITER,
                "ftol": CONTINUOUS_POLISH_FTOL,
                "gtol": CONTINUOUS_POLISH_GTOL,
                "maxls": 12,
            },
        )
        evaluations += int(result.nfev)
        successful += int(bool(result.success))
        candidate = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        candidate_value = float(-result.fun)
        accepted = accept_candidate is None or accept_candidate(candidate)
        if accepted and np.isfinite(candidate_value) and candidate_value > best_value:
            best_value = candidate_value
            best_point = candidate
    coarse_value = float(coarse_acquisition[coarse_index])
    improved = bool(best_value > coarse_value + CONTINUOUS_POLISH_MINIMUM_GAIN)
    return PolishResult(
        point=best_point,
        acquisition=best_value,
        coarse_acquisition=coarse_value,
        improved=improved,
        function_evaluations=evaluations,
        successful_starts=successful,
    )


def polish_lqei(
    model: SequentialFixedKernelGP,
    coefficient: float,
    incumbent: float,
    coarse_acquisition: FloatArray,
) -> PolishResult:
    """Polish the best finite-domain LQEI candidates in the unit hypercube."""

    def objective(point: FloatArray) -> tuple[float, FloatArray]:
        mean, variance, mean_gradient, variance_gradient = (
            model.posterior_with_gradients(point)
        )
        deviation = float(np.sqrt(max(variance, 0.0)))
        if deviation > 1e-14:
            deviation_gradient = variance_gradient / (2.0 * deviation)
        else:
            deviation_gradient = np.zeros(DIMENSION, dtype=float)
        value, gradient = _positive_part_with_gradient(
            incumbent - mean,
            deviation,
            -mean_gradient,
            deviation_gradient,
        )
        return -value, -gradient

    def batched_objective(
        points: FloatArray, indices: NDArray[np.int_]
    ) -> tuple[FloatArray, FloatArray]:
        means, variances, mean_gradients, variance_gradients = (
            model.posterior_with_gradients_batch(points, universe_indices=indices)
        )
        values = np.empty(points.shape[0], dtype=float)
        gradients = np.empty((points.shape[0], DIMENSION), dtype=float)
        for position in range(points.shape[0]):
            deviation = float(np.sqrt(max(variances[position], 0.0)))
            if deviation > 1e-14:
                deviation_gradient = (
                    variance_gradients[position] / (2.0 * deviation)
                )
            else:
                deviation_gradient = np.zeros(DIMENSION, dtype=float)
            value, gradient = _positive_part_with_gradient(
                incumbent - means[position],
                deviation,
                -mean_gradients[position],
                deviation_gradient,
            )
            values[position] = -value
            gradients[position] = -gradient
        return values, gradients

    del coefficient  # The coefficient is already represented by the incumbent.
    return _polish(
        model.universe,
        coarse_acquisition,
        objective,
        batched_objective=batched_objective,
    )


def polish_lqei_log(
    model: SequentialFixedKernelGP,
    incumbent: float,
    coarse_log_acquisition: FloatArray,
) -> PolishResult:
    """Polish log LQEI when the ordinary-scale acquisition underflows."""

    def objective(point: FloatArray) -> tuple[float, FloatArray]:
        mean, variance, mean_gradient, variance_gradient = (
            model.posterior_with_gradients(point)
        )
        deviation = float(np.sqrt(max(variance, 0.0)))
        if deviation > 1e-14:
            deviation_gradient = variance_gradient / (2.0 * deviation)
        else:
            deviation_gradient = np.zeros(DIMENSION, dtype=float)
        value, gradient = _log_positive_part_with_gradient(
            incumbent - mean,
            deviation,
            -mean_gradient,
            deviation_gradient,
        )
        return -value, -gradient

    def batched_objective(
        points: FloatArray, indices: NDArray[np.int_]
    ) -> tuple[FloatArray, FloatArray]:
        means, variances, mean_gradients, variance_gradients = (
            model.posterior_with_gradients_batch(points, universe_indices=indices)
        )
        values = np.empty(points.shape[0], dtype=float)
        gradients = np.empty((points.shape[0], DIMENSION), dtype=float)
        for position in range(points.shape[0]):
            deviation = float(np.sqrt(max(variances[position], 0.0)))
            if deviation > 1e-14:
                deviation_gradient = (
                    variance_gradients[position] / (2.0 * deviation)
                )
            else:
                deviation_gradient = np.zeros(DIMENSION, dtype=float)
            value, gradient = _log_positive_part_with_gradient(
                incumbent - means[position],
                deviation,
                -mean_gradients[position],
                deviation_gradient,
            )
            values[position] = -value
            gradients[position] = -gradient
        return values, gradients

    return _polish(
        model.universe,
        coarse_log_acquisition,
        objective,
        batched_objective=batched_objective,
    )


def polish_sko(
    response_model: SequentialFixedKernelGP,
    noise_model: SequentialFixedKernelGP,
    effective_best: float,
    minimum_noise_sd: float,
    maximum_noise_sd: float,
    coarse_acquisition: FloatArray,
) -> PolishResult:
    """Polish the 4D SKO acquisition with response/noise GP gradients."""

    def objective(point: FloatArray) -> tuple[float, FloatArray]:
        mean, variance, mean_gradient, variance_gradient = (
            response_model.posterior_with_gradients(point)
        )
        deviation = float(np.sqrt(max(variance, 0.0)))
        if deviation > 1e-14:
            deviation_gradient = variance_gradient / (2.0 * deviation)
        else:
            deviation_gradient = np.zeros(DIMENSION, dtype=float)
        expected_improvement, ei_gradient = _positive_part_with_gradient(
            effective_best - mean,
            deviation,
            -mean_gradient,
            deviation_gradient,
        )

        noise_mean, _, noise_mean_gradient, _ = (
            noise_model.posterior_with_gradients(point)
        )
        noise_sd = float(np.clip(noise_mean, minimum_noise_sd, maximum_noise_sd))
        if minimum_noise_sd < noise_mean < maximum_noise_sd:
            noise_sd_gradient = noise_mean_gradient
        else:
            noise_sd_gradient = np.zeros(DIMENSION, dtype=float)
        noise_variance = noise_sd**2
        noise_variance_gradient = 2.0 * noise_sd * noise_sd_gradient
        total = max(variance + noise_variance, np.finfo(float).eps)
        penalty = float(1.0 - noise_sd / np.sqrt(total))
        penalty_gradient = (
            -noise_sd_gradient / np.sqrt(total)
            + 0.5
            * noise_sd
            * (variance_gradient + noise_variance_gradient)
            / total**1.5
        )
        value = max(expected_improvement * penalty, 0.0)
        gradient = penalty * ei_gradient + expected_improvement * penalty_gradient
        return -value, -np.asarray(gradient, dtype=float)

    return _polish(response_model.universe, coarse_acquisition, objective)


def polish_eqi(
    response_model: SequentialFixedKernelGP,
    noise_model: SequentialFixedKernelGP,
    quantile_minimum: float,
    planned_replications: int,
    minimum_noise_sd: float,
    maximum_noise_sd: float,
    coarse_acquisition: FloatArray,
) -> PolishResult:
    """Polish EQI with exact response and auxiliary-noise GP gradients."""

    def objective(point: FloatArray) -> tuple[float, FloatArray]:
        mean, mse, mean_gradient, mse_gradient = (
            response_model.posterior_with_gradients(point)
        )
        noise_mean, _, noise_mean_gradient, _ = (
            noise_model.posterior_with_gradients(point)
        )
        predicted_sd = float(
            np.clip(noise_mean, minimum_noise_sd, maximum_noise_sd)
        )
        if minimum_noise_sd < noise_mean < maximum_noise_sd:
            noise_variance_gradient = 2.0 * predicted_sd * noise_mean_gradient
        else:
            noise_variance_gradient = np.zeros(DIMENSION, dtype=float)
        new_mean_noise = predicted_sd**2 / planned_replications
        total = max(mse + new_mean_noise, np.finfo(float).eps)
        deviation = float(mse / np.sqrt(total))
        total_gradient = (
            mse_gradient + noise_variance_gradient / planned_replications
        )
        deviation_gradient = (
            mse_gradient / np.sqrt(total)
            - 0.5 * mse * total_gradient / total**1.5
        )
        value, gradient = _positive_part_with_gradient(
            quantile_minimum - mean,
            deviation,
            -mean_gradient,
            deviation_gradient,
        )
        return -value, -gradient

    return _polish(response_model.universe, coarse_acquisition, objective)


def polish_tsso(
    screening_points: FloatArray,
    coarse_acquisition: FloatArray,
    incumbent: float,
    response_mean_with_gradient: Callable[
        [FloatArray], tuple[float, FloatArray]
    ],
    deterministic_model: SequentialFixedKernelGP,
    excluded_points: FloatArray,
) -> PolishResult:
    """Polish TSSO's own EI-like search rule over the continuous domain.

    TSSO combines the stochastic-Kriging posterior mean with deterministic
    Kriging uncertainty. The OCBA allocation that follows the search step is
    intentionally not changed by this optimizer.
    """

    excluded = np.asarray(excluded_points, dtype=float).reshape(-1, DIMENSION)

    def objective(point: FloatArray) -> tuple[float, FloatArray]:
        mean, mean_gradient = response_mean_with_gradient(point)
        _, variance, _, variance_gradient = (
            deterministic_model.posterior_with_gradients(point)
        )
        deviation = float(np.sqrt(max(variance, 0.0)))
        if deviation > 1e-14:
            deviation_gradient = variance_gradient / (2.0 * deviation)
        else:
            deviation_gradient = np.zeros(DIMENSION, dtype=float)
        value, gradient = _positive_part_with_gradient(
            incumbent - mean,
            deviation,
            -mean_gradient,
            deviation_gradient,
        )
        return -value, -gradient

    def is_new_design(point: FloatArray) -> bool:
        if excluded.size == 0:
            return True
        distances = np.max(np.abs(excluded - point), axis=1)
        return bool(np.all(distances > 1e-10))

    return _polish(
        np.asarray(screening_points, dtype=float),
        np.asarray(coarse_acquisition, dtype=float),
        objective,
        accept_candidate=is_new_design,
    )
