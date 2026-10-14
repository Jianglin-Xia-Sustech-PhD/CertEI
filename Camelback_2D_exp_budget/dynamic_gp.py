"""Dynamic maximum-likelihood stochastic Kriging with a Matérn-5/2 kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from study_config import LENGTHSCALE_BOUNDS, MATERN_NU


FloatArray = NDArray[np.float64]


def _as_points(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return array


def matern52_correlation(
    left: ArrayLike,
    right: ArrayLike,
    lengthscales: ArrayLike,
) -> FloatArray:
    """Anisotropic Matérn-5/2 correlation on the normalized domain."""

    x = _as_points(left)
    y = _as_points(right)
    ell = np.asarray(lengthscales, dtype=float).reshape(-1)
    if ell.shape != (2,) or np.any(~np.isfinite(ell)) or np.any(ell <= 0.0):
        raise ValueError("lengthscales must be two finite positive values")
    scaled = (x[:, np.newaxis, :] - y[np.newaxis, :, :]) / ell
    distance = np.sqrt(np.sum(scaled**2, axis=2))
    root_five_distance = np.sqrt(5.0) * distance
    return (
        1.0 + root_five_distance + root_five_distance**2 / 3.0
    ) * np.exp(-root_five_distance)


@dataclass(frozen=True)
class GPHyperparameters:
    constant_mean: float
    process_variance: float
    lengthscales: FloatArray
    negative_log_likelihood: float
    optimizer_success: bool
    optimizer_iterations: int
    regularization_penalty: float = 0.0
    penalized_objective: float = np.nan

    @property
    def log_optimization_parameters(self) -> FloatArray:
        return np.log(
            np.concatenate(([self.process_variance], self.lengthscales))
        )


@dataclass(frozen=True)
class DynamicGPFit:
    posterior_mean: FloatArray
    posterior_variance: FloatArray
    hyperparameters: GPHyperparameters


def predict_with_hyperparameters(
    points: ArrayLike,
    outputs: ArrayLike,
    mean_noise_variances: ArrayLike,
    prediction_points: ArrayLike,
    hyperparameters: GPHyperparameters,
    *,
    include_mean_estimation_variance: bool = True,
) -> DynamicGPFit:
    """Build a posterior with supplied fitted parameters and no reoptimization."""

    x = _as_points(points)
    prediction = _as_points(prediction_points)
    y = np.asarray(outputs, dtype=float).reshape(-1)
    noise = np.asarray(mean_noise_variances, dtype=float).reshape(-1)
    one = np.ones(y.size)
    covariance = (
        hyperparameters.process_variance
        * matern52_correlation(x, x, hyperparameters.lengthscales)
        + np.diag(noise)
    )
    factor = _stable_factor(covariance)
    residual = y - hyperparameters.constant_mean
    alpha = cho_solve(factor, residual, check_finite=False)
    cross = hyperparameters.process_variance * matern52_correlation(
        x, prediction, hyperparameters.lengthscales
    )
    posterior_mean = hyperparameters.constant_mean + cross.T @ alpha
    triangular = solve_triangular(factor[0], cross, lower=True, check_finite=False)
    posterior_variance = hyperparameters.process_variance - np.sum(
        triangular**2, axis=0
    )
    if include_mean_estimation_variance:
        inverse_one = cho_solve(factor, one, check_finite=False)
        mean_information = float(one @ inverse_one)
        # C is symmetric, so 1^T C^-1 K = (C^-1 1)^T K.  Reuse the
        # already-computed inverse_one instead of solving once per candidate.
        mean_residual = 1.0 - inverse_one @ cross
        posterior_variance += mean_residual**2 / mean_information
    return DynamicGPFit(
        posterior_mean=np.asarray(posterior_mean, dtype=float),
        posterior_variance=np.maximum(posterior_variance, 0.0),
        hyperparameters=hyperparameters,
    )


def _stable_factor(matrix: FloatArray) -> tuple[FloatArray, bool]:
    scale = max(float(np.max(np.diag(matrix))), 1.0)
    identity = np.eye(matrix.shape[0])
    last_error: Exception | None = None
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return cho_factor(
                matrix + relative_jitter * scale * identity,
                lower=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError as error:
            last_error = error
    raise np.linalg.LinAlgError("covariance factorization failed") from last_error


def fit_dynamic_gp(
    points: ArrayLike,
    outputs: ArrayLike,
    mean_noise_variances: ArrayLike,
    prediction_points: ArrayLike,
    *,
    previous_log_parameters: ArrayLike | None = None,
    process_variance_log_prior_mean: float | None = None,
    process_variance_log_prior_sd: float | None = None,
    lengthscale_log_prior_mean: ArrayLike | None = None,
    lengthscale_log_prior_sd: float | None = None,
    max_iterations: int = 20,
) -> DynamicGPFit:
    """Estimate beta, process variance and two lengthscales by MLE/MAP.

    The constant mean is profiled by generalized least squares.  Known local
    variances of the sample means enter as the diagonal intrinsic-noise term.
    Analytic covariance gradients keep the every-stage refits practical.  If
    a log-lengthscale prior is supplied, a weak isotropic Gaussian penalty is
    added to the log process variance and the two log-lengthscales; the
    constant mean stays unregularized.
    """

    x = _as_points(points)
    prediction = _as_points(prediction_points)
    y = np.asarray(outputs, dtype=float).reshape(-1)
    noise = np.asarray(mean_noise_variances, dtype=float).reshape(-1)
    if x.shape[0] < 2 or y.size != x.shape[0] or noise.size != x.shape[0]:
        raise ValueError("GP training arrays have incompatible sizes")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("GP outputs and noise variances must be finite and positive")

    if lengthscale_log_prior_mean is None:
        if lengthscale_log_prior_sd is not None:
            raise ValueError("lengthscale_log_prior_sd requires a prior mean")
        prior_mean = None
        prior_sd = None
    else:
        prior_mean = np.asarray(lengthscale_log_prior_mean, dtype=float).reshape(-1)
        if prior_mean.shape != (2,) or np.any(~np.isfinite(prior_mean)):
            raise ValueError("lengthscale_log_prior_mean must contain two values")
        if lengthscale_log_prior_sd is None or not np.isfinite(lengthscale_log_prior_sd):
            raise ValueError("a finite positive lengthscale_log_prior_sd is required")
        prior_sd = float(lengthscale_log_prior_sd)
        if prior_sd <= 0.0:
            raise ValueError("lengthscale_log_prior_sd must be positive")

    if process_variance_log_prior_mean is None:
        if process_variance_log_prior_sd is not None:
            raise ValueError(
                "process_variance_log_prior_sd requires a prior mean"
            )
        variance_prior_mean = None
        variance_prior_sd = None
    else:
        variance_prior_mean = float(process_variance_log_prior_mean)
        if not np.isfinite(variance_prior_mean):
            raise ValueError("process_variance_log_prior_mean must be finite")
        if (
            process_variance_log_prior_sd is None
            or not np.isfinite(process_variance_log_prior_sd)
        ):
            raise ValueError(
                "a finite positive process_variance_log_prior_sd is required"
            )
        variance_prior_sd = float(process_variance_log_prior_sd)
        if variance_prior_sd <= 0.0:
            raise ValueError("process_variance_log_prior_sd must be positive")

    differences = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    one = np.ones(y.size)
    identity = np.eye(y.size)
    output_variance = max(float(np.var(y, ddof=1)), float(np.median(noise)), 1e-8)
    variance_lower = max(output_variance * 1e-5, 1e-10)
    variance_upper = max(output_variance * 1e4, variance_lower * 100.0)
    length_lower, length_upper = LENGTHSCALE_BOUNDS
    bounds = [
        (np.log(variance_lower), np.log(variance_upper)),
        (np.log(length_lower), np.log(length_upper)),
        (np.log(length_lower), np.log(length_upper)),
    ]
    if previous_log_parameters is None:
        initial = np.log(np.asarray([output_variance, 0.25, 0.25]))
    else:
        initial = np.asarray(previous_log_parameters, dtype=float).reshape(-1)
        if initial.shape != (3,) or np.any(~np.isfinite(initial)):
            raise ValueError("previous_log_parameters must contain three values")
    initial = np.asarray(
        [np.clip(value, lower, upper) for value, (lower, upper) in zip(initial, bounds)],
        dtype=float,
    )

    def evaluate(parameters: FloatArray) -> tuple[float, FloatArray]:
        process_variance = float(np.exp(parameters[0]))
        lengthscales = np.exp(parameters[1:])
        scaled_squared = (differences / lengthscales) ** 2
        squared_distance = np.sum(scaled_squared, axis=2)
        distance = np.sqrt(squared_distance)
        root_five_distance = np.sqrt(5.0) * distance
        exponential = np.exp(-root_five_distance)
        correlation = (
            1.0 + root_five_distance + root_five_distance**2 / 3.0
        ) * exponential
        covariance = process_variance * correlation + np.diag(noise)
        try:
            factor = _stable_factor(covariance)
            inverse_y = cho_solve(factor, y, check_finite=False)
            inverse_one = cho_solve(factor, one, check_finite=False)
            denominator = float(one @ inverse_one)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise np.linalg.LinAlgError("invalid generalized-mean information")
            beta = float(one @ inverse_y / denominator)
            residual = y - beta
            alpha = cho_solve(factor, residual, check_finite=False)
            inverse = cho_solve(factor, identity, check_finite=False)
            log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
            nll = 0.5 * (
                log_determinant
                + float(residual @ alpha)
                + y.size * np.log(2.0 * np.pi)
            )
            score_matrix = inverse - np.outer(alpha, alpha)
            derivatives = [process_variance * correlation]
            for dimension in range(2):
                q_dimension = scaled_squared[:, :, dimension]
                derivative_correlation = (
                    5.0
                    / 3.0
                    * (1.0 + root_five_distance)
                    * exponential
                    * q_dimension
                )
                derivatives.append(process_variance * derivative_correlation)
            gradient = 0.5 * np.asarray(
                [np.sum(score_matrix * derivative) for derivative in derivatives]
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e100, np.zeros(3)
        if prior_mean is not None and prior_sd is not None:
            standardized = (parameters[1:] - prior_mean) / prior_sd
            nll += 0.5 * float(standardized @ standardized)
            gradient[1:] += (parameters[1:] - prior_mean) / prior_sd**2
        if variance_prior_mean is not None and variance_prior_sd is not None:
            standardized_variance = (
                parameters[0] - variance_prior_mean
            ) / variance_prior_sd
            nll += 0.5 * float(standardized_variance**2)
            gradient[0] += (
                parameters[0] - variance_prior_mean
            ) / variance_prior_sd**2
        if not np.isfinite(nll) or np.any(~np.isfinite(gradient)):
            return 1e100, np.zeros(3)
        return float(nll), np.asarray(gradient, dtype=float)

    result = minimize(
        evaluate,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(max_iterations), "maxls": 20, "ftol": 1e-8},
    )
    selected = np.asarray(result.x if np.all(np.isfinite(result.x)) else initial)
    selected_nll, _ = evaluate(selected)
    initial_nll, _ = evaluate(initial)
    if initial_nll < selected_nll:
        selected = initial
        selected_nll = initial_nll
    process_variance = float(np.exp(selected[0]))
    lengthscales = np.exp(selected[1:])
    regularization_penalty = 0.0
    if prior_mean is not None and prior_sd is not None:
        standardized = (selected[1:] - prior_mean) / prior_sd
        regularization_penalty += 0.5 * float(standardized @ standardized)
    if variance_prior_mean is not None and variance_prior_sd is not None:
        standardized_variance = (
            selected[0] - variance_prior_mean
        ) / variance_prior_sd
        regularization_penalty += 0.5 * float(standardized_variance**2)
    data_negative_log_likelihood = float(selected_nll - regularization_penalty)

    correlation = matern52_correlation(x, x, lengthscales)
    covariance = process_variance * correlation + np.diag(noise)
    factor = _stable_factor(covariance)
    inverse_y = cho_solve(factor, y, check_finite=False)
    inverse_one = cho_solve(factor, one, check_finite=False)
    mean_information = float(one @ inverse_one)
    beta = float(one @ inverse_y / mean_information)
    residual = y - beta
    alpha = cho_solve(factor, residual, check_finite=False)
    cross = process_variance * matern52_correlation(x, prediction, lengthscales)
    posterior_mean = beta + cross.T @ alpha
    triangular = solve_triangular(
        factor[0], cross, lower=True, check_finite=False
    )
    mean_residual = 1.0 - inverse_one @ cross
    posterior_variance = (
        process_variance
        - np.sum(triangular**2, axis=0)
        + mean_residual**2 / mean_information
    )
    posterior_variance = np.maximum(posterior_variance, 0.0)
    return DynamicGPFit(
        posterior_mean=np.asarray(posterior_mean, dtype=float),
        posterior_variance=np.asarray(posterior_variance, dtype=float),
        hyperparameters=GPHyperparameters(
            constant_mean=beta,
            process_variance=process_variance,
            lengthscales=np.asarray(lengthscales, dtype=float),
            negative_log_likelihood=data_negative_log_likelihood,
            optimizer_success=bool(result.success),
            optimizer_iterations=int(getattr(result, "nit", 0)),
            regularization_penalty=regularization_penalty,
            penalized_objective=float(selected_nll),
        ),
    )


def approximate_rkhs_norm(
    points: ArrayLike,
    values: ArrayLike,
    lengthscales: ArrayLike,
    process_variance: float,
) -> tuple[float, float]:
    """Return the norm under the same scaled Matérn kernel used by the GP."""

    x = _as_points(points)
    y = np.asarray(values, dtype=float).reshape(-1)
    variance = float(process_variance)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("process_variance must be finite and positive")
    covariance = variance * matern52_correlation(x, x, lengthscales)
    regularization = 1e-8 * variance * np.eye(x.shape[0])
    factor = _stable_factor(covariance + regularization)
    inverse_y = cho_solve(factor, y, check_finite=False)
    inverse_one = cho_solve(factor, np.ones(y.size), check_finite=False)
    intercept = float(np.sum(inverse_y) / np.sum(inverse_one))
    centered = y - intercept
    squared_norm = float(centered @ cho_solve(factor, centered, check_finite=False))
    return float(np.sqrt(max(squared_norm, 0.0))), intercept
