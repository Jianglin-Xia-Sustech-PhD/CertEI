"""Fast sequential exact GP updates with fixed kernel shape parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar

from rkhs_functions import KernelName, kernel_matrix


FloatArray = NDArray[np.float64]


def fit_initial_process_parameters(
    points: ArrayLike,
    outputs: ArrayLike,
    mean_variances: ArrayLike,
    kernel: KernelName,
    lengthscale: float,
) -> tuple[float, float]:
    """Profile a constant mean and process variance on the initial design only."""

    x = np.asarray(points, dtype=float)
    y = np.asarray(outputs, dtype=float).reshape(-1)
    noise = np.asarray(mean_variances, dtype=float).reshape(-1)
    correlation = kernel_matrix(x, x, kernel, lengthscale)
    one = np.ones(y.size)
    scale = max(float(np.var(y, ddof=1)), float(np.median(noise)), 1e-6)
    lower = np.log(scale * 1e-4)
    upper = np.log(scale * 1e2)

    def profile(log_variance: float, return_mean: bool = False) -> float:
        process_variance = float(np.exp(log_variance))
        covariance = process_variance * correlation + np.diag(noise)
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inverse_y = cho_solve(factor, y, check_finite=False)
            inverse_one = cho_solve(factor, one, check_finite=False)
        except np.linalg.LinAlgError:
            return float("inf")
        denominator = float(one @ inverse_one)
        if denominator <= 0.0:
            return float("inf")
        beta = float(one @ inverse_y / denominator)
        if return_mean:
            return beta
        residual = y - beta
        quadratic = float(residual @ cho_solve(factor, residual, check_finite=False))
        log_determinant = float(2.0 * np.sum(np.log(np.diag(factor[0]))))
        return log_determinant + quadratic

    result = minimize_scalar(
        profile,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 200},
    )
    selected_log_variance = float(result.x if result.success else np.log(scale))
    process_variance = float(np.exp(selected_log_variance))
    beta = float(profile(selected_log_variance, return_mean=True))
    return process_variance, beta


@dataclass
class SequentialFixedKernelGP:
    """Exact GP posterior over one fixed finite candidate universe.

    Cholesky rows and their projections onto the candidate universe are added
    incrementally. Posterior means and marginal variances are therefore updated
    without refitting hyperparameters or resolving the full GP at each step.
    """

    universe: FloatArray
    kernel: KernelName
    lengthscale: float
    process_variance: float
    constant_mean: float
    maximum_observations: int

    def __post_init__(self) -> None:
        self.universe = np.asarray(self.universe, dtype=float)
        if self.universe.ndim != 2 or self.universe.shape[1] != 2:
            raise ValueError("universe must have shape (m, 2)")
        if self.maximum_observations < 1:
            raise ValueError("maximum_observations must be positive")
        if self.process_variance <= 0.0:
            raise ValueError("process_variance must be positive")
        universe_size = self.universe.shape[0]
        self.posterior_mean = np.full(universe_size, self.constant_mean, dtype=float)
        self.posterior_variance = np.full(
            universe_size, self.process_variance, dtype=float
        )
        # Long O(1) schedules can contain ten thousand batches.  Capping the
        # active factor and periodically rebuilding from exact same-site
        # sufficient statistics avoids both cubic growth and roundoff drift.
        self._projection_capacity = min(
            self.maximum_observations,
            max(universe_size + 64, 3000),
        )
        self._projections = np.zeros(
            (self._projection_capacity, universe_size), dtype=float
        )
        self._training_indices = np.empty(self.maximum_observations, dtype=int)
        self._active_indices = np.empty(self._projection_capacity, dtype=int)
        self._precision_sum = np.zeros(universe_size, dtype=float)
        self._precision_weighted_sum = np.zeros(universe_size, dtype=float)
        self._active_count = 0
        self._observation_count = 0

    @property
    def count(self) -> int:
        return self._observation_count

    @property
    def training_indices(self) -> NDArray[np.int_]:
        return self._training_indices[: self._observation_count].copy()

    def _rebuild_from_sufficient_statistics(self) -> None:
        """Rebuild the exact posterior using one equivalent row per site."""

        indices = np.flatnonzero(self._precision_sum > 0.0)
        if indices.size >= self._projection_capacity:
            raise RuntimeError("projection capacity cannot hold unique sampled sites")
        observed_means = (
            self._precision_weighted_sum[indices] / self._precision_sum[indices]
        )
        noise_variances = 1.0 / self._precision_sum[indices]
        training_points = self.universe[indices]
        covariance = self.process_variance * kernel_matrix(
            training_points,
            training_points,
            self.kernel,
            self.lengthscale,
        )
        covariance.flat[:: indices.size + 1] += noise_variances
        cholesky = np.linalg.cholesky(covariance)
        cross = self.process_variance * kernel_matrix(
            training_points,
            self.universe,
            self.kernel,
            self.lengthscale,
        )
        projections = solve_triangular(
            cholesky,
            cross,
            lower=True,
            check_finite=False,
        )
        centered = observed_means - self.constant_mean
        transformed = solve_triangular(
            cholesky,
            centered,
            lower=True,
            check_finite=False,
        )
        self.posterior_mean = self.constant_mean + projections.T @ transformed
        self.posterior_variance = np.maximum(
            self.process_variance - np.sum(projections**2, axis=0),
            0.0,
        )
        self._projections[: indices.size] = projections
        self._active_indices[: indices.size] = indices
        self._active_count = int(indices.size)

    def add_observation(
        self, universe_index: int, observed_mean: float, mean_noise_variance: float
    ) -> None:
        if self._observation_count >= self.maximum_observations:
            raise RuntimeError("the GP observation capacity has been exhausted")
        if not 0 <= universe_index < self.universe.shape[0]:
            raise IndexError("universe_index is out of bounds")
        if not np.isfinite(mean_noise_variance) or mean_noise_variance <= 0.0:
            raise ValueError("mean_noise_variance must be finite and positive")
        if self._active_count >= self._projection_capacity:
            self._rebuild_from_sufficient_statistics()
        n = self._active_count
        selected = self.universe[universe_index : universe_index + 1]
        prior_cross = self.process_variance * kernel_matrix(
            self.universe, selected, self.kernel, self.lengthscale
        ).reshape(-1)
        if n == 0:
            conditional_cross = prior_cross
            schur = self.process_variance + mean_noise_variance
        else:
            # Each stored projection row is one row of L^{-1} K(X, universe).
            # Its column at the selected candidate is therefore exactly the
            # triangular solve L^{-1} K(X, selected) used by the conventional
            # incremental-Cholesky formula. Reusing that column avoids an
            # identical O(n^2) solve while preserving the exact GP update.
            triangular = self._projections[:n, universe_index]
            conditional_cross = (
                prior_cross - self._projections[:n].T @ triangular
            )
            schur = (
                self.process_variance
                + mean_noise_variance
                - float(triangular @ triangular)
            )
        numerical_floor = max(
            1e-12 * (self.process_variance + mean_noise_variance), 1e-14
        )
        schur = max(float(schur), numerical_floor)
        residual = float(observed_mean - self.posterior_mean[universe_index])
        self.posterior_mean += conditional_cross * (residual / schur)
        self.posterior_variance = np.maximum(
            self.posterior_variance - conditional_cross**2 / schur, 0.0
        )
        diagonal = float(np.sqrt(schur))
        self._projections[n] = conditional_cross / diagonal
        self._active_indices[n] = int(universe_index)
        self._training_indices[self._observation_count] = int(universe_index)
        precision = 1.0 / mean_noise_variance
        self._precision_sum[universe_index] += precision
        self._precision_weighted_sum[universe_index] += precision * observed_mean
        self._active_count += 1
        self._observation_count += 1
