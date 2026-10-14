"""Fast sequential exact GP updates with fixed kernel shape parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar

from rkhs_functions import KernelName, kernel_matrix
from study_config import DIMENSION


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
    """Exact GP posterior over an extensible candidate universe.

    Cholesky rows and their projections onto the candidate universe are added
    incrementally. Posterior means and marginal variances are therefore updated
    without refitting hyperparameters or resolving the full GP at each step.
    Arbitrary continuous queries use the same Cholesky factor, and polished
    points can be appended without rebuilding projections for existing points.
    """

    universe: FloatArray
    kernel: KernelName
    lengthscale: float
    process_variance: float
    constant_mean: float
    maximum_observations: int

    def __post_init__(self) -> None:
        self.universe = np.asarray(self.universe, dtype=float)
        if self.universe.ndim != 2 or self.universe.shape[1] != DIMENSION:
            raise ValueError(f"universe must have shape (m, {DIMENSION})")
        if self.maximum_observations < 1:
            raise ValueError("maximum_observations must be positive")
        if self.process_variance <= 0.0:
            raise ValueError("process_variance must be positive")
        universe_size = self.universe.shape[0]
        self.posterior_mean = np.full(universe_size, self.constant_mean, dtype=float)
        self.posterior_variance = np.full(
            universe_size, self.process_variance, dtype=float
        )
        # The log-log replication schedules can contain more than six thousand
        # batches.  Retaining one permanent Cholesky row for every batch would
        # use over a gigabyte per worker even though many batches revisit a
        # previously sampled site.  Periodic exact rebuilding replaces all
        # independent observations at the same site by their precision-weighted
        # Gaussian sufficient statistic.  This changes no GP posterior.
        self._row_capacity = min(self.maximum_observations, 2048)
        # Reserve modest headroom for polished continuous candidates and grow
        # only if one is actually appended.  Most LQEI paths retain the fixed
        # Sobol universe, so reserving one column per future batch needlessly
        # multiplies the projection matrix footprint on long log-log paths.
        self._column_capacity = universe_size + min(
            self.maximum_observations, 1024
        )
        self._cholesky = np.zeros(
            (self._row_capacity, self._row_capacity), dtype=float
        )
        self._projections = np.zeros(
            (self._row_capacity, self._column_capacity), dtype=float
        )
        self._training_indices = np.empty(self.maximum_observations, dtype=int)
        self._active_indices = np.empty(self._row_capacity, dtype=int)
        self._active_observed_means = np.empty(self._row_capacity, dtype=float)
        self._precision_sum = np.zeros(self._column_capacity, dtype=float)
        self._precision_weighted_sum = np.zeros(
            self._column_capacity, dtype=float
        )
        self._posterior_weight_cache = np.empty(0, dtype=float)
        self._posterior_weight_cache_count = -1
        self._factor_cache = np.empty((0, 0), dtype=float, order="F")
        self._factor_cache_count = -1
        self._active_count = 0
        self._observation_count = 0

    def _factor(self) -> FloatArray:
        """Share one LAPACK-ready factor copy across all solves in a round."""

        n = self._active_count
        if self._factor_cache_count != n:
            self._factor_cache = np.array(
                self._cholesky[:n, :n], dtype=float, order="F", copy=True
            )
            self._factor_cache_count = n
        return self._factor_cache

    def _ensure_column_capacity(self, required: int) -> None:
        if required <= self._column_capacity:
            return
        maximum = self.universe.shape[0] + (
            self.maximum_observations - self._observation_count
        )
        target = min(
            maximum,
            max(required, int(np.ceil(1.5 * self._column_capacity))),
        )
        if target < required:
            raise RuntimeError("continuous candidate capacity has been exhausted")
        projections = np.zeros((self._row_capacity, target), dtype=float)
        projections[:, : self._column_capacity] = self._projections
        precision_sum = np.zeros(target, dtype=float)
        precision_sum[: self._column_capacity] = self._precision_sum
        precision_weighted_sum = np.zeros(target, dtype=float)
        precision_weighted_sum[: self._column_capacity] = (
            self._precision_weighted_sum
        )
        self._projections = projections
        self._precision_sum = precision_sum
        self._precision_weighted_sum = precision_weighted_sum
        self._column_capacity = target

    @property
    def count(self) -> int:
        return self._observation_count

    @property
    def training_indices(self) -> NDArray[np.int_]:
        return self._training_indices[: self._observation_count].copy()

    def _ensure_row_capacity(self, required: int) -> None:
        if required <= self._row_capacity:
            return
        target = min(
            self.maximum_observations,
            max(required, int(np.ceil(1.5 * self._row_capacity))),
        )
        if target < required:
            raise RuntimeError("active GP factor capacity cannot be expanded")
        cholesky = np.zeros((target, target), dtype=float)
        cholesky[: self._active_count, : self._active_count] = self._cholesky[
            : self._active_count, : self._active_count
        ]
        projections = np.zeros((target, self._column_capacity), dtype=float)
        projections[: self._active_count] = self._projections[: self._active_count]
        active_indices = np.empty(target, dtype=int)
        active_indices[: self._active_count] = self._active_indices[
            : self._active_count
        ]
        active_means = np.empty(target, dtype=float)
        active_means[: self._active_count] = self._active_observed_means[
            : self._active_count
        ]
        self._cholesky = cholesky
        self._projections = projections
        self._active_indices = active_indices
        self._active_observed_means = active_means
        self._row_capacity = target

    def _rebuild_from_sufficient_statistics(self) -> None:
        """Rebuild using one exactly equivalent Gaussian row per sampled site."""

        indices = np.flatnonzero(
            self._precision_sum[: self.universe.shape[0]] > 0.0
        )
        self._ensure_row_capacity(int(indices.size) + 1)
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
        transformed = solve_triangular(
            cholesky,
            observed_means - self.constant_mean,
            lower=True,
            check_finite=False,
        )
        self.posterior_mean = self.constant_mean + projections.T @ transformed
        self.posterior_variance = np.maximum(
            self.process_variance - np.sum(projections**2, axis=0), 0.0
        )
        n = int(indices.size)
        self._cholesky.fill(0.0)
        self._cholesky[:n, :n] = cholesky
        self._projections[:n].fill(0.0)
        self._projections[:n, : self.universe.shape[0]] = projections
        self._active_indices[:n] = indices
        self._active_observed_means[:n] = observed_means
        self._active_count = n
        self._posterior_weight_cache_count = -1
        self._factor_cache_count = -1
        # Once duplicate observations have been compressed, leave enough room
        # before the next rebuild.  Without this adaptive headroom a nearly
        # full factor can trigger an expensive O(n^3) rebuild every few batches.
        if (
            self._row_capacity < self.maximum_observations
            and self._row_capacity - n < 1024
        ):
            self._ensure_row_capacity(
                min(
                    self.maximum_observations,
                    max(2 * self._row_capacity, n + 1024),
                )
            )

    def _posterior_weights(self) -> FloatArray:
        """Return the exact coefficient vector for continuous GP queries."""

        n = self._active_count
        if n == 0:
            return np.empty(0, dtype=float)
        if self._posterior_weight_cache_count == n:
            return self._posterior_weight_cache
        residual = self._active_observed_means[:n] - self.constant_mean
        self._posterior_weight_cache = cho_solve(
            (self._factor(), True),
            residual,
            check_finite=False,
        )
        self._posterior_weight_cache_count = n
        return self._posterior_weight_cache

    def posterior_at(self, points: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evaluate posterior mean and marginal variance at arbitrary points."""

        queries = np.asarray(points, dtype=float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != DIMENSION:
            raise ValueError(f"points must have shape (m, {DIMENSION})")
        n = self._active_count
        if n == 0:
            return (
                np.full(queries.shape[0], self.constant_mean, dtype=float),
                np.full(queries.shape[0], self.process_variance, dtype=float),
            )
        training = self.universe[self._active_indices[:n]]
        cross = self.process_variance * kernel_matrix(
            queries, training, self.kernel, self.lengthscale
        )
        mean = self.constant_mean + cross @ self._posterior_weights()
        triangular = solve_triangular(
            self._factor(),
            cross.T,
            lower=True,
            check_finite=False,
        )
        variance = np.maximum(
            self.process_variance - np.sum(triangular**2, axis=0),
            0.0,
        )
        return np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)

    def posterior_with_gradients(
        self, point: ArrayLike
    ) -> tuple[float, float, FloatArray, FloatArray]:
        """Evaluate a single posterior query and its coordinate gradients."""

        query = np.asarray(point, dtype=float).reshape(-1)
        if query.shape != (DIMENSION,):
            raise ValueError(f"point must have shape ({DIMENSION},)")
        means, variances, mean_gradients, variance_gradients = (
            self.posterior_with_gradients_batch(query.reshape(1, -1))
        )
        return (
            float(means[0]),
            float(variances[0]),
            mean_gradients[0],
            variance_gradients[0],
        )

    def posterior_with_gradients_batch(
        self, points: ArrayLike, *, universe_indices: ArrayLike | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate posterior values and gradients for several query points.

        Continuous acquisition polishing always starts from several points.
        Solving their triangular systems as one matrix operation is exactly the
        same GP calculation and is substantially faster on long paths.
        """

        queries = np.asarray(points, dtype=float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != DIMENSION:
            raise ValueError(f"points must have shape (m, {DIMENSION})")
        n = self._active_count
        if n == 0:
            zeros = np.zeros((queries.shape[0], DIMENSION), dtype=float)
            return (
                np.full(queries.shape[0], self.constant_mean, dtype=float),
                np.full(queries.shape[0], self.process_variance, dtype=float),
                zeros,
                zeros.copy(),
            )
        training = self.universe[self._active_indices[:n]]
        differences = queries[:, np.newaxis, :] - training[np.newaxis, :, :]
        radius_squared = np.sum(differences**2, axis=2) / self.lengthscale**2
        if self.kernel == "squared_exponential":
            values = np.exp(-0.5 * radius_squared)
            gradients = (
                -values[:, :, np.newaxis]
                * differences
                / self.lengthscale**2
            )
        elif self.kernel == "matern52":
            scaled = np.sqrt(5.0 * np.maximum(radius_squared, 0.0))
            exponential = np.exp(-scaled)
            values = (1.0 + scaled + scaled**2 / 3.0) * exponential
            gradients = (
                -5.0
                * (1.0 + scaled)[:, :, np.newaxis]
                * exponential[:, :, np.newaxis]
                * differences
                / (3.0 * self.lengthscale**2)
            )
        else:
            raise ValueError(
                "kernel must be 'matern52' or 'squared_exponential'"
            )
        cross = self.process_variance * values
        cross_gradient = self.process_variance * gradients
        weights = self._posterior_weights()
        means = self.constant_mean + cross @ weights
        mean_gradients = np.einsum(
            "mnd,n->md", cross_gradient, weights, optimize=True
        )
        if universe_indices is None:
            inverse_cross = cho_solve(
                (self._factor(), True),
                cross.T,
                check_finite=False,
            )
        else:
            indices = np.asarray(universe_indices, dtype=int).reshape(-1)
            if indices.size != queries.shape[0] or not np.array_equal(
                self.universe[indices], queries
            ):
                raise ValueError("universe indices do not match query points")
            # L^{-1} k(X, x) is cached. Only the upper-triangular solve remains.
            inverse_cross = solve_triangular(
                self._factor(),
                self._projections[:n, indices],
                lower=True,
                trans="T",
                check_finite=False,
            )
        variances = np.maximum(
            self.process_variance
            - np.einsum("mn,nm->m", cross, inverse_cross, optimize=True),
            0.0,
        )
        variance_gradients = -2.0 * np.einsum(
            "mnd,nm->md", cross_gradient, inverse_cross, optimize=True
        )
        return (
            np.asarray(means, dtype=float),
            np.asarray(variances, dtype=float),
            np.asarray(mean_gradients, dtype=float),
            np.asarray(variance_gradients, dtype=float),
        )

    def append_points(self, points: ArrayLike) -> NDArray[np.int_]:
        """Append continuous candidates and return their new universe indices."""

        additions = np.asarray(points, dtype=float)
        if additions.ndim == 1:
            additions = additions.reshape(1, -1)
        if additions.ndim != 2 or additions.shape[1] != DIMENSION:
            raise ValueError(f"points must have shape (m, {DIMENSION})")
        if not np.all(np.isfinite(additions)) or np.any(additions < 0.0) or np.any(
            additions > 1.0
        ):
            raise ValueError("appended points must lie in the unit hypercube")
        old_size = self.universe.shape[0]
        new_size = old_size + additions.shape[0]
        self._ensure_column_capacity(new_size)
        means, variances = self.posterior_at(additions)
        n = self._active_count
        if n:
            training = self.universe[self._active_indices[:n]]
            training_cross = self.process_variance * kernel_matrix(
                training, additions, self.kernel, self.lengthscale
            )
            self._projections[:n, old_size:new_size] = solve_triangular(
                self._factor(),
                training_cross,
                lower=True,
                check_finite=False,
            )
        self.universe = np.vstack((self.universe, additions))
        self.posterior_mean = np.concatenate((self.posterior_mean, means))
        self.posterior_variance = np.concatenate(
            (self.posterior_variance, variances)
        )
        return np.arange(old_size, new_size, dtype=int)

    def add_observation(
        self, universe_index: int, observed_mean: float, mean_noise_variance: float
    ) -> None:
        if self._observation_count >= self.maximum_observations:
            raise RuntimeError("the GP observation capacity has been exhausted")
        if not 0 <= universe_index < self.universe.shape[0]:
            raise IndexError("universe_index is out of bounds")
        if not np.isfinite(mean_noise_variance) or mean_noise_variance <= 0.0:
            raise ValueError("mean_noise_variance must be finite and positive")
        if self._active_count >= self._row_capacity:
            self._rebuild_from_sufficient_statistics()
        if self._active_count >= self._row_capacity:
            self._ensure_row_capacity(self._active_count + 1)
        n = self._active_count
        selected = self.universe[universe_index : universe_index + 1]
        prior_cross = self.process_variance * kernel_matrix(
            self.universe, selected, self.kernel, self.lengthscale
        ).reshape(-1)
        if n == 0:
            triangular = np.empty(0, dtype=float)
            conditional_cross = prior_cross
            schur = self.process_variance + mean_noise_variance
        else:
            # This exact L^{-1} k(X, x) vector is already maintained for every
            # universe point, including appended continuous candidates.
            triangular = self._projections[:n, universe_index].copy()
            conditional_cross = (
                prior_cross
                - self._projections[:n, : self.universe.shape[0]].T @ triangular
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
        if n:
            self._cholesky[n, :n] = triangular
        self._cholesky[n, n] = diagonal
        self._projections[n, : self.universe.shape[0]] = conditional_cross / diagonal
        self._active_indices[n] = int(universe_index)
        self._active_observed_means[n] = float(observed_mean)
        self._training_indices[self._observation_count] = int(universe_index)
        precision = 1.0 / mean_noise_variance
        self._precision_sum[universe_index] += precision
        self._precision_weighted_sum[universe_index] += precision * observed_mean
        self._active_count += 1
        self._observation_count += 1
        self._posterior_weight_cache_count = -1
        self._factor_cache_count = -1
