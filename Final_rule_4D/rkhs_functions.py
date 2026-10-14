"""Fixed finite-expansion RKHS objectives used by the LQEI study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from study_config import DIMENSION, NOISE_SD_MAX_MULTIPLIER, NOISE_SD_MIN_MULTIPLIER


FloatArray = NDArray[np.float64]
KernelName = Literal["matern52", "squared_exponential"]


def _as_points(values: ArrayLike) -> FloatArray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != DIMENSION:
        raise ValueError(f"points must have shape (n, {DIMENSION})")
    return points


def kernel_matrix(
    points: ArrayLike,
    centers: ArrayLike,
    kernel: KernelName,
    lengthscale: float = 0.2,
) -> FloatArray:
    """Evaluate the radial Matérn-5/2 or squared-exponential kernel."""

    x = _as_points(points)
    z = _as_points(centers)
    if not np.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("lengthscale must be finite and positive")
    differences = x[:, np.newaxis, :] - z[np.newaxis, :, :]
    radius_squared = np.sum(differences**2, axis=2) / lengthscale**2
    if kernel == "squared_exponential":
        return np.exp(-0.5 * radius_squared)
    if kernel != "matern52":
        raise ValueError("kernel must be 'matern52' or 'squared_exponential'")
    scaled = np.sqrt(5.0 * np.maximum(radius_squared, 0.0))
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def kernel_vector_and_gradient(
    point: ArrayLike,
    centers: ArrayLike,
    kernel: KernelName,
    lengthscale: float = 0.2,
) -> tuple[FloatArray, FloatArray]:
    """Return ``k(point, centers)`` and its gradient with respect to point.

    The gradient has shape ``(number_of_centers, DIMENSION)``.  This small,
    single-query routine is used by the continuous acquisition optimizer and
    deliberately shares the exact kernel parameterization with
    :func:`kernel_matrix`.
    """

    query = _as_points(point)
    if query.shape[0] != 1:
        raise ValueError("point must contain exactly one query location")
    z = _as_points(centers)
    if not np.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("lengthscale must be finite and positive")
    difference = query[0] - z
    radius_squared = np.sum(difference**2, axis=1) / lengthscale**2
    if kernel == "squared_exponential":
        values = np.exp(-0.5 * radius_squared)
        gradients = -values[:, np.newaxis] * difference / lengthscale**2
        return values, gradients
    if kernel != "matern52":
        raise ValueError("kernel must be 'matern52' or 'squared_exponential'")
    scaled = np.sqrt(5.0 * np.maximum(radius_squared, 0.0))
    exponential = np.exp(-scaled)
    values = (1.0 + scaled + scaled**2 / 3.0) * exponential
    gradients = (
        -5.0
        * (1.0 + scaled)[:, np.newaxis]
        * exponential[:, np.newaxis]
        * difference
        / (3.0 * lengthscale**2)
    )
    return values, gradients


@dataclass(frozen=True)
class FixedRKHSFunction:
    """An immutable finite weighted sum of kernel sections."""

    name: str
    kernel: KernelName
    lengthscale: float
    basis_points: FloatArray
    weights: FloatArray
    optimum_x: FloatArray
    optimum_value: float
    maximum_x: FloatArray
    maximum_value: float
    rkhs_norm: float
    seed: int

    @property
    def response_range(self) -> float:
        return float(self.maximum_value - self.optimum_value)

    @property
    def variance_min(self) -> float:
        return float((NOISE_SD_MIN_MULTIPLIER * self.response_range) ** 2)

    @property
    def variance_max(self) -> float:
        return float((NOISE_SD_MAX_MULTIPLIER * self.response_range) ** 2)

    def evaluate(self, points: ArrayLike, chunk_size: int = 8192) -> FloatArray:
        x = _as_points(points)
        values = np.empty(x.shape[0], dtype=float)
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            values[start:stop] = (
                kernel_matrix(
                    x[start:stop],
                    self.basis_points,
                    self.kernel,
                    self.lengthscale,
                )
                @ self.weights
            )
        return values

    def noise_standard_deviation(self, points: ArrayLike) -> FloatArray:
        """Half-noise heteroscedastic standard deviation."""

        values = self.evaluate(points)
        normalized = (values - self.optimum_value) / self.response_range
        # The extrema are numerical approximations. Clipping prevents tiny
        # optimizer-tolerance excursions from violating the declared bounds.
        normalized = np.clip(normalized, 0.0, 1.0)
        multiplier = NOISE_SD_MIN_MULTIPLIER + (
            NOISE_SD_MAX_MULTIPLIER - NOISE_SD_MIN_MULTIPLIER
        ) * normalized
        return self.response_range * multiplier


def save_fixed_function(function: FixedRKHSFunction, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        name=np.asarray(function.name),
        kernel=np.asarray(function.kernel),
        lengthscale=np.asarray(function.lengthscale),
        basis_points=function.basis_points,
        weights=function.weights,
        optimum_x=function.optimum_x,
        optimum_value=np.asarray(function.optimum_value),
        maximum_x=function.maximum_x,
        maximum_value=np.asarray(function.maximum_value),
        response_range=np.asarray(function.response_range),
        rkhs_norm=np.asarray(function.rkhs_norm),
        variance_min=np.asarray(function.variance_min),
        variance_max=np.asarray(function.variance_max),
        seed=np.asarray(function.seed, dtype=np.int64),
    )
    return destination


def load_fixed_function(path: str | Path) -> FixedRKHSFunction:
    with np.load(Path(path), allow_pickle=False) as data:
        return FixedRKHSFunction(
            name=str(data["name"].item()),
            kernel=str(data["kernel"].item()),  # type: ignore[arg-type]
            lengthscale=float(data["lengthscale"]),
            basis_points=np.asarray(data["basis_points"], dtype=float),
            weights=np.asarray(data["weights"], dtype=float),
            optimum_x=np.asarray(data["optimum_x"], dtype=float),
            optimum_value=float(data["optimum_value"]),
            maximum_x=np.asarray(data["maximum_x"], dtype=float),
            maximum_value=float(data["maximum_value"]),
            rkhs_norm=float(data["rkhs_norm"]),
            seed=int(data["seed"]),
        )
