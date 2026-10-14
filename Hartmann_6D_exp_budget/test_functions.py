"""Hartmann-6 objective and the matched analytic heteroscedastic noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from study_config import (
    DIMENSION,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
)


FloatArray = NDArray[np.float64]


def _points(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != DIMENSION:
        raise ValueError(f"points must have shape (n, {DIMENSION})")
    return array


_HARTMANN_ALPHA = np.asarray([1.0, 1.2, 3.0, 3.2], dtype=float)
_HARTMANN_A = np.asarray(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ],
    dtype=float,
)
_HARTMANN_P = 1.0e-4 * np.asarray(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ],
    dtype=float,
)


def hartmann6(points: ArrayLike) -> FloatArray:
    """Standard six-dimensional Hartmann function on ``[0, 1]^6``."""

    x = _points(points)
    squared = (x[:, np.newaxis, :] - _HARTMANN_P[np.newaxis, :, :]) ** 2
    exponent = -np.sum(_HARTMANN_A[np.newaxis, :, :] * squared, axis=2)
    return -np.sum(_HARTMANN_ALPHA[np.newaxis, :] * np.exp(exponent), axis=1)


@dataclass(frozen=True)
class AnalyticTestFunction:
    name: str
    display_name: str
    evaluator: Callable[[ArrayLike], FloatArray]
    optimum_points: FloatArray
    optimum_value: float
    candidate_optimum_points: FloatArray
    candidate_optimum_value: float
    maximum_value: float
    rkhs_norm: float
    reference_lengthscales: FloatArray

    def evaluate(self, points: ArrayLike) -> FloatArray:
        return self.evaluator(points)

    @property
    def response_range(self) -> float:
        return float(self.maximum_value - self.optimum_value)

    @property
    def output_location(self) -> float:
        return float(self.optimum_value)

    @property
    def output_scale(self) -> float:
        return self.response_range

    def standardize_outputs(self, values: ArrayLike) -> FloatArray:
        return (
            np.asarray(values, dtype=float) - self.output_location
        ) / self.output_scale

    def unstandardize_outputs(self, values: ArrayLike) -> FloatArray:
        return self.output_location + self.output_scale * np.asarray(values, dtype=float)

    def standardize_variances(self, values: ArrayLike) -> FloatArray:
        return np.asarray(values, dtype=float) / self.output_scale**2

    def unstandardize_variances(self, values: ArrayLike) -> FloatArray:
        return np.asarray(values, dtype=float) * self.output_scale**2

    @property
    def variance_min(self) -> float:
        return float((NOISE_SD_MIN_MULTIPLIER * self.response_range) ** 2)

    @property
    def variance_max(self) -> float:
        return float((NOISE_SD_MAX_MULTIPLIER * self.response_range) ** 2)

    @property
    def standardized_variance_min(self) -> float:
        return float(NOISE_SD_MIN_MULTIPLIER**2)

    @property
    def standardized_variance_max(self) -> float:
        return float(NOISE_SD_MAX_MULTIPLIER**2)

    def standardized_noise_sd_from_standardized_response(
        self,
        standardized_response: ArrayLike,
    ) -> FloatArray:
        """Return the true SD used only by the simulation data generator."""

        normalized = np.clip(np.asarray(standardized_response, dtype=float), 0.0, 1.0)
        return NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER * normalized

    def noise_standard_deviation(self, points: ArrayLike) -> FloatArray:
        standardized_response = self.standardize_outputs(self.evaluate(points))
        return self.output_scale * self.standardized_noise_sd_from_standardized_response(
            standardized_response
        )


_HARTMANN_OPTIMUM = np.asarray(
    [[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]],
    dtype=float,
)
_HARTMANN_OPTIMUM_VALUE = float(hartmann6(_HARTMANN_OPTIMUM)[0])


def function_without_rkhs_metadata(name: str) -> AnalyticTestFunction:
    """Return exact analytic metadata before numerical RKHS calibration."""

    if name == "hartmann6":
        return AnalyticTestFunction(
            name=name,
            display_name="Hartmann-6",
            evaluator=hartmann6,
            optimum_points=_HARTMANN_OPTIMUM.copy(),
            optimum_value=_HARTMANN_OPTIMUM_VALUE,
            candidate_optimum_points=_HARTMANN_OPTIMUM.copy(),
            candidate_optimum_value=_HARTMANN_OPTIMUM_VALUE,
            # Every summand is nonnegative, so zero is a rigorous upper bound
            # and the conventional response range is approximately 3.32237.
            maximum_value=0.0,
            rkhs_norm=float("nan"),
            reference_lengthscales=np.full(DIMENSION, np.nan),
        )
    raise ValueError(f"unknown test function: {name}")


def function_from_metadata(name: str, metadata: dict[str, object]) -> AnalyticTestFunction:
    base = function_without_rkhs_metadata(name)
    return AnalyticTestFunction(
        name=base.name,
        display_name=base.display_name,
        evaluator=base.evaluator,
        optimum_points=base.optimum_points,
        optimum_value=base.optimum_value,
        candidate_optimum_points=np.asarray(
            metadata["candidate_optimum_points"], dtype=float
        ),
        candidate_optimum_value=float(metadata["candidate_optimum_value"]),
        maximum_value=base.maximum_value,
        rkhs_norm=float(metadata["rkhs_norm"]),
        reference_lengthscales=np.asarray(
            metadata["reference_lengthscales"], dtype=float
        ),
    )
