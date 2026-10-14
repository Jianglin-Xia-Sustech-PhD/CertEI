"""The standardized two-dimensional Hills objective and noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from study_config import (
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
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return array


def hills(points: ArrayLike) -> FloatArray:
    """Minimization form of the Hills problem on normalized [0,1]^2.

    Wang et al. define the Hills function g as a maximization problem on
    [0,100]^2.  Negating it gives the minimization objective -g, with the same
    25 local optima and the unique global minimizer (0.9, 0.9).  Its range is
    exactly [-20,0].
    """

    u = _points(points)
    physical = 100.0 * u
    component = (
        10.0
        * np.sin(0.05 * np.pi * physical) ** 6
        / 2.0 ** (2.0 * ((physical - 90.0) / 50.0) ** 2)
    )
    return -np.sum(component, axis=1)


@dataclass(frozen=True)
class AnalyticTestFunction:
    name: str
    display_name: str
    evaluator: Callable[[ArrayLike], FloatArray]
    optimum_points: FloatArray
    optimum_value: float
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
        """Fixed location used for every GP fit of this analytic function."""

        return float(self.optimum_value)

    @property
    def output_scale(self) -> float:
        """Fixed scale used for every GP fit of this analytic function."""

        return self.response_range

    def standardize_outputs(self, values: ArrayLike) -> FloatArray:
        return (
            np.asarray(values, dtype=float) - self.output_location
        ) / self.output_scale

    def unstandardize_outputs(self, values: ArrayLike) -> FloatArray:
        return (
            self.output_location
            + self.output_scale * np.asarray(values, dtype=float)
        )

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


_HILLS_OPTIMUM = np.asarray([[0.9, 0.9]], dtype=float)


def function_without_rkhs_metadata(name: str) -> AnalyticTestFunction:
    """Return exact analytic metadata before the numerical RKHS norm is added."""

    if name == "hills":
        return AnalyticTestFunction(
            name=name,
            display_name="Hills (-g)",
            evaluator=hills,
            optimum_points=_HILLS_OPTIMUM.copy(),
            optimum_value=-20.0,
            maximum_value=0.0,
            rkhs_norm=float("nan"),
            reference_lengthscales=np.full(2, np.nan),
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
        maximum_value=base.maximum_value,
        rkhs_norm=float(metadata["rkhs_norm"]),
        reference_lengthscales=np.asarray(
            metadata["reference_lengthscales"], dtype=float
        ),
    )
