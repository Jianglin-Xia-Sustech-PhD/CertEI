"""Rescaled Branin and its linear response-linked noise model."""

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


def six_hump_camelback(points: ArrayLike) -> FloatArray:
    """Six-hump Camelback on a normalized [0,1]^2 input domain."""

    u = _points(points)
    x1 = 4.0 * u[:, 0] - 2.0
    x2 = 2.0 * u[:, 1] - 1.0
    return (
        4.0 * x1**2
        - 2.1 * x1**4
        + x1**6 / 3.0
        + x1 * x2
        - 4.0 * x2**2
        + 4.0 * x2**4
    )


def rescaled_branin(points: ArrayLike) -> FloatArray:
    """The rescaled Branin definition used by Jalali et al. (2017)."""

    u = _points(points)
    x1 = 15.0 * u[:, 0] - 5.0
    x2 = 15.0 * u[:, 1]
    return (
        (
            x2
            - 5.1 * x1**2 / (4.0 * np.pi**2)
            + 5.0 * x1 / np.pi
            - 6.0
        )
        ** 2
        + (10.0 - 10.0 / (8.0 * np.pi)) * np.cos(x1)
        - 44.81
    ) / 51.95


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


_CAMELBACK_OPTIMA = np.asarray(
    [
        [(0.08984201368301331 + 2.0) / 4.0, (-0.7126564032704135 + 1.0) / 2.0],
        [(-0.08984201368301331 + 2.0) / 4.0, (0.7126564032704135 + 1.0) / 2.0],
    ],
    dtype=float,
)

_BRANIN_OPTIMA = np.asarray(
    [
        [(-np.pi + 5.0) / 15.0, 12.275 / 15.0],
        [(np.pi + 5.0) / 15.0, 2.275 / 15.0],
        [(3.0 * np.pi + 5.0) / 15.0, 2.475 / 15.0],
    ],
    dtype=float,
)


def function_without_rkhs_metadata(name: str) -> AnalyticTestFunction:
    """Return exact analytic metadata before the numerical RKHS norm is added."""

    if name == "six_hump_camelback":
        return AnalyticTestFunction(
            name=name,
            display_name="Six-hump Camelback",
            evaluator=six_hump_camelback,
            optimum_points=_CAMELBACK_OPTIMA.copy(),
            optimum_value=-1.031628453489877,
            maximum_value=5.733333333333333,
            rkhs_norm=float("nan"),
            reference_lengthscales=np.full(2, np.nan),
        )
    if name == "rescaled_branin":
        return AnalyticTestFunction(
            name=name,
            display_name="Rescaled Branin",
            evaluator=rescaled_branin,
            optimum_points=_BRANIN_OPTIMA.copy(),
            optimum_value=-1.047393891092769,
            maximum_value=4.876209740358164,
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
