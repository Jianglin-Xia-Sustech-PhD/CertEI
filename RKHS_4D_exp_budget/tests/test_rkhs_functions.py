from __future__ import annotations

import numpy as np

from experiment_runner import FUNCTION_DIR
from rkhs_functions import (
    kernel_matrix,
    kernel_vector_and_gradient,
    load_fixed_function,
)
from study_config import NOISE_SD_MAX_MULTIPLIER, NOISE_SD_MIN_MULTIPLIER


def test_lower_noise_formula_and_variance_bounds() -> None:
    for function_name in ("rkhs_matern52_4d", "rkhs_se_4d"):
        function = load_fixed_function(FUNCTION_DIR / f"{function_name}.npz")
        response_range = function.response_range
        standard_deviations = function.noise_standard_deviation(
            np.vstack((function.optimum_x, function.maximum_x))
        )
        np.testing.assert_allclose(
            standard_deviations,
            response_range
            * np.asarray(
                [NOISE_SD_MIN_MULTIPLIER, NOISE_SD_MAX_MULTIPLIER]
            ),
            rtol=1e-7,
            atol=1e-10,
        )
        assert function.variance_min == (
            NOISE_SD_MIN_MULTIPLIER * response_range
        ) ** 2
        assert function.variance_max == (
            NOISE_SD_MAX_MULTIPLIER * response_range
        ) ** 2


def test_kernel_gradients_match_centered_finite_differences() -> None:
    rng = np.random.default_rng(41)
    point = rng.uniform(0.2, 0.8, size=4)
    centers = rng.uniform(0.0, 1.0, size=(7, 4))
    step = 1e-6
    for kernel in ("matern52", "squared_exponential"):
        values, gradients = kernel_vector_and_gradient(
            point, centers, kernel, 0.2
        )
        np.testing.assert_allclose(
            values,
            kernel_matrix(point, centers, kernel, 0.2).reshape(-1),
            atol=1e-14,
        )
        finite = np.empty_like(gradients)
        for coordinate in range(4):
            upper = point.copy()
            lower = point.copy()
            upper[coordinate] += step
            lower[coordinate] -= step
            finite[:, coordinate] = (
                kernel_matrix(upper, centers, kernel, 0.2).reshape(-1)
                - kernel_matrix(lower, centers, kernel, 0.2).reshape(-1)
            ) / (2.0 * step)
        np.testing.assert_allclose(gradients, finite, rtol=2e-5, atol=2e-7)
