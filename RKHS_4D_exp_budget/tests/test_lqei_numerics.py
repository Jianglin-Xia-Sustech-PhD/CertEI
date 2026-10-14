from __future__ import annotations

import numpy as np

from continuous_acquisition import _log_positive_part_with_gradient
from experiment_runner import _expected_improvement, _log_expected_improvement


def test_log_ei_matches_ordinary_ei_and_survives_extreme_tail() -> None:
    means = np.asarray([-0.5, 0.0, 0.4, 2.0])
    variances = np.asarray([0.09, 0.16, 0.25, 0.36])
    ordinary, incumbent = _expected_improvement(means, variances, 0.0)
    logarithmic = _log_expected_improvement(means, variances, incumbent)
    np.testing.assert_allclose(
        np.exp(logarithmic), ordinary, rtol=2e-10, atol=1e-300
    )

    extreme = _log_expected_improvement(
        np.asarray([100.0, 101.0]),
        np.asarray([0.01, 0.01]),
        -3.0,
    )
    assert np.all(np.isfinite(extreme))
    assert extreme[0] > extreme[1]


def test_log_ei_gradient_is_finite_in_the_underflow_tail() -> None:
    value, gradient = _log_positive_part_with_gradient(
        improvement=-103.0,
        deviation=0.1,
        improvement_gradient=np.asarray([0.2, -0.1, 0.3, 0.4]),
        deviation_gradient=np.asarray([0.01, 0.02, -0.01, 0.03]),
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
