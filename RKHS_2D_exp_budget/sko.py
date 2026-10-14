"""SKO acquisition adapted from the project's Python augmented-EI implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr


FloatArray = NDArray[np.float64]
SKO_INCUMBENT_SD_MULTIPLIER = 1.0


def stochastic_kriging_optimization_acquisition(
    sampled_means: ArrayLike,
    sampled_standard_deviations: ArrayLike,
    candidate_means: ArrayLike,
    candidate_variances: ArrayLike,
    candidate_noise_variances: ArrayLike,
) -> tuple[FloatArray, float]:
    """Return the SKO acquisition and its effective incumbent.

    This is the finite-candidate, fixed-GP adaptation of the acquisition in
    ``KrigingBasedHetero_sustech_python/AEI_python/aei/acquisition.py``.  For
    minimization, the effective incumbent is the posterior mean at the sampled
    point minimizing ``mean + posterior_sd``.  Ordinary expected improvement is
    then multiplied by the stochastic-kriging noise penalty

        1 - sqrt(noise_variance) / sqrt(posterior_variance + noise_variance).

    ``candidate_noise_variances`` are single-replication variances, matching
    the source implementation; they are not divided by the iteration budget.
    """

    observed_means = np.asarray(sampled_means, dtype=float).reshape(-1)
    observed_sds = np.asarray(
        sampled_standard_deviations, dtype=float
    ).reshape(-1)
    means = np.asarray(candidate_means, dtype=float).reshape(-1)
    variances = np.maximum(
        np.asarray(candidate_variances, dtype=float).reshape(-1), 0.0
    )
    noise_variances = np.maximum(
        np.asarray(candidate_noise_variances, dtype=float).reshape(-1), 0.0
    )

    if observed_means.size == 0:
        raise ValueError("at least one sampled point is required")
    if observed_sds.shape != observed_means.shape:
        raise ValueError("sampled posterior arrays must have matching shapes")
    if variances.shape != means.shape or noise_variances.shape != means.shape:
        raise ValueError("candidate prediction arrays must have matching shapes")
    for values in (observed_means, observed_sds, means, variances, noise_variances):
        if np.any(~np.isfinite(values)):
            raise ValueError("SKO inputs must be finite")
    if np.any(observed_sds < 0.0):
        raise ValueError("sampled posterior deviations must be nonnegative")

    incumbent_index = int(
        np.argmin(
            observed_means
            + SKO_INCUMBENT_SD_MULTIPLIER * observed_sds
        )
    )
    effective_best = float(observed_means[incumbent_index])

    deviations = np.sqrt(variances)
    improvement = effective_best - means
    expected_improvement = np.maximum(improvement, 0.0)
    positive = deviations > np.finfo(float).eps
    standardized = np.zeros_like(improvement)
    standardized[positive] = improvement[positive] / deviations[positive]
    density = np.exp(-0.5 * standardized[positive] ** 2) / np.sqrt(
        2.0 * np.pi
    )
    expected_improvement[positive] = (
        improvement[positive] * ndtr(standardized[positive])
        + deviations[positive] * density
    )

    total_uncertainty = variances + noise_variances
    penalty = np.ones_like(total_uncertainty)
    has_uncertainty = total_uncertainty > 0.0
    penalty[has_uncertainty] -= np.sqrt(
        noise_variances[has_uncertainty]
    ) / np.sqrt(total_uncertainty[has_uncertainty])
    acquisition = np.maximum(expected_improvement * penalty, 0.0)
    return acquisition, effective_best
