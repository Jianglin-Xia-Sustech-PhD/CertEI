"""Equation-(20) information-gain bound for the LQEI confidence sequence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from rkhs_functions import KernelName
from study_config import DIMENSION


FloatArray = NDArray[np.float64]
INFORMATION_GAIN_METHOD = "min_kernel_order_and_finite_batch_bounds"
INFORMATION_GAIN_CONSTANT = 1.0
MATERN_NU = 2.5


@dataclass(frozen=True)
class ConfidenceSchedule:
    """All deterministic quantities used to form Lambda_n, n=0,...,N_T."""

    delta: float
    information_gain_batch_size: int
    correction_batch_size: int
    batch_count: int
    variance_min: float
    rkhs_norm: float
    kappa_t: float
    kernel: KernelName
    effective_information_budget: FloatArray
    kernel_order_information_gain_bound: FloatArray
    finite_batch_information_gain_bound: FloatArray
    information_gain_upper_bound: FloatArray
    lambda_values: FloatArray


def chi_square_correction(batch_size: int, batch_count: int, delta: float) -> float:
    """Compute kappa_T from equation (4), using the smallest actual batch."""

    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    degrees_of_freedom = batch_size - 1
    alpha_t = delta / (3.0 * batch_count)
    lower_quantile = float(chi2.ppf(alpha_t, degrees_of_freedom))
    if not np.isfinite(lower_quantile) or lower_quantile <= 0.0:
        raise RuntimeError("invalid lower chi-square quantile")
    return max(1.0, degrees_of_freedom / lower_quantile)


def kernel_order_information_gain(
    kernel: KernelName,
    effective_information_budget: ArrayLike,
    *,
    dimension: int = DIMENSION,
    matern_nu: float = MATERN_NU,
) -> FloatArray:
    """Evaluate G_k(U) after setting its asymptotic big-O constant to one.

    The finite-sample extension log(1+U) is used so the sequence is real,
    nonnegative, and starts at zero even for the log-log ablation, where the
    first-batch U can be below one. This has the same asymptotic order as
    log(U).
    """

    u = np.asarray(effective_information_budget, dtype=float)
    if np.any(~np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("effective information budget must be finite and nonnegative")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    logarithm = np.log1p(u)
    if kernel == "squared_exponential":
        values = logarithm ** (dimension + 1)
    elif kernel == "matern52":
        if matern_nu <= 0.0:
            raise ValueError("matern_nu must be positive")
        denominator = 2.0 * matern_nu + dimension
        values = (
            u ** (dimension / denominator)
            * logarithm ** (2.0 * matern_nu / denominator)
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported kernel: {kernel}")
    values = np.asarray(values, dtype=float)
    values[u == 0.0] = 0.0
    return values


def finite_batch_information_gain(
    batch_indices: ArrayLike,
    *,
    batch_size: int,
    variance_min: float,
) -> FloatArray:
    """Evaluate the finite-batch term in equation (20) at every batch index.

    For a fixed replication number ``b``, Lemma 7 gives

        n / 2 * log(1 + b / nu_min).

    Applying it for ``n=0,...,N_T`` supplies the time-indexed upper bound
    required by Lambda_n and recovers equation (20) at ``n=N_T``.
    """

    indices = np.asarray(batch_indices, dtype=float)
    if np.any(~np.isfinite(indices)) or np.any(indices < 0.0):
        raise ValueError("batch indices must be finite and nonnegative")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if variance_min <= 0.0 or not np.isfinite(variance_min):
        raise ValueError("variance_min must be finite and positive")
    return np.asarray(
        0.5 * indices * np.log1p(batch_size / variance_min),
        dtype=float,
    )


def build_confidence_schedule(
    *,
    kernel: KernelName,
    information_gain_batch_size: int,
    correction_batch_size: int,
    batch_count: int,
    variance_min: float,
    rkhs_norm: float,
    delta: float,
) -> ConfidenceSchedule:
    """Build Lambda_n using the minimum information-gain bound in (20)."""

    if information_gain_batch_size < 2:
        raise ValueError("information_gain_batch_size must be at least two")
    if variance_min <= 0.0 or not np.isfinite(variance_min):
        raise ValueError("variance_min must be finite and positive")
    if rkhs_norm < 0.0 or not np.isfinite(rkhs_norm):
        raise ValueError("rkhs_norm must be finite and nonnegative")
    indices = np.arange(batch_count + 1, dtype=float)
    effective_budget = indices * information_gain_batch_size / variance_min
    kernel_order_bound = INFORMATION_GAIN_CONSTANT * kernel_order_information_gain(
        kernel,
        effective_budget,
    )
    finite_batch_bound = finite_batch_information_gain(
        indices,
        batch_size=information_gain_batch_size,
        variance_min=variance_min,
    )
    gamma_upper = np.minimum(kernel_order_bound, finite_batch_bound)
    kappa_t = chi_square_correction(
        correction_batch_size,
        batch_count,
        delta,
    )
    lambda_values = rkhs_norm + np.sqrt(
        2.0
        * kappa_t
        * (gamma_upper + np.log(3.0 / (2.0 * delta)))
    )
    return ConfidenceSchedule(
        delta=float(delta),
        information_gain_batch_size=int(information_gain_batch_size),
        correction_batch_size=int(correction_batch_size),
        batch_count=int(batch_count),
        variance_min=float(variance_min),
        rkhs_norm=float(rkhs_norm),
        kappa_t=float(kappa_t),
        kernel=kernel,
        effective_information_budget=effective_budget,
        kernel_order_information_gain_bound=kernel_order_bound,
        finite_batch_information_gain_bound=finite_batch_bound,
        information_gain_upper_bound=gamma_upper,
        lambda_values=lambda_values,
    )
