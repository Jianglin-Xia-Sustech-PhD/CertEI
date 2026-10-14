"""Registered design for the 2D squared-exponential LQEI cross-validation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int_]

FUNCTION_NAME = "rkhs_se_2d"
BUDGET_EXPONENTS = (10.5, 11.5)
CROSS_VALIDATION_BUDGETS = (36316, 98716)
TARGET_BUDGETS_BY_FUNCTION = {FUNCTION_NAME: CROSS_VALIDATION_BUDGETS}

N_INITIAL = 20
N_CANDIDATES = 2048
FORMAL_MACROS = 100
CONFIDENCE_DELTA = 0.05

# A new phase code gives this validation study fresh initial-design and noise
# streams while keeping every arm paired within macro replication.
CROSS_VALIDATION_PHASE_CODE = 1008
DESIGN_SEED = 20260831
NOISE_SEED = 20260901
CANDIDATE_SEED = 20260830
ANALYSIS_SEED = 20261008
BOOTSTRAP_RESAMPLES = 10_000

RESPONSE_CONSTANT_MEAN = 0.0
RESPONSE_PROCESS_VARIANCE = 1.0
RESPONSE_LENGTHSCALE = 0.2

LQEI_ACQUISITION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_RULE = (
    "posterior_mean_plus_0p2_terminal_lambda_posterior_sd"
)

NOISE_SD_MIN_MULTIPLIER = 0.75
NOISE_SD_MAX_MULTIPLIER = 3.0

RHO_BY_FUNCTION = {FUNCTION_NAME: 6.0}
REPLICATION_ORDERS = ("constant", "log_log", "log", "power_1p2")


def budget_exponent(total_budget: int) -> float:
    """Return the registered exponent k for T=round(exp(k))."""

    try:
        index = CROSS_VALIDATION_BUDGETS.index(int(total_budget))
    except ValueError as error:
        raise ValueError(f"unregistered cross-validation budget: {total_budget}") from error
    return float(BUDGET_EXPONENTS[index])


def round_up_five(value: float) -> int:
    return max(2, 5 * int(math.ceil(float(value) / 5.0)))


def rho_for_function(function_name: str) -> float:
    try:
        rho = float(RHO_BY_FUNCTION[function_name])
    except KeyError as error:
        raise ValueError(f"unknown function: {function_name}") from error
    if not 0.0 < rho < 20.0:
        raise ValueError("rho must lie in (0, 20)")
    return rho


def log_batch_size(total_budget: int, rho: float) -> int:
    """Return rho*ln(T), rounded up to a multiple of five."""

    if not 0.0 < rho < 20.0:
        raise ValueError("rho must lie in (0, 20)")
    return round_up_five(rho * math.log(total_budget))


def budgets_for_function(function_name: str) -> tuple[int, ...]:
    rho_for_function(function_name)
    return tuple(int(value) for value in TARGET_BUDGETS_BY_FUNCTION[function_name])


def replication_size(
    order: str,
    total_budget: int,
    *,
    rho: float,
    anchor_budget: int,
) -> int:
    """Return the same-rho replication-number rule without normalization."""

    del anchor_budget
    if order == "constant":
        return round_up_five(rho)
    if order == "log_log":
        if total_budget <= 1:
            raise ValueError("log-log replication requires T > 1")
        return round_up_five(rho * math.log(math.log(total_budget)))
    if order == "log":
        return log_batch_size(total_budget, rho)
    if order == "power_1p2":
        return round_up_five(rho * total_budget ** 0.5)
    raise ValueError(f"unknown replication order: {order}")


def make_batch_schedule(total_budget: int, batch_size: int) -> IntArray:
    """Consume the exact budget with balanced batches of size b_T or b_T+1."""

    number_of_batches, remainder = divmod(int(total_budget), int(batch_size))
    if number_of_batches <= N_INITIAL:
        raise ValueError(
            f"T={total_budget}, b_T={batch_size} leaves no sequential batch "
            f"after the {N_INITIAL} initial points"
        )
    schedule = np.full(number_of_batches, batch_size, dtype=int)
    extra_per_batch, extra_remainder = divmod(remainder, number_of_batches)
    schedule += extra_per_batch
    schedule[:extra_remainder] += 1
    if int(np.sum(schedule)) != int(total_budget):
        raise RuntimeError("batch schedule does not consume the exact budget")
    if int(np.max(schedule) - np.min(schedule)) > 1:
        raise RuntimeError("batch schedule is not balanced")
    return schedule


def make_mixed_batch_schedule(
    total_budget: int,
    sequential_batch_size: int,
    initial_replications: int,
) -> IntArray:
    """Mixed initial/sequential replication is outside this protocol."""

    del total_budget, sequential_batch_size, initial_replications
    raise ValueError("mixed initial/sequential replication is disabled")
