"""Configuration for the analytic-information-gain 2D LQEI study."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int_]

BUDGET_EXPONENTS = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5)
EXPONENTIAL_BUDGETS = (
    1808,
    2981,
    4915,
    8103,
    13360,
    22026,
    36316,
    59874,
    98716,
)
TARGET_BUDGETS_BY_FUNCTION = {
    "rkhs_matern52_2d": EXPONENTIAL_BUDGETS,
    "rkhs_se_2d": EXPONENTIAL_BUDGETS,
}
# Preserve the source study's three high-budget ablation points on the new
# exponential grid: e^10.5, e^11, and e^11.5.
ABLATION_BUDGETS = (36316, 59874, 98716)
N_INITIAL = 20
N_CANDIDATES = 2048
FORMAL_MACROS = 100
CONFIDENCE_DELTA = 0.05
FORMAL_PHASE_CODE = 198
DESIGN_SEED = 20260831
NOISE_SEED = 20260901
CANDIDATE_SEED = 20260830

RESPONSE_CONSTANT_MEAN = 0.0
RESPONSE_PROCESS_VARIANCE = 1.0
RESPONSE_LENGTHSCALE = 0.2

# Keep the accepted acquisition rule from restart 91 and replace the fixed
# terminal coefficient 3 by 0.2 * Lambda_{N_T}.  Lambda_{N_T} is resolved from
# each budget's confidence schedule in experiment_runner.py.
LQEI_ACQUISITION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_RULE = (
    "posterior_mean_plus_0p2_terminal_lambda_posterior_sd"
)

# Standard-deviation multipliers of the objective response range.  These are
# exactly one half of the original [1.5 R_f, 6 R_f] heteroscedastic envelope.
NOISE_SD_MIN_MULTIPLIER = 0.75
NOISE_SD_MAX_MULTIPLIER = 3.0

# The Matérn-5/2 batch coefficient is 8 for this trial.  The
# squared-exponential setting remains at 6.
RHO_BY_FUNCTION = {
    "rkhs_matern52_2d": 8.0,
    "rkhs_se_2d": 6.0,
}

REPLICATION_ORDERS = (
    "constant",
    "log",
    "log_log",
    "power_1p2",
)


def budget_exponent(total_budget: int) -> float:
    """Return k for a registered integer budget displayed as T=e^k."""

    try:
        index = EXPONENTIAL_BUDGETS.index(int(total_budget))
    except ValueError as error:
        raise ValueError(f"unregistered exponential budget: {total_budget}") from error
    return float(BUDGET_EXPONENTS[index])


def round_up_five(value: float) -> int:
    return max(2, 5 * int(math.ceil(float(value) / 5.0)))


def rho_for_function(function_name: str) -> float:
    try:
        rho = float(RHO_BY_FUNCTION[function_name])
    except KeyError as error:
        raise ValueError(f"unknown function: {function_name}") from error
    if not 0.0 < rho < 20.0:
        raise ValueError("every kernel-specific rho must lie in (0, 20)")
    return rho


def log_batch_size(total_budget: int, rho: float) -> int:
    """Return rho*ln(T), rounded up to a multiple of five."""

    if not 0.0 < rho < 20.0:
        raise ValueError("rho must lie in (0, 20)")
    return round_up_five(rho * math.log(total_budget))


@cache
def budgets_for_function(function_name: str) -> tuple[int, ...]:
    rho_for_function(function_name)
    try:
        registered = TARGET_BUDGETS_BY_FUNCTION[function_name]
    except KeyError as error:  # pragma: no cover - guarded by rho lookup above
        raise ValueError(f"unknown function: {function_name}") from error
    budgets = tuple(int(value) for value in registered)
    for budget in budgets:
        if budget <= N_INITIAL * log_batch_size(budget, rho_for_function(function_name)):
            raise RuntimeError("a requested budget leaves no sequential batch")
    return budgets


def replication_size(
    order: str,
    total_budget: int,
    *,
    rho: float,
    anchor_budget: int,
) -> int:
    """Return the requested same-rho replication-order rule.

    ``anchor_budget`` is retained so code from the preceding restart remains
    compatible.  It is intentionally unused: no common-anchor normalization is
    applied to the ablation orders.
    """

    del anchor_budget
    if order == "constant":
        return round_up_five(rho)
    if order == "log":
        return log_batch_size(total_budget, rho)
    if order == "log_log":
        if total_budget <= 1:
            raise ValueError("log-log replication requires total_budget > 1")
        return round_up_five(rho * math.log(math.log(total_budget)))
    if order == "power_1p2":
        return round_up_five(rho * total_budget ** (1.0 / 2.0))
    raise ValueError(f"unknown replication order: {order}")


def make_batch_schedule(total_budget: int, batch_size: int) -> IntArray:
    """Consume the exact budget with batches equal to b_T or b_T+1."""

    number_of_batches, remainder = divmod(total_budget, batch_size)
    if number_of_batches <= N_INITIAL:
        raise ValueError(
            f"T={total_budget}, b_T={batch_size} leaves no sequential batch "
            f"after the {N_INITIAL} initial points"
        )
    schedule = np.full(number_of_batches, batch_size, dtype=int)
    extra_per_batch, extra_remainder = divmod(remainder, number_of_batches)
    schedule += extra_per_batch
    schedule[:extra_remainder] += 1
    if int(np.sum(schedule)) != total_budget:
        raise RuntimeError("batch schedule does not consume the exact budget")
    if int(np.max(schedule) - np.min(schedule)) > 1:
        raise RuntimeError("batch schedule is not balanced")
    return schedule


def make_mixed_batch_schedule(
    total_budget: int,
    sequential_batch_size: int,
    initial_replications: int,
) -> IntArray:
    """Mixed initial/sequential batches are outside this experiment."""

    del total_budget, sequential_batch_size, initial_replications
    raise ValueError("mixed initial/sequential replication is disabled")
