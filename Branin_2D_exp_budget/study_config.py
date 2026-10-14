"""Authoritative configuration for the adjusted analytic-function comparison."""

from __future__ import annotations

import math
import os

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int_]

FUNCTION_NAMES = ("rescaled_branin",)
# Preserve the established Branin random streams from the earlier two-function
# comparison while changing only the noise relationship.
FUNCTION_SEED_CODES = {"rescaled_branin": 1}
ALGORITHMS = ("LQEI", "TSSO", "SKO", "EQI")
# Six requested budgets on the natural-exponential scale, capped at exp(10).
BUDGET_EXPONENTS = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
BUDGETS = tuple(int(round(math.exp(exponent))) for exponent in BUDGET_EXPONENTS)
# Preserve the source Branin GP/RKHS calibration protocol.
CALIBRATION_BUDGETS = (2981, 13360, 59874)

N_INITIAL = 20
N_CANDIDATES = 2048
# The code snapshot defaults to the selected 100-macro formal setting.
FORMAL_MACROS = int(os.environ.get("BRANIN_NEGATIVE_MACROS", "100"))
CALIBRATION_MACROS = 100
CALIBRATION_MACRO_OFFSET = 1000
REPLICATION_MULTIPLIER = float(
    os.environ.get("BRANIN_NEGATIVE_REPLICATION_MULTIPLIER", "8.0")
)
CONFIDENCE_DELTA = 0.05

DESIGN_SEED = 20260924
CANDIDATE_SEED = 20260925
NOISE_SEED = 20260926

# Negative-slope response-linked noise SD, sigma/R_f = a + b*z. The defaults
# reproduce the selected formal 4-3z setting; both remain overridable.
CANDIDATE_MEAN_STANDARDIZED_RESPONSE = 0.175193921550
NOISE_B_MULTIPLIER = float(os.environ.get("BRANIN_NEGATIVE_NOISE_B", "-3.0"))
NOISE_A_MULTIPLIER = float(os.environ.get("BRANIN_NEGATIVE_NOISE_A", "4.0"))
CANDIDATE_MEAN_NOISE_SD_MULTIPLIER = (
    NOISE_A_MULTIPLIER
    + CANDIDATE_MEAN_STANDARDIZED_RESPONSE * NOISE_B_MULTIPLIER
)
NOISE_SD_MIN_MULTIPLIER = min(
    NOISE_A_MULTIPLIER, NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER
)
NOISE_SD_MAX_MULTIPLIER = max(
    NOISE_A_MULTIPLIER, NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER
)
STUDY_STAGE = os.environ.get("BRANIN_NEGATIVE_STAGE", "formal").strip().lower()

if not np.isfinite(NOISE_B_MULTIPLIER) or NOISE_B_MULTIPLIER >= 0.0:
    raise ValueError("BRANIN_NEGATIVE_NOISE_B must be finite and strictly negative")
if not np.isfinite(NOISE_A_MULTIPLIER) or NOISE_A_MULTIPLIER <= 0.0:
    raise ValueError("BRANIN_NEGATIVE_NOISE_A must be positive and finite")
if not np.isfinite(REPLICATION_MULTIPLIER) or REPLICATION_MULTIPLIER <= 0.0:
    raise ValueError(
        "BRANIN_NEGATIVE_REPLICATION_MULTIPLIER must be positive and finite"
    )
if NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER <= 0.0:
    raise ValueError("a+b must be positive so the noise SD is positive on [0,1]")
if FORMAL_MACROS <= 0:
    raise ValueError("BRANIN_NEGATIVE_MACROS must be positive")
if STUDY_STAGE not in {"pilot", "formal"}:
    raise ValueError("BRANIN_NEGATIVE_STAGE must be 'pilot' or 'formal'")

# Keep the 100-macro confirmation independent of the 30-macro screening data.
MACRO_SEED_OFFSET = 0 if STUDY_STAGE == "pilot" else 10_000


def _number_slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


NOISE_PARAMETER_SLUG = (
    f"a_{_number_slug(NOISE_A_MULTIPLIER)}_"
    f"b_{_number_slug(NOISE_B_MULTIPLIER)}"
)
CONFIGURATION_SLUG = (
    NOISE_PARAMETER_SLUG
    if np.isclose(REPLICATION_MULTIPLIER, 8.0, rtol=0.0, atol=1e-12)
    else f"{NOISE_PARAMETER_SLUG}_rho_{_number_slug(REPLICATION_MULTIPLIER)}"
)
RESULT_GROUP = (
    f"{STUDY_STAGE}_six_budget_2048_candidates_"
    f"{CONFIGURATION_SLUG}_{FORMAL_MACROS}_macros"
)

LQEI_ACQUISITION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER = 0.2

EQI_BETA = 0.5
EQI_ALLOCATION_THRESHOLD = 0.5
SKO_INCUMBENT_SD_MULTIPLIER = 1.0
TSSO_MINIMUM_SEARCH_REPLICATIONS = 2

# The response GP uses a Matérn-5/2 covariance.  Outputs and intrinsic-noise
# variances are expressed in the fixed analytic scale
# (f - f_min) / (f_max - f_min) before fitting.  The two lengthscales, process
# variance and constant trend are estimated from data. SKO/EQI fit a second
# Matérn-5/2 GP to accumulated standardized sample standard deviations; the
# auxiliary noise GP is re-estimated after every sequential stage.
MATERN_NU = 2.5
LENGTHSCALE_BOUNDS = (0.02, 2.0)
INITIAL_MLE_MAXITER = 80
SEQUENTIAL_MLE_MAXITER = 12
# Gaussian prior on each sequential log-lengthscale, centred at that macro's
# unregularized initial-GP estimate.  A one-sigma interval allows a factor-e
# change, so this stabilizes extreme drift without freezing the MLE.
LENGTHSCALE_LOG_PRIOR_SD = 1.0
# The same weak factor-e log-scale prior is applied to the process variance.
# Its centre is the unregularized initial-GP estimate from the same macro.
PROCESS_VARIANCE_LOG_PRIOR_SD = 1.0


def round_up_five(value: float) -> int:
    return max(2, 5 * int(math.ceil(float(value) / 5.0)))


def replication_size(total_budget: int) -> int:
    """Return the common rho*log(T) nominal batch size."""

    return round_up_five(REPLICATION_MULTIPLIER * math.log(total_budget))


def make_batch_schedule(total_budget: int) -> IntArray:
    """Consume the exact budget with balanced batches nearest nominal b_T."""

    batch_size = replication_size(total_budget)
    number_of_batches, remainder = divmod(int(total_budget), batch_size)
    # When the remainder exceeds the number of full batches, b_T/b_T+1
    # cannot exhaust the budget.  Add one batch and balance at b_T-1/b_T.
    if remainder > number_of_batches:
        number_of_batches += 1
    if number_of_batches <= N_INITIAL:
        raise ValueError("the budget leaves no sequential batch")
    balanced_size, remainder = divmod(int(total_budget), number_of_batches)
    schedule = np.full(number_of_batches, balanced_size, dtype=int)
    schedule[:remainder] += 1
    if int(np.sum(schedule)) != int(total_budget):
        raise RuntimeError("the balanced schedule does not exhaust the budget")
    return schedule


def expected_sequential_stages(total_budget: int) -> int:
    return int(make_batch_schedule(total_budget).size - N_INITIAL)
