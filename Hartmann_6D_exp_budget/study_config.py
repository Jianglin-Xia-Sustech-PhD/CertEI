"""Configuration for the negative-slope Hartmann-6 comparison study."""

from __future__ import annotations

import math
import os

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int_]

FUNCTION_NAMES = ("hartmann6",)
FUNCTION_SEED_CODES = {"hartmann6": 0}
ALGORITHMS = ("LQEI", "TSSO", "SKO", "EQI")
BUDGET_EXPONENTS = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
BUDGETS = tuple(int(round(math.exp(exponent))) for exponent in BUDGET_EXPONENTS)
CALIBRATION_BUDGETS = BUDGETS

DIMENSION = 6
N_INITIAL = int(os.environ.get("HARTMANN_NEGATIVE_N_INITIAL", "28"))
N_CANDIDATES = 10_000
RKHS_REFERENCE_POINTS = 512
MIN_SEQUENTIAL_STAGES = 10
FORMAL_MACROS = int(os.environ.get("HARTMANN_NEGATIVE_MACROS", "100"))
CALIBRATION_MACROS = int(
    os.environ.get("HARTMANN_NEGATIVE_CALIBRATION_MACROS", "40")
)
CALIBRATION_MACRO_OFFSET = 1000
BF_CALIBRATION_QUANTILE = float(
    os.environ.get("HARTMANN_NEGATIVE_BF_QUANTILE", "0.75")
)
BF_SHRINKAGE_FACTOR = float(
    os.environ.get("HARTMANN_NEGATIVE_BF_SHRINKAGE", "0.5")
)
REPLICATION_MULTIPLIER = float(
    os.environ.get("HARTMANN_NEGATIVE_REPLICATION_MULTIPLIER", "8.0")
)
CONFIDENCE_DELTA = 0.05

DESIGN_SEED = 20260924
CANDIDATE_SEED = 20260925
NOISE_SEED = 20260926
CANDIDATE_DESIGN = os.environ.get(
    "HARTMANN_NEGATIVE_CANDIDATE_DESIGN", "sobol"
).strip().lower()

# sigma(x) / R_f = a + b z(x), z=(f-f_min)/R_f, with b<0.
# Hartmann-6 is concentrated near z=1 over the unit hypercube, so both
# endpoint values and the candidate-set mean are reported explicitly.
CANDIDATE_MEAN_STANDARDIZED_RESPONSE = 0.92204333248965
NOISE_A_MULTIPLIER = float(
    os.environ.get("HARTMANN_NEGATIVE_NOISE_A", "7.2")
)
NOISE_B_MULTIPLIER = float(
    os.environ.get("HARTMANN_NEGATIVE_NOISE_B", "-3.5")
)
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
STUDY_STAGE = os.environ.get("HARTMANN_NEGATIVE_STAGE", "formal").strip().lower()

if N_INITIAL < DIMENSION + 2:
    raise ValueError("HARTMANN_NEGATIVE_N_INITIAL is too small for the 6D GP")
if N_CANDIDATES != 10_000:
    raise ValueError("this study requires exactly 10000 candidates")
if CANDIDATE_DESIGN not in {"faure", "sobol"}:
    raise ValueError(
        "HARTMANN_NEGATIVE_CANDIDATE_DESIGN must be 'faure' or 'sobol'"
    )
if not np.isfinite(NOISE_B_MULTIPLIER) or NOISE_B_MULTIPLIER >= 0.0:
    raise ValueError("HARTMANN_NEGATIVE_NOISE_B must be finite and negative")
if not np.isfinite(NOISE_A_MULTIPLIER) or NOISE_A_MULTIPLIER <= 0.0:
    raise ValueError("HARTMANN_NEGATIVE_NOISE_A must be positive and finite")
if NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER <= 0.0:
    raise ValueError("a+b must be positive on z in [0,1]")
if not np.isfinite(REPLICATION_MULTIPLIER) or REPLICATION_MULTIPLIER <= 0.0:
    raise ValueError("the replication multiplier must be positive and finite")
if FORMAL_MACROS <= 0 or CALIBRATION_MACROS <= 0:
    raise ValueError("macro counts must be positive")
if not 0.5 <= BF_CALIBRATION_QUANTILE <= 1.0:
    raise ValueError("HARTMANN_NEGATIVE_BF_QUANTILE must be in [0.5, 1]")
if not np.isfinite(BF_SHRINKAGE_FACTOR) or not 0.0 < BF_SHRINKAGE_FACTOR <= 1.0:
    raise ValueError("HARTMANN_NEGATIVE_BF_SHRINKAGE must be in (0, 1]")
if STUDY_STAGE not in {"pilot", "formal"}:
    raise ValueError("HARTMANN_NEGATIVE_STAGE must be 'pilot' or 'formal'")

MACRO_SEED_OFFSET = 0 if STUDY_STAGE == "pilot" else 10_000


def _number_slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


NOISE_PARAMETER_SLUG = (
    f"a_{_number_slug(NOISE_A_MULTIPLIER)}_"
    f"b_{_number_slug(NOISE_B_MULTIPLIER)}"
)
BF_METHOD_SLUG = (
    "max"
    if np.isclose(BF_CALIBRATION_QUANTILE, 1.0)
    else f"q{int(round(100.0 * BF_CALIBRATION_QUANTILE))}"
)
if not np.isclose(BF_SHRINKAGE_FACTOR, 1.0):
    BF_METHOD_SLUG += f"_x{_number_slug(BF_SHRINKAGE_FACTOR)}"
_CANDIDATE_DESIGN_SLUG = (
    "" if CANDIDATE_DESIGN == "faure" else f"cand_{CANDIDATE_DESIGN}_"
)
CONFIGURATION_SLUG = (
    f"{_CANDIDATE_DESIGN_SLUG}ninit_{N_INITIAL}_{NOISE_PARAMETER_SLUG}_"
    f"rho_{_number_slug(REPLICATION_MULTIPLIER)}_bf_{BF_METHOD_SLUG}"
)
RESULT_GROUP = (
    f"{STUDY_STAGE}_six_budget_10000_candidates_"
    f"{CONFIGURATION_SLUG}_{FORMAL_MACROS}_macros"
)

LQEI_ACQUISITION_LAMBDA_MULTIPLIER = 0.2
LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER = 0.2

EQI_BETA = 0.5
EQI_ALLOCATION_THRESHOLD = 0.5
SKO_INCUMBENT_SD_MULTIPLIER = 1.0
TSSO_MINIMUM_SEARCH_REPLICATIONS = 2

MATERN_NU = 2.5
LENGTHSCALE_BOUNDS = (0.02, 2.0)
INITIAL_MLE_MAXITER = 80
SEQUENTIAL_MLE_MAXITER = 12
LENGTHSCALE_LOG_PRIOR_SD = 1.0
PROCESS_VARIANCE_LOG_PRIOR_SD = 1.0


def round_up_five(value: float) -> int:
    return max(2, 5 * int(math.ceil(float(value) / 5.0)))


def replication_size(total_budget: int) -> int:
    """Return the common rho*log(T) nominal batch size."""

    return round_up_five(REPLICATION_MULTIPLIER * math.log(total_budget))


def make_batch_schedule(total_budget: int) -> IntArray:
    """Consume the exact budget with balanced batches nearest rho*log(T)."""

    batch_size = replication_size(total_budget)
    number_of_batches, remainder = divmod(int(total_budget), batch_size)
    if remainder > number_of_batches:
        number_of_batches += 1
    # With a 10d initial design, rho=8 would exhaust the two smallest
    # budgets before any sequential search. Rebalance only when necessary so
    # every configuration retains at least ten sequential decisions.
    number_of_batches = max(number_of_batches, N_INITIAL + MIN_SEQUENTIAL_STAGES)
    balanced_size, remainder = divmod(int(total_budget), number_of_batches)
    schedule = np.full(number_of_batches, balanced_size, dtype=int)
    schedule[:remainder] += 1
    if int(np.sum(schedule)) != int(total_budget):
        raise RuntimeError("the balanced schedule does not exhaust the budget")
    return schedule


def expected_sequential_stages(total_budget: int) -> int:
    return int(make_batch_schedule(total_budget).size - N_INITIAL)
