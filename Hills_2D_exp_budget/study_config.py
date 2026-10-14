"""Authoritative configuration for the negative-noise Hills comparison."""

from __future__ import annotations

import math
import os

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int_]

FUNCTION_NAMES = ("hills",)
FUNCTION_SEED_CODES = {"hills": 0}
ALGORITHMS = ("LQEI", "TSSO", "SKO", "EQI")
BUDGET_EXPONENTS = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
BUDGETS = tuple(int(round(math.exp(exponent))) for exponent in BUDGET_EXPONENTS)
CALIBRATION_BUDGETS = (1808, 8103, 22026)

N_INITIAL = 20
N_CANDIDATES = 2048
FORMAL_MACROS = int(os.environ.get("HILLS_NEGATIVE_MACROS", "100"))
CALIBRATION_MACROS = 100
CALIBRATION_MACRO_OFFSET = 1000
REPLICATION_MULTIPLIER = float(
    os.environ.get("HILLS_NEGATIVE_REPLICATION_MULTIPLIER", "8.0")
)
CONFIDENCE_DELTA = 0.05

DESIGN_SEED = 20260924
CANDIDATE_SEED = 20260925
NOISE_SEED = 20260926

# Negative-slope response-linked noise SD, sigma/R_f = a + b*z, where the
# standardized minimization response z lies in [0,1].
NOISE_B_MULTIPLIER = float(os.environ.get("HILLS_NEGATIVE_NOISE_B", "-1.5"))
NOISE_A_MULTIPLIER = float(os.environ.get("HILLS_NEGATIVE_NOISE_A", "8.0"))
NOISE_SD_MIN_MULTIPLIER = min(
    NOISE_A_MULTIPLIER, NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER
)
NOISE_SD_MAX_MULTIPLIER = max(
    NOISE_A_MULTIPLIER, NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER
)
STUDY_STAGE = os.environ.get("HILLS_NEGATIVE_STAGE", "formal").strip().lower()

if not np.isfinite(NOISE_B_MULTIPLIER) or NOISE_B_MULTIPLIER >= 0.0:
    raise ValueError("HILLS_NEGATIVE_NOISE_B must be finite and strictly negative")
if not np.isfinite(NOISE_A_MULTIPLIER) or NOISE_A_MULTIPLIER <= 0.0:
    raise ValueError("HILLS_NEGATIVE_NOISE_A must be positive and finite")
if NOISE_A_MULTIPLIER + NOISE_B_MULTIPLIER <= 0.0:
    raise ValueError("a+b must be positive so the noise SD is positive on [0,1]")
if not np.isfinite(REPLICATION_MULTIPLIER) or REPLICATION_MULTIPLIER <= 0.0:
    raise ValueError(
        "HILLS_NEGATIVE_REPLICATION_MULTIPLIER must be positive and finite"
    )
if not np.isclose(REPLICATION_MULTIPLIER, 8.0):
    raise ValueError("This study fixes the replication multiplier rho at 8")
if FORMAL_MACROS <= 0:
    raise ValueError("HILLS_NEGATIVE_MACROS must be positive")
if STUDY_STAGE not in {"pilot", "formal"}:
    raise ValueError("HILLS_NEGATIVE_STAGE must be 'pilot' or 'formal'")

# The formal confirmation uses independent macroreplication streams.
MACRO_SEED_OFFSET = 0 if STUDY_STAGE == "pilot" else 10_000


def _number_slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


NOISE_PARAMETER_SLUG = (
    f"a_{_number_slug(NOISE_A_MULTIPLIER)}_"
    f"b_{_number_slug(NOISE_B_MULTIPLIER)}"
)
CONFIGURATION_SLUG = (
    f"{NOISE_PARAMETER_SLUG}_rho_{_number_slug(REPLICATION_MULTIPLIER)}"
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

# Both response and auxiliary-noise GPs use Matérn-5/2 kernels. Response-GP
# hyperparameters are initially estimated by MLE and then updated by weak MAP
# at every sequential stage. There is deliberately no hyperparameter-freezing
# rule in this study. SKO/EQI likewise re-estimate their auxiliary noise GP at
# every stage and are not supplied with the analytic noise relationship.
MATERN_NU = 2.5
LENGTHSCALE_BOUNDS = (0.02, 2.0)
INITIAL_MLE_MAXITER = 80
SEQUENTIAL_MLE_MAXITER = 12
LENGTHSCALE_LOG_PRIOR_SD = 1.0
PROCESS_VARIANCE_LOG_PRIOR_SD = 1.0


def round_up_five(value: float) -> int:
    return max(2, 5 * int(math.ceil(float(value) / 5.0)))


def replication_size(total_budget: int) -> int:
    """Return the common fixed-rho nominal batch size."""

    return round_up_five(REPLICATION_MULTIPLIER * math.log(total_budget))


def make_batch_schedule(total_budget: int) -> IntArray:
    """Consume the exact budget with balanced batches nearest nominal b_T."""

    batch_size = replication_size(total_budget)
    number_of_batches, remainder = divmod(int(total_budget), batch_size)
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
