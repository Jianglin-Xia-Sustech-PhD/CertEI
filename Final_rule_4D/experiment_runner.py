"""Execution engine for fixed-hyperparameter, confidence-matched LQEI."""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import qmc

from confidence import (
    INFORMATION_GAIN_CONSTANT,
    INFORMATION_GAIN_METHOD,
    ConfidenceSchedule,
    build_confidence_schedule,
)
from continuous_acquisition import PolishResult, polish_lqei, polish_lqei_log
from rkhs_functions import FixedRKHSFunction, load_fixed_function
from sequential_gp import SequentialFixedKernelGP
from study_config import (
    CANDIDATE_SEED,
    CONTINUOUS_POLISH_ENABLED,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_STARTS,
    DESIGN_SEED,
    DIMENSION,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_RULE,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    NOISE_SEED,
    RESPONSE_CONSTANT_MEAN,
    RESPONSE_LENGTHSCALE,
    RESPONSE_PROCESS_VARIANCE,
    make_batch_schedule,
    replication_size,
)


ROOT = Path(__file__).resolve().parent
FUNCTION_DIR = ROOT / "fixed_functions"
INPUT_DIR = ROOT / "shared_inputs"
RESULT_DIR = ROOT / "results"
CONFIDENCE_DIR = ROOT / "confidence_schedules"

INCUMBENT_RULE_DYNAMIC = "posterior_mean_minus_0p2_lambda_sd"
INCUMBENT_RULE_MEAN = "posterior_mean"
INCUMBENT_RULE_UPPER_ONE_SD = "posterior_mean_plus_one_sd"
INCUMBENT_RULES = (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)


@dataclass(frozen=True)
class ExperimentSpec:
    phase: str
    phase_code: int
    function_name: str
    total_budget: int
    replication_order: str
    replication_multiplier: float
    anchor_budget: int
    macro_count: int
    initial_point_count: int = N_INITIAL
    candidate_count: int = N_CANDIDATES
    confidence_delta: float = 0.05
    incumbent_rule: str = INCUMBENT_RULE_DYNAMIC

    @property
    def nominal_batch_size(self) -> int:
        return replication_size(
            self.replication_order,
            self.total_budget,
            rho=self.replication_multiplier,
            anchor_budget=self.anchor_budget,
        )

    @property
    def schedule(self) -> np.ndarray:
        return make_batch_schedule(
            self.total_budget,
            self.nominal_batch_size,
            self.initial_point_count,
        )

    @property
    def batch_size(self) -> int:
        """Smallest actual batch, used conservatively in Lambda_n."""

        return int(np.min(self.schedule))

    @property
    def slug(self) -> str:
        rho = str(self.replication_multiplier).replace(".", "p")
        delta = str(self.confidence_delta).replace(".", "p")
        incumbent_suffix = (
            ""
            if self.incumbent_rule == INCUMBENT_RULE_DYNAMIC
            else f"_incumbent_{self.incumbent_rule}"
        )
        return (
            f"{self.phase}/{self.function_name}/{self.replication_order}/"
            f"T_{self.total_budget}_rho_{rho}_delta_{delta}_dynamic_lambda"
            f"{incumbent_suffix}"
        )


def incumbent_sd_coefficient(rule: str, lambda_value: float) -> float:
    """Return signed c in the incumbent min_x(mu_n + c*sigma_n)."""

    if rule == INCUMBENT_RULE_DYNAMIC:
        return float(-LQEI_ACQUISITION_LAMBDA_MULTIPLIER * lambda_value)
    if rule == INCUMBENT_RULE_MEAN:
        return 0.0
    if rule == INCUMBENT_RULE_UPPER_ONE_SD:
        return 1.0
    raise ValueError(f"unknown incumbent rule: {rule}")


def ensure_candidate_pool(candidate_count: int = N_CANDIDATES) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = INPUT_DIR / f"sobol_candidates_{DIMENSION}d_{candidate_count}.npy"
    if destination.exists():
        points = np.load(destination)
        if points.shape == (candidate_count, DIMENSION):
            return destination
    exponent = int(math.ceil(math.log2(candidate_count)))
    sampler = qmc.Sobol(d=DIMENSION, scramble=True, seed=CANDIDATE_SEED)
    # Generate a complete power-of-two Sobol net and retain its deterministic
    # prefix. This supports the paper-matched 10,000-point finite domain.
    points = sampler.random_base2(m=exponent)[:candidate_count]
    np.save(destination, points)
    return destination


def _function_path(function_name: str) -> Path:
    return FUNCTION_DIR / f"{function_name}.npz"


def _initial_points(
    macro: int,
    phase_code: int,
    initial_point_count: int = N_INITIAL,
) -> np.ndarray:
    seed = np.random.SeedSequence([DESIGN_SEED, phase_code, macro])
    integer_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
    return qmc.LatinHypercube(d=DIMENSION, seed=integer_seed).random(
        initial_point_count
    )


def _confidence_path(spec: ExperimentSpec) -> Path:
    rho = str(spec.replication_multiplier).replace(".", "p")
    delta = str(spec.confidence_delta).replace(".", "p")
    return CONFIDENCE_DIR / (
        f"{spec.function_name}_{spec.replication_order}_T_{spec.total_budget}_"
        f"rho_{rho}_delta_{delta}_"
        f"phase_{spec.phase_code}_m_{spec.macro_count}.npz"
    )


def prepare_confidence_schedules(specs: Iterable[ExperimentSpec]) -> list[Path]:
    """Precompute deterministic Lambda schedules from the combined bound."""

    paths: list[Path] = []
    CONFIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        function = load_fixed_function(_function_path(spec.function_name))
        destination = _confidence_path(spec)
        if destination.exists():
            try:
                with np.load(destination, allow_pickle=False) as data:
                    valid = (
                        str(data["information_gain_method"])
                        == INFORMATION_GAIN_METHOD
                        and int(data["information_gain_batch_size"])
                        == spec.nominal_batch_size
                        and int(data["correction_batch_size"]) == spec.batch_size
                        and int(data["batch_count"]) == spec.schedule.size
                        and str(data["kernel"]) == function.kernel
                        and np.asarray(data["lambda_values"]).shape
                        == (spec.schedule.size + 1,)
                        and np.asarray(data["kernel_order_upper_bound"]).shape
                        == (spec.schedule.size + 1,)
                        and np.asarray(data["finite_batch_upper_bound"]).shape
                        == (spec.schedule.size + 1,)
                    )
            except (KeyError, OSError, ValueError):
                valid = False
            if valid:
                paths.append(destination)
                continue
        schedule = build_confidence_schedule(
            kernel=function.kernel,
            information_gain_batch_size=spec.nominal_batch_size,
            correction_batch_size=spec.batch_size,
            batch_count=int(spec.schedule.size),
            variance_min=function.variance_min,
            rkhs_norm=function.rkhs_norm,
            delta=spec.confidence_delta,
        )
        np.savez_compressed(
            destination,
            function_name=np.asarray(spec.function_name),
            replication_order=np.asarray(spec.replication_order),
            total_budget=np.asarray(spec.total_budget),
            phase_code=np.asarray(spec.phase_code),
            macro_count=np.asarray(spec.macro_count),
            candidate_count=np.asarray(spec.candidate_count),
            information_gain_method=np.asarray(INFORMATION_GAIN_METHOD),
            information_gain_constant=np.asarray(INFORMATION_GAIN_CONSTANT),
            information_gain_argument=np.asarray(
                "U_n = n * nominal_batch_size / variance_min"
            ),
            finite_batch_bound=np.asarray(
                "n/2 * log(1 + nominal_batch_size / variance_min)"
            ),
            delta=np.asarray(schedule.delta),
            information_gain_batch_size=np.asarray(
                schedule.information_gain_batch_size
            ),
            correction_batch_size=np.asarray(schedule.correction_batch_size),
            batch_count=np.asarray(schedule.batch_count),
            variance_min=np.asarray(schedule.variance_min),
            rkhs_norm=np.asarray(schedule.rkhs_norm),
            kappa_t=np.asarray(schedule.kappa_t),
            kernel=np.asarray(schedule.kernel),
            effective_information_budget=schedule.effective_information_budget,
            kernel_order_upper_bound=schedule.kernel_order_upper_bound,
            finite_batch_upper_bound=schedule.finite_batch_upper_bound,
            information_gain_upper_bound=schedule.information_gain_upper_bound,
            lambda_values=schedule.lambda_values,
        )
        paths.append(destination)
    return paths


def load_confidence_schedule(spec: ExperimentSpec) -> ConfidenceSchedule:
    path = _confidence_path(spec)
    if not path.exists():
        prepare_confidence_schedules([spec])
    with np.load(path, allow_pickle=False) as data:
        return ConfidenceSchedule(
            delta=float(data["delta"]),
            information_gain_batch_size=int(data["information_gain_batch_size"]),
            correction_batch_size=int(data["correction_batch_size"]),
            batch_count=int(data["batch_count"]),
            variance_min=float(data["variance_min"]),
            rkhs_norm=float(data["rkhs_norm"]),
            kappa_t=float(data["kappa_t"]),
            kernel=str(data["kernel"]),  # type: ignore[arg-type]
            effective_information_budget=np.asarray(
                data["effective_information_budget"], dtype=float
            ),
            kernel_order_upper_bound=np.asarray(
                data["kernel_order_upper_bound"], dtype=float
            ),
            finite_batch_upper_bound=np.asarray(
                data["finite_batch_upper_bound"], dtype=float
            ),
            information_gain_upper_bound=np.asarray(
                data["information_gain_upper_bound"], dtype=float
            ),
            lambda_values=np.asarray(data["lambda_values"], dtype=float),
        )


def _noise_generator(macro: int, phase_code: int, batch_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [NOISE_SEED, phase_code, macro, batch_index]
    )
    return np.random.Generator(np.random.Philox(sequence))


def _simulate_batch(
    function: FixedRKHSFunction,
    point: np.ndarray,
    replications: int,
    macro: int,
    phase_code: int,
    batch_index: int,
) -> tuple[float, float, float, float, float, np.ndarray]:
    true_value = float(function.evaluate(point.reshape(1, DIMENSION))[0])
    noise_sd = float(
        function.noise_standard_deviation(point.reshape(1, DIMENSION))[0]
    )
    rng = _noise_generator(macro, phase_code, batch_index)
    observations = true_value + noise_sd * rng.standard_normal(replications)
    sample_mean = float(np.mean(observations))
    sample_variance = float(np.var(observations, ddof=1))
    clipped_variance = float(
        np.clip(sample_variance, function.variance_min, function.variance_max)
    )
    return (
        true_value,
        noise_sd,
        sample_mean,
        sample_variance,
        clipped_variance,
        observations,
    )


def _expected_improvement(
    means: np.ndarray, variances: np.ndarray, incumbent_coefficient: float
) -> tuple[np.ndarray, float]:
    standard_deviation = np.sqrt(np.maximum(variances, 0.0))
    incumbent = float(
        np.min(means + incumbent_coefficient * standard_deviation)
    )
    improvement = incumbent - means
    values = np.maximum(improvement, 0.0)
    positive = standard_deviation > 1e-14
    standardized = np.zeros_like(improvement)
    standardized[positive] = improvement[positive] / standard_deviation[positive]
    density = np.exp(-0.5 * standardized[positive] ** 2) / np.sqrt(2.0 * np.pi)
    values[positive] = (
        improvement[positive] * ndtr(standardized[positive])
        + standard_deviation[positive] * density
    )
    return np.maximum(values, 0.0), incumbent


def _log_expected_improvement(
    means: np.ndarray, variances: np.ndarray, incumbent: float
) -> np.ndarray:
    """Evaluate log EI without underflow in extreme negative-tail cases."""

    standard_deviation = np.sqrt(np.maximum(variances, 0.0))
    improvement = incumbent - means
    values = np.full_like(improvement, -np.inf, dtype=float)
    deterministic = (standard_deviation <= 1e-14) & (improvement > 0.0)
    values[deterministic] = np.log(improvement[deterministic])
    positive = standard_deviation > 1e-14
    z = improvement[positive] / standard_deviation[positive]
    log_scaled_ei = np.empty_like(z)
    regular = z >= -8.0
    if np.any(regular):
        density = np.exp(-0.5 * z[regular] ** 2) / np.sqrt(2.0 * np.pi)
        scaled_ei = z[regular] * ndtr(z[regular]) + density
        log_scaled_ei[regular] = np.log(
            np.maximum(scaled_ei, np.finfo(float).tiny)
        )
    tail = ~regular
    if np.any(tail):
        tail_z = z[tail]
        tail_a = -tail_z
        moderate = tail_a < 1.0e4
        tail_values = np.empty_like(tail_z)
        if np.any(moderate):
            selected_z = tail_z[moderate]
            log_density = (
                -0.5 * selected_z**2 - 0.5 * np.log(2.0 * np.pi)
            )
            log_cdf = log_ndtr(selected_z)
            difference = np.exp(log_density - log_cdf) + selected_z
            tail_values[moderate] = log_cdf + np.log(
                np.maximum(difference, np.finfo(float).tiny)
            )
        if np.any(~moderate):
            selected_a = tail_a[~moderate]
            inverse_square = 1.0 / selected_a**2
            correction = np.maximum(
                1.0 - 3.0 * inverse_square + 15.0 * inverse_square**2,
                np.finfo(float).tiny,
            )
            log_density = (
                -0.5 * selected_a**2 - 0.5 * np.log(2.0 * np.pi)
            )
            tail_values[~moderate] = (
                log_density - 2.0 * np.log(selected_a) + np.log(correction)
            )
        log_scaled_ei[tail] = tail_values
    values[positive] = np.log(standard_deviation[positive]) + log_scaled_ei
    return values


def _recommendation(
    model: SequentialFixedKernelGP, coefficient: float
) -> tuple[int, float, float]:
    sampled = np.unique(model.training_indices)
    standard_deviation = np.sqrt(
        np.maximum(model.posterior_variance[sampled], 0.0)
    )
    score = model.posterior_mean[sampled] + coefficient * standard_deviation
    local = int(np.argmin(score))
    selected_index = int(sampled[local])
    return (
        selected_index,
        float(model.posterior_mean[selected_index]),
        float(standard_deviation[local]),
    )


def _polished_universe_index(
    model: SequentialFixedKernelGP,
    result: PolishResult,
) -> tuple[int, str]:
    """Reuse an existing point or append a genuinely improved polished point."""

    if not result.improved:
        raise ValueError("only improved polish results may be appended")
    distances = np.max(np.abs(model.universe - result.point), axis=1)
    matches = np.flatnonzero(distances <= 1e-10)
    if matches.size:
        return int(matches[0]), "continuous_existing"
    index = int(model.append_points(result.point)[0])
    return index, "continuous_new"


def _macro_path(spec: ExperimentSpec, macro: int) -> Path:
    return RESULT_DIR / spec.slug / "macro_results" / f"macro_{macro:03d}.npz"


def run_one_macro(spec: ExperimentSpec, macro: int, overwrite: bool = False) -> dict[str, object]:
    destination = _macro_path(spec, macro)
    if destination.exists() and not overwrite:
        with np.load(destination, allow_pickle=False) as data:
            return {
                "macro": macro,
                "normalized_gap": float(data["normalized_gap"]),
                "raw_gap": float(data["raw_gap"]),
                "status": "reused",
            }

    function = load_fixed_function(_function_path(spec.function_name))
    candidate_pool = np.load(ensure_candidate_pool(spec.candidate_count))
    initial_count = spec.initial_point_count
    initial_points = _initial_points(macro, spec.phase_code, initial_count)
    universe = np.vstack((initial_points, candidate_pool))
    schedule = spec.schedule
    batch_count = schedule.size
    confidence = load_confidence_schedule(spec)
    if (
        confidence.batch_count != batch_count
        or confidence.information_gain_batch_size != spec.nominal_batch_size
        or confidence.correction_batch_size != spec.batch_size
    ):
        raise RuntimeError("confidence schedule does not match the experiment schedule")
    lambda_values = confidence.lambda_values

    selected_indices = np.empty(batch_count, dtype=int)
    replications = schedule.copy()
    true_values = np.empty(batch_count)
    true_noise_sd = np.empty(batch_count)
    sample_means = np.empty(batch_count)
    sample_variances = np.empty(batch_count)
    clipped_variances = np.empty(batch_count)
    incumbent_history = np.full(batch_count, np.nan)
    maximum_lqei_history = np.full(batch_count, np.nan)
    coarse_maximum_lqei_history = np.full(batch_count, np.nan)
    log_maximum_lqei_history = np.full(batch_count, np.nan)
    continuous_polish_improved_history = np.zeros(batch_count, dtype=bool)
    continuous_polish_evaluations_history = np.zeros(batch_count, dtype=int)
    continuous_polish_successful_starts_history = np.zeros(batch_count, dtype=int)
    selection_source_history = np.full(batch_count, "initial", dtype="<U24")
    recommendation_index_history = np.empty(batch_count, dtype=int)
    recommendation_true_value_history = np.empty(batch_count)
    raw_observations: list[np.ndarray] = []

    for batch_index in range(initial_count):
        selected_indices[batch_index] = batch_index
        simulation = _simulate_batch(
            function,
            universe[batch_index],
            int(schedule[batch_index]),
            macro,
            spec.phase_code,
            batch_index,
        )
        true_values[batch_index] = simulation[0]
        true_noise_sd[batch_index] = simulation[1]
        sample_means[batch_index] = simulation[2]
        sample_variances[batch_index] = simulation[3]
        clipped_variances[batch_index] = simulation[4]
        raw_observations.append(simulation[5])

    if not np.isclose(function.lengthscale, RESPONSE_LENGTHSCALE):
        raise RuntimeError(
            f"{function.name} has lengthscale={function.lengthscale}, expected "
            f"the fixed response lengthscale {RESPONSE_LENGTHSCALE}"
        )
    process_variance = RESPONSE_PROCESS_VARIANCE
    constant_mean = RESPONSE_CONSTANT_MEAN
    model = SequentialFixedKernelGP(
        universe=universe,
        kernel=function.kernel,
        lengthscale=RESPONSE_LENGTHSCALE,
        process_variance=process_variance,
        constant_mean=constant_mean,
        maximum_observations=batch_count,
    )
    for batch_index in range(initial_count):
        model.add_observation(
            batch_index,
            sample_means[batch_index],
            clipped_variances[batch_index] / schedule[batch_index],
        )
        recommendation_index, _, _ = _recommendation(
            model,
            float(
                LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
                * lambda_values[batch_index + 1]
            ),
        )
        recommendation_index_history[batch_index] = recommendation_index
        recommendation_true_value_history[batch_index] = float(
            function.evaluate(
                model.universe[recommendation_index].reshape(1, DIMENSION)
            )[0]
        )

    for batch_index in range(initial_count, batch_count):
        current_incumbent_coefficient = incumbent_sd_coefficient(
            spec.incumbent_rule,
            float(lambda_values[batch_index]),
        )
        acquisition, incumbent = _expected_improvement(
            model.posterior_mean,
            model.posterior_variance,
            current_incumbent_coefficient,
        )
        if not np.all(np.isfinite(acquisition)):
            raise FloatingPointError("EI must be finite")
        coarse_value = float(np.max(acquisition))
        if coarse_value > 0.0:
            selected_index = int(np.argmax(acquisition))
            selected_value = coarse_value
            selected_log_value = float(np.log(coarse_value))
            selection_source = "sobol_coarse"
            if CONTINUOUS_POLISH_ENABLED:
                polish = polish_lqei(
                    model,
                    current_incumbent_coefficient,
                    incumbent,
                    acquisition,
                )
                continuous_polish_improved_history[batch_index] = polish.improved
                continuous_polish_evaluations_history[batch_index] = (
                    polish.function_evaluations
                )
                continuous_polish_successful_starts_history[batch_index] = (
                    polish.successful_starts
                )
                if polish.improved:
                    selected_index, selection_source = _polished_universe_index(
                        model, polish
                    )
                    selected_value = polish.acquisition
                    selected_log_value = float(np.log(selected_value))
        else:
            log_acquisition = _log_expected_improvement(
                model.posterior_mean,
                model.posterior_variance,
                incumbent,
            )
            if not np.any(np.isfinite(log_acquisition)):
                raise FloatingPointError("log EI has no finite value")
            selected_index = int(np.argmax(log_acquisition))
            selected_log_value = float(log_acquisition[selected_index])
            selected_value = float(np.exp(selected_log_value))
            selection_source = "sobol_coarse_log"
            if CONTINUOUS_POLISH_ENABLED:
                polish = polish_lqei_log(model, incumbent, log_acquisition)
                continuous_polish_improved_history[batch_index] = polish.improved
                continuous_polish_evaluations_history[batch_index] = (
                    polish.function_evaluations
                )
                continuous_polish_successful_starts_history[batch_index] = (
                    polish.successful_starts
                )
                if polish.improved:
                    selected_index, selection_source = _polished_universe_index(
                        model, polish
                    )
                    selected_log_value = polish.acquisition
                    selected_value = float(np.exp(selected_log_value))
        selected_indices[batch_index] = selected_index
        incumbent_history[batch_index] = incumbent
        coarse_maximum_lqei_history[batch_index] = coarse_value
        maximum_lqei_history[batch_index] = selected_value
        log_maximum_lqei_history[batch_index] = selected_log_value
        selection_source_history[batch_index] = selection_source
        simulation = _simulate_batch(
            function,
            model.universe[selected_index],
            int(schedule[batch_index]),
            macro,
            spec.phase_code,
            batch_index,
        )
        true_values[batch_index] = simulation[0]
        true_noise_sd[batch_index] = simulation[1]
        sample_means[batch_index] = simulation[2]
        sample_variances[batch_index] = simulation[3]
        clipped_variances[batch_index] = simulation[4]
        raw_observations.append(simulation[5])
        model.add_observation(
            selected_index,
            sample_means[batch_index],
            clipped_variances[batch_index] / schedule[batch_index],
        )
        recommendation_index, _, _ = _recommendation(
            model,
            float(
                LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
                * lambda_values[batch_index + 1]
            ),
        )
        recommendation_index_history[batch_index] = recommendation_index
        recommendation_true_value_history[batch_index] = float(
            function.evaluate(
                model.universe[recommendation_index].reshape(1, DIMENSION)
            )[0]
        )

    recommendation_coefficient = float(
        LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER * lambda_values[batch_count]
    )
    recommendation_index, recommendation_mean, recommendation_sd = _recommendation(
        model, recommendation_coefficient
    )
    universe = model.universe
    recommendation_x = universe[recommendation_index].copy()
    recommendation_value = float(
        function.evaluate(recommendation_x.reshape(1, DIMENSION))[0]
    )
    raw_gap = max(recommendation_value - function.optimum_value, 0.0)
    normalized_gap = raw_gap / function.response_range
    offsets = np.zeros(batch_count + 1, dtype=int)
    offsets[1:] = np.cumsum(schedule)
    observations = np.concatenate(raw_observations)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        macro=np.asarray(macro),
        function_name=np.asarray(spec.function_name),
        total_budget=np.asarray(spec.total_budget),
        batch_size=np.asarray(spec.batch_size),
        nominal_batch_size=np.asarray(spec.nominal_batch_size),
        replication_order=np.asarray(spec.replication_order),
        replication_multiplier=np.asarray(spec.replication_multiplier),
        initial_point_count=np.asarray(initial_count),
        confidence_delta=np.asarray(spec.confidence_delta),
        confidence_method=np.asarray(INFORMATION_GAIN_METHOD),
        information_gain_constant=np.asarray(INFORMATION_GAIN_CONSTANT),
        information_gain_argument=np.asarray(
            "U_n = n * nominal_batch_size / variance_min"
        ),
        finite_batch_bound=np.asarray(
            "n/2 * log(1 + nominal_batch_size / variance_min)"
        ),
        kappa_t=np.asarray(confidence.kappa_t),
        effective_information_budget=confidence.effective_information_budget,
        kernel_order_upper_bound=confidence.kernel_order_upper_bound,
        finite_batch_upper_bound=confidence.finite_batch_upper_bound,
        information_gain_upper_bound=confidence.information_gain_upper_bound,
        lambda_values=lambda_values,
        incumbent_rule=np.asarray(spec.incumbent_rule),
        incumbent_sd_coefficient_history=np.asarray(
            [
                incumbent_sd_coefficient(spec.incumbent_rule, value)
                for value in lambda_values[:-1]
            ],
            dtype=float,
        ),
        acquisition_lambda_multiplier=np.asarray(
            LQEI_ACQUISITION_LAMBDA_MULTIPLIER
            if spec.incumbent_rule == INCUMBENT_RULE_DYNAMIC
            else np.nan
        ),
        acquisition_optimizer=np.asarray(
            f"sobol_{spec.candidate_count}_plus_"
            f"{CONTINUOUS_POLISH_STARTS}start_lbfgsb"
        ),
        continuous_polish_enabled=np.asarray(CONTINUOUS_POLISH_ENABLED),
        continuous_polish_starts=np.asarray(CONTINUOUS_POLISH_STARTS),
        continuous_polish_maxiter=np.asarray(CONTINUOUS_POLISH_MAXITER),
        recommendation_rule=np.asarray(LQEI_RECOMMENDATION_RULE),
        recommendation_lambda_multiplier=np.asarray(
            LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
        ),
        recommendation_coefficient=np.asarray(recommendation_coefficient),
        initial_points=initial_points,
        candidate_count=np.asarray(spec.candidate_count),
        universe_points=universe,
        continuous_points_added=np.asarray(
            universe.shape[0] - candidate_pool.shape[0] - initial_count
        ),
        selected_indices=selected_indices,
        selected_points=universe[selected_indices],
        replications=replications,
        true_values=true_values,
        true_noise_sd=true_noise_sd,
        noise_sd_min_multiplier=np.asarray(NOISE_SD_MIN_MULTIPLIER),
        noise_sd_max_multiplier=np.asarray(NOISE_SD_MAX_MULTIPLIER),
        sample_means=sample_means,
        sample_variances=sample_variances,
        clipped_variances=clipped_variances,
        sample_mean_variances=clipped_variances / schedule,
        raw_observation_offsets=offsets,
        raw_observations=observations,
        incumbent_history=incumbent_history,
        coarse_maximum_lqei_history=coarse_maximum_lqei_history,
        maximum_lqei_history=maximum_lqei_history,
        log_maximum_lqei_history=log_maximum_lqei_history,
        continuous_polish_improved_history=continuous_polish_improved_history,
        continuous_polish_evaluations_history=continuous_polish_evaluations_history,
        continuous_polish_successful_starts_history=continuous_polish_successful_starts_history,
        selection_source_history=selection_source_history,
        recommendation_index_history=recommendation_index_history,
        recommendation_true_value_history=recommendation_true_value_history,
        recommendation_index=np.asarray(recommendation_index),
        recommendation_x=recommendation_x,
        recommendation_value=np.asarray(recommendation_value),
        recommendation_posterior_mean=np.asarray(recommendation_mean),
        recommendation_posterior_sd=np.asarray(recommendation_sd),
        optimum_x=function.optimum_x,
        optimum_value=np.asarray(function.optimum_value),
        raw_gap=np.asarray(raw_gap),
        normalized_gap=np.asarray(normalized_gap),
        fitted_process_variance=np.asarray(process_variance),
        fitted_constant_mean=np.asarray(constant_mean),
        fitted_lengthscale=np.asarray(RESPONSE_LENGTHSCALE),
        response_hyperparameters_fixed=np.asarray(True),
        design_seed=np.asarray(DESIGN_SEED),
        noise_seed=np.asarray(NOISE_SEED),
        phase_code=np.asarray(spec.phase_code),
    )
    return {
        "macro": macro,
        "normalized_gap": normalized_gap,
        "raw_gap": raw_gap,
        "status": "completed",
    }


def _write_configuration_summary(spec: ExperimentSpec) -> Path:
    directory = RESULT_DIR / spec.slug
    macro_files = sorted((directory / "macro_results").glob("macro_*.npz"))
    if len(macro_files) != spec.macro_count:
        raise RuntimeError(
            f"{spec.slug} has {len(macro_files)} macros; expected {spec.macro_count}"
        )
    rows: list[dict[str, object]] = []
    gaps = []
    for path in macro_files:
        with np.load(path, allow_pickle=False) as data:
            row = {
                "macro": int(data["macro"]),
                "raw_gap": float(data["raw_gap"]),
                "recommendation_x1": float(data["recommendation_x"][0]),
                "recommendation_x2": float(data["recommendation_x"][1]),
                "recommendation_x3": float(data["recommendation_x"][2]),
                "recommendation_x4": float(data["recommendation_x"][3]),
                "recommendation_value": float(data["recommendation_value"]),
                "fitted_process_variance": float(data["fitted_process_variance"]),
                "fitted_constant_mean": float(data["fitted_constant_mean"]),
            }
            # Normalization remains an internal pilot device so that the two
            # differently scaled RKHS functions can be pooled for tuning.  It
            # is deliberately omitted from formal result displays.
            if spec.phase == "pilot":
                row["normalized_gap"] = float(data["normalized_gap"])
            rows.append(row)
            gaps.append(float(row["raw_gap"]))
    with (directory / "macro_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray(gaps)
    confidence = load_confidence_schedule(spec)
    summary = {
        "specification": asdict(spec),
        "metric": "raw_gap",
        "nominal_batch_size": spec.nominal_batch_size,
        "confidence_batch_size": spec.batch_size,
        "number_of_batches": int(
            spec.schedule.size
        ),
        "initial_point_count": spec.initial_point_count,
        "mean_gap": float(np.mean(values)),
        "median_gap": float(np.median(values)),
        "q25_gap": float(np.quantile(values, 0.25)),
        "q75_gap": float(np.quantile(values, 0.75)),
        "q90_gap": float(np.quantile(values, 0.90)),
        "minimum_gap": float(np.min(values)),
        "maximum_gap": float(np.max(values)),
        "confidence_method": INFORMATION_GAIN_METHOD,
        "information_gain_constant": INFORMATION_GAIN_CONSTANT,
        "information_gain_argument": "U_n = n * nominal_batch_size / variance_min",
        "finite_batch_bound": (
            "n/2 * log(1 + nominal_batch_size / variance_min)"
        ),
        "kappa_t": confidence.kappa_t,
        "lambda_0": float(confidence.lambda_values[0]),
        "lambda_after_initial_design": float(
            confidence.lambda_values[spec.initial_point_count]
        ),
        "lambda_terminal": float(confidence.lambda_values[-1]),
        "incumbent_rule": spec.incumbent_rule,
        "incumbent_formula": {
            INCUMBENT_RULE_DYNAMIC: "min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            INCUMBENT_RULE_MEAN: "min_x(mu_n)",
            INCUMBENT_RULE_UPPER_ONE_SD: "min_x(mu_n + sigma_n)",
        }[spec.incumbent_rule],
        "acquisition_lambda_multiplier": (
            LQEI_ACQUISITION_LAMBDA_MULTIPLIER
            if spec.incumbent_rule == INCUMBENT_RULE_DYNAMIC
            else None
        ),
        "acquisition_optimizer": {
            "global_screen": f"{N_CANDIDATES}-point scrambled Sobol pool",
            "continuous_method": "L-BFGS-B",
            "starts": CONTINUOUS_POLISH_STARTS,
            "maximum_iterations_per_start": CONTINUOUS_POLISH_MAXITER,
            "bounds": [[0.0, 1.0]] * DIMENSION,
        },
        "recommendation_rule": LQEI_RECOMMENDATION_RULE,
        "recommendation_lambda_multiplier": (
            LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
        ),
        "terminal_recommendation_coefficient": float(
            LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
            * confidence.lambda_values[-1]
        ),
        "effective_U_terminal": float(
            confidence.effective_information_budget[-1]
        ),
        "kernel_order_terminal_upper_bound": float(
            confidence.kernel_order_upper_bound[-1]
        ),
        "finite_batch_terminal_upper_bound": float(
            confidence.finite_batch_upper_bound[-1]
        ),
        "gamma_terminal_upper_bound": float(
            confidence.information_gain_upper_bound[-1]
        ),
        "response_gp_hyperparameters": {
            "constant_mean_beta": RESPONSE_CONSTANT_MEAN,
            "process_variance_tau_squared": RESPONSE_PROCESS_VARIANCE,
            "lengthscale": RESPONSE_LENGTHSCALE,
            "fixed": True,
        },
    }
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return directory / "summary.json"


def run_specs(
    specs: Iterable[ExperimentSpec],
    *,
    workers: int = 8,
    overwrite: bool = False,
    macro_limit: int | None = None,
) -> None:
    selected_specs = list(specs)
    prepare_confidence_schedules(selected_specs)
    if macro_limit is not None and macro_limit < 1:
        raise ValueError("macro_limit must be positive")
    jobs = [
        (spec, macro)
        for spec in selected_specs
        for macro in range(1, min(spec.macro_count, macro_limit or spec.macro_count) + 1)
        if overwrite or not _macro_path(spec, macro).exists()
    ]
    print(
        f"Running {len(jobs)} macro jobs across {len(selected_specs)} configurations "
        f"with {workers} workers.",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one_macro, spec, macro, overwrite): (spec, macro)
                for spec, macro in jobs
            }
            completed = 0
            for future in as_completed(futures):
                spec, macro = futures[future]
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"Completed {completed}/{len(futures)} macro jobs; "
                        f"latest={spec.function_name}/{spec.replication_order}/"
                        f"T={spec.total_budget}/macro={macro}",
                        flush=True,
                    )
    if macro_limit is None or all(macro_limit >= spec.macro_count for spec in selected_specs):
        for spec in selected_specs:
            _write_configuration_summary(spec)
