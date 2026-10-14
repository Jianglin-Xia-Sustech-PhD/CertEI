"""Run four algorithms on Camelback while varying negative linear noise."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, ndtr
from scipy.stats import chi2, qmc

from dynamic_gp import (
    DynamicGPFit,
    fit_dynamic_gp,
    predict_with_hyperparameters,
)
from prepare_inputs import METADATA_PATH, load_candidates, load_function, prepare_inputs
from study_config import (
    ALGORITHMS,
    BUDGET_EXPONENTS,
    BUDGETS,
    CANDIDATE_SEED,
    CONFIDENCE_DELTA,
    DESIGN_SEED,
    EQI_ALLOCATION_THRESHOLD,
    FORMAL_MACROS,
    FUNCTION_NAMES,
    FUNCTION_SEED_CODES,
    INITIAL_MLE_MAXITER,
    LENGTHSCALE_LOG_PRIOR_SD,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    MACRO_SEED_OFFSET,
    N_INITIAL,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    NOISE_SEED,
    PROCESS_VARIANCE_LOG_PRIOR_SD,
    REPLICATION_MULTIPLIER,
    RESULT_GROUP,
    SEQUENTIAL_MLE_MAXITER,
    SKO_INCUMBENT_SD_MULTIPLIER,
    TSSO_MINIMUM_SEARCH_REPLICATIONS,
    make_batch_schedule,
    replication_size,
)
from test_functions import AnalyticTestFunction


FloatArray = NDArray[np.float64]
AlgorithmName = Literal["LQEI", "TSSO", "SKO", "EQI"]

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"


@dataclass(frozen=True)
class ExperimentSpec:
    function_name: str
    algorithm: AlgorithmName
    total_budget: int
    macro_count: int = FORMAL_MACROS

    @property
    def schedule(self) -> NDArray[np.int_]:
        return make_batch_schedule(self.total_budget)

    @property
    def nominal_batch_size(self) -> int:
        return replication_size(self.total_budget)

    @property
    def slug(self) -> str:
        return (
            f"{RESULT_GROUP}/{self.function_name}/{self.algorithm}/"
            f"T_{self.total_budget}"
        )


def all_specs(macro_count: int = FORMAL_MACROS) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(function_name, algorithm, budget, macro_count)
        for function_name in FUNCTION_NAMES
        for budget in BUDGETS
        for algorithm in ALGORITHMS
    ]


def _initial_points(function_name: str, macro: int) -> FloatArray:
    function_code = FUNCTION_SEED_CODES[function_name]
    seed = np.random.SeedSequence(
        [DESIGN_SEED, function_code, macro + MACRO_SEED_OFFSET]
    )
    integer_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
    return qmc.LatinHypercube(d=2, seed=integer_seed).random(N_INITIAL)


def _expected_positive_part(
    improvement: FloatArray,
    deviation: FloatArray,
) -> FloatArray:
    values = np.maximum(improvement, 0.0)
    positive = deviation > 1e-14
    standardized = np.zeros_like(improvement)
    standardized[positive] = improvement[positive] / deviation[positive]
    density = np.exp(-0.5 * standardized[positive] ** 2) / np.sqrt(2.0 * np.pi)
    values[positive] = (
        improvement[positive] * ndtr(standardized[positive])
        + deviation[positive] * density
    )
    return np.maximum(values, 0.0)


def _log_expected_improvement(
    means: FloatArray,
    variances: FloatArray,
    incumbent: float,
) -> FloatArray:
    deviation = np.sqrt(np.maximum(variances, 0.0))
    improvement = incumbent - means
    values = np.full_like(improvement, -np.inf)
    deterministic = (deviation <= 1e-14) & (improvement > 0.0)
    values[deterministic] = np.log(improvement[deterministic])
    positive = deviation > 1e-14
    z = improvement[positive] / deviation[positive]
    log_scaled = np.empty_like(z)
    regular = z >= -8.0
    if np.any(regular):
        density = np.exp(-0.5 * z[regular] ** 2) / np.sqrt(2.0 * np.pi)
        scaled = z[regular] * ndtr(z[regular]) + density
        log_scaled[regular] = np.log(np.maximum(scaled, np.finfo(float).tiny))
    tail = ~regular
    if np.any(tail):
        tail_z = z[tail]
        log_density = -0.5 * tail_z**2 - 0.5 * np.log(2.0 * np.pi)
        log_cdf = log_ndtr(tail_z)
        difference = np.exp(log_density - log_cdf) + tail_z
        log_scaled[tail] = log_cdf + np.log(
            np.maximum(difference, np.finfo(float).tiny)
        )
    values[positive] = np.log(deviation[positive]) + log_scaled
    return values


def _integer_allocation(weights: FloatArray, budget: int) -> NDArray[np.int_]:
    raw = budget * weights / np.sum(weights)
    allocation = np.floor(raw).astype(int)
    remainder = int(budget - np.sum(allocation))
    if remainder:
        order = np.argsort(-(raw - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation


def _allocate_ocba(
    current_replications: NDArray[np.int_],
    sample_means: FloatArray,
    mean_variances: FloatArray,
    additional_budget: int,
) -> NDArray[np.int_]:
    if additional_budget <= 0:
        return current_replications.copy()
    best = int(np.argmin(sample_means))
    gaps = sample_means - sample_means[best]
    tied = np.isclose(gaps, 0.0, rtol=0.0, atol=1e-12)
    if np.count_nonzero(tied) > 1:
        weights = tied.astype(float)
    else:
        safe_variances = np.maximum(mean_variances, np.finfo(float).eps)
        weights = np.zeros_like(sample_means)
        nonbest = np.arange(sample_means.size) != best
        weights[nonbest] = safe_variances[nonbest] / gaps[nonbest] ** 2
        weights[best] = np.sqrt(safe_variances[best]) * np.sqrt(
            np.sum(weights[nonbest] ** 2 / safe_variances[nonbest])
        )
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
            weights = np.ones_like(sample_means)
    return current_replications + _integer_allocation(weights, additional_budget)


def _tsso_stage_split(
    iteration_budget: int,
    stage: int,
    stage_count: int,
) -> tuple[int, int]:
    allocation_budget = int(round(iteration_budget * stage / (stage_count + 1)))
    allocation_budget = min(
        allocation_budget,
        iteration_budget - TSSO_MINIMUM_SEARCH_REPLICATIONS,
    )
    return iteration_budget - allocation_budget, allocation_budget


def _lambda_schedule(
    function: AnalyticTestFunction,
    schedule: NDArray[np.int_],
    nominal_batch_size: int,
) -> FloatArray:
    batch_count = int(schedule.size)
    smallest_batch = int(np.min(schedule))
    alpha = CONFIDENCE_DELTA / (3.0 * batch_count)
    degrees = smallest_batch - 1
    lower_quantile = float(chi2.ppf(alpha, degrees))
    kappa = max(1.0, degrees / lower_quantile)
    indices = np.arange(batch_count + 1, dtype=float)
    effective_budget = (
        indices * nominal_batch_size / function.standardized_variance_min
    )
    kernel_bound = np.zeros_like(effective_budget)
    positive = effective_budget > 0.0
    kernel_bound[positive] = (
        effective_budget[positive] ** (2.0 / 7.0)
        * np.log1p(effective_budget[positive]) ** (5.0 / 7.0)
    )
    finite_bound = (
        0.5
        * indices
        * np.log1p(nominal_batch_size / function.standardized_variance_min)
    )
    information_bound = np.minimum(kernel_bound, finite_bound)
    return function.rkhs_norm + np.sqrt(
        2.0
        * kappa
        * (information_bound + np.log(3.0 / (2.0 * CONFIDENCE_DELTA)))
    )


def _hyperparameter_row(fit: DynamicGPFit) -> FloatArray:
    hp = fit.hyperparameters
    return np.asarray(
        [
            hp.constant_mean,
            hp.process_variance,
            hp.lengthscales[0],
            hp.lengthscales[1],
            hp.negative_log_likelihood,
            float(hp.optimizer_success),
            float(hp.optimizer_iterations),
            hp.regularization_penalty,
            hp.penalized_objective,
        ],
        dtype=float,
    )


class MacroState:
    def __init__(
        self,
        spec: ExperimentSpec,
        macro: int,
        function: AnalyticTestFunction,
        universe: FloatArray,
    ) -> None:
        self.spec = spec
        self.macro = macro
        self.seed_macro = macro + MACRO_SEED_OFFSET
        self.function = function
        self.universe = universe
        size = universe.shape[0]
        self.counts = np.zeros(size, dtype=int)
        self.sums = np.zeros(size)
        self.sums_of_squares = np.zeros(size)
        self.event_indices: list[int] = []
        self.event_types: list[str] = []
        self.event_stages: list[int] = []
        self.event_replications: list[int] = []
        self.event_true_values: list[float] = []
        self.event_true_noise_sds: list[float] = []
        self.event_sample_means: list[float] = []
        self.event_sample_variances: list[float] = []
        self.event_acquisitions: list[float] = []
        self.response_fit: DynamicGPFit | None = None
        self.noise_fit: DynamicGPFit | None = None
        self.response_log_parameters: FloatArray | None = None
        self.noise_log_parameters: FloatArray | None = None
        self.response_process_variance_log_prior_mean: float | None = None
        self.response_lengthscale_log_prior_mean: FloatArray | None = None
        self.response_hyperparameter_history: list[FloatArray] = []
        self.noise_hyperparameter_history: list[FloatArray] = []

    @property
    def budget(self) -> int:
        return int(np.sum(self.event_replications))

    def observed_indices(self) -> NDArray[np.int_]:
        return np.flatnonzero(self.counts > 0)

    def aggregate_mean(self, index: int) -> float:
        return float(self.sums[index] / self.counts[index])

    def aggregate_variance(self, index: int) -> float:
        count = int(self.counts[index])
        if count < 2:
            return self.function.variance_max
        centered_sum = self.sums_of_squares[index] - self.sums[index] ** 2 / count
        unbiased = max(float(centered_sum / (count - 1)), 0.0)
        return float(
            np.clip(
                unbiased,
                self.function.variance_min,
                self.function.variance_max,
            )
        )

    def simulate(
        self,
        index: int,
        replications: int,
        event_type: str,
        stage: int,
        acquisition: float = np.nan,
    ) -> None:
        if replications <= 0:
            return
        true_value = float(self.function.evaluate(self.universe[index])[0])
        noise_sd = float(
            self.function.noise_standard_deviation(self.universe[index])[0]
        )
        function_code = FUNCTION_SEED_CODES[self.spec.function_name]
        event_id = len(self.event_indices)
        seed = np.random.SeedSequence(
            [NOISE_SEED, function_code, self.seed_macro, event_id]
        )
        rng = np.random.Generator(np.random.Philox(seed))
        sample_mean = true_value + noise_sd * rng.standard_normal() / np.sqrt(
            replications
        )
        if replications >= 2:
            sample_variance = (
                noise_sd**2
                * rng.chisquare(replications - 1)
                / (replications - 1)
            )
        else:
            sample_variance = 0.0
        self.counts[index] += int(replications)
        self.sums[index] += replications * sample_mean
        self.sums_of_squares[index] += (
            (replications - 1) * sample_variance
            + replications * sample_mean**2
        )
        self.event_indices.append(int(index))
        self.event_types.append(event_type)
        self.event_stages.append(int(stage))
        self.event_replications.append(int(replications))
        self.event_true_values.append(true_value)
        self.event_true_noise_sds.append(noise_sd)
        self.event_sample_means.append(float(sample_mean))
        self.event_sample_variances.append(float(sample_variance))
        self.event_acquisitions.append(float(acquisition))

    def _training_arrays(self) -> tuple[NDArray[np.int_], FloatArray, FloatArray]:
        observed = self.observed_indices()
        means = np.asarray([self.aggregate_mean(index) for index in observed])
        mean_variances = np.asarray(
            [self.aggregate_variance(index) / self.counts[index] for index in observed]
        )
        return observed, means, mean_variances

    def refit_response(self, *, initial: bool = False) -> None:
        observed, means, mean_variances = self._training_arrays()
        standardized_means = self.function.standardize_outputs(means)
        standardized_mean_variances = self.function.standardize_variances(
            mean_variances
        )
        self.response_fit = fit_dynamic_gp(
            self.universe[observed],
            standardized_means,
            standardized_mean_variances,
            self.universe,
            previous_log_parameters=self.response_log_parameters,
            process_variance_log_prior_mean=(
                None
                if initial
                else self.response_process_variance_log_prior_mean
            ),
            process_variance_log_prior_sd=(
                None if initial else PROCESS_VARIANCE_LOG_PRIOR_SD
            ),
            lengthscale_log_prior_mean=(
                None if initial else self.response_lengthscale_log_prior_mean
            ),
            lengthscale_log_prior_sd=(
                None if initial else LENGTHSCALE_LOG_PRIOR_SD
            ),
            max_iterations=(
                INITIAL_MLE_MAXITER if initial else SEQUENTIAL_MLE_MAXITER
            ),
        )
        self.response_log_parameters = (
            self.response_fit.hyperparameters.log_optimization_parameters
        )
        if initial:
            self.response_process_variance_log_prior_mean = float(
                np.log(self.response_fit.hyperparameters.process_variance)
            )
            self.response_lengthscale_log_prior_mean = np.log(
                self.response_fit.hyperparameters.lengthscales
            )
        self.response_hyperparameter_history.append(
            _hyperparameter_row(self.response_fit)
        )

    def refit_noise(self, *, initial: bool = False) -> None:
        """Fit the auxiliary GP to accumulated standardized sample SDs."""

        observed = self.observed_indices()
        aggregate_variances = np.asarray(
            [self.aggregate_variance(index) for index in observed]
        )
        standardized_variances = self.function.standardize_variances(
            aggregate_variances
        )
        sample_sds = np.sqrt(standardized_variances)
        target_variances = np.asarray(
            [
                standardized_variances[local]
                / (2.0 * max(int(self.counts[index]) - 1, 1))
                for local, index in enumerate(observed)
            ]
        )
        self.noise_fit = fit_dynamic_gp(
            self.universe[observed],
            sample_sds,
            target_variances,
            self.universe,
            previous_log_parameters=self.noise_log_parameters,
            max_iterations=(
                INITIAL_MLE_MAXITER if initial else SEQUENTIAL_MLE_MAXITER
            ),
        )
        self.noise_log_parameters = (
            self.noise_fit.hyperparameters.log_optimization_parameters
        )
        self.noise_hyperparameter_history.append(
            _hyperparameter_row(self.noise_fit)
        )

    def predicted_noise_variances(self) -> FloatArray:
        """Return learned standardized single-replication variances."""

        if self.noise_fit is None:
            raise RuntimeError("the auxiliary noise GP has not been fitted")
        predicted_sd = np.clip(
            self.noise_fit.posterior_mean,
            NOISE_SD_MIN_MULTIPLIER,
            NOISE_SD_MAX_MULTIPLIER,
        )
        return predicted_sd**2


def _sampled_recommendation(
    state: MacroState,
    coefficient: float,
) -> tuple[int, float, float]:
    if state.response_fit is None:
        raise RuntimeError("response GP is not fitted")
    sampled = state.observed_indices()
    deviations = np.sqrt(state.response_fit.posterior_variance[sampled])
    scores = state.response_fit.posterior_mean[sampled] + coefficient * deviations
    local = int(np.argmin(scores))
    index = int(sampled[local])
    return index, float(state.response_fit.posterior_mean[index]), float(deviations[local])


def _sko_acquisition(state: MacroState) -> FloatArray:
    if state.response_fit is None:
        raise RuntimeError("response GP is not fitted")
    sampled = state.observed_indices()
    means = state.response_fit.posterior_mean
    variances = np.maximum(state.response_fit.posterior_variance, 0.0)
    sampled_sds = np.sqrt(variances[sampled])
    incumbent_local = int(
        np.argmin(means[sampled] + SKO_INCUMBENT_SD_MULTIPLIER * sampled_sds)
    )
    incumbent = float(means[sampled[incumbent_local]])
    deviation = np.sqrt(variances)
    expected_improvement = _expected_positive_part(incumbent - means, deviation)
    noise_variances = state.predicted_noise_variances()
    total = variances + noise_variances
    penalty = np.ones_like(total)
    positive = total > 0.0
    penalty[positive] -= np.sqrt(noise_variances[positive]) / np.sqrt(total[positive])
    return np.maximum(expected_improvement * penalty, 0.0)


def _eqi_acquisition(state: MacroState, planned_replications: int) -> FloatArray:
    if state.response_fit is None:
        raise RuntimeError("response GP is not fitted")
    sampled = state.observed_indices()
    means = state.response_fit.posterior_mean
    mse = np.maximum(state.response_fit.posterior_variance, 0.0)
    quantile_minimum = float(np.min(means[sampled]))
    new_mean_noise = state.predicted_noise_variances() / planned_replications
    total = mse + new_mean_noise
    deviation = np.zeros_like(mse)
    positive = total > 0.0
    deviation[positive] = mse[positive] / np.sqrt(total[positive])
    return _expected_positive_part(quantile_minimum - means, deviation)


def _run_lqei(
    state: MacroState,
    schedule: NDArray[np.int_],
    lambda_values: FloatArray,
) -> tuple[int, float, float]:
    for batch_index in range(N_INITIAL, schedule.size):
        assert state.response_fit is not None
        means = state.response_fit.posterior_mean
        variances = state.response_fit.posterior_variance
        deviations = np.sqrt(np.maximum(variances, 0.0))
        coefficient = LQEI_ACQUISITION_LAMBDA_MULTIPLIER * lambda_values[batch_index]
        incumbent = float(np.min(means - coefficient * deviations))
        acquisition = _expected_positive_part(incumbent - means, deviations)
        if float(np.max(acquisition)) > 0.0:
            selected = int(np.argmax(acquisition))
        else:
            selected = int(
                np.argmax(_log_expected_improvement(means, variances, incumbent))
            )
        state.simulate(
            selected,
            int(schedule[batch_index]),
            "lqei_infill",
            batch_index - N_INITIAL + 1,
            float(acquisition[selected]),
        )
        state.refit_response()
    terminal_coefficient = (
        LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER * lambda_values[-1]
    )
    return _sampled_recommendation(state, terminal_coefficient)


def _run_sko(
    state: MacroState,
    schedule: NDArray[np.int_],
) -> tuple[int, float, float]:
    for batch_index in range(N_INITIAL, schedule.size):
        acquisition = _sko_acquisition(state)
        selected = int(np.argmax(acquisition))
        state.simulate(
            selected,
            int(schedule[batch_index]),
            "sko_infill",
            batch_index - N_INITIAL + 1,
            float(acquisition[selected]),
        )
        state.refit_response()
        state.refit_noise()
    return _sampled_recommendation(state, SKO_INCUMBENT_SD_MULTIPLIER)


def _run_eqi(
    state: MacroState,
    schedule: NDArray[np.int_],
) -> tuple[int, float, float]:
    locked_index: int | None = None
    locked_reference = 0.0
    for batch_index in range(N_INITIAL, schedule.size):
        acquisition = _eqi_acquisition(state, int(schedule[batch_index]))
        if (
            locked_index is not None
            and acquisition[locked_index]
            > EQI_ALLOCATION_THRESHOLD * locked_reference
        ):
            selected = locked_index
            event_type = "eqi_allocation"
        else:
            selected = int(np.argmax(acquisition))
            locked_index = selected
            locked_reference = float(acquisition[selected])
            event_type = "eqi_new_selection"
        state.simulate(
            selected,
            int(schedule[batch_index]),
            event_type,
            batch_index - N_INITIAL + 1,
            float(acquisition[selected]),
        )
        state.refit_response()
        state.refit_noise()
    return _sampled_recommendation(state, 0.0)


def _run_tsso(
    state: MacroState,
    schedule: NDArray[np.int_],
) -> tuple[int, float, float]:
    candidate_offset = N_INITIAL
    available = np.ones(load_candidates(state.spec.function_name).shape[0], dtype=bool)
    stage_budgets = schedule[N_INITIAL:]
    stage_count = int(stage_budgets.size)
    for stage, iteration_budget in enumerate(stage_budgets, start=1):
        assert state.response_fit is not None
        search_budget, allocation_budget = _tsso_stage_split(
            int(iteration_budget), stage, stage_count
        )
        observed = state.observed_indices()
        observed_means_raw = np.asarray(
            [state.aggregate_mean(index) for index in observed]
        )
        observed_means = state.function.standardize_outputs(observed_means_raw)
        deterministic_noise = np.full(
            observed.size,
            max(1e-10 * state.response_fit.hyperparameters.process_variance, 1e-12),
        )
        deterministic = predict_with_hyperparameters(
            state.universe[observed],
            observed_means,
            deterministic_noise,
            state.universe,
            state.response_fit.hyperparameters,
        )
        current_best = int(observed[int(np.argmin(observed_means_raw))])
        incumbent = float(state.response_fit.posterior_mean[current_best])
        available_local = np.flatnonzero(available)
        available_global = candidate_offset + available_local
        candidate_means = state.response_fit.posterior_mean[available_global]
        candidate_sds = np.sqrt(deterministic.posterior_variance[available_global])
        acquisition = _expected_positive_part(
            incumbent - candidate_means,
            candidate_sds,
        )
        selected_local = int(available_local[int(np.argmax(acquisition))])
        selected = candidate_offset + selected_local
        available[selected_local] = False
        state.simulate(
            selected,
            search_budget,
            "tsso_search",
            stage,
            float(np.max(acquisition)),
        )

        observed = state.observed_indices()
        current = state.counts[observed].copy()
        means = np.asarray([state.aggregate_mean(index) for index in observed])
        mean_variances = np.asarray(
            [state.aggregate_variance(index) / state.counts[index] for index in observed]
        )
        targets = _allocate_ocba(current, means, mean_variances, allocation_budget)
        for local_index, target in enumerate(targets):
            additional = int(target - current[local_index])
            if additional > 0:
                state.simulate(
                    int(observed[local_index]),
                    additional,
                    "tsso_ocba",
                    stage,
                )
        state.refit_response()

    observed = state.observed_indices()
    sample_means = np.asarray([state.aggregate_mean(index) for index in observed])
    selected = int(observed[int(np.argmin(sample_means))])
    assert state.response_fit is not None
    return selected, float(state.response_fit.posterior_mean[selected]), float("nan")


def _macro_path(spec: ExperimentSpec, macro: int) -> Path:
    return RESULT_DIR / spec.slug / "macro_results" / f"macro_{macro:03d}.npz"


def run_one_macro(
    spec: ExperimentSpec,
    macro: int,
    overwrite: bool = False,
) -> dict[str, object]:
    destination = _macro_path(spec, macro)
    if destination.exists() and not overwrite:
        try:
            with np.load(destination, allow_pickle=False) as data:
                return {
                    "macro": macro,
                    "normalized_gap": float(data["normalized_gap"]),
                    "status": "reused",
                }
        except (OSError, KeyError, ValueError):
            destination.unlink(missing_ok=True)

    function = load_function(spec.function_name)
    candidates = load_candidates(spec.function_name)
    initial_points = _initial_points(spec.function_name, macro)
    universe = np.vstack((initial_points, candidates))
    schedule = spec.schedule
    state = MacroState(spec, macro, function, universe)
    for index in range(N_INITIAL):
        state.simulate(index, int(schedule[index]), "initial_design", 0)
    state.refit_response(initial=True)
    if spec.algorithm in ("SKO", "EQI"):
        state.refit_noise(initial=True)

    lambda_values = _lambda_schedule(function, schedule, spec.nominal_batch_size)
    if spec.algorithm == "LQEI":
        recommendation = _run_lqei(state, schedule, lambda_values)
    elif spec.algorithm == "SKO":
        recommendation = _run_sko(state, schedule)
    elif spec.algorithm == "EQI":
        recommendation = _run_eqi(state, schedule)
    elif spec.algorithm == "TSSO":
        recommendation = _run_tsso(state, schedule)
    else:  # pragma: no cover
        raise ValueError(f"unsupported algorithm: {spec.algorithm}")
    if state.budget != spec.total_budget:
        raise RuntimeError(
            f"{spec.algorithm} used {state.budget}, expected {spec.total_budget}"
        )
    (
        recommendation_index,
        recommendation_mean_standardized,
        recommendation_sd_standardized,
    ) = recommendation
    recommendation_mean = float(
        function.unstandardize_outputs(recommendation_mean_standardized)
    )
    recommendation_sd = float(
        function.output_scale * recommendation_sd_standardized
    )
    recommendation_x = universe[recommendation_index]
    recommendation_value = float(function.evaluate(recommendation_x)[0])
    raw_gap = max(recommendation_value - function.optimum_value, 0.0)
    normalized_gap = raw_gap / function.response_range
    observed = state.observed_indices()
    aggregate_means = np.asarray([state.aggregate_mean(index) for index in observed])
    aggregate_variances = np.asarray(
        [state.aggregate_variance(index) for index in observed]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        macro=np.asarray(macro),
        macro_seed_id=np.asarray(macro + MACRO_SEED_OFFSET),
        function_name=np.asarray(spec.function_name),
        algorithm=np.asarray(spec.algorithm),
        total_budget=np.asarray(spec.total_budget),
        budget_used=np.asarray(state.budget),
        nominal_batch_size=np.asarray(spec.nominal_batch_size),
        schedule=schedule,
        initial_points=initial_points,
        candidate_count=np.asarray(candidates.shape[0]),
        optimum_points=function.optimum_points,
        optimum_value=np.asarray(function.optimum_value),
        maximum_value=np.asarray(function.maximum_value),
        response_range=np.asarray(function.response_range),
        output_location=np.asarray(function.output_location),
        output_scale=np.asarray(function.output_scale),
        gp_output_standardization=np.asarray(
            "(f - f_min) / (f_max - f_min)"
        ),
        rkhs_norm=np.asarray(function.rkhs_norm),
        lambda_values=lambda_values,
        recommendation_index=np.asarray(recommendation_index),
        recommendation_x=recommendation_x,
        recommendation_posterior_mean=np.asarray(recommendation_mean),
        recommendation_posterior_sd=np.asarray(recommendation_sd),
        recommendation_posterior_mean_standardized=np.asarray(
            recommendation_mean_standardized
        ),
        recommendation_posterior_sd_standardized=np.asarray(
            recommendation_sd_standardized
        ),
        recommendation_value=np.asarray(recommendation_value),
        raw_gap=np.asarray(raw_gap),
        normalized_gap=np.asarray(normalized_gap),
        observed_indices=observed,
        observed_points=universe[observed],
        observed_replications=state.counts[observed],
        observed_sample_means=aggregate_means,
        observed_sample_variances=aggregate_variances,
        event_indices=np.asarray(state.event_indices, dtype=int),
        event_points=universe[np.asarray(state.event_indices, dtype=int)],
        event_types=np.asarray(state.event_types),
        event_stages=np.asarray(state.event_stages, dtype=int),
        event_replications=np.asarray(state.event_replications, dtype=int),
        event_true_values=np.asarray(state.event_true_values),
        event_true_noise_sds=np.asarray(state.event_true_noise_sds),
        event_sample_means=np.asarray(state.event_sample_means),
        event_sample_variances=np.asarray(state.event_sample_variances),
        event_acquisition_values=np.asarray(state.event_acquisitions),
        response_hyperparameter_columns=np.asarray(
            ["constant_mean", "process_variance", "lengthscale_1", "lengthscale_2", "nll", "optimizer_success", "optimizer_iterations", "regularization_penalty", "penalized_objective"]
        ),
        response_hyperparameter_history=np.asarray(
            state.response_hyperparameter_history
        ),
        noise_hyperparameter_columns=np.asarray(
            ["constant_mean", "process_variance", "lengthscale_1", "lengthscale_2", "nll", "optimizer_success", "optimizer_iterations", "regularization_penalty", "penalized_objective"]
        ),
        noise_hyperparameter_history=np.asarray(
            state.noise_hyperparameter_history
        ),
        response_hyperparameters_refit_every_stage=np.asarray(True),
        noise_hyperparameters_refit_every_stage=np.asarray(
            spec.algorithm in ("SKO", "EQI")
        ),
        auxiliary_noise_gp_used=np.asarray(spec.algorithm in ("SKO", "EQI")),
        analytic_noise_prediction_used=np.asarray(False),
        noise_gp_output=np.asarray("standardized sample standard deviation"),
        kernel=np.asarray("matern52"),
        response_lengthscale_log_prior_mean=np.asarray(
            state.response_lengthscale_log_prior_mean
        ),
        response_process_variance_log_prior_mean=np.asarray(
            state.response_process_variance_log_prior_mean
        ),
        lengthscale_log_prior_sd=np.asarray(LENGTHSCALE_LOG_PRIOR_SD),
        process_variance_log_prior_sd=np.asarray(
            PROCESS_VARIANCE_LOG_PRIOR_SD
        ),
        noise_sd_min_multiplier=np.asarray(NOISE_SD_MIN_MULTIPLIER),
        noise_sd_max_multiplier=np.asarray(NOISE_SD_MAX_MULTIPLIER),
        noise_a_multiplier=np.asarray(NOISE_A_MULTIPLIER),
        noise_b_multiplier=np.asarray(NOISE_B_MULTIPLIER),
    )
    return {
        "macro": macro,
        "normalized_gap": normalized_gap,
        "status": "completed",
    }


def summarize_spec(spec: ExperimentSpec) -> Path:
    directory = RESULT_DIR / spec.slug
    files = sorted((directory / "macro_results").glob("macro_*.npz"))
    if len(files) != spec.macro_count:
        raise RuntimeError(
            f"{spec.slug} has {len(files)} macros, expected {spec.macro_count}"
        )
    rows: list[dict[str, object]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "macro": int(data["macro"]),
                    "raw_gap": float(data["raw_gap"]),
                    "normalized_gap": float(data["normalized_gap"]),
                    "recommendation_value": float(data["recommendation_value"]),
                    "recommendation_x1": float(data["recommendation_x"][0]),
                    "recommendation_x2": float(data["recommendation_x"][1]),
                    "unique_designs": int(data["observed_indices"].size),
                    "event_count": int(data["event_indices"].size),
                    "budget_used": int(data["budget_used"]),
                }
            )
    summary_path = directory / "macro_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gaps = np.asarray([row["normalized_gap"] for row in rows], dtype=float)
    summary = {
        "specification": asdict(spec),
        "kernel": "Matérn-5/2 with ARD lengthscales",
        "gp_output_standardization": "(f - f_min) / (f_max - f_min)",
        "response_gp_hyperparameters": (
            "constant mean, process variance and both lengthscales estimated "
            "by unregularized MLE after the initial design and weakly regularized "
            "MAP on log process variance and both log lengthscales after every "
            "sequential stage on the fixed standardized scale"
        ),
        "noise_prediction": (
            "auxiliary Matérn-5/2 GP fitted to accumulated standardized sample SDs; "
            "all hyperparameters re-estimated after every stage"
            if spec.algorithm in ("SKO", "EQI")
            else "not required"
        ),
        "auxiliary_noise_gp_used": spec.algorithm in ("SKO", "EQI"),
        "analytic_noise_prediction_used": False,
        "normalized_gap": {
            "mean": float(np.mean(gaps)),
            "standard_deviation": float(np.std(gaps, ddof=1)),
            "median": float(np.median(gaps)),
            "q25": float(np.quantile(gaps, 0.25)),
            "q75": float(np.quantile(gaps, 0.75)),
            "minimum": float(np.min(gaps)),
            "maximum": float(np.max(gaps)),
        },
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary_path


def run_experiments(
    specs: Iterable[ExperimentSpec],
    *,
    workers: int,
    overwrite: bool = False,
) -> None:
    selected = list(specs)
    prepare_inputs()
    jobs = [
        (spec, macro)
        for spec in selected
        for macro in range(1, spec.macro_count + 1)
        if overwrite or not _macro_path(spec, macro).exists()
    ]
    print(f"Running {len(jobs)} macro jobs with {workers} workers", flush=True)
    completed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one_macro, spec, macro, overwrite): (spec, macro)
                for spec, macro in jobs
            }
            for future in as_completed(futures):
                spec, macro = futures[future]
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    print(
                        f"Completed {completed}/{len(jobs)}; latest="
                        f"{spec.function_name}/{spec.algorithm}/T={spec.total_budget}/"
                        f"macro={macro}; gap={float(result['normalized_gap']):.6g}",
                        flush=True,
                    )
    for spec in selected:
        summarize_spec(spec)
    protocol = {
        "study": (
            "Six-budget Six-hump Camelback comparison with pure Sobol candidates, "
            "linear response-linked noise, a weakly regularized response "
            "GP, and learned auxiliary noise GPs"
        ),
        "functions": list(FUNCTION_NAMES),
        "algorithms": list(ALGORITHMS),
        "budgets": list(BUDGETS),
        "budget_exponents": list(BUDGET_EXPONENTS),
        "macros_per_configuration": selected[0].macro_count if selected else 0,
        "macro_seed_offset": MACRO_SEED_OFFSET,
        "initial_design_size": N_INITIAL,
        "candidate_count": int(load_candidates(FUNCTION_NAMES[0]).shape[0]),
        "candidate_rule": (
            "pure scrambled Sobol points on [0,1]^2; analytic global optima "
            "are not injected"
        ),
        "analytic_optima_injected": False,
        "replication_rule": "balanced rho*log(T)",
        "replication_multiplier": REPLICATION_MULTIPLIER,
        "noise_standard_deviation": (
            f"R_f*({NOISE_A_MULTIPLIER:.12g} + "
            f"{NOISE_B_MULTIPLIER:.12g}*z(x))"
        ),
        "noise_a_multiplier": NOISE_A_MULTIPLIER,
        "noise_b_multiplier": NOISE_B_MULTIPLIER,
        "input_metadata": str(METADATA_PATH.relative_to(ROOT)),
        "response_gp": (
            "Matérn-5/2; initial fit by MLE, then every-stage MAP with "
            "N(log(initial process variance), 1^2) and "
            "N(log(initial lengthscale), 1^2) priors; "
            "all fits use (f-f_min)/(f_max-f_min) and correspondingly scaled noise"
        ),
        "noise_prediction": (
            "SKO/EQI fit a Matérn-5/2 auxiliary GP to accumulated standardized "
            "sample SDs; the constant mean, process variance, and two ARD "
            "lengthscales are re-estimated after the initial design and every "
            "sequential stage. Predictions are clipped to the true SD envelope. The "
            "analytic noise-response relation is not supplied."
        ),
        "rkhs_norm_calibration": (
            "15x15 finite-grid norm of the standardized analytic output under a "
            "Matérn-5/2 kernel whose ARD lengthscales and process variance are "
            "pooled initial-GP medians from held-out macros 1001--1100"
        ),
        "unchanged_lqei_settings": {
            "acquisition_lambda_multiplier": LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
            "recommendation_lambda_multiplier": LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
            "dedicated_identification_stage": False,
        },
    }
    (RESULT_DIR / RESULT_GROUP / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
