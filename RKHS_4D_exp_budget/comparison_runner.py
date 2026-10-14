"""Budget-matched SKO, EQI, and TSSO experiments on fixed 4D RKHS functions."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

from continuous_acquisition import polish_eqi, polish_sko, polish_tsso
from experiment_runner import (
    FUNCTION_DIR,
    RESULT_DIR,
    _initial_points,
    _noise_generator,
    ensure_candidate_pool,
)
from rkhs_functions import (
    FixedRKHSFunction,
    kernel_matrix,
    kernel_vector_and_gradient,
    load_fixed_function,
)
from sequential_gp import SequentialFixedKernelGP, fit_initial_process_parameters
from sko import (
    SKO_INCUMBENT_SD_MULTIPLIER,
    stochastic_kriging_optimization_acquisition,
)
from study_config import (
    CONTINUOUS_POLISH_ENABLED,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_STARTS,
    DIMENSION,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    N_INITIAL,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    RESPONSE_CONSTANT_MEAN,
    RESPONSE_LENGTHSCALE,
    RESPONSE_PROCESS_VARIANCE,
    make_batch_schedule,
    make_mixed_batch_schedule,
    replication_size,
)


AlgorithmName = Literal["SKO", "EQI", "TSSO"]
COMPARISON_ALGORITHMS: tuple[AlgorithmName, ...] = ("SKO", "EQI", "TSSO")
EQI_BETA = 0.5
EQI_ALLOCATION_THRESHOLD = 0.5
TSSO_MINIMUM_SEARCH_REPLICATIONS = 2


@dataclass(frozen=True)
class _TSSOMeanPosterior:
    """Aggregated stochastic-Kriging mean used by continuous TSSO search."""

    observed_points: np.ndarray
    weights: np.ndarray
    kernel: str
    lengthscale: float
    process_variance: float
    constant_mean: float

    def mean_with_gradient(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        values, gradients = kernel_vector_and_gradient(
            point,
            self.observed_points,
            self.kernel,  # type: ignore[arg-type]
            self.lengthscale,
        )
        cross = self.process_variance * values
        cross_gradient = self.process_variance * gradients
        mean = float(self.constant_mean + cross @ self.weights)
        mean_gradient = np.asarray(cross_gradient.T @ self.weights, dtype=float)
        return mean, mean_gradient


@dataclass(frozen=True)
class ComparisonSpec:
    algorithm: AlgorithmName
    phase_code: int
    function_name: str
    total_budget: int
    replication_multiplier: float
    anchor_budget: int
    macro_count: int
    candidate_count: int
    initial_replications: int | None = None
    result_group: str = "comparison"

    @property
    def nominal_batch_size(self) -> int:
        return replication_size(
            "log",
            self.total_budget,
            rho=self.replication_multiplier,
            anchor_budget=self.anchor_budget,
        )

    @property
    def schedule(self) -> np.ndarray:
        if self.initial_replications is None:
            return make_batch_schedule(self.total_budget, self.nominal_batch_size)
        return make_mixed_batch_schedule(
            self.total_budget,
            self.nominal_batch_size,
            self.initial_replications,
        )

    @property
    def sequential_stage_count(self) -> int:
        return int(self.schedule.size - N_INITIAL)

    @property
    def slug(self) -> str:
        rho = str(self.replication_multiplier).replace(".", "p")
        initial = (
            "" if self.initial_replications is None else f"_b0_{self.initial_replications}"
        )
        return (
            f"{self.result_group}/{self.function_name}/{self.algorithm}/"
            f"T_{self.total_budget}_rho_{rho}{initial}"
        )


def _expected_positive_part(
    improvement: np.ndarray, deviation: np.ndarray
) -> np.ndarray:
    values = np.maximum(improvement, 0.0)
    positive = deviation > np.finfo(float).eps
    standardized = np.zeros_like(improvement)
    standardized[positive] = improvement[positive] / deviation[positive]
    density = np.exp(-0.5 * standardized[positive] ** 2) / np.sqrt(2.0 * np.pi)
    values[positive] = (
        improvement[positive] * ndtr(standardized[positive])
        + deviation[positive] * density
    )
    return np.maximum(values, 0.0)


def _integer_allocation(weights: np.ndarray, budget: int) -> np.ndarray:
    raw = budget * weights / np.sum(weights)
    allocation = np.floor(raw).astype(int)
    remainder = budget - int(np.sum(allocation))
    if remainder:
        order = np.argsort(-(raw - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation


def _allocate_ocba(
    current_replications: np.ndarray,
    sample_means: np.ndarray,
    mean_variances: np.ndarray,
    additional_budget: int,
) -> np.ndarray:
    """Return exact integer OCBA target totals (the OCBAv2 rule)."""

    if additional_budget == 0:
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


class _MacroState:
    """One macro's observations, aggregate statistics, and sequential GPs."""

    def __init__(
        self,
        spec: ComparisonSpec,
        macro: int,
        function: FixedRKHSFunction,
        universe: np.ndarray,
        response_model: SequentialFixedKernelGP,
        noise_model: SequentialFixedKernelGP | None,
    ) -> None:
        self.spec = spec
        self.macro = macro
        self.function = function
        self.universe = universe
        self.response_model = response_model
        self.noise_model = noise_model
        size = universe.shape[0]
        self.counts = np.zeros(size, dtype=int)
        self.sums = np.zeros(size, dtype=float)
        self.sums_of_squares = np.zeros(size, dtype=float)
        self.event_indices: list[int] = []
        self.event_types: list[str] = []
        self.event_stages: list[int] = []
        self.replications: list[int] = []
        self.true_values: list[float] = []
        self.true_noise_sds: list[float] = []
        self.sample_means: list[float] = []
        self.sample_variances: list[float] = []
        self.clipped_variances: list[float] = []
        self.acquisition_values: list[float] = []
        self.coarse_acquisition_values: list[float] = []
        self.polish_improved: list[bool] = []
        self.polish_evaluations: list[int] = []
        self.polish_successful_starts: list[int] = []
        self.raw_observations: list[np.ndarray] = []

    @property
    def event_count(self) -> int:
        return len(self.event_indices)

    @property
    def budget(self) -> int:
        return int(np.sum(self.replications))

    def aggregate_mean(self, universe_index: int) -> float:
        return float(self.sums[universe_index] / self.counts[universe_index])

    def aggregate_variance(self, universe_index: int) -> float:
        count = int(self.counts[universe_index])
        if count < 2:
            return self.function.variance_max
        numerator = (
            self.sums_of_squares[universe_index]
            - self.sums[universe_index] ** 2 / count
        )
        variance = max(float(numerator / (count - 1)), 0.0)
        return float(
            np.clip(
                variance,
                self.function.variance_min,
                self.function.variance_max,
            )
        )

    def observed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.counts > 0)

    def append_continuous_point(self, point: np.ndarray) -> int:
        """Append one polished design to the state and every active GP."""

        location = np.asarray(point, dtype=float).reshape(1, DIMENSION)
        old_size = self.universe.shape[0]
        response_index = int(self.response_model.append_points(location)[0])
        if response_index != old_size:
            raise RuntimeError("response GP returned an inconsistent append index")
        if self.noise_model is not None:
            noise_index = int(self.noise_model.append_points(location)[0])
            if noise_index != old_size:
                raise RuntimeError("noise GP returned an inconsistent append index")
        self.universe = np.vstack((self.universe, location))
        self.counts = np.append(self.counts, 0)
        self.sums = np.append(self.sums, 0.0)
        self.sums_of_squares = np.append(self.sums_of_squares, 0.0)
        return old_size

    def simulate_and_update(
        self,
        universe_index: int,
        replications: int,
        event_type: str,
        stage: int,
        acquisition_value: float = np.nan,
        coarse_acquisition_value: float = np.nan,
        polish_improved: bool = False,
        polish_evaluations: int = 0,
        polish_successful_starts: int = 0,
        update_response_model: bool = True,
    ) -> None:
        point = self.universe[universe_index]
        true_value = float(
            self.function.evaluate(point.reshape(1, DIMENSION))[0]
        )
        noise_sd = float(
            self.function.noise_standard_deviation(
                point.reshape(1, DIMENSION)
            )[0]
        )
        rng = _noise_generator(
            self.macro,
            self.spec.phase_code,
            self.event_count,
        )
        observations = true_value + noise_sd * rng.standard_normal(replications)
        sample_mean = float(np.mean(observations))
        if replications >= 2:
            sample_variance = float(np.var(observations, ddof=1))
            clipped_variance = float(
                np.clip(
                    sample_variance,
                    self.function.variance_min,
                    self.function.variance_max,
                )
            )
        else:
            sample_variance = float("nan")
            clipped_variance = self.aggregate_variance(universe_index)

        self.counts[universe_index] += replications
        self.sums[universe_index] += float(np.sum(observations))
        self.sums_of_squares[universe_index] += float(observations @ observations)
        self.event_indices.append(int(universe_index))
        self.event_types.append(event_type)
        self.event_stages.append(stage)
        self.replications.append(int(replications))
        self.true_values.append(true_value)
        self.true_noise_sds.append(noise_sd)
        self.sample_means.append(sample_mean)
        self.sample_variances.append(sample_variance)
        self.clipped_variances.append(clipped_variance)
        self.acquisition_values.append(float(acquisition_value))
        self.coarse_acquisition_values.append(float(coarse_acquisition_value))
        self.polish_improved.append(bool(polish_improved))
        self.polish_evaluations.append(int(polish_evaluations))
        self.polish_successful_starts.append(int(polish_successful_starts))
        self.raw_observations.append(observations)

        if update_response_model:
            self.response_model.add_observation(
                universe_index,
                sample_mean,
                clipped_variance / replications,
            )
        if self.noise_model is not None:
            sd_target = float(np.sqrt(clipped_variance))
            sd_estimation_variance = clipped_variance / (
                2.0 * max(replications - 1, 1)
            )
            self.noise_model.add_observation(
                universe_index,
                sd_target,
                sd_estimation_variance,
            )


def _build_initial_state(
    spec: ComparisonSpec,
    macro: int,
    maximum_observations: int,
    with_noise_model: bool,
) -> tuple[_MacroState, np.ndarray]:
    function = load_fixed_function(FUNCTION_DIR / f"{spec.function_name}.npz")
    candidates = np.load(ensure_candidate_pool(spec.candidate_count))
    initial_points = _initial_points(macro, spec.phase_code)
    universe = np.vstack((initial_points, candidates))
    full_schedule = spec.schedule
    initial_schedule = full_schedule[:N_INITIAL]

    initial_means = np.empty(N_INITIAL)
    initial_variances = np.empty(N_INITIAL)
    initial_raw: list[np.ndarray] = []
    for index in range(N_INITIAL):
        point = initial_points[index]
        true_value = float(function.evaluate(point.reshape(1, DIMENSION))[0])
        noise_sd = float(
            function.noise_standard_deviation(point.reshape(1, DIMENSION))[0]
        )
        rng = _noise_generator(macro, spec.phase_code, index)
        observations = true_value + noise_sd * rng.standard_normal(
            int(initial_schedule[index])
        )
        sample_variance = float(np.var(observations, ddof=1))
        initial_means[index] = float(np.mean(observations))
        initial_variances[index] = float(
            np.clip(sample_variance, function.variance_min, function.variance_max)
        )
        initial_raw.append(observations)

    if not np.isclose(function.lengthscale, RESPONSE_LENGTHSCALE):
        raise RuntimeError(
            f"{function.name} has lengthscale={function.lengthscale}, expected "
            f"the fixed response lengthscale {RESPONSE_LENGTHSCALE}"
        )
    response_variance = RESPONSE_PROCESS_VARIANCE
    response_mean = RESPONSE_CONSTANT_MEAN
    response_model = SequentialFixedKernelGP(
        universe=universe,
        kernel=function.kernel,
        lengthscale=RESPONSE_LENGTHSCALE,
        process_variance=response_variance,
        constant_mean=response_mean,
        maximum_observations=maximum_observations,
    )
    noise_model: SequentialFixedKernelGP | None = None
    if with_noise_model:
        sd_targets = np.sqrt(initial_variances)
        sd_target_variances = initial_variances / (
            2.0 * np.maximum(initial_schedule - 1, 1)
        )
        noise_variance, noise_mean = fit_initial_process_parameters(
            initial_points,
            sd_targets,
            sd_target_variances,
            function.kernel,
            function.lengthscale,
        )
        noise_model = SequentialFixedKernelGP(
            universe=universe,
            kernel=function.kernel,
            lengthscale=function.lengthscale,
            process_variance=noise_variance,
            constant_mean=noise_mean,
            maximum_observations=maximum_observations,
        )

    state = _MacroState(
        spec,
        macro,
        function,
        universe,
        response_model,
        noise_model,
    )
    for index in range(N_INITIAL):
        observations = initial_raw[index]
        count = int(initial_schedule[index])
        state.counts[index] = count
        state.sums[index] = float(np.sum(observations))
        state.sums_of_squares[index] = float(observations @ observations)
        state.event_indices.append(index)
        state.event_types.append("initial")
        state.event_stages.append(0)
        state.replications.append(count)
        state.true_values.append(float(function.evaluate(initial_points[index:index+1])[0]))
        state.true_noise_sds.append(
            float(function.noise_standard_deviation(initial_points[index:index+1])[0])
        )
        state.sample_means.append(float(initial_means[index]))
        state.sample_variances.append(float(np.var(observations, ddof=1)))
        state.clipped_variances.append(float(initial_variances[index]))
        state.acquisition_values.append(float("nan"))
        state.coarse_acquisition_values.append(float("nan"))
        state.polish_improved.append(False)
        state.polish_evaluations.append(0)
        state.polish_successful_starts.append(0)
        state.raw_observations.append(observations)
        response_model.add_observation(
            index,
            initial_means[index],
            initial_variances[index] / count,
        )
        if noise_model is not None:
            noise_model.add_observation(
                index,
                float(np.sqrt(initial_variances[index])),
                float(initial_variances[index] / (2.0 * max(count - 1, 1))),
            )
    return state, full_schedule


def _predicted_single_replication_variances(state: _MacroState) -> np.ndarray:
    if state.noise_model is None:
        raise RuntimeError("this acquisition requires an auxiliary noise model")
    minimum_sd = NOISE_SD_MIN_MULTIPLIER * state.function.response_range
    maximum_sd = NOISE_SD_MAX_MULTIPLIER * state.function.response_range
    predicted_sd = np.clip(
        state.noise_model.posterior_mean,
        minimum_sd,
        maximum_sd,
    )
    return predicted_sd**2


def _recommend_posterior(
    state: _MacroState, uncertainty_coefficient: float
) -> tuple[int, float, float]:
    sampled = state.observed_indices()
    deviations = np.sqrt(
        np.maximum(state.response_model.posterior_variance[sampled], 0.0)
    )
    scores = (
        state.response_model.posterior_mean[sampled]
        + uncertainty_coefficient * deviations
    )
    local = int(np.argmin(scores))
    index = int(sampled[local])
    return (
        index,
        float(state.response_model.posterior_mean[index]),
        float(deviations[local]),
    )


def _run_sko(state: _MacroState, schedule: np.ndarray) -> tuple[int, float, float]:
    for batch_index in range(N_INITIAL, schedule.size):
        sampled = state.observed_indices()
        sampled_means = state.response_model.posterior_mean[sampled]
        sampled_sds = np.sqrt(
            np.maximum(state.response_model.posterior_variance[sampled], 0.0)
        )
        means = state.response_model.posterior_mean
        variances = np.maximum(state.response_model.posterior_variance, 0.0)
        noise_variances = _predicted_single_replication_variances(state)
        acquisition, effective_best = stochastic_kriging_optimization_acquisition(
            sampled_means,
            sampled_sds,
            means,
            variances,
            noise_variances,
        )
        selected = int(np.argmax(acquisition))
        selected_value = float(acquisition[selected])
        coarse_value = selected_value
        event_type = "sko_infill_sobol"
        polish_improved = False
        polish_evaluations = 0
        polish_successful_starts = 0
        if CONTINUOUS_POLISH_ENABLED:
            if state.noise_model is None:
                raise RuntimeError("SKO continuous polishing requires a noise GP")
            polish = polish_sko(
                state.response_model,
                state.noise_model,
                effective_best,
                NOISE_SD_MIN_MULTIPLIER * state.function.response_range,
                NOISE_SD_MAX_MULTIPLIER * state.function.response_range,
                acquisition,
            )
            polish_improved = polish.improved
            polish_evaluations = polish.function_evaluations
            polish_successful_starts = polish.successful_starts
            if polish.improved:
                distances = np.max(np.abs(state.universe - polish.point), axis=1)
                matches = np.flatnonzero(distances <= 1e-10)
                if matches.size:
                    selected = int(matches[0])
                    event_type = "sko_continuous_existing"
                else:
                    selected = state.append_continuous_point(polish.point)
                    event_type = "sko_continuous_new"
                selected_value = polish.acquisition
        state.simulate_and_update(
            selected,
            int(schedule[batch_index]),
            event_type,
            batch_index - N_INITIAL + 1,
            selected_value,
            coarse_acquisition_value=coarse_value,
            polish_improved=polish_improved,
            polish_evaluations=polish_evaluations,
            polish_successful_starts=polish_successful_starts,
        )
    return _recommend_posterior(state, SKO_INCUMBENT_SD_MULTIPLIER)


def _eqi_acquisition(state: _MacroState, planned_replications: int) -> np.ndarray:
    sampled = state.observed_indices()
    means = state.response_model.posterior_mean
    mse = np.maximum(state.response_model.posterior_variance, 0.0)
    # beta=0.5, so Phi^{-1}(beta)=0 and the posterior quantile is the mean.
    quantile_minimum = float(np.min(means[sampled]))
    new_mean_noise = (
        _predicted_single_replication_variances(state) / planned_replications
    )
    total = mse + new_mean_noise
    deviation = np.zeros_like(mse)
    positive = total > 0.0
    deviation[positive] = mse[positive] / np.sqrt(total[positive])
    return _expected_positive_part(quantile_minimum - means, deviation)


def _run_eqi(state: _MacroState, schedule: np.ndarray) -> tuple[int, float, float]:
    locked_index: int | None = None
    locked_reference = 0.0
    for batch_index in range(N_INITIAL, schedule.size):
        planned_replications = int(schedule[batch_index])
        acquisition = _eqi_acquisition(state, planned_replications)
        coarse_value = float(np.max(acquisition))
        polish_improved = False
        polish_evaluations = 0
        polish_successful_starts = 0
        if (
            locked_index is not None
            and acquisition[locked_index]
            > EQI_ALLOCATION_THRESHOLD * locked_reference
        ):
            selected = locked_index
            selected_value = float(acquisition[selected])
            event_type = "eqi_allocation"
        else:
            selected = int(np.argmax(acquisition))
            selected_value = float(acquisition[selected])
            event_type = "eqi_new_selection_sobol"
            if CONTINUOUS_POLISH_ENABLED:
                sampled = state.observed_indices()
                quantile_minimum = float(
                    np.min(state.response_model.posterior_mean[sampled])
                )
                if state.noise_model is None:
                    raise RuntimeError("EQI continuous polishing requires a noise GP")
                polish = polish_eqi(
                    state.response_model,
                    state.noise_model,
                    quantile_minimum,
                    planned_replications,
                    NOISE_SD_MIN_MULTIPLIER * state.function.response_range,
                    NOISE_SD_MAX_MULTIPLIER * state.function.response_range,
                    acquisition,
                )
                polish_improved = polish.improved
                polish_evaluations = polish.function_evaluations
                polish_successful_starts = polish.successful_starts
                if polish.improved:
                    distances = np.max(
                        np.abs(state.universe - polish.point), axis=1
                    )
                    matches = np.flatnonzero(distances <= 1e-10)
                    if matches.size:
                        selected = int(matches[0])
                        event_type = "eqi_continuous_existing"
                    else:
                        selected = state.append_continuous_point(polish.point)
                        event_type = "eqi_continuous_new"
                    selected_value = polish.acquisition
            locked_index = selected
            locked_reference = selected_value
        state.simulate_and_update(
            selected,
            planned_replications,
            event_type,
            batch_index - N_INITIAL + 1,
            selected_value,
            coarse_acquisition_value=coarse_value,
            polish_improved=polish_improved,
            polish_evaluations=polish_evaluations,
            polish_successful_starts=polish_successful_starts,
        )
    return _recommend_posterior(state, 0.0)


def _tsso_stage_split(
    iteration_budget: int,
    stage: int,
    stage_count: int,
) -> tuple[int, int]:
    """Split one matched batch between TSSO search and OCBA allocation."""

    if stage_count < 1 or not 1 <= stage <= stage_count:
        raise ValueError("stage must lie in [1, stage_count]")
    if iteration_budget < TSSO_MINIMUM_SEARCH_REPLICATIONS:
        raise ValueError("iteration budget is smaller than the minimum search batch")
    allocation_budget = int(round(iteration_budget * stage / (stage_count + 1)))
    allocation_budget = min(
        allocation_budget,
        iteration_budget - TSSO_MINIMUM_SEARCH_REPLICATIONS,
    )
    search_budget = iteration_budget - allocation_budget
    return search_budget, allocation_budget


def _refresh_tsso_response_mean(
    state: _MacroState,
    cross_covariance: np.ndarray,
    observed_order: list[int],
) -> _TSSOMeanPosterior:
    """Refit the fixed stochastic-Kriging mean on aggregated unique designs.

    OCBA can update many previously sampled points in one stage.  Aggregating
    those replications before this refit is algebraically equivalent to using
    one sample-mean observation per design and keeps the GP dimension equal to
    the number of unique designs rather than the number of allocation events.
    TSSO uses deterministic Kriging MSE for search, so only the stochastic
    posterior mean is needed here.
    """

    observed = np.asarray(observed_order, dtype=int)
    if observed.size != state.observed_indices().size:
        raise RuntimeError("TSSO covariance cache does not match observed designs")
    means = np.asarray([state.aggregate_mean(index) for index in observed])
    mean_variances = np.asarray(
        [state.aggregate_variance(index) / state.counts[index] for index in observed]
    )
    covariance = cross_covariance[observed, : observed.size].copy()
    covariance = 0.5 * (covariance + covariance.T)
    covariance.flat[:: covariance.shape[0] + 1] += mean_variances
    factor = cho_factor(covariance, lower=True, check_finite=False)
    residual = means - state.response_model.constant_mean
    weights = cho_solve(factor, residual, check_finite=False)
    state.response_model.posterior_mean = (
        state.response_model.constant_mean
        + cross_covariance[: state.universe.shape[0], : observed.size] @ weights
    )
    return _TSSOMeanPosterior(
        observed_points=state.universe[observed].copy(),
        weights=np.asarray(weights, dtype=float),
        kernel=state.function.kernel,
        lengthscale=state.function.lengthscale,
        process_variance=state.response_model.process_variance,
        constant_mean=state.response_model.constant_mean,
    )


def _run_tsso(state: _MacroState, schedule: np.ndarray) -> tuple[int, float, float]:
    iteration_budgets = np.asarray(schedule[N_INITIAL:], dtype=int)
    stage_count = int(iteration_budgets.size)
    if stage_count != state.spec.sequential_stage_count:
        raise RuntimeError("TSSO stage count does not match the common batch schedule")
    deterministic = SequentialFixedKernelGP(
        universe=state.universe,
        kernel=state.function.kernel,
        lengthscale=state.function.lengthscale,
        process_variance=state.response_model.process_variance,
        constant_mean=state.response_model.constant_mean,
        maximum_observations=int(schedule.size),
    )
    # Only covariances from the current universe to observed designs are
    # needed. A full candidate-universe covariance prevents continuous point
    # appends and cost roughly 0.8 GB per worker. This rectangular cache grows
    # by one column per search stage and gives the same posterior mean.
    initial_universe_size = state.universe.shape[0]
    maximum_universe_size = initial_universe_size + stage_count
    maximum_observed = N_INITIAL + stage_count
    cross_covariance = np.empty(
        (maximum_universe_size, maximum_observed), dtype=float
    )
    cross_covariance[:initial_universe_size, :N_INITIAL] = (
        state.response_model.process_variance
        * kernel_matrix(
            state.universe,
            state.universe[:N_INITIAL],
            state.function.kernel,
            state.function.lengthscale,
        )
    )
    numerical_noise = max(1e-12 * state.response_model.process_variance, 1e-14)
    for index in range(N_INITIAL):
        deterministic.add_observation(
            index,
            state.aggregate_mean(index),
            numerical_noise,
        )
    candidate_offset = N_INITIAL
    available = np.ones(state.spec.candidate_count, dtype=bool)
    observed_order = list(range(N_INITIAL))
    mean_posterior = _refresh_tsso_response_mean(
        state, cross_covariance, observed_order
    )

    for stage, iteration_budget in enumerate(iteration_budgets, start=1):
        search_budget, allocation_budget = _tsso_stage_split(
            int(iteration_budget), stage, stage_count
        )
        available_local = np.flatnonzero(available)
        available_global = candidate_offset + available_local
        observed = state.observed_indices()
        observed_means = np.asarray(
            [state.aggregate_mean(index) for index in observed]
        )
        current_best = int(observed[int(np.argmin(observed_means))])
        incumbent = float(state.response_model.posterior_mean[current_best])
        candidate_means = state.response_model.posterior_mean[available_global]
        candidate_sds = np.sqrt(
            np.maximum(deterministic.posterior_variance[available_global], 0.0)
        )
        acquisition = _expected_positive_part(
            incumbent - candidate_means,
            candidate_sds,
        )
        local = int(np.argmax(acquisition))
        selected_local = int(available_local[local])
        selected = candidate_offset + selected_local
        selected_value = float(acquisition[local])
        coarse_value = selected_value
        polish_improved = False
        polish_evaluations = 0
        polish_successful_starts = 0
        event_type = "tsso_search_sobol"
        if CONTINUOUS_POLISH_ENABLED:
            polish = polish_tsso(
                state.universe[available_global],
                acquisition,
                incumbent,
                mean_posterior.mean_with_gradient,
                deterministic,
                state.universe[np.asarray(observed_order, dtype=int)],
            )
            polish_improved = polish.improved
            polish_evaluations = polish.function_evaluations
            polish_successful_starts = polish.successful_starts
            if polish.improved:
                distances = np.max(np.abs(state.universe - polish.point), axis=1)
                matches = np.flatnonzero(distances <= 1e-10)
                if matches.size:
                    selected = int(matches[0])
                    event_type = "tsso_search_continuous_existing"
                else:
                    selected = state.append_continuous_point(polish.point)
                    deterministic_index = int(
                        deterministic.append_points(polish.point)[0]
                    )
                    if deterministic_index != selected:
                        raise RuntimeError(
                            "deterministic TSSO GP returned an inconsistent append index"
                        )
                    current_observed = np.asarray(observed_order, dtype=int)
                    cross_covariance[selected, : current_observed.size] = (
                        state.response_model.process_variance
                        * kernel_matrix(
                            state.universe[selected : selected + 1],
                            state.universe[current_observed],
                            state.function.kernel,
                            state.function.lengthscale,
                        ).reshape(-1)
                    )
                    event_type = "tsso_search_continuous_new"
                selected_value = polish.acquisition

        if candidate_offset <= selected < candidate_offset + state.spec.candidate_count:
            available[selected - candidate_offset] = False
        state.simulate_and_update(
            selected,
            search_budget,
            event_type,
            stage,
            selected_value,
            coarse_acquisition_value=coarse_value,
            polish_improved=polish_improved,
            polish_evaluations=polish_evaluations,
            polish_successful_starts=polish_successful_starts,
            update_response_model=False,
        )
        deterministic.add_observation(
            selected,
            state.aggregate_mean(selected),
            numerical_noise,
        )
        column = len(observed_order)
        cross_covariance[: state.universe.shape[0], column] = (
            state.response_model.process_variance
            * kernel_matrix(
                state.universe,
                state.universe[selected : selected + 1],
                state.function.kernel,
                state.function.lengthscale,
            ).reshape(-1)
        )
        observed_order.append(selected)

        observed = state.observed_indices()
        current = state.counts[observed]
        means = np.asarray([state.aggregate_mean(index) for index in observed])
        mean_variances = np.asarray(
            [state.aggregate_variance(index) / state.counts[index] for index in observed]
        )
        targets = _allocate_ocba(
            current,
            means,
            mean_variances,
            allocation_budget,
        )
        for local_index, target in enumerate(targets):
            additional = int(target - current[local_index])
            if additional > 0:
                state.simulate_and_update(
                    int(observed[local_index]),
                    additional,
                    "tsso_ocba",
                    stage,
                    update_response_model=False,
                )

        mean_posterior = _refresh_tsso_response_mean(
            state, cross_covariance, observed_order
        )

    observed = state.observed_indices()
    observed_means = np.asarray(
        [state.aggregate_mean(index) for index in observed]
    )
    selected = int(observed[int(np.argmin(observed_means))])
    return selected, float(state.response_model.posterior_mean[selected]), float("nan")


def _macro_path(spec: ComparisonSpec, macro: int) -> Path:
    return RESULT_DIR / spec.slug / "macro_results" / f"macro_{macro:03d}.npz"


def run_one_comparison_macro(
    spec: ComparisonSpec,
    macro: int,
    overwrite: bool = False,
) -> dict[str, object]:
    destination = _macro_path(spec, macro)
    if destination.exists() and not overwrite:
        with np.load(destination, allow_pickle=False) as data:
            return {
                "macro": macro,
                "raw_gap": float(data["raw_gap"]),
                "status": "reused",
            }

    schedule = spec.schedule
    maximum_observations = int(schedule.size)
    state, schedule = _build_initial_state(
        spec,
        macro,
        maximum_observations,
        with_noise_model=spec.algorithm in ("SKO", "EQI"),
    )
    if spec.algorithm == "SKO":
        recommendation_index, posterior_mean, posterior_sd = _run_sko(state, schedule)
    elif spec.algorithm == "EQI":
        recommendation_index, posterior_mean, posterior_sd = _run_eqi(state, schedule)
    elif spec.algorithm == "TSSO":
        recommendation_index, posterior_mean, posterior_sd = _run_tsso(state, schedule)
    else:  # pragma: no cover - protected by the Literal configuration
        raise ValueError(f"unsupported algorithm: {spec.algorithm}")

    if state.budget != spec.total_budget:
        raise RuntimeError(
            f"{spec.algorithm} macro {macro} used {state.budget}, expected "
            f"{spec.total_budget}"
        )
    recommendation_x = state.universe[recommendation_index].copy()
    recommendation_value = float(
        state.function.evaluate(recommendation_x.reshape(1, DIMENSION))[0]
    )
    raw_gap = max(recommendation_value - state.function.optimum_value, 0.0)
    normalized_gap = raw_gap / state.function.response_range
    replications = np.asarray(state.replications, dtype=int)
    offsets = np.zeros(replications.size + 1, dtype=int)
    offsets[1:] = np.cumsum(replications)
    raw_observations = np.concatenate(state.raw_observations)
    if raw_observations.size != spec.total_budget:
        raise RuntimeError("raw observation count does not equal the exact budget")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        algorithm=np.asarray(spec.algorithm),
        macro=np.asarray(macro),
        function_name=np.asarray(spec.function_name),
        total_budget=np.asarray(spec.total_budget),
        nominal_batch_size=np.asarray(spec.nominal_batch_size),
        replication_multiplier=np.asarray(spec.replication_multiplier),
        initial_points=state.universe[:N_INITIAL],
        candidate_count=np.asarray(spec.candidate_count),
        universe_points=state.universe,
        acquisition_optimizer=np.asarray(
            (
                f"sobol_{spec.candidate_count}_plus_"
                f"{CONTINUOUS_POLISH_STARTS}start_lbfgsb"
            )
            if spec.algorithm in ("SKO", "EQI", "TSSO")
            else "registered_discrete"
        ),
        continuous_polish_enabled=np.asarray(
            CONTINUOUS_POLISH_ENABLED
            and spec.algorithm in ("SKO", "EQI", "TSSO")
        ),
        continuous_polish_starts=np.asarray(
            CONTINUOUS_POLISH_STARTS
            if spec.algorithm in ("SKO", "EQI", "TSSO")
            else 0
        ),
        continuous_polish_maxiter=np.asarray(
            CONTINUOUS_POLISH_MAXITER
            if spec.algorithm in ("SKO", "EQI", "TSSO")
            else 0
        ),
        continuous_points_added=np.asarray(
            state.universe.shape[0] - spec.candidate_count - N_INITIAL
        ),
        event_indices=np.asarray(state.event_indices, dtype=int),
        event_points=state.universe[np.asarray(state.event_indices, dtype=int)],
        event_types=np.asarray(state.event_types),
        event_stages=np.asarray(state.event_stages, dtype=int),
        replications=replications,
        true_values=np.asarray(state.true_values),
        true_noise_sd=np.asarray(state.true_noise_sds),
        noise_sd_min_multiplier=np.asarray(NOISE_SD_MIN_MULTIPLIER),
        noise_sd_max_multiplier=np.asarray(NOISE_SD_MAX_MULTIPLIER),
        sample_means=np.asarray(state.sample_means),
        sample_variances=np.asarray(state.sample_variances),
        clipped_variances=np.asarray(state.clipped_variances),
        acquisition_values=np.asarray(state.acquisition_values),
        coarse_acquisition_values=np.asarray(state.coarse_acquisition_values),
        polish_improved=np.asarray(state.polish_improved),
        polish_evaluations=np.asarray(state.polish_evaluations),
        polish_successful_starts=np.asarray(state.polish_successful_starts),
        raw_observation_offsets=offsets,
        raw_observations=raw_observations,
        unique_indices=state.observed_indices(),
        unique_points=state.universe[state.observed_indices()],
        unique_replications=state.counts[state.observed_indices()],
        unique_sample_means=np.asarray(
            [state.aggregate_mean(index) for index in state.observed_indices()]
        ),
        recommendation_index=np.asarray(recommendation_index),
        recommendation_x=recommendation_x,
        recommendation_value=np.asarray(recommendation_value),
        recommendation_posterior_mean=np.asarray(posterior_mean),
        recommendation_posterior_sd=np.asarray(posterior_sd),
        optimum_x=state.function.optimum_x,
        optimum_value=np.asarray(state.function.optimum_value),
        raw_gap=np.asarray(raw_gap),
        normalized_gap=np.asarray(normalized_gap),
        response_process_variance=np.asarray(RESPONSE_PROCESS_VARIANCE),
        response_constant_mean=np.asarray(RESPONSE_CONSTANT_MEAN),
        response_lengthscale=np.asarray(RESPONSE_LENGTHSCALE),
        response_hyperparameters_fixed=np.asarray(True),
        auxiliary_noise_gp_initial_fit_only=np.asarray(
            spec.algorithm in ("SKO", "EQI")
        ),
        sko_incumbent_sd_multiplier=np.asarray(
            SKO_INCUMBENT_SD_MULTIPLIER if spec.algorithm == "SKO" else np.nan
        ),
        sko_noise_penalty_uses_single_replication_variance=np.asarray(
            spec.algorithm == "SKO"
        ),
        phase_code=np.asarray(spec.phase_code),
        tsso_stage_count=np.asarray(
            spec.sequential_stage_count if spec.algorithm == "TSSO" else -1
        ),
        tsso_stage_budgets=(
            schedule[N_INITIAL:].copy()
            if spec.algorithm == "TSSO"
            else np.empty(0, dtype=int)
        ),
        tsso_stage_budget_matches_lqei=np.asarray(spec.algorithm == "TSSO"),
    )
    return {"macro": macro, "raw_gap": raw_gap, "status": "completed"}


def _write_configuration_summary(spec: ComparisonSpec) -> Path:
    directory = RESULT_DIR / spec.slug
    files = sorted((directory / "macro_results").glob("macro_*.npz"))
    if len(files) != spec.macro_count:
        raise RuntimeError(
            f"{spec.slug} has {len(files)} macros; expected {spec.macro_count}"
        )
    rows: list[dict[str, object]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "macro": int(data["macro"]),
                    "raw_gap": float(data["raw_gap"]),
                    "recommendation_value": float(data["recommendation_value"]),
                    "event_count": int(np.asarray(data["replications"]).size),
                    "unique_design_count": int(np.asarray(data["unique_indices"]).size),
                    **{
                        f"recommendation_x{coordinate + 1}": float(
                            data["recommendation_x"][coordinate]
                        )
                        for coordinate in range(DIMENSION)
                    },
                }
            )
    with (directory / "macro_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gaps = np.asarray([float(row["raw_gap"]) for row in rows])
    summary = {
        "specification": asdict(spec),
        "metric": "raw_gap",
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "q25_gap": float(np.quantile(gaps, 0.25)),
        "q75_gap": float(np.quantile(gaps, 0.75)),
        "q90_gap": float(np.quantile(gaps, 0.90)),
        "minimum_gap": float(np.min(gaps)),
        "maximum_gap": float(np.max(gaps)),
        "response_gp_hyperparameters": {
            "constant_mean_beta": RESPONSE_CONSTANT_MEAN,
            "process_variance_tau_squared": RESPONSE_PROCESS_VARIANCE,
            "lengthscale": RESPONSE_LENGTHSCALE,
            "fixed": True,
        },
        "acquisition_optimizer": (
            {
                "global_screen": f"{spec.candidate_count}-point scrambled Sobol pool",
                "continuous_method": "L-BFGS-B",
                "starts": CONTINUOUS_POLISH_STARTS,
                "maximum_iterations_per_start": CONTINUOUS_POLISH_MAXITER,
            }
            if spec.algorithm in ("SKO", "EQI", "TSSO")
            else {"method": "registered discrete search"}
        ),
        "auxiliary_noise_gp_policy": (
            "Fit constant mean and process variance on the initial design only; "
            "keep them and the kernel lengthscale fixed during sequential updates."
            if spec.algorithm in ("SKO", "EQI")
            else None
        ),
        "sko": (
            {
                "incumbent": "posterior mean at argmin(mean + posterior_sd)",
                "incumbent_sd_multiplier": SKO_INCUMBENT_SD_MULTIPLIER,
                "noise_penalty": (
                    "1 - sqrt(single_replication_noise_variance) / "
                    "sqrt(posterior_variance + single_replication_noise_variance)"
                ),
                "recommendation": "argmin(mean + posterior_sd) over sampled points",
                "response_gp_hyperparameters_refit": False,
                "noise_gp_hyperparameters_refit_after_initial_design": False,
            }
            if spec.algorithm == "SKO"
            else None
        ),
        "tsso_stage_count": (
            spec.sequential_stage_count if spec.algorithm == "TSSO" else None
        ),
        "tsso_stage_budgets": (
            spec.schedule[N_INITIAL:].tolist() if spec.algorithm == "TSSO" else None
        ),
        "tsso_stage_budget_policy": (
            f"Same exact O(log T) batch schedule as LQEI; the first {N_INITIAL} batches "
            "initialize the design and every remaining batch is one TSSO stage."
            if spec.algorithm == "TSSO"
            else None
        ),
    }
    destination = directory / "summary.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return destination


def run_comparison_specs(
    specs: Iterable[ComparisonSpec],
    *,
    workers: int = 8,
    overwrite: bool = False,
    macro_limit: int | None = None,
    write_protocol_file: bool = True,
) -> None:
    selected_specs = list(specs)
    if macro_limit is not None and macro_limit < 1:
        raise ValueError("macro_limit must be positive")
    jobs = [
        (spec, macro)
        for spec in selected_specs
        for macro in range(1, min(spec.macro_count, macro_limit or spec.macro_count) + 1)
        if overwrite or not _macro_path(spec, macro).exists()
    ]
    print(
        f"Running {len(jobs)} comparison macro jobs across "
        f"{len(selected_specs)} configurations with {workers} workers.",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one_comparison_macro, spec, macro, overwrite): (
                    spec,
                    macro,
                )
                for spec, macro in jobs
            }
            completed = 0
            for future in as_completed(futures):
                spec, macro = futures[future]
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"Completed {completed}/{len(futures)}; "
                        f"latest={spec.function_name}/{spec.algorithm}/"
                        f"T={spec.total_budget}/macro={macro}",
                        flush=True,
                    )
    if macro_limit is None or all(macro_limit >= spec.macro_count for spec in selected_specs):
        for spec in selected_specs:
            _write_configuration_summary(spec)
        if write_protocol_file:
            write_comparison_protocol(selected_specs)


def write_comparison_protocol(
    specs: Iterable[ComparisonSpec],
) -> Path:
    selected_specs = list(specs)
    destination = RESULT_DIR / "comparison" / "protocol.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithms_run": list(COMPARISON_ALGORITHMS),
        "lqei_reference": (
            "formal O(log T); acquisition uses "
            f"{LQEI_ACQUISITION_LAMBDA_MULTIPLIER:g} times the dynamic "
            "equation-(20) minimum-bound Lambda_n; "
            "the terminal recommendation minimizes posterior_mean + "
            "0.2*Lambda_N*posterior_sd; rho is kernel-specific"
        ),
        "common_random_numbers": (
            "Same phase code, macro index, initial design, candidate pool, and "
            "event-index Philox stream as the LQEI formal study."
        ),
        "budget_matching": (
            "SKO and EQI use the same exact O(log T) batch schedule as LQEI. "
            "Each SKO iteration receives one complete matched batch. "
            "EQI is the fixed-batch adaptation of its original dynamic "
            "selection/allocation rule. TSSO uses that same exact schedule: "
            f"the first {N_INITIAL} batches initialize the design and each remaining "
            "batch is one sequential stage, split between search and OCBA."
        ),
        "sko": {
            "registered_name": "SKO",
            "source": (
                "Adapted from KrigingBasedHetero_sustech_python/AEI_python/"
                "aei/acquisition.py"
            ),
            "incumbent": "posterior mean at argmin(mean + posterior_sd)",
            "incumbent_sd_multiplier": SKO_INCUMBENT_SD_MULTIPLIER,
            "noise_model": (
                "Auxiliary GP fitted on the initial sample SDs; all GP "
                "hyperparameters remain fixed during sequential updates."
            ),
            "acquisition_optimizer": (
                f"Registered {DIMENSION}D Sobol screen plus "
                f"{CONTINUOUS_POLISH_STARTS}-start bounded L-BFGS-B"
            ),
            "iteration_budget_policy": "one complete common batch per iteration",
            "recommendation": "argmin(mean + posterior_sd) over sampled points",
        },
        "eqi": {
            "beta": EQI_BETA,
            "allocation_threshold": EQI_ALLOCATION_THRESHOLD,
            "noise_model": "Auxiliary fixed-kernel GP fitted to sample SDs.",
        },
        "tsso": {
            "stage_count_policy": f"len(common_batch_schedule) - {N_INITIAL}",
            "stage_budget_policy": "one complete common batch per stage",
            "minimum_search_replications": TSSO_MINIMUM_SEARCH_REPLICATIONS,
            "search_mse": "Deterministic kriging MSE with stochastic-kriging mean.",
            "search_optimizer": (
                f"Registered Sobol screen plus {CONTINUOUS_POLISH_STARTS}-start "
                "bounded L-BFGS-B"
            ),
            "allocation": "Exact integer OCBA with a linearly increasing stage share.",
        },
        "configurations": [asdict(spec) for spec in selected_specs],
    }
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return destination
