"""Search a 4D homoscedastic RKHS scenario for terminal-rule separation.

The response is always a finite Matérn-5/2 kernel expansion.  LQEI uses the
same fixed response-GP hyperparameters, log replication order, and dynamic
``mu - 0.2 Lambda sigma`` incumbent as ``new_921_4d_exp_budget``.  Every final
rule postprocesses the same sampled points and the same terminal posterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent

from confidence import build_confidence_schedule  # noqa: E402
from continuous_acquisition import polish_lqei, polish_lqei_log  # noqa: E402
from experiment_runner import (  # noqa: E402
    _expected_improvement,
    _log_expected_improvement,
    _polished_universe_index,
)
from rkhs_functions import kernel_matrix  # noqa: E402
from sequential_gp import SequentialFixedKernelGP  # noqa: E402
from study_config import make_batch_schedule, replication_size  # noqa: E402


DIMENSION = 4
KERNEL = "matern52"
LENGTHSCALE = 0.2
PROCESS_VARIANCE = 1.0
CONSTANT_MEAN = 0.0
N_INITIAL = 20
RHO = 8.0
DELTA = 0.05
ACQUISITION_LAMBDA_MULTIPLIER = 0.2
NOISE_LEVELS = {"medium": 2.5, "high": 3.0}
LOW_BUDGETS = (1408, 1808, 2322)
MID_HIGH_BUDGETS = (4915, 13360, 36316, 98716)
FORMAL_BUDGETS = (*LOW_BUDGETS, *MID_HIGH_BUDGETS)
REQUESTED_GRID_BUDGETS = (1808, 2981, 4915, 8103, 13360, 22026, 36316, 98716)
ADDITIONAL_BUDGETS = (2981, 8103, 22026)

RULE_PLUS = "posterior_mean_plus_0p2_lambda_sd"
RULE_PLUS_ONE = "posterior_mean_plus_1p0_lambda_sd"
RULE_MEAN = "posterior_mean"
RULE_MINUS = "posterior_mean_minus_10_sd"
RULE_MINUS_SYMMETRIC = "posterior_mean_minus_0p2_lambda_sd"
RULES = (RULE_PLUS, RULE_PLUS_ONE, RULE_MEAN, RULE_MINUS)
RULE_LABELS = {
    RULE_PLUS: "+0.2 Lambda",
    RULE_PLUS_ONE: "+1.0 Lambda",
    RULE_MEAN: "posterior mean",
    RULE_MINUS: "-10 sd",
    RULE_MINUS_SYMMETRIC: "-0.2 Lambda",
}

CANDIDATE_SEED = 20260924
DESIGN_SEED = 20260925
NOISE_SEED = 20260926
SEARCH_PHASE = 92410
CONFIRM_PHASE = 92420
FORMAL_PHASE = 92430
EXTENSION_PHASE = 92440
ADDITIONAL_PHASE = 92450

SAFE_CENTER = np.full(DIMENSION, 0.30)
OPTIMAL_CENTER = np.full(DIMENSION, 0.72)
DECOY_CENTERS = np.asarray(
    [
        [0.76, 0.24, 0.76, 0.24],
        [0.24, 0.76, 0.24, 0.76],
        [0.78, 0.78, 0.22, 0.22],
        [0.22, 0.22, 0.78, 0.78],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    safe_radius: float
    optimal_weight: float
    decoy_weight: float


@dataclass(frozen=True)
class Objective:
    scenario: Scenario
    centers: np.ndarray
    weights: np.ndarray
    optimum_x: np.ndarray
    optimum_value: float
    maximum_x: np.ndarray
    maximum_value: float
    rkhs_norm: float

    @property
    def response_range(self) -> float:
        return float(self.maximum_value - self.optimum_value)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return kernel_matrix(points, self.centers, KERNEL, LENGTHSCALE) @ self.weights


def scenario_grid() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for radius in (0.05, 0.08, 0.11):
        for optimal_weight in (10.0, 11.0, 12.0, 13.0, 14.0):
            for decoy_weight in (6.0, 7.5, 9.0):
                name = (
                    f"three_tier_r{int(100 * radius):02d}_"
                    f"o{str(optimal_weight).replace('.', 'p')}_"
                    f"d{str(decoy_weight).replace('.', 'p')}"
                )
                scenarios.append(Scenario(name, radius, optimal_weight, decoy_weight))
    return scenarios


def _raw_expansion(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    centers = [SAFE_CENTER.tolist()]
    for coordinate in range(DIMENSION):
        for sign in (-1.0, 1.0):
            point = SAFE_CENTER.copy()
            point[coordinate] += sign * scenario.safe_radius
            centers.append(point.tolist())
    centers.append(OPTIMAL_CENTER.tolist())
    centers.extend(DECOY_CENTERS.tolist())
    weights = [*([-1.0] * (1 + 2 * DIMENSION)), -scenario.optimal_weight]
    weights.extend([-scenario.decoy_weight] * DECOY_CENTERS.shape[0])
    return np.asarray(centers, dtype=float), np.asarray(weights, dtype=float)


def _extreme(
    centers: np.ndarray, weights: np.ndarray, *, maximize: bool, thorough: bool
) -> tuple[np.ndarray, float]:
    sign = -1.0 if maximize else 1.0

    def objective(point: np.ndarray) -> float:
        value = float(
            (kernel_matrix(point.reshape(1, DIMENSION), centers, KERNEL, LENGTHSCALE) @ weights).item()
        )
        return sign * value

    count = 65536 if thorough else 8192
    sampler = qmc.Sobol(d=DIMENSION, scramble=True, seed=1701 + int(maximize))
    points = sampler.random_base2(int(math.log2(count)))
    points = np.vstack((points, centers, SAFE_CENTER, OPTIMAL_CENTER, DECOY_CENTERS))
    values = kernel_matrix(points, centers, KERNEL, LENGTHSCALE) @ weights
    order = np.argsort(-values if maximize else values)[:16]
    best_x = points[order[0]].copy()
    best_value = float(values[order[0]])
    for index in order:
        result = minimize(
            objective,
            points[index],
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * DIMENSION,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 300},
        )
        value = sign * float(result.fun)
        if (maximize and value > best_value) or (not maximize and value < best_value):
            best_x = np.asarray(result.x, dtype=float)
            best_value = value
    return best_x, best_value


def make_objective(scenario: Scenario, *, thorough: bool = False) -> Objective:
    centers, raw_weights = _raw_expansion(scenario)
    min_x, minimum = _extreme(centers, raw_weights, maximize=False, thorough=thorough)
    max_x, maximum = _extreme(centers, raw_weights, maximize=True, thorough=thorough)
    raw_range = maximum - minimum
    weights = raw_weights / raw_range
    min_x, minimum = _extreme(centers, weights, maximize=False, thorough=thorough)
    max_x, maximum = _extreme(centers, weights, maximize=True, thorough=thorough)
    gram = kernel_matrix(centers, centers, KERNEL, LENGTHSCALE)
    return Objective(
        scenario=scenario,
        centers=centers,
        weights=weights,
        optimum_x=min_x,
        optimum_value=float(minimum),
        maximum_x=max_x,
        maximum_value=float(maximum),
        rkhs_norm=float(np.sqrt(weights @ gram @ weights)),
    )


def refine_objective(function: Objective) -> Objective:
    min_x, minimum = _extreme(
        function.centers, function.weights, maximize=False, thorough=True
    )
    max_x, maximum = _extreme(
        function.centers, function.weights, maximize=True, thorough=True
    )
    gram = kernel_matrix(function.centers, function.centers, KERNEL, LENGTHSCALE)
    return Objective(
        scenario=function.scenario,
        centers=function.centers,
        weights=function.weights,
        optimum_x=min_x,
        optimum_value=minimum,
        maximum_x=max_x,
        maximum_value=maximum,
        rkhs_norm=float(np.sqrt(function.weights @ gram @ function.weights)),
    )


def objective_to_dict(function: Objective) -> dict[str, object]:
    return {
        "scenario": asdict(function.scenario),
        "centers": function.centers.tolist(),
        "weights": function.weights.tolist(),
        "optimum_x": function.optimum_x.tolist(),
        "optimum_value": function.optimum_value,
        "maximum_x": function.maximum_x.tolist(),
        "maximum_value": function.maximum_value,
        "rkhs_norm": function.rkhs_norm,
    }


def objective_from_dict(data: dict[str, object]) -> Objective:
    return Objective(
        scenario=Scenario(**data["scenario"]),  # type: ignore[arg-type]
        centers=np.asarray(data["centers"], dtype=float),
        weights=np.asarray(data["weights"], dtype=float),
        optimum_x=np.asarray(data["optimum_x"], dtype=float),
        optimum_value=float(data["optimum_value"]),
        maximum_x=np.asarray(data["maximum_x"], dtype=float),
        maximum_value=float(data["maximum_value"]),
        rkhs_norm=float(data["rkhs_norm"]),
    )


def _candidate_pool(count: int) -> np.ndarray:
    exponent = int(round(math.log2(count)))
    if 2**exponent != count:
        raise ValueError("candidate count must be a power of two")
    return qmc.Sobol(d=DIMENSION, scramble=True, seed=CANDIDATE_SEED).random_base2(exponent)


def _initial_points(macro: int, phase: int) -> np.ndarray:
    sequence = np.random.SeedSequence([DESIGN_SEED, phase, macro])
    seeds = sequence.generate_state(3, dtype=np.uint32)
    global_points = qmc.LatinHypercube(d=DIMENSION, seed=int(seeds[0])).random(4)
    safe_points = qmc.LatinHypercube(d=DIMENSION, seed=int(seeds[1])).random(8)
    safe_points = SAFE_CENTER - 0.075 + 0.15 * safe_points
    optimal_points = qmc.LatinHypercube(d=DIMENSION, seed=int(seeds[2])).random(2)
    optimal_points = OPTIMAL_CENTER - 0.04 + 0.08 * optimal_points
    anchors = np.vstack((SAFE_CENTER, OPTIMAL_CENTER, DECOY_CENTERS))
    return np.vstack((global_points, safe_points, optimal_points, anchors))


def _schedule(total_budget: int) -> np.ndarray:
    nominal = replication_size("log", total_budget, rho=RHO, anchor_budget=total_budget)
    return make_batch_schedule(total_budget, nominal, N_INITIAL)


def _rng(macro: int, phase: int, batch: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.Philox(np.random.SeedSequence([NOISE_SEED, phase, macro, batch]))
    )


def run_macro(
    function: Objective,
    noise_level: str,
    total_budget: int,
    candidate_count: int,
    macro: int,
    phase: int,
    *,
    continuous: bool,
    noise_multiplier: float | None = None,
    final_rules: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    multiplier = (
        NOISE_LEVELS[noise_level]
        if noise_multiplier is None
        else float(noise_multiplier)
    )
    noise_sd = multiplier * function.response_range
    noise_variance = noise_sd**2
    schedule = _schedule(total_budget)
    nominal = replication_size("log", total_budget, rho=RHO, anchor_budget=total_budget)
    confidence = build_confidence_schedule(
        kernel=KERNEL,
        information_gain_batch_size=nominal,
        correction_batch_size=int(np.min(schedule)),
        batch_count=int(schedule.size),
        variance_min=noise_variance,
        rkhs_norm=function.rkhs_norm,
        delta=DELTA,
    )
    universe = np.vstack((_initial_points(macro, phase), _candidate_pool(candidate_count)))
    model = SequentialFixedKernelGP(
        universe=universe,
        kernel=KERNEL,
        lengthscale=LENGTHSCALE,
        process_variance=PROCESS_VARIANCE,
        constant_mean=CONSTANT_MEAN,
        maximum_observations=int(schedule.size),
    )
    for batch_index, batch_size_value in enumerate(schedule):
        if batch_index < N_INITIAL:
            selected = batch_index
        else:
            coefficient = float(
                -ACQUISITION_LAMBDA_MULTIPLIER * confidence.lambda_values[batch_index]
            )
            acquisition, incumbent = _expected_improvement(
                model.posterior_mean, model.posterior_variance, coefficient
            )
            maximum = float(np.max(acquisition))
            if maximum > 0.0:
                selected = int(np.argmax(acquisition))
                if continuous:
                    polished = polish_lqei(model, coefficient, incumbent, acquisition)
                    if polished.improved:
                        selected, _ = _polished_universe_index(model, polished)
            else:
                log_acquisition = _log_expected_improvement(
                    model.posterior_mean, model.posterior_variance, incumbent
                )
                selected = int(np.argmax(log_acquisition))
                if continuous:
                    polished = polish_lqei_log(model, incumbent, log_acquisition)
                    if polished.improved:
                        selected, _ = _polished_universe_index(model, polished)
        point = model.universe[selected]
        truth = float(function.evaluate(point.reshape(1, DIMENSION))[0])
        batch_size = int(batch_size_value)
        observed_mean = float(
            truth + noise_sd / np.sqrt(batch_size) * _rng(macro, phase, batch_index).standard_normal()
        )
        model.add_observation(selected, observed_mean, noise_variance / batch_size)

    sampled = np.unique(model.training_indices)
    sampled_points = model.universe[sampled]
    truth = function.evaluate(sampled_points)
    sampled_sd = np.sqrt(np.maximum(model.posterior_variance[sampled], 0.0))
    coefficients = {
        RULE_PLUS: float(0.2 * confidence.lambda_values[-1]),
        RULE_PLUS_ONE: float(confidence.lambda_values[-1]),
        RULE_MEAN: 0.0,
        RULE_MINUS: -10.0,
        RULE_MINUS_SYMMETRIC: float(-0.2 * confidence.lambda_values[-1]),
    }
    selected_rules = RULES if final_rules is None else final_rules
    unknown_rules = set(selected_rules) - set(coefficients)
    if unknown_rules:
        raise ValueError(f"Unknown final rules: {sorted(unknown_rules)}")
    rows: list[dict[str, object]] = []
    for rule in selected_rules:
        local = int(np.argmin(model.posterior_mean[sampled] + coefficients[rule] * sampled_sd))
        gap = max(float(truth[local]) - function.optimum_value, 0.0)
        point = sampled_points[local]
        rows.append(
            {
                "scenario": function.scenario.name,
                "noise_level": noise_level,
                "noise_sd_multiplier": multiplier,
                "total_budget": total_budget,
                "candidate_count": candidate_count,
                "macro": macro,
                "rule": rule,
                "rule_label": RULE_LABELS[rule],
                "raw_gap": gap,
                "terminal_lambda": float(confidence.lambda_values[-1]),
                "final_sd_coefficient": coefficients[rule],
                "recommendation_x1": float(point[0]),
                "recommendation_x2": float(point[1]),
                "recommendation_x3": float(point[2]),
                "recommendation_x4": float(point[3]),
                "recommendation_mean": float(model.posterior_mean[sampled[local]]),
                "recommendation_sd": float(sampled_sd[local]),
                "batch_count": int(schedule.size),
                "acquisition_count": int(schedule.size - N_INITIAL),
                "sampled_unique_count": int(sampled.size),
                "continuous_polish": continuous,
            }
        )
    return rows


def _worker(payload: tuple[dict[str, object], str, int, int, int, int, bool]) -> list[dict[str, object]]:
    data, noise, budget, candidates, macro, phase, continuous = payload
    return run_macro(
        objective_from_dict(data), noise, budget, candidates, macro, phase, continuous=continuous
    )


def run_payloads(
    payloads: list[tuple[dict[str, object], str, int, int, int, int, bool]], workers: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    keys = sorted(
        {(r["scenario"], r["noise_level"], int(r["total_budget"]), r["rule"]) for r in rows}
    )
    for scenario, noise, budget, rule in keys:
        values = np.asarray(
            [
                float(r["raw_gap"])
                for r in rows
                if r["scenario"] == scenario
                and r["noise_level"] == noise
                and int(r["total_budget"]) == budget
                and r["rule"] == rule
            ]
        )
        result.append(
            {
                "scenario": scenario,
                "noise_level": noise,
                "total_budget": budget,
                "rule": rule,
                "count": values.size,
                "mean_raw_gap": float(np.mean(values)),
                "median_raw_gap": float(np.median(values)),
                "standard_error": float(np.std(values, ddof=1) / np.sqrt(values.size)),
            }
        )
    return result


def rank_screen(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranking: list[dict[str, object]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        by_rule = {
            rule: np.mean([float(row["raw_gap"]) for row in subset if row["rule"] == rule])
            for rule in RULES
        }
        cells_won = 0
        available_budgets = sorted({int(row["total_budget"]) for row in subset})
        for noise in NOISE_LEVELS:
            for budget in available_budgets:
                means = {
                    rule: np.mean(
                        [
                            float(row["raw_gap"])
                            for row in subset
                            if row["noise_level"] == noise
                            and int(row["total_budget"]) == budget
                            and row["rule"] == rule
                        ]
                    )
                    for rule in (RULE_PLUS, RULE_PLUS_ONE, RULE_MEAN)
                }
                cells_won += int(means[RULE_PLUS] == min(means.values()))
        margin = min(
            by_rule[RULE_PLUS_ONE] - by_rule[RULE_PLUS],
            by_rule[RULE_MEAN] - by_rule[RULE_PLUS],
        )
        ranking.append(
            {
                "scenario": scenario,
                "plus_mean": by_rule[RULE_PLUS],
                "plus_one_mean": by_rule[RULE_PLUS_ONE],
                "mean_rule_mean": by_rule[RULE_MEAN],
                "minus_mean": by_rule[RULE_MINUS],
                "minimum_margin": margin,
                "cells_won": cells_won,
            }
        )
    return sorted(ranking, key=lambda row: (row["cells_won"], row["minimum_margin"]), reverse=True)


def run_screen(workers: int) -> None:
    functions = [make_objective(scenario) for scenario in scenario_grid()]
    functions = [
        function
        for function in functions
        if np.max(np.abs(function.optimum_x - OPTIMAL_CENTER)) < 0.12
    ]
    (HERE / "screening").mkdir(parents=True, exist_ok=True)
    with (HERE / "screening" / "objectives.json").open("w", encoding="utf-8") as handle:
        json.dump([objective_to_dict(function) for function in functions], handle, indent=2)
    payloads = [
        (objective_to_dict(function), noise, budget, 512, macro, SEARCH_PHASE, False)
        for function in functions
        for noise in NOISE_LEVELS
        for budget in LOW_BUDGETS[:2]
        for macro in range(1, 13)
    ]
    rows = run_payloads(payloads, workers)
    write_csv(HERE / "screening" / "all_recommendations.csv", rows)
    write_csv(HERE / "screening" / "summary.csv", summarize(rows))
    ranking = rank_screen(rows)
    write_csv(HERE / "screening" / "ranking.csv", ranking)
    print(json.dumps(ranking[:10], indent=2))


def run_confirmation(workers: int) -> None:
    source = HERE / "screening_round2"
    with (source / "objectives.json").open(encoding="utf-8") as handle:
        functions = [objective_from_dict(item) for item in json.load(handle)]
    with (source / "ranking.csv").open(newline="", encoding="utf-8") as handle:
        ranking = list(csv.DictReader(handle))
    selected_names = [row["scenario"] for row in ranking[:6]]
    selected = [function for function in functions if function.scenario.name in selected_names]
    output = HERE / "confirmation_round1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "objectives.json").open("w", encoding="utf-8") as handle:
        json.dump([objective_to_dict(function) for function in selected], handle, indent=2)
    payloads = [
        (objective_to_dict(function), noise, budget, 1024, macro, CONFIRM_PHASE, True)
        for function in selected
        for noise in NOISE_LEVELS
        for budget in LOW_BUDGETS
        for macro in range(1, 31)
    ]
    rows = run_payloads(payloads, workers)
    write_csv(output / "all_recommendations.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    confirmed_ranking = rank_screen(rows)
    write_csv(output / "ranking.csv", confirmed_ranking)
    print(json.dumps(confirmed_ranking, indent=2))


def _confirmed_winner() -> Objective:
    source = HERE / "confirmation_round1"
    with (source / "ranking.csv").open(newline="", encoding="utf-8") as handle:
        winner_name = next(csv.DictReader(handle))["scenario"]
    with (source / "objectives.json").open(encoding="utf-8") as handle:
        functions = [objective_from_dict(item) for item in json.load(handle)]
    winner = next(
        function for function in functions if function.scenario.name == winner_name
    )
    return refine_objective(winner)


def run_formal(workers: int) -> None:
    function = _confirmed_winner()
    output = HERE / "formal_low_budget"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selected_function.json").open("w", encoding="utf-8") as handle:
        json.dump(objective_to_dict(function), handle, indent=2)
    payloads = [
        (objective_to_dict(function), noise, budget, 2048, macro, FORMAL_PHASE, True)
        for noise in NOISE_LEVELS
        for budget in LOW_BUDGETS
        for macro in range(1, 251)
    ]
    rows = run_payloads(payloads, workers)
    write_csv(output / "all_recommendations.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    print(json.dumps(rank_screen(rows), indent=2))


def _extension_checkpoint(noise: str, budget: int, macro: int) -> Path:
    return (
        HERE
        / "formal_extension"
        / "trajectory_results"
        / noise
        / f"T_{budget}"
        / f"macro_{macro:03d}.json"
    )


def run_extension(workers: int) -> None:
    with (HERE / "formal_low_budget" / "selected_function.json").open(
        encoding="utf-8"
    ) as handle:
        function = objective_from_dict(json.load(handle))
    function_data = objective_to_dict(function)
    jobs: list[
        tuple[
            tuple[dict[str, object], str, int, int, int, int, bool],
            str,
            int,
            int,
        ]
    ] = []
    for noise in NOISE_LEVELS:
        for budget in MID_HIGH_BUDGETS:
            for macro in range(1, 101):
                checkpoint = _extension_checkpoint(noise, budget, macro)
                if not checkpoint.exists():
                    jobs.append(
                        (
                            (
                                function_data,
                                noise,
                                budget,
                                2048,
                                macro,
                                EXTENSION_PHASE,
                                True,
                            ),
                            noise,
                            budget,
                            macro,
                        )
                    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_metadata = {
                executor.submit(_worker, payload): (noise, budget, macro)
                for payload, noise, budget, macro in jobs
            }
            for completed, future in enumerate(as_completed(future_metadata), 1):
                noise, budget, macro = future_metadata[future]
                rows = future.result()
                destination = _extension_checkpoint(noise, budget, macro)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(rows) + "\n", encoding="utf-8")
                if completed % 25 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)} new trajectories", flush=True)
    rows: list[dict[str, object]] = []
    for noise in NOISE_LEVELS:
        for budget in MID_HIGH_BUDGETS:
            for macro in range(1, 101):
                checkpoint = _extension_checkpoint(noise, budget, macro)
                if not checkpoint.exists():
                    raise RuntimeError(f"missing extension trajectory: {checkpoint}")
                rows.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
    output = HERE / "formal_extension"
    write_csv(output / "all_recommendations.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    print(f"assembled {len(rows) // len(RULES)} extension paths", flush=True)


def _additional_checkpoint(noise: str, budget: int, macro: int) -> Path:
    return (
        HERE
        / "formal_additional_budgets"
        / "trajectory_results"
        / noise
        / f"T_{budget}"
        / f"macro_{macro:03d}.json"
    )


def run_additional_budgets(workers: int) -> None:
    with (HERE / "formal_low_budget" / "selected_function.json").open(
        encoding="utf-8"
    ) as handle:
        function = objective_from_dict(json.load(handle))
    function_data = objective_to_dict(function)
    jobs: list[
        tuple[
            tuple[dict[str, object], str, int, int, int, int, bool],
            str,
            int,
            int,
        ]
    ] = []
    for noise in NOISE_LEVELS:
        for budget in ADDITIONAL_BUDGETS:
            for macro in range(1, 101):
                checkpoint = _additional_checkpoint(noise, budget, macro)
                if not checkpoint.exists():
                    jobs.append(
                        (
                            (
                                function_data,
                                noise,
                                budget,
                                2048,
                                macro,
                                ADDITIONAL_PHASE,
                                True,
                            ),
                            noise,
                            budget,
                            macro,
                        )
                    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_metadata = {
                executor.submit(_worker, payload): (noise, budget, macro)
                for payload, noise, budget, macro in jobs
            }
            for completed, future in enumerate(as_completed(future_metadata), 1):
                noise, budget, macro = future_metadata[future]
                rows = future.result()
                destination = _additional_checkpoint(noise, budget, macro)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(rows) + "\n", encoding="utf-8")
                if completed % 25 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)} new trajectories", flush=True)
    rows: list[dict[str, object]] = []
    for noise in NOISE_LEVELS:
        for budget in ADDITIONAL_BUDGETS:
            for macro in range(1, 101):
                checkpoint = _additional_checkpoint(noise, budget, macro)
                if not checkpoint.exists():
                    raise RuntimeError(f"missing additional trajectory: {checkpoint}")
                rows.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
    output = HERE / "formal_additional_budgets"
    write_csv(output / "all_recommendations.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    print(f"assembled {len(rows) // len(RULES)} additional paths", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("screen", "confirm", "formal", "extension", "additional"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.command == "screen":
        run_screen(args.workers)
    elif args.command == "confirm":
        run_confirmation(args.workers)
    elif args.command == "formal":
        run_formal(args.workers)
    elif args.command == "extension":
        run_extension(args.workers)
    elif args.command == "additional":
        run_additional_budgets(args.workers)


if __name__ == "__main__":
    main()
