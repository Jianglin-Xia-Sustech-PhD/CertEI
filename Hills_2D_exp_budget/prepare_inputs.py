"""Create candidates and calibrate LQEI's RKHS norm from initial GPs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from dynamic_gp import fit_dynamic_gp, posterior_mean_rkhs_norm
from study_config import (
    CALIBRATION_BUDGETS,
    CALIBRATION_MACRO_OFFSET,
    CALIBRATION_MACROS,
    CANDIDATE_SEED,
    CONFIGURATION_SLUG,
    DESIGN_SEED,
    FUNCTION_NAMES,
    FUNCTION_SEED_CODES,
    INITIAL_MLE_MAXITER,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    NOISE_SEED,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    make_batch_schedule,
)
from test_functions import function_without_rkhs_metadata


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs" / CONFIGURATION_SLUG
METADATA_PATH = INPUT_DIR / "function_metadata.json"


def _candidate_path(function_name: str) -> Path:
    return INPUT_DIR / f"{function_name}_sobol_candidates_{N_CANDIDATES}.npy"


def _build_candidates(function_name: str) -> np.ndarray:
    exponent = int(round(np.log2(N_CANDIDATES)))
    if 2**exponent != N_CANDIDATES:
        raise ValueError("N_CANDIDATES must be a power of two")
    function_code = FUNCTION_SEED_CODES[function_name]
    sampler = qmc.Sobol(d=2, scramble=True, seed=CANDIDATE_SEED + function_code)
    candidates = sampler.random_base2(m=exponent)
    return np.asarray(candidates, dtype=float)


def _initial_points(function_name: str, macro: int) -> np.ndarray:
    function_code = FUNCTION_SEED_CODES[function_name]
    seed = np.random.SeedSequence([DESIGN_SEED, function_code, macro])
    integer_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
    return qmc.LatinHypercube(d=2, seed=integer_seed).random(N_INITIAL)


def _initial_fit_kernel(
    function_name: str,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """Return a pooled median kernel fitted on fixed standardized outputs."""

    function = function_without_rkhs_metadata(function_name)
    function_code = FUNCTION_SEED_CODES[function_name]
    pooled_lengthscales: list[np.ndarray] = []
    pooled_process_variances: list[float] = []
    pooled_constant_means: list[float] = []
    pooled_effective_norms: list[float] = []
    calibration_records: list[dict[str, object]] = []
    by_budget: dict[str, object] = {}
    for budget in CALIBRATION_BUDGETS:
        schedule = make_batch_schedule(budget)
        fitted_lengthscales: list[np.ndarray] = []
        fitted_process_variances: list[float] = []
        fitted_effective_norms: list[float] = []
        for calibration_index in range(1, CALIBRATION_MACROS + 1):
            macro = CALIBRATION_MACRO_OFFSET + calibration_index
            points = _initial_points(function_name, macro)
            true_values = function.evaluate(points)
            noise_sds = function.noise_standard_deviation(points)
            sample_means = np.empty(N_INITIAL)
            mean_variances = np.empty(N_INITIAL)
            for event_id in range(N_INITIAL):
                replications = int(schedule[event_id])
                seed = np.random.SeedSequence(
                    [NOISE_SEED, function_code, macro, event_id]
                )
                rng = np.random.Generator(np.random.Philox(seed))
                sample_means[event_id] = (
                    true_values[event_id]
                    + noise_sds[event_id]
                    * rng.standard_normal()
                    / np.sqrt(replications)
                )
                sample_variance = (
                    noise_sds[event_id] ** 2
                    * rng.chisquare(replications - 1)
                    / (replications - 1)
                )
                sample_variance = float(
                    np.clip(
                        sample_variance,
                        function.variance_min,
                        function.variance_max,
                    )
                )
                mean_variances[event_id] = sample_variance / replications
            fit = fit_dynamic_gp(
                points,
                function.standardize_outputs(sample_means),
                function.standardize_variances(mean_variances),
                points,
                max_iterations=INITIAL_MLE_MAXITER,
            )
            effective_norm = posterior_mean_rkhs_norm(
                points,
                function.standardize_outputs(sample_means),
                function.standardize_variances(mean_variances),
                fit.hyperparameters,
            )
            fitted_lengthscales.append(fit.hyperparameters.lengthscales)
            fitted_process_variances.append(fit.hyperparameters.process_variance)
            fitted_effective_norms.append(effective_norm)
            pooled_constant_means.append(fit.hyperparameters.constant_mean)
            calibration_records.append(
                {
                    "budget": int(budget),
                    "macro": int(macro),
                    "posterior_mean_rkhs_norm": float(effective_norm),
                    "constant_mean": float(fit.hyperparameters.constant_mean),
                    "process_variance": float(fit.hyperparameters.process_variance),
                    "lengthscale_1": float(fit.hyperparameters.lengthscales[0]),
                    "lengthscale_2": float(fit.hyperparameters.lengthscales[1]),
                }
            )
        lengthscale_values = np.asarray(fitted_lengthscales, dtype=float)
        variance_values = np.asarray(fitted_process_variances, dtype=float)
        norm_values = np.asarray(fitted_effective_norms, dtype=float)
        pooled_lengthscales.extend(lengthscale_values)
        pooled_process_variances.extend(variance_values)
        pooled_effective_norms.extend(norm_values)
        by_budget[str(budget)] = {
            "lengthscale_median": np.median(lengthscale_values, axis=0).tolist(),
            "lengthscale_q25": np.quantile(lengthscale_values, 0.25, axis=0).tolist(),
            "lengthscale_q75": np.quantile(lengthscale_values, 0.75, axis=0).tolist(),
            "process_variance_median": float(np.median(variance_values)),
            "process_variance_q25": float(np.quantile(variance_values, 0.25)),
            "process_variance_q75": float(np.quantile(variance_values, 0.75)),
            "posterior_mean_rkhs_norm_q25": float(np.quantile(norm_values, 0.25)),
            "posterior_mean_rkhs_norm_median": float(np.median(norm_values)),
            "posterior_mean_rkhs_norm_q75": float(np.quantile(norm_values, 0.75)),
            "posterior_mean_rkhs_norm_max": float(np.max(norm_values)),
        }
    pooled_lengths = np.asarray(pooled_lengthscales, dtype=float)
    pooled_variances = np.asarray(pooled_process_variances, dtype=float)
    pooled_means = np.asarray(pooled_constant_means, dtype=float)
    pooled_norms = np.asarray(pooled_effective_norms, dtype=float)
    median_lengthscales = np.median(pooled_lengths, axis=0)
    median_process_variance = float(np.median(pooled_variances))
    details = {
        "macro_ids": [
            CALIBRATION_MACRO_OFFSET + 1,
            CALIBRATION_MACRO_OFFSET + CALIBRATION_MACROS,
        ],
        "macros_per_budget": CALIBRATION_MACROS,
        "budgets": list(CALIBRATION_BUDGETS),
        "per_budget": by_budget,
        "pooled_lengthscale_median": median_lengthscales.tolist(),
        "pooled_lengthscale_q25": np.quantile(
            pooled_lengths, 0.25, axis=0
        ).tolist(),
        "pooled_lengthscale_q75": np.quantile(
            pooled_lengths, 0.75, axis=0
        ).tolist(),
        "pooled_process_variance_median": median_process_variance,
        "pooled_process_variance_q25": float(np.quantile(pooled_variances, 0.25)),
        "pooled_process_variance_q75": float(np.quantile(pooled_variances, 0.75)),
        "pooled_constant_mean_median": float(np.median(pooled_means)),
        "posterior_mean_rkhs_norm_q10": float(np.quantile(pooled_norms, 0.10)),
        "posterior_mean_rkhs_norm_q25": float(np.quantile(pooled_norms, 0.25)),
        "posterior_mean_rkhs_norm_median": float(np.median(pooled_norms)),
        "posterior_mean_rkhs_norm_q75": float(np.quantile(pooled_norms, 0.75)),
        "posterior_mean_rkhs_norm_q90": float(np.quantile(pooled_norms, 0.90)),
        "posterior_mean_rkhs_norm_max": float(np.max(pooled_norms)),
        "posterior_mean_rkhs_norm_rule": (
            "maximum across all held-out initial-GP posterior-mean RKHS norms"
        ),
        "calibration_records": calibration_records,
        "output_standardization": "(f - f_min) / (f_max - f_min)",
    }
    return (
        np.asarray(median_lengthscales, dtype=float),
        median_process_variance,
        details,
    )


def prepare_inputs(overwrite: bool = False) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists() and not overwrite:
        candidate_paths = [_candidate_path(name) for name in FUNCTION_NAMES]
        if all(path.exists() for path in candidate_paths):
            return METADATA_PATH

    metadata: dict[str, object] = {
        "candidate_count": N_CANDIDATES,
        "candidate_rule": (
            "pure scrambled Sobol points on [0,1]^2; no analytic global "
            "minimizers are injected"
        ),
        "analytic_optima_injected": False,
        "noise_relationship": (
            f"sigma/R_f = {NOISE_A_MULTIPLIER:.12g} + "
            f"{NOISE_B_MULTIPLIER:.12g}*z(x)"
        ),
        "noise_a_multiplier": NOISE_A_MULTIPLIER,
        "noise_b_multiplier": NOISE_B_MULTIPLIER,
        "reference_rkhs_rule": (
            "maximum of 300 noise-regularized initial-GP posterior-mean RKHS "
            "norms after fixed output standardization; the fits use held-out "
            "calibration macros 1001--1100 at three representative budgets"
        ),
        "functions": {},
    }
    for name in FUNCTION_NAMES:
        function = function_without_rkhs_metadata(name)
        candidates = _build_candidates(name)
        candidate_path = _candidate_path(name)
        np.save(candidate_path, candidates)
        lengthscales, process_variance, calibration = _initial_fit_kernel(name)
        rkhs_norm = float(calibration["posterior_mean_rkhs_norm_max"])
        intercept = float(calibration["pooled_constant_mean_median"])
        calibration_path = INPUT_DIR / f"{name}_bf_calibration_records.csv"
        calibration_records = calibration["calibration_records"]
        with calibration_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(calibration_records[0])
            )
            writer.writeheader()
            writer.writerows(calibration_records)
        checksum = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        metadata["functions"][name] = {
            "display_name": function.display_name,
            "optimum_points": function.optimum_points.tolist(),
            "optimum_value": function.optimum_value,
            "maximum_value": function.maximum_value,
            "response_range": function.response_range,
            "output_location": function.output_location,
            "output_scale": function.output_scale,
            "output_standardization": "(f - f_min) / (f_max - f_min)",
            "noise_sd_min": NOISE_SD_MIN_MULTIPLIER * function.response_range,
            "noise_sd_max": NOISE_SD_MAX_MULTIPLIER * function.response_range,
            "standardized_noise_sd_min": NOISE_SD_MIN_MULTIPLIER,
            "standardized_noise_sd_max": NOISE_SD_MAX_MULTIPLIER,
            "reference_lengthscales": lengthscales.tolist(),
            "reference_process_variance": process_variance,
            "reference_constant_mean": intercept,
            "reference_rkhs_intercept_standardized": intercept,
            "rkhs_norm": rkhs_norm,
            "rkhs_norm_definition": (
                "maximum held-out initial-GP posterior-mean RKHS norm"
            ),
            "bf_calibration_file": calibration_path.name,
            "initial_kernel_calibration": calibration,
            "candidate_file": candidate_path.name,
            "candidate_sha256": checksum,
        }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return METADATA_PATH


def load_function(function_name: str):
    from test_functions import function_from_metadata

    prepare_inputs()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return function_from_metadata(function_name, metadata["functions"][function_name])


def load_candidates(function_name: str) -> np.ndarray:
    prepare_inputs()
    return np.asarray(np.load(_candidate_path(function_name)), dtype=float)


if __name__ == "__main__":
    print(prepare_inputs(overwrite=True))
