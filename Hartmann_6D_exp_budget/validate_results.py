"""Strict structural validation for the completed formal experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_runner import RESULT_DIR, RESULT_GROUP, all_specs
from study_config import (
    DIMENSION,
    FORMAL_MACROS,
    LENGTHSCALE_LOG_PRIOR_SD,
    MACRO_SEED_OFFSET,
    N_INITIAL,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    PROCESS_VARIANCE_LOG_PRIOR_SD,
)


ROOT = Path(__file__).resolve().parent


def validate() -> Path:
    specifications = all_specs(FORMAL_MACROS)
    configuration_count = len(specifications)
    report: dict[str, object] = {
        "expected_configurations": configuration_count,
        "expected_macros_per_configuration": FORMAL_MACROS,
        "expected_total_macros": configuration_count * FORMAL_MACROS,
        "configurations": [],
    }
    total_files = 0
    for spec in specifications:
        directory = RESULT_DIR / spec.slug / "macro_results"
        files = sorted(directory.glob("macro_*.npz"))
        if len(files) != FORMAL_MACROS:
            raise RuntimeError(f"{spec.slug}: found {len(files)} macro files")
        response_successes = 0
        response_fits = 0
        noise_successes = 0
        noise_fits = 0
        gaps: list[float] = []
        for expected_macro, path in enumerate(files, start=1):
            with np.load(path, allow_pickle=False) as data:
                if int(data["macro"]) != expected_macro:
                    raise RuntimeError(f"unexpected macro number in {path}")
                if "macro_seed_id" in data.files:
                    if int(data["macro_seed_id"]) != expected_macro + MACRO_SEED_OFFSET:
                        raise RuntimeError(f"unexpected macro seed id in {path}")
                elif MACRO_SEED_OFFSET != 0:
                    raise RuntimeError(f"missing macro seed id in {path}")
                if int(data["budget_used"]) != spec.total_budget:
                    raise RuntimeError(f"budget mismatch in {path}")
                if int(np.sum(data["event_replications"])) != spec.total_budget:
                    raise RuntimeError(f"event budget mismatch in {path}")
                if str(data["kernel"]) != "matern52":
                    raise RuntimeError(f"kernel mismatch in {path}")
                if str(data["gp_output_standardization"]) != (
                    "(f - f_min) / (f_max - f_min)"
                ):
                    raise RuntimeError(f"output-scale mismatch in {path}")
                expected_auxiliary_noise = spec.algorithm in ("SKO", "EQI")
                if bool(data["auxiliary_noise_gp_used"]) != expected_auxiliary_noise:
                    raise RuntimeError(f"auxiliary noise GP mismatch in {path}")
                if bool(data["analytic_noise_prediction_used"]):
                    raise RuntimeError(f"analytic noise flag mismatch in {path}")
                if not np.isclose(
                    float(data["noise_sd_min_multiplier"]),
                    NOISE_SD_MIN_MULTIPLIER,
                ) or not np.isclose(
                    float(data["noise_sd_max_multiplier"]),
                    NOISE_SD_MAX_MULTIPLIER,
                ):
                    raise RuntimeError(f"noise envelope mismatch in {path}")
                if not np.isclose(
                    float(data["noise_a_multiplier"]), NOISE_A_MULTIPLIER
                ) or not np.isclose(
                    float(data["noise_b_multiplier"]), NOISE_B_MULTIPLIER
                ):
                    raise RuntimeError(f"noise coefficient mismatch in {path}")
                response = np.asarray(data["response_hyperparameter_history"])
                expected_refits = int(data["schedule"].size - N_INITIAL + 1)
                expected_columns = DIMENSION + 7
                success_column = DIMENSION + 3
                penalty_column = DIMENSION + 5
                if response.shape != (expected_refits, expected_columns):
                    raise RuntimeError(f"response history mismatch in {path}")
                if np.any(~np.isfinite(response)) or np.any(
                    response[:, 1 : DIMENSION + 2] <= 0.0
                ):
                    raise RuntimeError(f"invalid response hyperparameters in {path}")
                response_successes += int(np.sum(response[:, success_column]))
                response_fits += response.shape[0]
                noise = np.asarray(data["noise_hyperparameter_history"])
                if expected_auxiliary_noise:
                    if noise.shape != (expected_refits, expected_columns):
                        raise RuntimeError(f"noise history mismatch in {path}")
                    if np.any(~np.isfinite(noise)) or np.any(
                        noise[:, 1 : DIMENSION + 2] <= 0.0
                    ):
                        raise RuntimeError(f"invalid noise hyperparameters in {path}")
                    if np.any(noise[:, penalty_column] != 0.0):
                        raise RuntimeError(f"unexpected noise-GP regularization in {path}")
                    noise_successes += int(np.sum(noise[:, success_column]))
                    noise_fits += noise.shape[0]
                elif noise.size != 0:
                    raise RuntimeError(f"unexpected noise history in {path}")
                if response[0, penalty_column] != 0.0 or np.any(
                    response[1:, penalty_column] < 0.0
                ):
                    raise RuntimeError(f"invalid MAP penalties in {path}")
                if not np.isclose(
                    float(data["lengthscale_log_prior_sd"]),
                    LENGTHSCALE_LOG_PRIOR_SD,
                ):
                    raise RuntimeError(f"lengthscale prior mismatch in {path}")
                if not np.isclose(
                    float(data["process_variance_log_prior_sd"]),
                    PROCESS_VARIANCE_LOG_PRIOR_SD,
                ):
                    raise RuntimeError(f"process-variance prior mismatch in {path}")
                initial_log_variance = np.log(response[0, 1])
                if not np.isclose(
                    float(data["response_process_variance_log_prior_mean"]),
                    initial_log_variance,
                ):
                    raise RuntimeError(
                        f"process-variance prior centre mismatch in {path}"
                    )
                gap = float(data["normalized_gap"])
                if not np.isfinite(gap) or gap < 0.0:
                    raise RuntimeError(f"invalid normalized gap in {path}")
                gaps.append(gap)
        total_files += len(files)
        report["configurations"].append(
            {
                "function": spec.function_name,
                "algorithm": spec.algorithm,
                "budget": spec.total_budget,
                "macro_files": len(files),
                "mean_normalized_gap": float(np.mean(gaps)),
                "response_optimizer_success_fraction": (
                    response_successes / response_fits
                ),
                "auxiliary_noise_gp_used": spec.algorithm in ("SKO", "EQI"),
                "analytic_noise_prediction_used": False,
                "noise_optimizer_success_fraction": (
                    noise_successes / noise_fits if noise_fits else None
                ),
            }
        )
    expected_total = configuration_count * FORMAL_MACROS
    if total_files != expected_total:
        raise RuntimeError(f"expected {expected_total} files, found {total_files}")
    report["validated_total_macros"] = total_files
    report["status"] = "passed"
    destination = ROOT / "analysis" / RESULT_GROUP / "validation_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(validate())
