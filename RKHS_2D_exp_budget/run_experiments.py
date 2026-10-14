"""Run the equation-(20) minimum-bound LQEI study and comparisons."""

from __future__ import annotations

# Each independent macro uses one numerical-library thread. Set before NumPy
# loads so process parallelism does not oversubscribe the desktop.
import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import scipy

from comparison_runner import (
    COMPARISON_ALGORITHMS,
    ComparisonSpec,
    run_comparison_specs,
)
from confidence import (
    INFORMATION_GAIN_CONSTANT,
    INFORMATION_GAIN_METHOD,
)
from experiment_runner import (
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    ROOT,
    ExperimentSpec,
    prepare_confidence_schedules,
    run_specs,
)
from study_config import (
    ABLATION_BUDGETS,
    CANDIDATE_SEED,
    CONFIDENCE_DELTA,
    DESIGN_SEED,
    FORMAL_MACROS,
    FORMAL_PHASE_CODE,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_RULE,
    N_CANDIDATES,
    NOISE_SEED,
    budgets_for_function,
    rho_for_function,
)


FUNCTION_NAMES = ("rkhs_matern52_2d", "rkhs_se_2d")
LQEI_RESULT_GROUP = "formal_min_bound_0p2_lambda_acquisition_0p2_lambda_final"
COMPARISON_RESULT_GROUP = "formal_algorithm_comparison_log_batches"
ABLATION_RESULT_GROUP = "replication_order_ablation_min_bound"
INCUMBENT_ABLATION_RESULT_GROUP = "incumbent_rule_ablation_min_bound"
INCUMBENT_ABLATION_RULES = (
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
ABLATION_NONBENCHMARK_ORDERS = (
    "constant",
    "log_log",
    "power_1p2",
)


def lqei_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            phase=LQEI_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_order="log",
            replication_multiplier=rho_for_function(name),
            anchor_budget=budgets_for_function(name)[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for name in FUNCTION_NAMES
        for budget in budgets_for_function(name)
    ]


def ablation_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            phase=ABLATION_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_order=order,
            replication_multiplier=rho_for_function(name),
            anchor_budget=ABLATION_BUDGETS[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for name in FUNCTION_NAMES
        for order in ABLATION_NONBENCHMARK_ORDERS
        for budget in ABLATION_BUDGETS
    ]


def comparison_specs() -> list[ComparisonSpec]:
    return [
        ComparisonSpec(
            algorithm=algorithm,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_multiplier=rho_for_function(name),
            anchor_budget=budgets_for_function(name)[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            result_group=COMPARISON_RESULT_GROUP,
        )
        for name in FUNCTION_NAMES
        for algorithm in COMPARISON_ALGORITHMS
        for budget in budgets_for_function(name)
    ]


def incumbent_ablation_specs() -> list[ExperimentSpec]:
    """Additional incumbent rules; the dynamic -0.2 Lambda arm reuses main LQEI."""

    return [
        ExperimentSpec(
            phase=INCUMBENT_ABLATION_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_order="log",
            replication_multiplier=rho_for_function(name),
            anchor_budget=budgets_for_function(name)[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
            incumbent_rule=rule,
        )
        for name in FUNCTION_NAMES
        for rule in INCUMBENT_ABLATION_RULES
        for budget in budgets_for_function(name)
    ]


def write_protocol(workers: int) -> None:
    protocol_path = ROOT / "protocol.json"
    previous = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    now = datetime.now(timezone.utc).isoformat()
    paths = list(ROOT.glob("*.py")) + list((ROOT / "fixed_functions").glob("*"))
    paths += list((ROOT / "shared_inputs").glob("*")) + [ROOT / "reference/paper.md"]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    protocol = {
        "created_utc": previous.get("created_utc", now),
        "updated_utc": now,
        "study": "2D LQEI with equation-(20) minimum information-gain bound",
        "lqei": {
            "information_gain_method": INFORMATION_GAIN_METHOD,
            "information_gain_constant": INFORMATION_GAIN_CONSTANT,
            "effective_argument": "U_n(b) = n*b/nu_min",
            "finite_sample_log": "log(1 + U_n)",
            "squared_exponential_d2": "G_SE(U) = log(1+U)^3",
            "matern52_d2": "G_Mat(U) = U^(2/7)*log(1+U)^(5/7)",
            "finite_batch_bound": "n/2 * log(1 + b/nu_min)",
            "information_gain_bound": (
                "min{G_k(U_n), n/2 * log(1 + b/nu_min)}"
            ),
            "lambda_formula": (
                "B_f + sqrt(2*kappa_T*(min{G_k(U_n), "
                "n/2*log(1+b/nu_min)} + log(3/(2*delta))))"
            ),
            "acquisition_lambda_multiplier": LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
            "incumbent": "min_x [mu_n(x) - 0.2*Lambda_n*sigma_n(x)]",
            "recommendation_rule": LQEI_RECOMMENDATION_RULE,
            "recommendation_lambda_multiplier": LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
            "recommendation": "argmin over sampled points of mu_N + 0.2*Lambda_N*sigma_N",
        },
        "incumbent_ablation": {
            "varied_component_only": "EI incumbent",
            "rules": [
                "min_x(mu_n - 0.2*Lambda_n*sigma_n) [reused main LQEI]",
                "min_x(mu_n)",
                "min_x(mu_n + sigma_n)",
            ],
            "final_recommendation_unchanged": (
                "argmin over sampled points of mu_N + 0.2*Lambda_N*sigma_N"
            ),
            "common_random_numbers": True,
        },
        "final_recommendation_ablation": {
            "varied_component_only": "final sampled-point recommendation rule",
            "rules": [
                "argmin_x(mu_N + 0.2*Lambda_N*sigma_N) [reused main output]",
                "argmin_x(mu_N)",
                "argmin_x(mu_N - 10*sigma_N)",
            ],
            "incumbent_unchanged": (
                "min_x(mu_n - 0.2*Lambda_n*sigma_n)"
            ),
            "postprocessed_from_same_terminal_posterior": True,
            "new_simulations_required": False,
        },
        "replication": {
            "main": "rho*log(T)",
            "ablation": [
                "rho",
                "rho*log(log(T))",
                "rho*log(T)",
                "rho*T^(1/2)",
            ],
            "same_numeric_rho_without_anchor_normalization": True,
            "balanced_schedules_exhaust_each_total_budget": True,
            "log_benchmark_reused_from_main_lqei": True,
        },
        "numerical_stability": {
            "same_site_observation_compression": (
                "independent Gaussian batch means at an identical design are "
                "combined by exact precision-weighted sufficient statistics"
            ),
            "posterior_rebuild": (
                "the exact fixed-hyperparameter GP posterior is periodically "
                "rebuilt from those sufficient statistics on very long paths"
            ),
            "ei_underflow": (
                "when ordinary EI values all underflow to zero, the same EI "
                "argmax is evaluated in the log domain"
            ),
        },
        "budgets_by_function": {
            name: list(budgets_for_function(name)) for name in FUNCTION_NAMES
        },
        "ablation_budgets": list(ABLATION_BUDGETS),
        "macro_count_per_configuration": FORMAL_MACROS,
        "comparison_algorithms": list(COMPARISON_ALGORITHMS),
        "sko": {
            "registered_name": "SKO",
            "source": (
                "KrigingBasedHetero_sustech_python/AEI_python/"
                "aei/acquisition.py"
            ),
            "response_gp_hyperparameters_fixed": True,
            "noise_gp_hyperparameters_fit_on_initial_design_only": True,
            "batch_schedule": "same exact balanced rho*log(T) schedule as LQEI",
            "iteration_budget": "one complete matched batch per iteration",
        },
        "seeds": {
            "design": DESIGN_SEED,
            "noise": NOISE_SEED,
            "candidate": CANDIDATE_SEED,
            "phase_code": FORMAL_PHASE_CODE,
        },
        "lqei_specifications": [asdict(spec) for spec in lqei_specs()],
        "comparison_specifications": [asdict(spec) for spec in comparison_specs()],
        "ablation_specifications": [asdict(spec) for spec in ablation_specs()],
        "incumbent_ablation_specifications": [
            asdict(spec) for spec in incumbent_ablation_specs()
        ],
        "workers": workers,
        "numerical_threads_per_worker": 1,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "sha256": hashes,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=(
            "confidence",
            "lqei",
            "comparison",
            "formal",
            "incumbent_ablation",
            "final_recommendation_ablation",
            "ablation",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.phase != "analyze":
        write_protocol(args.workers)
    if args.phase in ("confidence", "formal", "all"):
        prepare_confidence_schedules(lqei_specs())
    if args.phase in ("lqei", "formal", "all"):
        run_specs(lqei_specs(), workers=args.workers, overwrite=args.overwrite)
    if args.phase in ("comparison", "formal", "all"):
        run_comparison_specs(
            comparison_specs(), workers=args.workers, overwrite=args.overwrite
        )
    if args.phase in ("ablation", "all"):
        run_specs(ablation_specs(), workers=args.workers, overwrite=args.overwrite)
        if args.phase == "ablation":
            from analyze_replication_order_ablation import main as analyze_replication

            analyze_replication()
    if args.phase == "incumbent_ablation":
        run_specs(
            incumbent_ablation_specs(),
            workers=args.workers,
            overwrite=args.overwrite,
        )
        from analyze_incumbent_ablation import main as analyze_incumbent

        analyze_incumbent()
    if args.phase == "final_recommendation_ablation":
        from analyze_final_recommendation_ablation import main as analyze_final_rule

        analyze_final_rule()
    if args.phase in ("analyze", "formal", "all"):
        from analyze_results import main as analyze

        analyze()
    elapsed = time.perf_counter() - started
    print(f"Finished {args.phase} in {elapsed:.1f} seconds.", flush=True)
    if args.phase in ("formal", "all"):
        protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        first_started = datetime.fromisoformat(protocol["created_utc"])
        payload = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "last_invocation_elapsed_seconds": elapsed,
            "total_wall_seconds_since_first_start": (
                datetime.now(timezone.utc) - first_started
            ).total_seconds(),
            "lqei_macro_count": len(lqei_specs()) * FORMAL_MACROS,
            "comparison_macro_count": len(comparison_specs()) * FORMAL_MACROS,
            "additional_ablation_macro_count": (
                len(ablation_specs()) * FORMAL_MACROS if args.phase == "all" else 0
            ),
            "total_macro_count": (
                len(lqei_specs())
                + len(comparison_specs())
                + (len(ablation_specs()) if args.phase == "all" else 0)
            )
            * FORMAL_MACROS,
            "configuration_count": (
                len(lqei_specs())
                + len(comparison_specs())
                + (len(ablation_specs()) if args.phase == "all" else 0)
            ),
        }
        (ROOT / "run_completion.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
