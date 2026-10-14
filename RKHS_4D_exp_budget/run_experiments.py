"""Run the four-dimensional combined-bound LQEI study and its comparisons."""

from __future__ import annotations

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
from confidence import INFORMATION_GAIN_CONSTANT, INFORMATION_GAIN_METHOD
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
    ABLATION_N_INITIAL,
    BUDGET_EXPONENTS,
    CANDIDATE_SEED,
    CONFIDENCE_DELTA,
    DESIGN_SEED,
    FORMAL_MACROS,
    FORMAL_PHASE_CODE,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_RULE,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_SEED,
    REPLICATION_ORDERS,
    budgets_for_function,
    rho_for_function,
)


FUNCTION_NAMES = ("rkhs_matern52_4d", "rkhs_se_4d")
LQEI_RESULT_GROUP = "formal_min_bound_0p2_lambda_acquisition_0p2_lambda_final"
SMOKE_RESULT_GROUP = "smoke_se_min_bound_stability"
COMPARISON_RESULT_GROUP = "formal_algorithm_comparison_log_batches"
ABLATION_RESULT_GROUP = "replication_order_ablation"
INCUMBENT_ABLATION_RESULT_GROUP = "incumbent_rule_ablation_min_bound"
INCUMBENT_ABLATION_RULES = (
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
ABLATION_ORDERS = REPLICATION_ORDERS
SMOKE_MACROS = 10
SMOKE_BUDGETS = (budgets_for_function(FUNCTION_NAMES[0])[3], budgets_for_function(FUNCTION_NAMES[0])[-1])
ABLATION_SMOKE_MACROS = 2


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
            initial_point_count=N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for name in FUNCTION_NAMES
        for budget in budgets_for_function(name)
    ]


def main_lqei_specs() -> list[ExperimentSpec]:
    return lqei_specs()


def smoke_specs() -> list[ExperimentSpec]:
    """Paired SE checks at a moderate and the largest requested budget."""

    name = "rkhs_se_4d"
    return [
        ExperimentSpec(
            phase=SMOKE_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_order="log",
            replication_multiplier=rho_for_function(name),
            anchor_budget=budgets_for_function(name)[0],
            macro_count=SMOKE_MACROS,
            initial_point_count=N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for budget in SMOKE_BUDGETS
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
            initial_point_count=ABLATION_N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for name in FUNCTION_NAMES
        for order in ABLATION_ORDERS
        for budget in ABLATION_BUDGETS
    ]


def constant_ablation_specs() -> list[ExperimentSpec]:
    """O(1) arm of the full four-order replication ablation."""

    return [spec for spec in ablation_specs() if spec.replication_order == "constant"]


def constant_workers_for_spec(spec: ExperimentSpec, requested_workers: int) -> int:
    """Bound memory use on the longest SE paths without changing any samples."""

    if spec.function_name == "rkhs_se_4d":
        if spec.total_budget >= ABLATION_BUDGETS[-1]:
            return min(requested_workers, 2)
        if spec.total_budget >= ABLATION_BUDGETS[-2]:
            return min(requested_workers, 4)
    return requested_workers


def ablation_smoke_specs() -> list[ExperimentSpec]:
    """Worst-case numerical check before launching the full ablation."""

    return [
        spec
        for spec in ablation_specs()
        if spec.function_name == "rkhs_se_4d"
        and spec.replication_order == "log_log"
        and spec.total_budget == ABLATION_BUDGETS[-1]
    ]


def comparison_specs() -> list[ComparisonSpec]:
    return [
        ComparisonSpec(
            algorithm=algorithm,  # type: ignore[arg-type]
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=budget,
            replication_multiplier=rho_for_function(name),
            anchor_budget=budgets_for_function(name)[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            initial_replications=None,
            result_group=COMPARISON_RESULT_GROUP,
        )
        for name in FUNCTION_NAMES
        for algorithm in COMPARISON_ALGORITHMS
        for budget in budgets_for_function(name)
    ]


def incumbent_ablation_specs() -> list[ExperimentSpec]:
    """Additional incumbent rules; the dynamic arm reuses main LQEI."""

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
            initial_point_count=N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
            incumbent_rule=rule,
        )
        for name in FUNCTION_NAMES
        for rule in INCUMBENT_ABLATION_RULES
        for budget in budgets_for_function(name)
    ]


def write_protocol(workers: int) -> None:
    destination = ROOT / "protocol.json"
    previous = (
        json.loads(destination.read_text(encoding="utf-8"))
        if destination.exists()
        else {}
    )
    now = datetime.now(timezone.utc).isoformat()
    paths = list(ROOT.glob("*.py"))
    paths += list((ROOT / "fixed_functions").glob("*"))
    paths += list((ROOT / "shared_inputs").glob("*"))
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    protocol = {
        "created_utc": previous.get("created_utc", now),
        "updated_utc": now,
        "study": "4D continuous LQEI with a dynamic minimum information-gain bound",
        "information_gain": {
            "method": INFORMATION_GAIN_METHOD,
            "constant": INFORMATION_GAIN_CONSTANT,
            "argument": "U_n = n*b_T/variance_min",
            "squared_exponential_d4": "G(U)=log(1+U)^5",
            "matern52_d4": "G(U)=U^(4/9)*log(1+U)^(5/9)",
            "finite_batch": "n/2*log(1+b_T/variance_min)",
            "combined": "min{G_k(U_n), n/2*log(1+b_T/variance_min)}",
        },
        "lqei": {
            "lambda_formula": "B_f + sqrt(2*kappa_T*(min_bound_n+log(3/(2*delta))))",
            "acquisition_lambda_multiplier": LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
            "incumbent": "min_x [mu_n(x)-0.2*Lambda_n*sigma_n(x)]",
            "recommendation_rule": LQEI_RECOMMENDATION_RULE,
            "recommendation_lambda_multiplier": LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
            "recommendation": "argmin over sampled points of mu_N+0.2*Lambda_N*sigma_N",
        },
        "incumbent_ablation": {
            "varied_component_only": "EI incumbent",
            "rules": [
                "min_x(mu_n - 0.2*Lambda_n*sigma_n) [reused main LQEI]",
                "min_x(mu_n)",
                "min_x(mu_n + sigma_n)",
            ],
            "final_recommendation_unchanged": (
                "argmin over sampled points of mu_N+0.2*Lambda_N*sigma_N"
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
            "incumbent_unchanged": "min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            "postprocessed_from_same_terminal_posterior": True,
            "new_simulations_required": False,
        },
        "budgets": list(budgets_for_function(FUNCTION_NAMES[0])),
        "budget_exponents": list(BUDGET_EXPONENTS),
        "budget_rounding": "nearest integer to exp(exponent)",
        "main_initial_point_count": N_INITIAL,
        "replication": {
            "main_and_comparisons": "rho*log(T)",
            "ablation_budgets": list(ABLATION_BUDGETS),
            "ablation_orders": [
                "rho",
                "rho*log(log(T))",
                "rho*log(T)",
                "rho*T^(1/2)",
            ],
            "rounding": "each nominal replication number is rounded up to a multiple of five",
            "constant_nominal_batch_size": 10,
            "constant_actual_batch_size": "balanced 10 or 11, summing exactly to T",
            "constant_execution_worker_caps": {
                f"SE_T_{ABLATION_BUDGETS[-1]}": 2,
                f"SE_T_{ABLATION_BUDGETS[-2]}": 4,
            },
            "ablation_initial_point_count": ABLATION_N_INITIAL,
            "same_numeric_rho_without_anchor_normalization": True,
            "balanced_schedules_exhaust_each_total_budget": True,
            "ablation_log_benchmark_is_run_separately": True,
        },
        "macro_count_per_configuration": FORMAL_MACROS,
        "smoke_validation": {
            "function": "rkhs_se_4d",
            "budgets": list(SMOKE_BUDGETS),
            "macros_per_budget": SMOKE_MACROS,
            "criteria": [
                "no exact-zero sequential LQEI acquisition values",
                "terminal recommendations are not systematically locked to the initial design",
            ],
        },
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
            "acquisition_optimizer": (
                "same 8192-point 4D Sobol screen plus 10-start bounded "
                "L-BFGS-B policy"
            ),
        },
        "seeds": {
            "design": DESIGN_SEED,
            "noise": NOISE_SEED,
            "candidate": CANDIDATE_SEED,
            "phase_code": FORMAL_PHASE_CODE,
        },
        "lqei_specifications": [asdict(spec) for spec in lqei_specs()],
        "smoke_specifications": [asdict(spec) for spec in smoke_specs()],
        "comparison_specifications": [asdict(spec) for spec in comparison_specs()],
        "ablation_specifications": [asdict(spec) for spec in ablation_specs()],
        "incumbent_ablation_specifications": [
            asdict(spec) for spec in incumbent_ablation_specs()
        ],
        "workers": workers,
        "numerical_threads_per_worker": 1,
        "numerical_implementation": {
            "same_site_compression": "exact precision-weighted Gaussian sufficient statistics",
            "adaptive_capacity": "storage and rebuild frequency only; posterior unchanged",
            "continuous_start_queries": "batch the same 10 starts without changing optimization settings",
            "shared_factor": "one LAPACK-ready factor copy per posterior round",
            "candidate_projections": "reuse exact cached triangular projections for candidate observations and optimizer starts",
            "ei_underflow": "log-EI fallback preserves the mathematical EI maximizer",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "sha256": hashes,
    }
    destination.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=(
            "smoke",
            "ablation_smoke",
            "confidence",
            "lqei",
            "comparisons",
            "formal",
            "incumbent_ablation",
            "final_recommendation_ablation",
            "ablation",
            "constant_ablation",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.phase != "analyze":
        write_protocol(args.workers)
    all_lqei = lqei_specs() + ablation_specs()
    if args.phase == "smoke":
        run_specs(smoke_specs(), workers=args.workers, overwrite=args.overwrite)
        from validate_smoke import main as validate_smoke

        validate_smoke()
    if args.phase == "ablation_smoke":
        run_specs(
            ablation_smoke_specs(),
            workers=args.workers,
            overwrite=args.overwrite,
            macro_limit=ABLATION_SMOKE_MACROS,
        )
        from validate_ablation_smoke import main as validate_ablation_smoke

        validate_ablation_smoke()
    if args.phase in ("confidence", "formal", "all"):
        confidence_specs = lqei_specs() if args.phase == "formal" else all_lqei
        paths = prepare_confidence_schedules(confidence_specs)
        print(f"Prepared {len(paths)} confidence schedules.", flush=True)
    if args.phase in ("lqei", "formal", "all"):
        run_specs(lqei_specs(), workers=args.workers, overwrite=args.overwrite)
    if args.phase in ("comparisons", "formal", "all"):
        run_comparison_specs(
            comparison_specs(), workers=args.workers, overwrite=args.overwrite
        )
    if args.phase in ("ablation", "all"):
        # Run O(1) configurations individually so the longest SE paths respect
        # the original memory caps.  Execution order does not change samples.
        nonconstant = [s for s in ablation_specs() if s.replication_order != "constant"]
        run_specs(nonconstant, workers=args.workers, overwrite=args.overwrite)
        for spec in constant_ablation_specs():
            workers = constant_workers_for_spec(spec, args.workers)
            run_specs([spec], workers=workers, overwrite=args.overwrite)
        if args.phase == "ablation":
            from analyze_replication_order_ablation import main as analyze_replication

            analyze_replication()
    if args.phase == "constant_ablation":
        for spec in constant_ablation_specs():
            workers = constant_workers_for_spec(spec, args.workers)
            print(
                f"Constant extension configuration: {spec.function_name}/"
                f"T={spec.total_budget}, workers={workers}",
                flush=True,
            )
            run_specs([spec], workers=workers, overwrite=args.overwrite)
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
    if args.phase in ("analyze", "all"):
        from analyze_results import main as analyze

        analyze()
    if args.phase == "formal":
        from analyze_completed_main import main as analyze_formal

        analyze_formal()
    elapsed = time.perf_counter() - started
    print(f"Finished {args.phase} in {elapsed:.1f} seconds.", flush=True)
    if args.phase in ("formal", "all"):
        include_ablation = args.phase == "all"
        payload = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "last_invocation_elapsed_seconds": elapsed,
            "lqei_macro_count": len(lqei_specs()) * FORMAL_MACROS,
            "comparison_macro_count": len(comparison_specs()) * FORMAL_MACROS,
            "ablation_macro_count": (
                len(ablation_specs()) * FORMAL_MACROS if include_ablation else 0
            ),
            "total_macro_count": (
                len(lqei_specs())
                + len(comparison_specs())
                + (len(ablation_specs()) if include_ablation else 0)
            )
            * FORMAL_MACROS,
            "configuration_count": (
                len(lqei_specs())
                + len(comparison_specs())
                + (len(ablation_specs()) if include_ablation else 0)
            ),
        }
        (ROOT / "run_completion.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
