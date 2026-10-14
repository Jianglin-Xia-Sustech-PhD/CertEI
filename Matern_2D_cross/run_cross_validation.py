"""Run and resume the complete 2D Matérn LQEI factorial experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    ExperimentSpec,
    _macro_path,
    run_specs,
)
from study_config import (
    CONFIDENCE_DELTA,
    CROSS_VALIDATION_BUDGETS,
    CROSS_VALIDATION_PHASE_CODE,
    FORMAL_MACROS,
    FUNCTION_NAME,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    N_CANDIDATES,
    N_INITIAL,
    REPLICATION_ORDERS,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
RESULT_GROUP = "cross_validation_trajectories"
INCUMBENT_RULES = (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
FINAL_RECOMMENDATION_RULES = (
    "posterior_mean_plus_0p2_lambda_sd",
    "posterior_mean",
    "posterior_mean_minus_0p2_lambda_sd",
)


def cross_validation_specs() -> list[ExperimentSpec]:
    """Return the registered 3 incumbent x 4 order x 2 budget paths."""

    rho = rho_for_function(FUNCTION_NAME)
    return [
        ExperimentSpec(
            phase=RESULT_GROUP,
            phase_code=CROSS_VALIDATION_PHASE_CODE,
            function_name=FUNCTION_NAME,
            total_budget=budget,
            replication_order=order,
            replication_multiplier=rho,
            anchor_budget=CROSS_VALIDATION_BUDGETS[0],
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
            incumbent_rule=incumbent,
        )
        for budget in CROSS_VALIDATION_BUDGETS
        for incumbent in INCUMBENT_RULES
        for order in REPLICATION_ORDERS
    ]


def write_protocol() -> Path:
    specs = cross_validation_specs()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "2D Matern-5/2 LQEI incumbent/final-rule/replication-order cross-validation",
        "function": FUNCTION_NAME,
        "budgets": list(CROSS_VALIDATION_BUDGETS),
        "macro_count_per_trajectory_configuration": FORMAL_MACROS,
        "initial_point_count": N_INITIAL,
        "candidate_count": N_CANDIDATES,
        "phase_code": CROSS_VALIDATION_PHASE_CODE,
        "fresh_random_streams": True,
        "common_random_numbers_within_macro": True,
        "incumbent_rules": [
            "min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            "min_x(mu_n)",
            "min_x(mu_n + sigma_n)",
        ],
        "final_recommendation_rules": [
            "argmin sampled(mu_N + 0.2*Lambda_N*sigma_N)",
            "argmin sampled(mu_N)",
            "argmin sampled(mu_N - 0.2*Lambda_N*sigma_N)",
        ],
        "replication_orders": list(REPLICATION_ORDERS),
        "acquisition_lambda_multiplier": LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
        "final_recommendation_lambda_multiplier": (
            LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
        ),
        "trajectory_configuration_count": len(specs),
        "reported_combination_count": (
            len(specs) * len(FINAL_RECOMMENDATION_RULES)
        ),
        "simulated_trajectory_count": len(specs) * FORMAL_MACROS,
        "postprocessed_recommendation_count": (
            len(specs) * FORMAL_MACROS * len(FINAL_RECOMMENDATION_RULES)
        ),
        "target_combination": {
            "incumbent_rule": INCUMBENT_RULE_DYNAMIC,
            "final_recommendation_rule": FINAL_RECOMMENDATION_RULES[0],
            "replication_order": "log",
        },
        "trajectory_specifications": [asdict(spec) for spec in specs],
    }
    destination = ROOT / "protocol.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
        stream.write("\n")
    return destination


def status() -> dict[str, object]:
    specs = cross_validation_specs()
    complete = 0
    by_budget: dict[str, int] = {str(value): 0 for value in CROSS_VALIDATION_BUDGETS}
    by_configuration: list[dict[str, object]] = []
    for spec in specs:
        count = sum(
            _macro_path(spec, macro).exists()
            for macro in range(1, spec.macro_count + 1)
        )
        complete += count
        by_budget[str(spec.total_budget)] += count
        by_configuration.append(
            {
                "budget": spec.total_budget,
                "incumbent_rule": spec.incumbent_rule,
                "replication_order": spec.replication_order,
                "completed_macros": count,
                "expected_macros": spec.macro_count,
            }
        )
    return {
        "completed_trajectory_count": complete,
        "expected_trajectory_count": len(specs) * FORMAL_MACROS,
        "by_budget": by_budget,
        "configurations": by_configuration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("run", "analyze", "all", "status", "protocol"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_path = write_protocol()
    if args.phase == "protocol":
        print(protocol_path)
        return
    if args.phase == "status":
        print(json.dumps(status(), indent=2))
        return
    if args.phase in {"run", "all"}:
        run_specs(
            cross_validation_specs(),
            workers=args.workers,
            overwrite=args.overwrite,
        )
    if args.phase in {"analyze", "all"}:
        from analyze_cross_validation import main as analyze

        analyze()


if __name__ == "__main__":
    main()
