from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_cross_validation import (  # noqa: E402
    TARGET_FINAL,
    TARGET_INCUMBENT,
    TARGET_ORDER,
    _final_coefficient,
)
from experiment_runner import INCUMBENT_RULE_DYNAMIC  # noqa: E402
from run_cross_validation import (  # noqa: E402
    FINAL_RECOMMENDATION_RULES,
    cross_validation_specs,
)
from study_config import (  # noqa: E402
    CROSS_VALIDATION_BUDGETS,
    FORMAL_MACROS,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    N_INITIAL,
    REPLICATION_ORDERS,
)


def test_factorial_design_has_24_unique_trajectory_configurations() -> None:
    specs = cross_validation_specs()
    assert len(specs) == 24
    assert len({spec.slug for spec in specs}) == 24
    assert {spec.total_budget for spec in specs} == set(CROSS_VALIDATION_BUDGETS)
    assert {spec.replication_order for spec in specs} == set(REPLICATION_ORDERS)
    assert all(spec.macro_count == FORMAL_MACROS for spec in specs)
    assert len(specs) * len(FINAL_RECOMMENDATION_RULES) == 72


def test_every_replication_order_is_feasible_and_uses_exact_budget() -> None:
    for spec in cross_validation_specs():
        assert spec.schedule.size > N_INITIAL
        assert int(np.sum(spec.schedule)) == spec.total_budget
        assert int(np.max(spec.schedule) - np.min(spec.schedule)) <= 1


def test_symmetric_final_rule_coefficients() -> None:
    terminal_lambda = 17.5
    multiplier = LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
    assert np.isclose(
        _final_coefficient("posterior_mean_plus_0p2_lambda_sd", terminal_lambda),
        multiplier * terminal_lambda,
    )
    assert _final_coefficient("posterior_mean", terminal_lambda) == 0.0
    assert np.isclose(
        _final_coefficient("posterior_mean_minus_0p2_lambda_sd", terminal_lambda),
        -multiplier * terminal_lambda,
    )


def test_preregistered_target_is_present_once_per_budget() -> None:
    assert TARGET_INCUMBENT == INCUMBENT_RULE_DYNAMIC
    assert TARGET_ORDER == "log"
    assert TARGET_FINAL == "posterior_mean_plus_0p2_lambda_sd"
    for budget in CROSS_VALIDATION_BUDGETS:
        matches = [
            spec
            for spec in cross_validation_specs()
            if spec.total_budget == budget
            and spec.incumbent_rule == TARGET_INCUMBENT
            and spec.replication_order == TARGET_ORDER
        ]
        assert len(matches) == 1
