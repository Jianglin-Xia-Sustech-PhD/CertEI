from __future__ import annotations

import numpy as np

from analyze_replication_order_ablation import (
    COLORS,
    ORDER_FORMULAS,
    ORDER_LABELS,
    ORDER_SHORT_LABELS,
    ORDERS,
    _summary,
)
from run_experiments import ablation_specs, constant_ablation_specs, constant_workers_for_spec


def test_four_order_plots_and_labels_include_constant() -> None:
    assert ORDERS == ("constant", "log_log", "log", "power_1p2")
    for mapping in (COLORS, ORDER_FORMULAS, ORDER_LABELS, ORDER_SHORT_LABELS):
        assert set(mapping) == set(ORDERS)
    assert ORDER_SHORT_LABELS["constant"] == "O(1)"
    assert ORDER_FORMULAS["constant"] == "rho"


def test_all_four_orders_use_exponential_budgets() -> None:
    specs = ablation_specs()
    assert len(specs) == 24
    assert len(constant_ablation_specs()) == 6
    old = [spec for spec in specs if spec.replication_order != "constant"]
    assert len(old) == 18
    for spec in specs:
        assert spec.macro_count == 100
        assert spec.initial_point_count == 20
        assert spec.candidate_count == 8192
        assert spec.confidence_delta == 0.05
        assert int(spec.schedule.sum()) == spec.total_budget
    expected_old_batches = {
        ("rkhs_matern52_4d", "log_log"): (20, 20, 20),
        ("rkhs_matern52_4d", "log"): (85, 90, 95),
        ("rkhs_matern52_4d", "power_1p2"): (1525, 1960, 2515),
        ("rkhs_se_4d", "log_log"): (15, 15, 15),
        ("rkhs_se_4d", "log"): (65, 70, 70),
        ("rkhs_se_4d", "power_1p2"): (1145, 1470, 1890),
    }
    budgets = (36316, 59874, 98716)
    for spec in old:
        assert spec.nominal_batch_size == expected_old_batches[
            (spec.function_name, spec.replication_order)
        ][budgets.index(spec.total_budget)]


def test_constant_summary_reports_actual_budget_and_average_gap() -> None:
    spec = constant_ablation_specs()[0]
    gaps = np.asarray([0.1, 0.2, 0.3])
    row = _summary(
        spec.function_name, spec.replication_order, spec.total_budget, gaps, spec
    )
    assert row["batch_formula"] == "rho"
    assert row["nominal_replication_number"] == 10
    assert row["minimum_actual_replication_number"] == 10
    assert row["maximum_actual_replication_number"] == 11
    assert row["number_of_adaptive_batches"] == 3611
    assert row["exact_budget_used"] == 36316
    assert np.isclose(row["average_gap"], 0.2)


def test_long_se_paths_use_bounded_execution_concurrency() -> None:
    for spec in constant_ablation_specs():
        workers = constant_workers_for_spec(spec, 8)
        expected = 8
        if spec.function_name == "rkhs_se_4d":
            expected = {36316: 8, 59874: 4, 98716: 2}[spec.total_budget]
        assert workers == expected
        assert constant_workers_for_spec(spec, 1) == 1
