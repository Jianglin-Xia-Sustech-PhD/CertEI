from __future__ import annotations

import numpy as np

from e8_budget_extension import (
    E8_BUDGET,
    E8_BUDGET_EXPONENT,
    e8_comparison_specs,
    e8_incumbent_specs,
    e8_lqei_specs,
)
from experiment_runner import (
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
from run_experiments import COMPARISON_ALGORITHMS, FUNCTION_NAMES, lqei_specs
from study_config import FORMAL_MACROS, N_INITIAL


def test_e8_budget_and_exact_schedules_are_registered() -> None:
    assert E8_BUDGET_EXPONENT == 8.0
    assert E8_BUDGET == 2981
    specs = e8_lqei_specs()
    assert len(specs) == len(FUNCTION_NAMES) == 2
    sequential_counts = {}
    for spec in specs:
        assert spec.total_budget == E8_BUDGET
        assert spec.macro_count == FORMAL_MACROS
        assert spec.initial_point_count == N_INITIAL
        assert int(np.sum(spec.schedule)) == E8_BUDGET
        assert spec.schedule.size > N_INITIAL
        sequential_counts[spec.function_name] = spec.schedule.size - N_INITIAL
    assert sequential_counts == {
        "rkhs_matern52_4d": 5,
        "rkhs_se_4d": 19,
    }


def test_e8_algorithm_comparison_matches_main_schedule() -> None:
    main = {spec.function_name: spec for spec in e8_lqei_specs()}
    comparisons = e8_comparison_specs()
    assert len(comparisons) == len(FUNCTION_NAMES) * len(COMPARISON_ALGORITHMS)
    assert {spec.algorithm for spec in comparisons} == set(COMPARISON_ALGORITHMS)
    for spec in comparisons:
        np.testing.assert_array_equal(spec.schedule, main[spec.function_name].schedule)
        assert spec.total_budget == E8_BUDGET
        assert spec.macro_count == FORMAL_MACROS


def test_e8_incumbent_ablation_changes_only_the_incumbent() -> None:
    main = {spec.function_name: spec for spec in e8_lqei_specs()}
    specs = e8_incumbent_specs()
    assert len(specs) == len(FUNCTION_NAMES) * 2
    assert {spec.incumbent_rule for spec in specs} == {
        INCUMBENT_RULE_MEAN,
        INCUMBENT_RULE_UPPER_ONE_SD,
    }
    for spec in specs:
        reference = main[spec.function_name]
        assert spec.phase_code == reference.phase_code
        assert spec.replication_multiplier == reference.replication_multiplier
        assert spec.initial_point_count == reference.initial_point_count
        assert spec.candidate_count == reference.candidate_count
        np.testing.assert_array_equal(spec.schedule, reference.schedule)


def test_extension_does_not_change_the_registered_original_grid() -> None:
    original = sorted({spec.total_budget for spec in lqei_specs()})
    assert original == [4915, 8103, 13360, 22026, 36316, 59874, 98716]
    assert E8_BUDGET not in original
