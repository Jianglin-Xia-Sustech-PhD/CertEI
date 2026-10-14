from __future__ import annotations

import numpy as np

from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    _expected_improvement,
    incumbent_sd_coefficient,
)
from run_experiments import incumbent_ablation_specs, lqei_specs
from study_config import FORMAL_MACROS, FORMAL_PHASE_CODE, N_INITIAL


def test_incumbent_rules_replace_the_entire_signed_coefficient() -> None:
    assert incumbent_sd_coefficient(INCUMBENT_RULE_DYNAMIC, 10.0) == -2.0
    assert incumbent_sd_coefficient(INCUMBENT_RULE_MEAN, 10.0) == 0.0
    assert incumbent_sd_coefficient(INCUMBENT_RULE_UPPER_ONE_SD, 10.0) == 1.0


def test_expected_improvement_uses_the_registered_global_incumbent() -> None:
    means = np.asarray([0.3, -0.1, 0.2])
    variances = np.asarray([0.04, 0.09, 0.01])
    _, mean_incumbent = _expected_improvement(means, variances, 0.0)
    _, upper_incumbent = _expected_improvement(means, variances, 1.0)
    assert np.isclose(mean_incumbent, -0.1)
    assert np.isclose(upper_incumbent, 0.2)


def test_incumbent_ablation_reuses_every_non_incumbent_setting() -> None:
    main = {
        (spec.function_name, spec.total_budget): spec for spec in lqei_specs()
    }
    specs = incumbent_ablation_specs()
    assert len(specs) == 28
    assert {spec.incumbent_rule for spec in specs} == {
        INCUMBENT_RULE_MEAN,
        INCUMBENT_RULE_UPPER_ONE_SD,
    }
    for spec in specs:
        benchmark = main[(spec.function_name, spec.total_budget)]
        assert spec.phase_code == benchmark.phase_code == FORMAL_PHASE_CODE
        assert spec.initial_point_count == benchmark.initial_point_count == N_INITIAL
        assert spec.macro_count == benchmark.macro_count == FORMAL_MACROS
        assert spec.replication_order == benchmark.replication_order == "log"
        assert spec.replication_multiplier == benchmark.replication_multiplier
        np.testing.assert_array_equal(spec.schedule, benchmark.schedule)
        assert spec.slug != benchmark.slug
