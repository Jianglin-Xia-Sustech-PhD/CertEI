from __future__ import annotations

from study_config import (
    ABLATION_BUDGETS,
    ABLATION_N_INITIAL,
    CONTINUOUS_POLISH_ENABLED,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_STARTS,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    N_CANDIDATES,
    N_INITIAL,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
    REPLICATION_ORDERS,
    RESPONSE_CONSTANT_MEAN,
    RESPONSE_LENGTHSCALE,
    RESPONSE_PROCESS_VARIANCE,
    budgets_for_function,
    make_batch_schedule,
    replication_size,
    rho_for_function,
)


def test_kernel_specific_rhos_are_registered() -> None:
    assert rho_for_function("rkhs_matern52_4d") == 8.0
    assert rho_for_function("rkhs_se_4d") == 6.0


def test_response_gp_and_continuous_optimizer_are_unchanged() -> None:
    assert RESPONSE_CONSTANT_MEAN == 0.0
    assert RESPONSE_PROCESS_VARIANCE == 1.0
    assert RESPONSE_LENGTHSCALE == 0.2
    assert CONTINUOUS_POLISH_ENABLED
    assert CONTINUOUS_POLISH_STARTS == 10
    assert CONTINUOUS_POLISH_MAXITER == 10


def test_requested_lqei_multipliers_are_registered() -> None:
    assert LQEI_ACQUISITION_LAMBDA_MULTIPLIER == 0.2
    assert LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER == 0.2


def test_noise_envelope_is_unchanged() -> None:
    assert NOISE_SD_MIN_MULTIPLIER == 1.5 / 2.0
    assert NOISE_SD_MAX_MULTIPLIER == 6.0 / 2.0


def test_requested_main_budgets_and_log_schedules() -> None:
    assert N_INITIAL == 40
    assert ABLATION_N_INITIAL == 20
    assert N_CANDIDATES == 8192
    expected = (4915, 8103, 13360, 22026, 36316, 59874, 98716)
    assert budgets_for_function("rkhs_matern52_4d") == expected
    assert budgets_for_function("rkhs_se_4d") == expected
    for function_name in ("rkhs_matern52_4d", "rkhs_se_4d"):
        rho = rho_for_function(function_name)
        for budget in expected:
            batch = replication_size("log", budget, rho=rho, anchor_budget=expected[0])
            schedule = make_batch_schedule(budget, batch, N_INITIAL)
            assert int(schedule.sum()) == budget
            assert schedule.size > N_INITIAL


def test_ablation_uses_direct_same_rho_orders_and_exact_budgets() -> None:
    assert ABLATION_BUDGETS == (36316, 59874, 98716)
    assert REPLICATION_ORDERS == ("constant", "log_log", "log", "power_1p2")
    expected_at_first_ablation_budget = {
        (8.0, "constant"): 10,
        (6.0, "constant"): 10,
        (8.0, "log_log"): 20,
        (8.0, "log"): 85,
        (8.0, "power_1p2"): 1525,
        (6.0, "log_log"): 15,
        (6.0, "log"): 65,
        (6.0, "power_1p2"): 1145,
    }
    for (rho, order), expected in expected_at_first_ablation_budget.items():
        assert (
            replication_size(order, ABLATION_BUDGETS[0], rho=rho, anchor_budget=ABLATION_BUDGETS[0])
            == expected
        )
    for rho in (8.0, 6.0):
        for order in REPLICATION_ORDERS:
            for budget in ABLATION_BUDGETS:
                batch = replication_size(
                    order,
                    budget,
                    rho=rho,
                    anchor_budget=ABLATION_BUDGETS[0],
                )
                schedule = make_batch_schedule(
                    budget,
                    batch,
                    ABLATION_N_INITIAL,
                )
                assert int(schedule.sum()) == budget
                assert schedule.size > ABLATION_N_INITIAL


def test_constant_replication_is_budget_independent() -> None:
    from run_experiments import constant_ablation_specs

    specs = constant_ablation_specs()
    assert len(specs) == 6
    for spec in specs:
        assert spec.replication_order == "constant"
        assert spec.nominal_batch_size == 10
        assert set(spec.schedule).issubset({10, 11})
        assert int(spec.schedule.sum()) == spec.total_budget
        assert spec.schedule.size == spec.total_budget // 10
        assert spec.initial_point_count == ABLATION_N_INITIAL
