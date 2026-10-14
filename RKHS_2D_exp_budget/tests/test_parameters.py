import numpy as np
from scipy.linalg import solve_triangular

from confidence import (
    build_confidence_schedule,
    finite_batch_information_gain,
    kernel_order_information_gain,
)
from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    _expected_improvement,
    _log_expected_improvement,
    _recommendation,
    incumbent_sd_coefficient,
)
from run_experiments import ablation_specs, incumbent_ablation_specs, lqei_specs
from rkhs_functions import kernel_matrix
from sequential_gp import SequentialFixedKernelGP
from analyze_final_recommendation_ablation import (
    RULE_CURRENT,
    RULE_MEAN,
    RULE_MINUS_TEN_SD,
    _coefficient as final_recommendation_coefficient,
)
from study_config import (
    ABLATION_BUDGETS,
    BUDGET_EXPONENTS,
    EXPONENTIAL_BUDGETS,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    TARGET_BUDGETS_BY_FUNCTION,
    budget_exponent,
    make_batch_schedule,
    replication_size,
)


def test_requested_multipliers_are_registered():
    assert LQEI_ACQUISITION_LAMBDA_MULTIPLIER == 0.2
    assert LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER == 0.2


def test_incumbent_ablation_replaces_the_entire_signed_coefficient():
    assert incumbent_sd_coefficient(INCUMBENT_RULE_DYNAMIC, 10.0) == -2.0
    assert incumbent_sd_coefficient(INCUMBENT_RULE_MEAN, 10.0) == 0.0
    assert incumbent_sd_coefficient(INCUMBENT_RULE_UPPER_ONE_SD, 10.0) == 1.0
    means = np.array([0.4, 0.2])
    variances = np.array([0.25, 0.01])
    assert np.isclose(_expected_improvement(means, variances, -2.0)[1], -0.6)
    assert np.isclose(_expected_improvement(means, variances, 0.0)[1], 0.2)
    assert np.isclose(_expected_improvement(means, variances, 1.0)[1], 0.3)


def test_log_ei_matches_regular_formula_and_survives_extreme_tail():
    means = np.array([-0.5, 0.0, 0.4, 2.0])
    variances = np.array([0.09, 0.16, 0.25, 0.36])
    incumbent = float(np.min(means))
    direct, _ = _expected_improvement(means, variances, 0.0)
    log_values = _log_expected_improvement(means, variances, incumbent)
    # _expected_improvement with coefficient zero forms this same incumbent.
    np.testing.assert_allclose(np.exp(log_values), direct, rtol=2e-10, atol=1e-300)

    extreme_means = np.array([100.0, 101.0])
    extreme_variances = np.array([0.01, 0.01])
    extreme = _log_expected_improvement(extreme_means, extreme_variances, -3.0)
    assert np.all(np.isfinite(extreme))
    assert extreme[0] > extreme[1]


def test_incumbent_ablation_uses_main_schedules_and_distinct_paths():
    main = {
        (spec.function_name, spec.total_budget): spec for spec in lqei_specs()
    }
    specs = incumbent_ablation_specs()
    assert len(specs) == 36
    assert {spec.incumbent_rule for spec in specs} == {
        INCUMBENT_RULE_MEAN,
        INCUMBENT_RULE_UPPER_ONE_SD,
    }
    assert len({spec.slug for spec in specs}) == len(specs)
    for spec in specs:
        np.testing.assert_array_equal(
            spec.schedule,
            main[(spec.function_name, spec.total_budget)].schedule,
        )


def test_recommendation_uses_supplied_scaled_terminal_coefficient():
    class Posterior:
        training_indices = np.array([0, 1, 1])
        posterior_mean = np.array([0.0, 0.1])
        posterior_variance = np.array([0.04, 0.0])

    assert _recommendation(Posterior(), coefficient=0.2)[0] == 0
    assert _recommendation(Posterior(), coefficient=2.0)[0] == 1


def test_final_recommendation_ablation_signed_coefficients():
    assert final_recommendation_coefficient(RULE_CURRENT, 20.0) == 4.0
    assert final_recommendation_coefficient(RULE_MEAN, 20.0) == 0.0
    assert final_recommendation_coefficient(RULE_MINUS_TEN_SD, 20.0) == -10.0


def test_projection_column_update_matches_incremental_cholesky_reference():
    rng = np.random.default_rng(314159)
    universe = rng.uniform(size=(24, 2))
    indices = rng.integers(0, universe.shape[0], size=35)
    observations = rng.normal(size=indices.size)
    noise_variances = rng.uniform(0.01, 0.2, size=indices.size)
    process_variance = 1.3
    constant_mean = -0.2
    model = SequentialFixedKernelGP(
        universe=universe,
        kernel="matern52",
        lengthscale=0.2,
        process_variance=process_variance,
        constant_mean=constant_mean,
        maximum_observations=indices.size,
    )
    cholesky = np.zeros((indices.size, indices.size))
    projections = np.zeros((indices.size, universe.shape[0]))
    reference_mean = np.full(universe.shape[0], constant_mean)
    reference_variance = np.full(universe.shape[0], process_variance)

    for n, (index, observation, noise) in enumerate(
        zip(indices, observations, noise_variances)
    ):
        selected = universe[index : index + 1]
        prior_cross = process_variance * kernel_matrix(
            universe, selected, "matern52", 0.2
        ).reshape(-1)
        if n == 0:
            triangular = np.empty(0)
            conditional_cross = prior_cross
            schur = process_variance + noise
        else:
            training_cross = process_variance * kernel_matrix(
                universe[indices[:n]], selected, "matern52", 0.2
            ).reshape(-1)
            triangular = solve_triangular(
                cholesky[:n, :n], training_cross, lower=True
            )
            conditional_cross = prior_cross - projections[:n].T @ triangular
            schur = process_variance + noise - float(triangular @ triangular)
        residual = observation - reference_mean[index]
        reference_mean += conditional_cross * residual / schur
        reference_variance = np.maximum(
            reference_variance - conditional_cross**2 / schur, 0.0
        )
        diagonal = np.sqrt(schur)
        if n:
            cholesky[n, :n] = triangular
        cholesky[n, n] = diagonal
        projections[n] = conditional_cross / diagonal

        model.add_observation(int(index), float(observation), float(noise))
        np.testing.assert_allclose(model.posterior_mean, reference_mean, atol=1e-11)
        np.testing.assert_allclose(
            model.posterior_variance, reference_variance, atol=1e-11
        )


def test_sufficient_statistic_rebuild_matches_full_duplicate_observation_gp():
    rng = np.random.default_rng(271828)
    universe = rng.uniform(size=(7, 2))
    indices = rng.integers(0, universe.shape[0], size=55)
    observations = rng.normal(size=indices.size)
    noise_variances = rng.uniform(0.03, 0.3, size=indices.size)
    process_variance = 0.9
    constant_mean = 0.15
    model = SequentialFixedKernelGP(
        universe=universe,
        kernel="squared_exponential",
        lengthscale=0.25,
        process_variance=process_variance,
        constant_mean=constant_mean,
        maximum_observations=indices.size,
    )
    # Force several rebuilds in this small exact-equivalence test.
    model._projection_capacity = 11

    for stop, (index, observation, noise) in enumerate(
        zip(indices, observations, noise_variances), start=1
    ):
        model.add_observation(int(index), float(observation), float(noise))
        active_indices = indices[:stop]
        training_points = universe[active_indices]
        covariance = process_variance * kernel_matrix(
            training_points,
            training_points,
            "squared_exponential",
            0.25,
        )
        covariance.flat[:: stop + 1] += noise_variances[:stop]
        cross = process_variance * kernel_matrix(
            training_points,
            universe,
            "squared_exponential",
            0.25,
        )
        centered = observations[:stop] - constant_mean
        solved_response = np.linalg.solve(covariance, centered)
        solved_cross = np.linalg.solve(covariance, cross)
        expected_mean = constant_mean + cross.T @ solved_response
        expected_variance = np.maximum(
            process_variance - np.sum(cross * solved_cross, axis=0), 0.0
        )
        np.testing.assert_allclose(model.posterior_mean, expected_mean, atol=2e-10)
        np.testing.assert_allclose(
            model.posterior_variance, expected_variance, atol=2e-10
        )
    assert model.count == indices.size
    np.testing.assert_array_equal(model.training_indices, indices)


def test_requested_budgets_are_registered():
    assert BUDGET_EXPONENTS == (
        7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5
    )
    assert EXPONENTIAL_BUDGETS == (
        1808, 2981, 4915, 8103, 13360, 22026, 36316, 59874, 98716
    )
    assert TARGET_BUDGETS_BY_FUNCTION["rkhs_matern52_2d"] == EXPONENTIAL_BUDGETS
    assert TARGET_BUDGETS_BY_FUNCTION["rkhs_se_2d"] == EXPONENTIAL_BUDGETS
    for exponent, budget in zip(BUDGET_EXPONENTS, EXPONENTIAL_BUDGETS):
        assert budget_exponent(budget) == exponent
    assert ABLATION_BUDGETS == (36316, 59874, 98716)


def test_unit_constant_kernel_order_information_gain():
    u = np.array([0.0, 2.0, 10.0])
    logarithm = np.log1p(u)
    np.testing.assert_allclose(
        kernel_order_information_gain("squared_exponential", u),
        logarithm**3,
    )
    np.testing.assert_allclose(
        kernel_order_information_gain("matern52", u),
        u ** (2.0 / 7.0) * logarithm ** (5.0 / 7.0),
    )


def test_ablation_uses_same_rho_without_anchor_normalization():
    assert replication_size(
        "constant", 36316, rho=8.0, anchor_budget=36316
    ) == 10
    assert replication_size(
        "log_log", 36316, rho=8.0, anchor_budget=36316
    ) == 20
    assert replication_size(
        "power_1p2", 36316, rho=8.0, anchor_budget=36316
    ) == 1525
    assert replication_size(
        "constant", 36316, rho=6.0, anchor_budget=36316
    ) == 10
    assert replication_size(
        "log_log", 36316, rho=6.0, anchor_budget=36316
    ) == 15
    assert replication_size(
        "power_1p2", 36316, rho=6.0, anchor_budget=36316
    ) == 1145


def test_replication_order_ablation_has_only_requested_additional_arms():
    specs = ablation_specs()
    assert len(specs) == 18
    assert {spec.replication_order for spec in specs} == {
        "constant",
        "log_log",
        "power_1p2",
    }
    assert {spec.total_budget for spec in specs} == set(ABLATION_BUDGETS)
    assert len({spec.slug for spec in specs}) == len(specs)
    for spec in specs:
        assert int(np.sum(spec.schedule)) == spec.total_budget


def test_every_requested_schedule_exhausts_its_budget():
    for rho in (6.0, 8.0):
        for order in (
            "constant",
            "log",
            "log_log",
            "power_1p2",
        ):
            for budget in ABLATION_BUDGETS:
                size = replication_size(
                    order, budget, rho=rho, anchor_budget=ABLATION_BUDGETS[0]
                )
                schedule = make_batch_schedule(budget, size)
                assert int(np.sum(schedule)) == budget
                assert schedule.size > 20


def test_lambda_uses_equation_20_minimum_information_gain_bound():
    schedule = build_confidence_schedule(
        kernel="matern52",
        information_gain_batch_size=80,
        correction_batch_size=80,
        batch_count=25,
        variance_min=20.0,
        rkhs_norm=7.0,
        delta=0.05,
    )
    indices = np.arange(26)
    expected_u = np.arange(26) * 80.0 / 20.0
    expected_kernel_order = (
        expected_u ** (2.0 / 7.0) * np.log1p(expected_u) ** (5.0 / 7.0)
    )
    expected_finite_batch = 0.5 * indices * np.log1p(80.0 / 20.0)
    expected_minimum = np.minimum(expected_kernel_order, expected_finite_batch)
    np.testing.assert_allclose(schedule.effective_information_budget, expected_u)
    np.testing.assert_allclose(
        schedule.kernel_order_information_gain_bound,
        expected_kernel_order,
    )
    np.testing.assert_allclose(
        schedule.finite_batch_information_gain_bound,
        expected_finite_batch,
    )
    np.testing.assert_allclose(schedule.information_gain_upper_bound, expected_minimum)
    assert np.any(expected_finite_batch[1:] < expected_kernel_order[1:])
    assert np.any(expected_kernel_order[1:] < expected_finite_batch[1:])


def test_finite_batch_bound_matches_equation_20_term():
    indices = np.array([0, 1, 7, 25])
    expected = 0.5 * indices * np.log1p(80.0 / 20.0)
    np.testing.assert_allclose(
        finite_batch_information_gain(
            indices,
            batch_size=80,
            variance_min=20.0,
        ),
        expected,
    )
