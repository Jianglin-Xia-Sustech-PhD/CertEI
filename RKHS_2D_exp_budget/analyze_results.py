"""Validate and summarize the equation-(20) minimum-bound LQEI experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from comparison_runner import _macro_path as comparison_macro_path
from confidence import (
    INFORMATION_GAIN_CONSTANT,
    INFORMATION_GAIN_METHOD,
    finite_batch_information_gain,
    kernel_order_information_gain,
)
from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    RESULT_DIR,
    _macro_path,
    incumbent_sd_coefficient,
    load_confidence_schedule,
)
from rkhs_functions import load_fixed_function
from run_experiments import (
    ABLATION_NONBENCHMARK_ORDERS,
    ABLATION_RESULT_GROUP,
    COMPARISON_RESULT_GROUP,
    FUNCTION_NAMES,
    ablation_specs,
    comparison_specs,
    lqei_specs,
)
from study_config import (
    ABLATION_BUDGETS,
    FORMAL_MACROS,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    N_INITIAL,
    budget_exponent,
    budgets_for_function,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"
ALGORITHMS = ("LQEI", "SKO", "EQI", "TSSO")
ABLATION_ORDERS = ("log",) + ABLATION_NONBENCHMARK_ORDERS
ORDER_LABELS = {
    "log": "O(log T)",
    "log_log": "O(log log T)",
    "power_1p3": "O(T^(1/3))",
    "power_2p5": "O(T^(2/5))",
    "power_1p2": "O(T^(1/2)) [stress test]",
}
ORDER_FORMULAS = {
    "log": "rho*log(T)",
    "log_log": "rho*log(log(T))",
    "power_1p3": "rho*T^(1/3)",
    "power_2p5": "rho*T^(2/5)",
    "power_1p2": "rho*T^(1/2)",
}
FUNCTION_LABELS = {
    "rkhs_matern52_2d": "Matérn-5/2 (2D)",
    "rkhs_se_2d": "Squared exponential (2D)",
}
COLORS = {
    "LQEI": "#0072B2",
    "SKO": "#CC79A7",
    "EQI": "#D55E00",
    "TSSO": "#009E73",
}
PRIMARY_ABLATION_ORDERS = ("log", "log_log", "power_1p2")
PRIMARY_ABLATION_LABELS = {
    "log": "O(log T)",
    "log_log": "O(log log T)",
    "power_1p2": "O(T^(1/2))",
}
ABLATION_COLORS = {
    "log": "#0072B2",
    "log_log": "#E69F00",
    "power_1p2": "#D55E00",
}
THEORY_SLOPES = {"rkhs_matern52_2d": -3.0 / 14.0, "rkhs_se_2d": -0.5}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    function_name: str,
    algorithm: str,
    budget: int,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "function": function_name,
        "algorithm": algorithm,
        "total_budget": budget,
        "macro_count": values.size,
        "average_gap": float(np.mean(values)),
        "std_gap": float(np.std(values, ddof=1)),
        "se_average_gap": float(np.std(values, ddof=1) / np.sqrt(values.size)),
        "median_gap": float(np.median(values)),
        "q25_gap": float(np.quantile(values, 0.25)),
        "q75_gap": float(np.quantile(values, 0.75)),
        "minimum_gap": float(np.min(values)),
        "maximum_gap": float(np.max(values)),
    }


def _validate_lqei_spec(spec, provenance: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    function = load_fixed_function(ROOT / "fixed_functions" / f"{spec.function_name}.npz")
    confidence = load_confidence_schedule(spec)
    expected_kernel_order_bound = INFORMATION_GAIN_CONSTANT * kernel_order_information_gain(
        function.kernel,
        confidence.effective_information_budget,
    )
    expected_finite_batch_bound = finite_batch_information_gain(
        np.arange(confidence.batch_count + 1),
        batch_size=confidence.information_gain_batch_size,
        variance_min=confidence.variance_min,
    )
    expected_gamma = np.minimum(
        expected_kernel_order_bound,
        expected_finite_batch_bound,
    )
    np.testing.assert_allclose(
        confidence.kernel_order_information_gain_bound,
        expected_kernel_order_bound,
    )
    np.testing.assert_allclose(
        confidence.finite_batch_information_gain_bound,
        expected_finite_batch_bound,
    )
    np.testing.assert_allclose(confidence.information_gain_upper_bound, expected_gamma)
    expected_lambda = function.rkhs_norm + np.sqrt(
        2.0
        * confidence.kappa_t
        * (
            confidence.information_gain_upper_bound
            + np.log(3.0 / (2.0 * spec.confidence_delta))
        )
    )
    np.testing.assert_allclose(confidence.lambda_values, expected_lambda)
    if int(np.sum(spec.schedule)) != spec.total_budget:
        raise AssertionError("LQEI schedule does not exhaust the exact budget")

    gaps = np.empty(spec.macro_count)
    rows: list[dict[str, object]] = []
    for macro in range(1, spec.macro_count + 1):
        path = _macro_path(spec, macro)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if str(data["confidence_method"]) != INFORMATION_GAIN_METHOD:
                raise AssertionError("wrong information-gain method")
            if str(data["replication_order"]) != spec.replication_order:
                raise AssertionError("wrong replication order")
            if int(data["total_budget"]) != spec.total_budget:
                raise AssertionError("wrong total budget")
            np.testing.assert_array_equal(data["replications"], spec.schedule)
            if np.asarray(data["raw_observations"]).size != spec.total_budget:
                raise AssertionError("raw observation count does not match budget")
            np.testing.assert_allclose(data["lambda_values"], expected_lambda)
            np.testing.assert_allclose(
                data["kernel_order_information_gain_bound"],
                expected_kernel_order_bound,
            )
            np.testing.assert_allclose(
                data["finite_batch_information_gain_bound"],
                expected_finite_batch_bound,
            )
            np.testing.assert_allclose(
                data["information_gain_upper_bound"], expected_gamma
            )
            expected_signed_incumbent_coefficients = np.asarray(
                [
                    incumbent_sd_coefficient(spec.incumbent_rule, value)
                    for value in expected_lambda[:-1]
                ]
            )
            np.testing.assert_allclose(
                data["acquisition_coefficient_history"],
                -expected_signed_incumbent_coefficients,
            )
            if "log_maximum_lqei_history" in data:
                maximum_ei = np.asarray(data["maximum_lqei_history"], dtype=float)
                log_maximum_ei = np.asarray(
                    data["log_maximum_lqei_history"], dtype=float
                )
                if maximum_ei.shape != (confidence.batch_count,):
                    raise AssertionError("maximum EI history has the wrong shape")
                if log_maximum_ei.shape != maximum_ei.shape:
                    raise AssertionError("log EI history has the wrong shape")
                if not np.all(np.isfinite(log_maximum_ei[N_INITIAL:])):
                    raise AssertionError("sequential log EI history is not finite")
                positive_ei = maximum_ei[N_INITIAL:] > 0.0
                np.testing.assert_allclose(
                    log_maximum_ei[N_INITIAL:][positive_ei],
                    np.log(maximum_ei[N_INITIAL:][positive_ei]),
                )
            if "incumbent_rule" in data:
                if str(data["incumbent_rule"]) != spec.incumbent_rule:
                    raise AssertionError("wrong incumbent rule")
                np.testing.assert_allclose(
                    data["incumbent_sd_coefficient_history"],
                    expected_signed_incumbent_coefficients,
                )
            elif spec.incumbent_rule != INCUMBENT_RULE_DYNAMIC:
                raise AssertionError("incumbent rule metadata is missing")
            coefficient = float(
                LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER * expected_lambda[-1]
            )
            if not np.isclose(float(data["recommendation_coefficient"]), coefficient):
                raise AssertionError("wrong terminal recommendation coefficient")
            sampled = np.asarray(data["sampled_unique_indices"], dtype=int)
            means = np.asarray(data["sampled_posterior_mean"], dtype=float)
            sds = np.asarray(data["sampled_posterior_sd"], dtype=float)
            expected_index = int(sampled[int(np.argmin(means + coefficient * sds))])
            if int(data["recommendation_index"]) != expected_index:
                raise AssertionError("terminal recommendation is not the stored argmin")
            gap = float(data["raw_gap"])
            if not np.isfinite(gap) or gap < 0.0:
                raise AssertionError("invalid LQEI GAP")
            gaps[macro - 1] = gap
            rows.append(
                {
                    "provenance": provenance,
                    "function": spec.function_name,
                    "algorithm": "LQEI",
                    "replication_order": spec.replication_order,
                    "total_budget": spec.total_budget,
                    "macro": macro,
                    "raw_gap": gap,
                    "recommendation_x1": float(data["recommendation_x"][0]),
                    "recommendation_x2": float(data["recommendation_x"][1]),
                }
            )
    return gaps, rows


def collect_and_validate(*, include_ablation: bool = False):
    budgets_by_function = {
        name: budgets_for_function(name) for name in FUNCTION_NAMES
    }
    algorithm_matrices = {
        (name, algorithm): np.empty(
            (FORMAL_MACROS, len(budgets_by_function[name]))
        )
        for name in FUNCTION_NAMES
        for algorithm in ALGORITHMS
    }
    ablation_matrices = {
        (name, order): np.empty((FORMAL_MACROS, len(ABLATION_BUDGETS)))
        for name in FUNCTION_NAMES
        for order in ABLATION_ORDERS
    }
    macro_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []

    main_by_key = {(spec.function_name, spec.total_budget): spec for spec in lqei_specs()}
    for spec in lqei_specs():
        gaps, rows = _validate_lqei_spec(spec, "main")
        macro_rows.extend(rows)
        budget_index = budgets_by_function[spec.function_name].index(
            spec.total_budget
        )
        algorithm_matrices[(spec.function_name, "LQEI")][:, budget_index] = gaps
        if spec.total_budget in ABLATION_BUDGETS:
            ablation_index = ABLATION_BUDGETS.index(spec.total_budget)
            ablation_matrices[(spec.function_name, "log")][:, ablation_index] = gaps
        confidence = load_confidence_schedule(spec)
        lambda_rows.append(
            {
                "study": "main",
                "function": spec.function_name,
                "replication_order": spec.replication_order,
                "total_budget": spec.total_budget,
                "rho": spec.replication_multiplier,
                "nominal_batch_size": spec.nominal_batch_size,
                "minimum_actual_batch_size": int(np.min(spec.schedule)),
                "maximum_actual_batch_size": int(np.max(spec.schedule)),
                "batch_count": spec.schedule.size,
                "variance_min": confidence.variance_min,
                "effective_U_terminal": float(confidence.effective_information_budget[-1]),
                "Gk_terminal": float(
                    confidence.kernel_order_information_gain_bound[-1]
                ),
                "finite_batch_terminal": float(
                    confidence.finite_batch_information_gain_bound[-1]
                ),
                "information_gain_upper_bound_terminal": float(
                    confidence.information_gain_upper_bound[-1]
                ),
                "active_bound_terminal": (
                    "kernel_order"
                    if confidence.kernel_order_information_gain_bound[-1]
                    <= confidence.finite_batch_information_gain_bound[-1]
                    else "finite_batch"
                ),
                "kappa_T": confidence.kappa_t,
                "lambda_0": float(confidence.lambda_values[0]),
                "lambda_terminal": float(confidence.lambda_values[-1]),
                "terminal_0p2_lambda": float(0.2 * confidence.lambda_values[-1]),
            }
        )

    for spec in (ablation_specs() if include_ablation else ()):
        gaps, rows = _validate_lqei_spec(spec, "ablation")
        macro_rows.extend(rows)
        budget_index = ABLATION_BUDGETS.index(spec.total_budget)
        ablation_matrices[(spec.function_name, spec.replication_order)][
            :, budget_index
        ] = gaps
        confidence = load_confidence_schedule(spec)
        lambda_rows.append(
            {
                "study": "ablation",
                "function": spec.function_name,
                "replication_order": spec.replication_order,
                "total_budget": spec.total_budget,
                "rho": spec.replication_multiplier,
                "nominal_batch_size": spec.nominal_batch_size,
                "minimum_actual_batch_size": int(np.min(spec.schedule)),
                "maximum_actual_batch_size": int(np.max(spec.schedule)),
                "batch_count": spec.schedule.size,
                "variance_min": confidence.variance_min,
                "effective_U_terminal": float(confidence.effective_information_budget[-1]),
                "Gk_terminal": float(
                    confidence.kernel_order_information_gain_bound[-1]
                ),
                "finite_batch_terminal": float(
                    confidence.finite_batch_information_gain_bound[-1]
                ),
                "information_gain_upper_bound_terminal": float(
                    confidence.information_gain_upper_bound[-1]
                ),
                "active_bound_terminal": (
                    "kernel_order"
                    if confidence.kernel_order_information_gain_bound[-1]
                    <= confidence.finite_batch_information_gain_bound[-1]
                    else "finite_batch"
                ),
                "kappa_T": confidence.kappa_t,
                "lambda_0": float(confidence.lambda_values[0]),
                "lambda_terminal": float(confidence.lambda_values[-1]),
                "terminal_0p2_lambda": float(0.2 * confidence.lambda_values[-1]),
            }
        )

    for spec in comparison_specs():
        budget_index = budgets_by_function[spec.function_name].index(
            spec.total_budget
        )
        gaps = np.empty(spec.macro_count)
        for macro in range(1, spec.macro_count + 1):
            path = comparison_macro_path(spec, macro)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                if str(data["algorithm"]) != spec.algorithm:
                    raise AssertionError("comparison algorithm metadata mismatch")
                if int(data["total_budget"]) != spec.total_budget:
                    raise AssertionError("comparison budget metadata mismatch")
                replications = np.asarray(data["replications"], dtype=int)
                if int(np.sum(replications)) != spec.total_budget:
                    raise AssertionError("comparison did not exhaust exact budget")
                if np.asarray(data["raw_observations"]).size != spec.total_budget:
                    raise AssertionError("comparison raw observation count mismatch")
                if not bool(data["response_hyperparameters_fixed"]):
                    raise AssertionError("comparison response GP was not fixed")
                if spec.algorithm == "SKO":
                    if not bool(data["auxiliary_noise_gp_initial_fit_only"]):
                        raise AssertionError("SKO noise GP was not fixed after initialization")
                    if not np.isclose(
                        float(data["sko_incumbent_sd_multiplier"]), 1.0
                    ):
                        raise AssertionError("SKO incumbent coefficient mismatch")
                    if not bool(
                        data["sko_noise_penalty_uses_single_replication_variance"]
                    ):
                        raise AssertionError("SKO noise penalty metadata mismatch")
                gap = float(data["raw_gap"])
                if not np.isfinite(gap) or gap < 0.0:
                    raise AssertionError("invalid comparison GAP")
                gaps[macro - 1] = gap
                macro_rows.append(
                    {
                        "provenance": "comparison",
                        "function": spec.function_name,
                        "algorithm": spec.algorithm,
                        "replication_order": "log",
                        "total_budget": spec.total_budget,
                        "macro": macro,
                        "raw_gap": gap,
                        "recommendation_x1": float(data["recommendation_x"][0]),
                        "recommendation_x2": float(data["recommendation_x"][1]),
                    }
                )
        algorithm_matrices[(spec.function_name, spec.algorithm)][:, budget_index] = gaps

    algorithm_summary = []
    for name in FUNCTION_NAMES:
        for algorithm in ALGORITHMS:
            for index, budget in enumerate(budgets_by_function[name]):
                algorithm_summary.append(
                    _summary(
                        name,
                        algorithm,
                        budget,
                        algorithm_matrices[(name, algorithm)][:, index],
                    )
                )

    ablation_summary = []
    for name in (FUNCTION_NAMES if include_ablation else ()):
        for order in ABLATION_ORDERS:
            if order == "log":
                specs = [
                    main_by_key[(name, budget)] for budget in ABLATION_BUDGETS
                ]
            else:
                specs = [
                    spec
                    for spec in ablation_specs()
                    if spec.function_name == name and spec.replication_order == order
                ]
                specs.sort(key=lambda item: item.total_budget)
            for index, (budget, spec) in enumerate(zip(ABLATION_BUDGETS, specs)):
                row = _summary(
                    name,
                    "LQEI",
                    budget,
                    ablation_matrices[(name, order)][:, index],
                )
                row.update(
                    {
                        "replication_order": order,
                        "order_label": ORDER_LABELS[order],
                        "batch_formula": ORDER_FORMULAS[order],
                        "rho": rho_for_function(name),
                        "nominal_batch_size": spec.nominal_batch_size,
                        "batch_count": spec.schedule.size,
                        "exact_budget_used": int(np.sum(spec.schedule)),
                    }
                )
                ablation_summary.append(row)

    validation = {
        "passed": True,
        "information_gain_method": INFORMATION_GAIN_METHOD,
        "information_gain_constant": INFORMATION_GAIN_CONSTANT,
        "main_lqei_macro_count": len(lqei_specs()) * FORMAL_MACROS,
        "comparison_macro_count": len(comparison_specs()) * FORMAL_MACROS,
        "additional_ablation_macro_count": (
            len(ablation_specs()) * FORMAL_MACROS if include_ablation else 0
        ),
        "total_macro_count": len(macro_rows),
        "checks": [
            "Every requested macro result exists exactly once",
            "Every run consumes its exact total budget",
            "Every LQEI kernel-order bound, finite-batch bound, equation-(20) minimum, Lambda_n, 0.2*Lambda_n acquisition history, and terminal coefficient match the registered formula",
            "Every LQEI final recommendation is the exact stored-posterior argmin",
            "SKO, EQI, and TSSO retain their registered implementations and use the common log-batch schedule",
            (
                "Ablation orders use the same numerical kernel-specific rho without anchor normalization"
                if include_ablation
                else "No ablation configurations are included in this formal comparison run"
            ),
        ],
    }
    return (
        algorithm_matrices,
        ablation_matrices,
        macro_rows,
        algorithm_summary,
        ablation_summary,
        lambda_rows,
        validation,
    )


def plot_figures(algorithm_matrices):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    slope_rows = []
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    for axis, name in zip(axes, FUNCTION_NAMES):
        budgets = np.asarray(budgets_for_function(name), dtype=float)
        values = algorithm_matrices[(name, "LQEI")]
        means = np.mean(values, axis=0)
        errors = np.std(values, axis=0, ddof=1) / np.sqrt(FORMAL_MACROS)
        fit = linregress(np.log(budgets), np.log(means))
        reference = means[0] * (budgets / budgets[0]) ** THEORY_SLOPES[name]
        axis.errorbar(
            budgets,
            means,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS["LQEI"],
            label=f"LQEI mean GAP (slope={fit.slope:.3f})",
        )
        axis.plot(
            budgets,
            reference,
            linestyle="--",
            color="#666666",
            label=f"reference slope={THEORY_SLOPES[name]:.3f}",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(
            budgets,
            [rf"$e^{{{budget_exponent(int(budget)):g}}}$" for budget in budgets],
        )
        axis.set_title(FUNCTION_LABELS[name])
        axis.set_xlabel("Total simulation budget T")
        axis.set_ylabel("Average raw GAP")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)
        slope_rows.append(
            {
                "function": name,
                "empirical_log_log_slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r_squared": float(fit.rvalue**2),
                "reference_slope": THEORY_SLOPES[name],
                "budget_min": int(budgets.min()),
                "budget_max": int(budgets.max()),
            }
        )
    figure.suptitle(
        "LQEI convergence with equation-(20) minimum information-gain bound\n"
        "100 macro-replications per budget"
    )
    for extension in ("png", "pdf"):
        figure.savefig(FIGURE_DIR / f"lqei_convergence_rate.{extension}", dpi=240)
    plt.close(figure)

    maximum_budget_count = max(
        len(budgets_for_function(name)) for name in FUNCTION_NAMES
    )
    figure, axes = plt.subplots(
        2,
        maximum_budget_count,
        figsize=(3.05 * maximum_budget_count, 8.1),
        constrained_layout=True,
        squeeze=False,
    )
    for row, name in enumerate(FUNCTION_NAMES):
        budgets = np.asarray(budgets_for_function(name), dtype=int)
        for column, budget in enumerate(budgets.astype(int)):
            axis = axes[row, column]
            data = [
                algorithm_matrices[(name, algorithm)][:, column]
                for algorithm in ALGORITHMS
            ]
            means = [float(np.mean(values)) for values in data]
            box = axis.boxplot(
                data,
                labels=ALGORITHMS,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 4,
                },
            )
            for patch_box, algorithm in zip(box["boxes"], ALGORITHMS):
                patch_box.set_facecolor(COLORS[algorithm])
                patch_box.set_alpha(0.72)
            axis.set_title(
                rf"$T=e^{{{budget_exponent(int(budget)):g}}}$" + f"\n({budget:,})"
            )
            axis.tick_params(axis="x", labelrotation=28, labelsize=8)
            axis.grid(True, axis="y", alpha=0.22)
            if column == 0:
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP")
            text = "\n".join(
                f"{algorithm}: {mean:.3g}" for algorithm, mean in zip(ALGORITHMS, means)
            )
            axis.text(
                0.04,
                0.96,
                text,
                transform=axis.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
        for column in range(len(budgets), maximum_budget_count):
            axes[row, column].set_visible(False)
    figure.suptitle(
        "Same-budget algorithm comparison (diamond = average GAP)\n"
        "LQEI, SKO, EQI, and TSSO use the common O(log T) batch schedule"
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"algorithm_comparison_same_budget_boxplots.{extension}",
            dpi=240,
        )
    plt.close(figure)
    return slope_rows


def plot_ablation_figure(ablation_matrices) -> None:
    """Plot a simple grouped-bar comparison of the three primary orders."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    budgets = np.asarray(ABLATION_BUDGETS, dtype=int)
    x_positions = np.arange(budgets.size, dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.2),
        constrained_layout=True,
        sharey=False,
    )
    for axis, function_name in zip(axes, FUNCTION_NAMES):
        for order, offset in zip(PRIMARY_ABLATION_ORDERS, offsets):
            values = ablation_matrices[(function_name, order)]
            means = np.mean(values, axis=0)
            bars = axis.bar(
                x_positions + offset,
                means,
                width=width,
                color=ABLATION_COLORS[order],
                label=PRIMARY_ABLATION_LABELS[order],
            )
            axis.bar_label(bars, fmt="%.4f", padding=2, fontsize=7)
        axis.set_xticks(x_positions, [f"{budget:,}" for budget in budgets])
        axis.set_xlabel("Total simulation budget T")
        axis.set_ylabel("Average raw GAP")
        axis.set_title(FUNCTION_LABELS[function_name])
        axis.grid(True, axis="y", alpha=0.2)
        axis.legend(fontsize=8.5)
    figure.suptitle(
        "LQEI replication-order ablation: average GAP (lower is better)",
        fontsize=14,
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"replication_order_ablation_average_gap.{extension}",
            dpi=240,
        )
    plt.close(figure)


def write_ablation_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# LQEI replication-number-order ablation",
        "",
        "All entries are average raw GAP over 100 macros. Each schedule consumes the exact displayed total budget. The same numerical kernel-specific rho is used without anchor normalization.",
        "",
    ]
    for name in FUNCTION_NAMES:
        lines.extend(
            [
                f"## {FUNCTION_LABELS[name]}",
                "",
                "| Order | rho | T=30,000 | T=40,000 | T=60,000 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for order in ABLATION_ORDERS:
            selected = [
                row
                for row in rows
                if row["function"] == name and row["replication_order"] == order
            ]
            selected.sort(key=lambda row: int(row["total_budget"]))
            values = " | ".join(f"{float(row['average_gap']):.8f}" for row in selected)
            lines.append(
                f"| {ORDER_LABELS[order]} | {rho_for_function(name):g} | {values} |"
            )
        lines.append("")
        lines.extend(
            [
                "Batch sizes / batch counts:",
                "",
                "| Order | 30,000 | 40,000 | 60,000 |",
                "|---|---:|---:|---:|",
            ]
        )
        for order in ABLATION_ORDERS:
            selected = [
                row
                for row in rows
                if row["function"] == name and row["replication_order"] == order
            ]
            selected.sort(key=lambda row: int(row["total_budget"]))
            values = " | ".join(
                f"{int(row['nominal_batch_size'])} / {int(row['batch_count'])}"
                for row in selected
            )
            lines.append(f"| {ORDER_LABELS[order]} | {values} |")
        lines.append("")
    (ANALYSIS_DIR / "replication_order_ablation_average_gap.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_report(
    algorithm_summary: list[dict[str, object]],
    ablation_summary: list[dict[str, object]],
    slopes: list[dict[str, object]],
    validation: dict[str, object],
) -> None:
    lines = [
        "# 2D LQEI with equation-(20) minimum information-gain bound",
        "",
        "The LQEI confidence sequence takes the pointwise minimum of the unit-constant numerical kernel-order proxy",
        "$G_{SE}(U)=\\log(1+U)^3$ and "
        "$G_{Mat\\acute{e}rn}(U)=U^{2/7}\\log(1+U)^{5/7}$, with "
        "$U_n=n b_T/\\nu_{min}$, and the finite-batch bound "
        "$n\\log(1+b_T/\\nu_{min})/2$. Acquisition and terminal recommendation use "
        "$0.2\\Lambda_n$ and $0.2\\Lambda_{N_T}$, respectively.",
        "",
        "## Empirical convergence slopes",
        "",
        "| Function | Slope | R-squared | Reference |",
        "|---|---:|---:|---:|",
    ]
    for row in slopes:
        lines.append(
            f"| {FUNCTION_LABELS[str(row['function'])]} | "
            f"{float(row['empirical_log_log_slope']):.4f} | "
            f"{float(row['r_squared']):.4f} | "
            f"{float(row['reference_slope']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"All {int(validation['total_macro_count']):,} macro results passed the registered formula, exact-budget, metadata, and recommendation checks.",
            "",
            "## Outputs",
            "",
            "- `figures/lqei_convergence_rate.png` and `.pdf`",
            "- `figures/algorithm_comparison_same_budget_boxplots.png` and `.pdf`",
            "- `algorithm_comparison_summary.csv`",
            "- `lambda_schedule_summary.csv`",
            "- `all_macro_results.csv` and `gap_matrices.npz`",
            "- `validation_report.json`",
        ]
    )
    (ANALYSIS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (
        algorithm_matrices,
        ablation_matrices,
        macro_rows,
        algorithm_summary,
        ablation_summary,
        lambda_rows,
        validation,
    ) = collect_and_validate()
    slopes = plot_figures(algorithm_matrices)
    _write_csv(ANALYSIS_DIR / "all_macro_results.csv", macro_rows)
    _write_csv(ANALYSIS_DIR / "algorithm_comparison_summary.csv", algorithm_summary)
    _write_csv(ANALYSIS_DIR / "lambda_schedule_summary.csv", lambda_rows)
    _write_csv(ANALYSIS_DIR / "convergence_slopes.csv", slopes)
    matrix_payload = {
        f"algorithm__{name}__{algorithm}": values
        for (name, algorithm), values in algorithm_matrices.items()
    }
    for name in FUNCTION_NAMES:
        matrix_payload[f"main_budgets__{name}"] = np.asarray(
            budgets_for_function(name), dtype=int
        )
    np.savez_compressed(ANALYSIS_DIR / "gap_matrices.npz", **matrix_payload)
    (ANALYSIS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(algorithm_summary, ablation_summary, slopes, validation)
    print(
        f"Validated {validation['total_macro_count']:,} macros and wrote all tables and figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
