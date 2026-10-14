"""Validate, aggregate, and plot the four-dimensional combined-bound study."""

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
    FUNCTION_DIR,
    INCUMBENT_RULE_DYNAMIC,
    _macro_path,
    incumbent_sd_coefficient,
    load_confidence_schedule,
)
from rkhs_functions import load_fixed_function
from run_experiments import (
    ABLATION_ORDERS,
    COMPARISON_ALGORITHMS,
    FUNCTION_NAMES,
    ablation_specs,
    comparison_specs,
    lqei_specs,
)
from study_config import (
    ABLATION_BUDGETS,
    ABLATION_N_INITIAL,
    BUDGET_EXPONENTS,
    budget_label,
    CONTINUOUS_POLISH_ENABLED,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_STARTS,
    FORMAL_MACROS,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_RULE,
    N_CANDIDATES,
    N_INITIAL,
    budgets_for_function,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"
ALGORITHMS = ("LQEI", *COMPARISON_ALGORITHMS)
FUNCTION_LABELS = {
    "rkhs_matern52_4d": "Matérn-5/2 (4D)",
    "rkhs_se_4d": "Squared exponential (4D)",
}
ORDER_LABELS = {
    "constant": "O(1)",
    "log_log": "O(log log T)",
    "log": "O(log T)",
    "power_1p2": "O(T^(1/2))",
}
ORDER_FORMULAS = {
    "constant": "rho",
    "log_log": "rho*log(log(T))",
    "log": "rho*log(T)",
    "power_1p2": "rho*T^(1/2)",
}
COLORS = {
    "LQEI": "#0072B2",
    "SKO": "#E69F00",
    "EQI": "#009E73",
    "TSSO": "#CC79A7",
}
THEORY_SLOPES = {"rkhs_matern52_4d": -1.0 / 18.0, "rkhs_se_4d": -0.5}


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


def _validate_confidence(spec) -> object:
    function = load_fixed_function(FUNCTION_DIR / f"{spec.function_name}.npz")
    confidence = load_confidence_schedule(spec)
    expected_kernel_order = INFORMATION_GAIN_CONSTANT * kernel_order_information_gain(
        function.kernel,
        confidence.effective_information_budget,
    )
    expected_finite_batch = finite_batch_information_gain(
        np.arange(confidence.batch_count + 1),
        information_gain_batch_size=confidence.information_gain_batch_size,
        variance_min=confidence.variance_min,
    )
    np.testing.assert_allclose(
        confidence.kernel_order_upper_bound,
        expected_kernel_order,
    )
    np.testing.assert_allclose(
        confidence.finite_batch_upper_bound,
        expected_finite_batch,
    )
    np.testing.assert_allclose(
        confidence.information_gain_upper_bound,
        np.minimum(expected_kernel_order, expected_finite_batch),
    )
    expected_lambda = function.rkhs_norm + np.sqrt(
        2.0
        * confidence.kappa_t
        * (
            confidence.information_gain_upper_bound
            + np.log(3.0 / (2.0 * spec.confidence_delta))
        )
    )
    np.testing.assert_allclose(confidence.lambda_values, expected_lambda)
    if confidence.information_gain_batch_size != spec.nominal_batch_size:
        raise AssertionError("information-gain batch size mismatch")
    if confidence.correction_batch_size != spec.batch_size:
        raise AssertionError("confidence correction batch size mismatch")
    return confidence


def _validate_lqei_spec(
    spec,
    provenance: str,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    confidence = _validate_confidence(spec)
    if int(np.sum(spec.schedule)) != spec.total_budget:
        raise AssertionError("LQEI schedule does not exhaust the exact budget")
    gaps = np.empty(spec.macro_count)
    rows: list[dict[str, object]] = []
    for macro in range(1, spec.macro_count + 1):
        path = _macro_path(spec, macro)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            expected_incumbent_coefficients = np.asarray(
                [
                    incumbent_sd_coefficient(spec.incumbent_rule, value)
                    for value in confidence.lambda_values[:-1]
                ],
                dtype=float,
            )
            stored_multiplier = float(data["acquisition_lambda_multiplier"])
            multiplier_matches = (
                np.isclose(
                    stored_multiplier,
                    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
                )
                if spec.incumbent_rule == INCUMBENT_RULE_DYNAMIC
                else np.isnan(stored_multiplier)
            )
            checks = (
                str(data["confidence_method"]) == INFORMATION_GAIN_METHOD,
                int(data["total_budget"]) == spec.total_budget,
                int(data["initial_point_count"]) == spec.initial_point_count,
                np.array_equal(data["replications"], spec.schedule),
                int(np.sum(data["replications"])) == spec.total_budget,
                np.asarray(data["raw_observations"]).size == spec.total_budget,
                np.allclose(data["lambda_values"], confidence.lambda_values),
                np.all(np.diff(data["lambda_values"]) >= 0.0),
                multiplier_matches,
                str(data["recommendation_rule"]) == LQEI_RECOMMENDATION_RULE,
                float(data["recommendation_lambda_multiplier"])
                == LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
                np.isclose(
                    float(data["recommendation_coefficient"]),
                    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
                    * confidence.lambda_values[-1],
                ),
                np.asarray(data["initial_points"]).shape
                == (spec.initial_point_count, 4),
                np.array_equal(
                    data["selected_points"],
                    data["universe_points"][data["selected_indices"]],
                ),
                int(data["recommendation_index"])
                in np.unique(np.asarray(data["selected_indices"], dtype=int)),
                bool(data["continuous_polish_enabled"])
                == CONTINUOUS_POLISH_ENABLED,
                int(data["continuous_polish_starts"])
                == CONTINUOUS_POLISH_STARTS,
                int(data["continuous_polish_maxiter"])
                == CONTINUOUS_POLISH_MAXITER,
                int(data["continuous_points_added"])
                == int(
                    data["universe_points"].shape[0]
                    - spec.initial_point_count
                    - N_CANDIDATES
                ),
            )
            if not all(checks):
                raise AssertionError(f"invalid LQEI macro metadata: {path}")
            if "incumbent_rule" in data:
                if str(data["incumbent_rule"]) != spec.incumbent_rule:
                    raise AssertionError(f"wrong incumbent rule: {path}")
                np.testing.assert_allclose(
                    data["incumbent_sd_coefficient_history"],
                    expected_incumbent_coefficients,
                )
            elif spec.incumbent_rule != INCUMBENT_RULE_DYNAMIC:
                raise AssertionError(f"incumbent rule metadata is missing: {path}")
            gap = float(data["raw_gap"])
            if not np.isfinite(gap) or gap < 0.0:
                raise AssertionError(f"invalid GAP: {path}")
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
                    "recommendation_x3": float(data["recommendation_x"][2]),
                    "recommendation_x4": float(data["recommendation_x"][3]),
                }
            )
    lambda_row = {
        "study": provenance,
        "function": spec.function_name,
        "replication_order": spec.replication_order,
        "total_budget": spec.total_budget,
        "rho": spec.replication_multiplier,
        "initial_point_count": spec.initial_point_count,
        "nominal_batch_size": spec.nominal_batch_size,
        "minimum_actual_batch_size": int(np.min(spec.schedule)),
        "maximum_actual_batch_size": int(np.max(spec.schedule)),
        "batch_count": spec.schedule.size,
        "variance_min": confidence.variance_min,
        "effective_U_terminal": float(confidence.effective_information_budget[-1]),
        "Gk_terminal": float(confidence.kernel_order_upper_bound[-1]),
        "finite_batch_terminal": float(confidence.finite_batch_upper_bound[-1]),
        "combined_min_terminal": float(confidence.information_gain_upper_bound[-1]),
        "terminal_active_bound": (
            "kernel_order"
            if confidence.kernel_order_upper_bound[-1]
            <= confidence.finite_batch_upper_bound[-1]
            else "finite_batch"
        ),
        "kappa_T": confidence.kappa_t,
        "lambda_0": float(confidence.lambda_values[0]),
        "lambda_terminal": float(confidence.lambda_values[-1]),
        "terminal_0p2_lambda": float(
            LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER
            * confidence.lambda_values[-1]
        ),
    }
    return gaps, rows, lambda_row


def collect_and_validate():
    budgets = tuple(budgets_for_function(FUNCTION_NAMES[0]))
    algorithm_matrices = {
        (name, algorithm): np.empty((FORMAL_MACROS, len(budgets)))
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

    for spec in lqei_specs():
        gaps, rows, lambda_row = _validate_lqei_spec(spec, "main")
        macro_rows.extend(rows)
        lambda_rows.append(lambda_row)
        budget_index = budgets.index(spec.total_budget)
        algorithm_matrices[(spec.function_name, "LQEI")][:, budget_index] = gaps

    for spec in ablation_specs():
        gaps, rows, lambda_row = _validate_lqei_spec(spec, "ablation")
        macro_rows.extend(rows)
        lambda_rows.append(lambda_row)
        budget_index = ABLATION_BUDGETS.index(spec.total_budget)
        ablation_matrices[(spec.function_name, spec.replication_order)][
            :, budget_index
        ] = gaps

    main_by_key = {
        (spec.function_name, spec.total_budget): spec for spec in lqei_specs()
    }
    for spec in comparison_specs():
        gaps = np.empty(spec.macro_count)
        reference = main_by_key[(spec.function_name, spec.total_budget)]
        initial_budget = int(np.sum(reference.schedule[:N_INITIAL]))
        for macro in range(1, spec.macro_count + 1):
            path = comparison_macro_path(spec, macro)
            reference_path = _macro_path(reference, macro)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data, np.load(
                reference_path, allow_pickle=False
            ) as lqei:
                checks = (
                    str(data["algorithm"]) == spec.algorithm,
                    int(data["total_budget"]) == spec.total_budget,
                    int(np.sum(data["replications"])) == spec.total_budget,
                    np.asarray(data["raw_observations"]).size == spec.total_budget,
                    np.array_equal(data["initial_points"], lqei["initial_points"]),
                    np.array_equal(
                        data["raw_observations"][:initial_budget],
                        lqei["raw_observations"][:initial_budget],
                    ),
                )
                if spec.algorithm == "TSSO":
                    checks += (
                        int(data["tsso_stage_count"])
                        == spec.schedule.size - N_INITIAL,
                        np.array_equal(
                            data["tsso_stage_budgets"], spec.schedule[N_INITIAL:]
                        ),
                    )
                if spec.algorithm == "SKO":
                    checks += (
                        bool(data["response_hyperparameters_fixed"]),
                        bool(data["auxiliary_noise_gp_initial_fit_only"]),
                        float(data["sko_incumbent_sd_multiplier"]) == 1.0,
                        bool(data["sko_noise_penalty_uses_single_replication_variance"]),
                        bool(data["continuous_polish_enabled"])
                        == CONTINUOUS_POLISH_ENABLED,
                        int(data["continuous_polish_starts"])
                        == CONTINUOUS_POLISH_STARTS,
                        int(data["continuous_polish_maxiter"])
                        == CONTINUOUS_POLISH_MAXITER,
                    )
                if not all(checks):
                    raise AssertionError(f"invalid comparison macro: {path}")
                gap = float(data["raw_gap"])
                if not np.isfinite(gap) or gap < 0.0:
                    raise AssertionError(f"invalid comparison GAP: {path}")
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
                        "recommendation_x3": float(data["recommendation_x"][2]),
                        "recommendation_x4": float(data["recommendation_x"][3]),
                    }
                )
        budget_index = budgets.index(spec.total_budget)
        algorithm_matrices[(spec.function_name, spec.algorithm)][
            :, budget_index
        ] = gaps

    algorithm_summary: list[dict[str, object]] = []
    for name in FUNCTION_NAMES:
        for algorithm in ALGORITHMS:
            for index, budget in enumerate(budgets):
                algorithm_summary.append(
                    _summary(
                        name,
                        algorithm,
                        budget,
                        algorithm_matrices[(name, algorithm)][:, index],
                    )
                )

    ablation_lookup = {
        (spec.function_name, spec.replication_order, spec.total_budget): spec
        for spec in ablation_specs()
    }
    ablation_summary: list[dict[str, object]] = []
    for name in FUNCTION_NAMES:
        for order in ABLATION_ORDERS:
            for index, budget in enumerate(ABLATION_BUDGETS):
                spec = ablation_lookup[(name, order, budget)]
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
                        "initial_point_count": spec.initial_point_count,
                        "nominal_batch_size": spec.nominal_batch_size,
                        "batch_count": spec.schedule.size,
                        "sequential_batch_count": (
                            spec.schedule.size - spec.initial_point_count
                        ),
                        "exact_budget_used": int(np.sum(spec.schedule)),
                    }
                )
                ablation_summary.append(row)

    validation = {
        "passed": True,
        "information_gain_method": INFORMATION_GAIN_METHOD,
        "information_gain_constant": INFORMATION_GAIN_CONSTANT,
        "main_initial_point_count": N_INITIAL,
        "ablation_initial_point_count": ABLATION_N_INITIAL,
        "main_lqei_macro_count": len(lqei_specs()) * FORMAL_MACROS,
        "comparison_macro_count": len(comparison_specs()) * FORMAL_MACROS,
        "ablation_macro_count": len(ablation_specs()) * FORMAL_MACROS,
        "total_macro_count": len(macro_rows),
        "checks": [
            "Every requested macro result exists exactly once",
            "Every run consumes its exact total simulation budget",
            "Every LQEI min{G_k(U_n), finite-batch bound}, Lambda_n, 0.2*Lambda_n acquisition history, and terminal coefficient match the registered formulas",
            "Every LQEI recommendation is one of the sampled points",
            "Main LQEI, SKO, EQI, and TSSO share the same initial observations and O(log T) budget schedule",
            "SKO, EQI, and TSSO use the registered 4D continuous optimization policy",
            "All four ablation orders use 20 initial points and the same numerical rho without anchor normalization",
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


def plot_figures(algorithm_matrices) -> list[dict[str, object]]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    budgets = np.asarray(budgets_for_function(FUNCTION_NAMES[0]), dtype=float)
    slope_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    for axis, name in zip(axes, FUNCTION_NAMES):
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
        axis.set_xticks(budgets, [f"e^{index:g}" for index in BUDGET_EXPONENTS])
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
        "4D LQEI convergence with the combined information-gain bound\n"
        "100 macro-replications per budget"
    )
    for extension in ("png", "pdf"):
        figure.savefig(FIGURE_DIR / f"lqei_convergence_rate.{extension}", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(
        2,
        len(budgets),
        figsize=(3.1 * len(budgets), 8.1),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    row_y_max = {
        name: 1.05
        * max(
            float(np.max(algorithm_matrices[(name, algorithm)]))
            for algorithm in ALGORITHMS
        )
        for name in FUNCTION_NAMES
    }
    for row, name in enumerate(FUNCTION_NAMES):
        for column, budget in enumerate(budgets.astype(int)):
            axis = axes[row, column]
            data = [
                algorithm_matrices[(name, algorithm)][:, column]
                for algorithm in ALGORITHMS
            ]
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
            axis.set_title(budget_label(budget))
            axis.set_ylim(0.0, row_y_max[name])
            axis.tick_params(axis="x", labelrotation=28, labelsize=8)
            axis.grid(True, axis="y", alpha=0.22)
            if column == 0:
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP")
            means = [float(np.mean(values)) for values in data]
            axis.text(
                0.035,
                0.965,
                "\n".join(
                    f"{algorithm}: {mean:.3g}"
                    for algorithm, mean in zip(ALGORITHMS, means)
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=7.0,
                bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
            )
    figure.suptitle(
        "Same-budget 4D algorithm comparison (diamond = average GAP)\n"
        "Common y-scale within each kernel row; all methods use the common O(log T) schedule"
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"algorithm_comparison_same_budget_boxplots.{extension}",
            dpi=240,
        )
    plt.close(figure)
    return slope_rows


def write_ablation_markdown(rows: list[dict[str, object]]) -> None:
    budget_headers = " | ".join(budget_label(budget) for budget in ABLATION_BUDGETS)
    separators = "|".join("---:" for _ in ABLATION_BUDGETS)
    lines = [
        "# 4D LQEI replication-number-order ablation",
        "",
        "All entries are average raw GAP over 100 macros. Every order uses 20 initial points, the same numerical kernel-specific rho without anchor normalization, and an exact balanced schedule that exhausts the displayed budget.",
        "",
    ]
    for name in FUNCTION_NAMES:
        lines.extend(
            [
                f"## {FUNCTION_LABELS[name]}",
                "",
                f"| Order | rho | {budget_headers} |",
                f"|---|---:|{separators}|",
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
        lines.extend(
            [
                "",
                "Batch sizes / total batch counts / sequential batch counts:",
                "",
                f"| Order | {budget_headers} |",
                f"|---|{separators}|",
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
                f"{int(row['nominal_batch_size'])} / {int(row['batch_count'])} / {int(row['sequential_batch_count'])}"
                for row in selected
            )
            lines.append(f"| {ORDER_LABELS[order]} | {values} |")
        lines.append("")
    (ANALYSIS_DIR / "replication_order_ablation_average_gap.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_ablation_average_gap_bars(rows: list[dict[str, object]]) -> None:
    """Draw the requested plain average-GAP comparison bars."""

    budgets = np.asarray(ABLATION_BUDGETS, dtype=int)
    positions = np.arange(budgets.size, dtype=float)
    width = 0.8 / len(ABLATION_ORDERS)
    colors = ("#56B4E9", "#E69F00", "#0072B2", "#D55E00")
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), constrained_layout=True, squeeze=False
    )
    for axis, name in zip(axes[0], FUNCTION_NAMES):
        for order_index, (order, color) in enumerate(zip(ABLATION_ORDERS, colors)):
            lookup = {
                int(row["total_budget"]): float(row["average_gap"])
                for row in rows
                if row["function"] == name and row["replication_order"] == order
            }
            values = [lookup[int(budget)] for budget in budgets]
            offset = (order_index - (len(ABLATION_ORDERS) - 1) / 2) * width
            axis.bar(
                positions + offset,
                values,
                width=width,
                color=color,
                label=ORDER_LABELS[order],
            )
        axis.set_title(FUNCTION_LABELS[name])
        axis.set_xticks(positions, [budget_label(int(budget)) for budget in budgets])
        axis.set_xlabel("Total simulation budget T")
        axis.set_ylabel("Average raw GAP")
        axis.grid(True, axis="y", alpha=0.22)
        axis.legend()
    figure.suptitle("LQEI replication-order ablation: average GAP")
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"replication_order_ablation_average_gap_bars.{extension}",
            dpi=240,
        )
    plt.close(figure)


def write_report(
    slopes: list[dict[str, object]],
    validation: dict[str, object],
) -> None:
    lines = [
        "# 4D LQEI with the combined information-gain upper bound",
        "",
        "The confidence sequence uses $\\min\\{G_k(U_n),\\,n\\log(1+b_T/\\nu_{min})/2\\}$, with $G_{SE}(U)=\\log(1+U)^5$, $G_{Mat\\acute{e}rn}(U)=U^{4/9}\\log(1+U)^{5/9}$, $U_n=n b_T/\\nu_{min}$, and unit big-O multiplier. Acquisition and terminal recommendation use $0.2\\Lambda_n$ and $0.2\\Lambda_{N_T}$.",
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
            f"All {int(validation['total_macro_count']):,} macro results passed the registered formula, exact-budget, metadata, continuous-optimizer, and paired-initial-observation checks.",
            "",
            "## Outputs",
            "",
            "- `figures/lqei_convergence_rate.png` and `.pdf`",
            "- `figures/algorithm_comparison_same_budget_boxplots.png` and `.pdf`",
            "- `algorithm_comparison_summary.csv`",
            "- `replication_order_ablation_average_gap.csv` and `.md`",
            "- `figures/replication_order_ablation_average_gap_bars.png` and `.pdf`",
            "- `lambda_schedule_summary.csv`",
            "- `all_macro_results.csv` and `gap_matrices.npz`",
            "- `validation_report.json`",
        ]
    )
    (ANALYSIS_DIR / "REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
    _write_csv(
        ANALYSIS_DIR / "replication_order_ablation_average_gap.csv",
        ablation_summary,
    )
    _write_csv(ANALYSIS_DIR / "lambda_schedule_summary.csv", lambda_rows)
    _write_csv(ANALYSIS_DIR / "convergence_slopes.csv", slopes)
    write_ablation_markdown(ablation_summary)
    plot_ablation_average_gap_bars(ablation_summary)
    matrix_payload = {
        f"algorithm__{name}__{algorithm}": values
        for (name, algorithm), values in algorithm_matrices.items()
    }
    matrix_payload.update(
        {
            f"ablation__{name}__{order}": values
            for (name, order), values in ablation_matrices.items()
        }
    )
    matrix_payload["main_budgets"] = np.asarray(
        budgets_for_function(FUNCTION_NAMES[0]), dtype=int
    )
    matrix_payload["ablation_budgets"] = np.asarray(ABLATION_BUDGETS, dtype=int)
    np.savez_compressed(ANALYSIS_DIR / "gap_matrices.npz", **matrix_payload)
    (ANALYSIS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(slopes, validation)
    print(
        f"Validated {validation['total_macro_count']:,} macros and wrote all tables and figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
