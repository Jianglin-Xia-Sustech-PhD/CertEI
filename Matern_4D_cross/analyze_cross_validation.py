"""Validate, postprocess, summarize, and plot the factorial LQEI study."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    INPUT_DIR,
    _macro_path,
    incumbent_sd_coefficient,
)
from rkhs_functions import load_fixed_function
from run_cross_validation import (
    FINAL_RECOMMENDATION_RULES,
    cross_validation_specs,
)
from study_config import (
    ANALYSIS_SEED,
    BOOTSTRAP_RESAMPLES,
    CROSS_VALIDATION_BUDGETS,
    CROSS_VALIDATION_PHASE_CODE,
    DIMENSION,
    FORMAL_MACROS,
    FUNCTION_NAME,
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
    N_CANDIDATES,
    N_INITIAL,
    REPLICATION_ORDERS,
    budget_exponent,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"

INCUMBENT_RULES = (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
INCUMBENT_LABELS = {
    INCUMBENT_RULE_DYNAMIC: "mu - 0.2 Lambda_n sigma",
    INCUMBENT_RULE_MEAN: "mu",
    INCUMBENT_RULE_UPPER_ONE_SD: "mu + sigma",
}
FINAL_LABELS = {
    "posterior_mean_plus_0p2_lambda_sd": "mu_N + 0.2 Lambda_N sigma",
    "posterior_mean": "mu_N",
    "posterior_mean_minus_0p2_lambda_sd": "mu_N - 0.2 Lambda_N sigma",
}
ORDER_LABELS = {
    "constant": "O(1)",
    "log_log": "O(log log T)",
    "log": "O(log T)",
    "power_1p2": "O(T^(1/2))",
}

TARGET_INCUMBENT = INCUMBENT_RULE_DYNAMIC
TARGET_FINAL = "posterior_mean_plus_0p2_lambda_sd"
TARGET_ORDER = "log"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _final_coefficient(rule: str, terminal_lambda: float) -> float:
    if rule == "posterior_mean_plus_0p2_lambda_sd":
        return float(LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER * terminal_lambda)
    if rule == "posterior_mean":
        return 0.0
    if rule == "posterior_mean_minus_0p2_lambda_sd":
        return float(-LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER * terminal_lambda)
    raise ValueError(f"unknown final recommendation rule: {rule}")


def _bootstrap_indices(budget: int) -> np.ndarray:
    seed = np.random.SeedSequence([ANALYSIS_SEED, int(budget)])
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        FORMAL_MACROS,
        size=(BOOTSTRAP_RESAMPLES, FORMAL_MACROS),
        dtype=np.int16,
    )


def _mean_ci(values: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, float]:
    boot_means = np.mean(values[bootstrap_indices], axis=1)
    lower, upper = np.quantile(boot_means, (0.025, 0.975))
    return float(lower), float(upper)


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value"]))
    running = 0.0
    total = len(rows)
    for position, index in enumerate(ordered):
        adjusted = min(1.0, (total - position) * float(rows[index]["p_value"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p_value"] = running


def collect_and_validate():
    candidate_path = (
        INPUT_DIR / f"sobol_candidates_{DIMENSION}d_{N_CANDIDATES}.npy"
    )
    candidate_pool = np.asarray(np.load(candidate_path), dtype=float)
    if candidate_pool.shape != (N_CANDIDATES, DIMENSION):
        raise AssertionError("candidate pool has the wrong shape")
    function = load_fixed_function(ROOT / "fixed_functions" / f"{FUNCTION_NAME}.npz")

    specifications = cross_validation_specs()
    expected_paths = len(specifications) * FORMAL_MACROS
    all_rows: list[dict[str, object]] = []
    gap_vectors: dict[tuple[int, str, str, str], np.ndarray] = {}
    initial_design_by_macro: dict[int, np.ndarray] = {}
    checked_paths = 0

    for spec in specifications:
        vector_by_final = {
            rule: np.empty(FORMAL_MACROS, dtype=float)
            for rule in FINAL_RECOMMENDATION_RULES
        }
        for macro in range(1, FORMAL_MACROS + 1):
            path = _macro_path(spec, macro)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                if str(data["function_name"]) != FUNCTION_NAME:
                    raise AssertionError("function metadata mismatch")
                if int(data["total_budget"]) != spec.total_budget:
                    raise AssertionError("budget metadata mismatch")
                if str(data["replication_order"]) != spec.replication_order:
                    raise AssertionError("replication-order metadata mismatch")
                if str(data["incumbent_rule"]) != spec.incumbent_rule:
                    raise AssertionError("incumbent metadata mismatch")
                if int(data["phase_code"]) != CROSS_VALIDATION_PHASE_CODE:
                    raise AssertionError("phase-code metadata mismatch")
                if int(data["candidate_count"]) != N_CANDIDATES:
                    raise AssertionError("candidate-count metadata mismatch")
                replications = np.asarray(data["replications"], dtype=int)
                if not np.array_equal(replications, spec.schedule):
                    raise AssertionError("stored replication schedule mismatch")
                if int(np.sum(replications)) != spec.total_budget:
                    raise AssertionError("trajectory does not use its exact budget")

                initial_points = np.asarray(data["initial_points"], dtype=float)
                if initial_points.shape != (N_INITIAL, DIMENSION):
                    raise AssertionError("initial design has the wrong shape")
                if macro not in initial_design_by_macro:
                    initial_design_by_macro[macro] = initial_points.copy()
                elif not np.array_equal(initial_design_by_macro[macro], initial_points):
                    raise AssertionError("initial designs are not paired across arms")

                lambda_values = np.asarray(data["lambda_values"], dtype=float)
                expected_incumbent = np.asarray(
                    [
                        incumbent_sd_coefficient(spec.incumbent_rule, value)
                        for value in lambda_values[:-1]
                    ],
                    dtype=float,
                )
                stored_incumbent = np.asarray(
                    data["incumbent_sd_coefficient_history"], dtype=float
                )
                if not np.allclose(stored_incumbent, expected_incumbent):
                    raise AssertionError("incumbent coefficient history mismatch")
                if spec.incumbent_rule == INCUMBENT_RULE_DYNAMIC:
                    if not np.isclose(
                        float(data["acquisition_lambda_multiplier"]),
                        LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
                    ):
                        raise AssertionError("acquisition multiplier mismatch")

                sampled = np.asarray(data["sampled_unique_indices"], dtype=int)
                means = np.asarray(data["sampled_posterior_mean"], dtype=float)
                sds = np.asarray(data["sampled_posterior_sd"], dtype=float)
                if not (sampled.size == means.size == sds.size):
                    raise AssertionError("terminal posterior arrays are misaligned")
                if not np.array_equal(sampled, np.unique(sampled)):
                    raise AssertionError("sampled indices are not sorted and unique")
                if np.any(~np.isfinite(means)) or np.any(~np.isfinite(sds)):
                    raise AssertionError("terminal posterior is not finite")
                if np.any(sds < 0.0):
                    raise AssertionError("terminal posterior sd is negative")

                universe = np.vstack((initial_points, candidate_pool))
                if sampled.size == 0 or sampled[0] < 0 or sampled[-1] >= universe.shape[0]:
                    raise AssertionError("sampled index outside the candidate universe")
                sampled_points = universe[sampled]
                terminal_lambda = float(lambda_values[-1])

                for final_rule in FINAL_RECOMMENDATION_RULES:
                    coefficient = _final_coefficient(final_rule, terminal_lambda)
                    local = int(np.argmin(means + coefficient * sds))
                    recommendation_index = int(sampled[local])
                    recommendation_x = sampled_points[local]
                    recommendation_value = float(
                        function.evaluate(
                            recommendation_x.reshape(1, DIMENSION)
                        )[0]
                    )
                    gap = max(recommendation_value - function.optimum_value, 0.0)
                    if final_rule == TARGET_FINAL:
                        if recommendation_index != int(data["recommendation_index"]):
                            raise AssertionError("stored final recommendation mismatch")
                        if not np.isclose(gap, float(data["raw_gap"])):
                            raise AssertionError("stored GAP mismatch")
                    vector_by_final[final_rule][macro - 1] = gap
                    is_target = (
                        spec.incumbent_rule == TARGET_INCUMBENT
                        and spec.replication_order == TARGET_ORDER
                        and final_rule == TARGET_FINAL
                    )
                    all_rows.append(
                        {
                            "function": FUNCTION_NAME,
                            "total_budget": spec.total_budget,
                            "budget_exponent": budget_exponent(spec.total_budget),
                            "macro": macro,
                            "incumbent_rule": spec.incumbent_rule,
                            "incumbent_label": INCUMBENT_LABELS[spec.incumbent_rule],
                            "replication_order": spec.replication_order,
                            "replication_order_label": ORDER_LABELS[spec.replication_order],
                            "final_recommendation_rule": final_rule,
                            "final_recommendation_label": FINAL_LABELS[final_rule],
                            "terminal_lambda": terminal_lambda,
                            "final_sd_coefficient": coefficient,
                            "raw_gap": gap,
                            "recommendation_index": recommendation_index,
                            "recommendation_x1": float(recommendation_x[0]),
                            "recommendation_x2": float(recommendation_x[1]),
                            "recommendation_x3": float(recommendation_x[2]),
                            "recommendation_x4": float(recommendation_x[3]),
                            "recommendation_posterior_mean": float(means[local]),
                            "recommendation_posterior_sd": float(sds[local]),
                            "is_target_combination": is_target,
                        }
                    )
            checked_paths += 1
        for final_rule, values in vector_by_final.items():
            key = (
                spec.total_budget,
                spec.incumbent_rule,
                spec.replication_order,
                final_rule,
            )
            if key in gap_vectors:
                raise AssertionError(f"duplicate combination: {key}")
            gap_vectors[key] = values

    if checked_paths != expected_paths:
        raise AssertionError("trajectory count mismatch")
    if len(gap_vectors) != 72:
        raise AssertionError("expected exactly 72 reported combinations")
    if len(all_rows) != 72 * FORMAL_MACROS:
        raise AssertionError("macro-level recommendation count mismatch")
    validation = {
        "passed": True,
        "validated_trajectory_count": checked_paths,
        "expected_trajectory_count": expected_paths,
        "validated_recommendation_count": len(all_rows),
        "combination_count": len(gap_vectors),
        "paired_initial_design_macro_count": len(initial_design_by_macro),
        "checks": [
            "All 24 trajectory configurations contain 100 macros",
            "Every trajectory consumes its exact registered budget",
            "Function, budget, phase, incumbent, order, and candidate metadata match",
            "All arms share the same initial design within each macro",
            "The saved dynamic incumbent coefficient sequence is reproduced",
            "The saved +0.2 Lambda final recommendation and raw GAP are reproduced",
            "All three final rules are rescored on the same terminal posterior",
        ],
    }
    return gap_vectors, all_rows, validation


def summarize(gap_vectors: dict[tuple[int, str, str, str], np.ndarray]):
    summary_rows: list[dict[str, object]] = []
    bootstrap_by_budget = {
        budget: _bootstrap_indices(budget) for budget in CROSS_VALIDATION_BUDGETS
    }
    for key, values in gap_vectors.items():
        budget, incumbent, order, final_rule = key
        lower, upper = _mean_ci(values, bootstrap_by_budget[budget])
        summary_rows.append(
            {
                "function": FUNCTION_NAME,
                "total_budget": budget,
                "budget_exponent": budget_exponent(budget),
                "incumbent_rule": incumbent,
                "incumbent_label": INCUMBENT_LABELS[incumbent],
                "replication_order": order,
                "replication_order_label": ORDER_LABELS[order],
                "final_recommendation_rule": final_rule,
                "final_recommendation_label": FINAL_LABELS[final_rule],
                "macro_count": int(values.size),
                "average_gap": float(np.mean(values)),
                "mean_gap_ci95_lower": lower,
                "mean_gap_ci95_upper": upper,
                "std_gap": float(np.std(values, ddof=1)),
                "se_average_gap": float(np.std(values, ddof=1) / np.sqrt(values.size)),
                "median_gap": float(np.median(values)),
                "q25_gap": float(np.quantile(values, 0.25)),
                "q75_gap": float(np.quantile(values, 0.75)),
                "minimum_gap": float(np.min(values)),
                "maximum_gap": float(np.max(values)),
                "is_target_combination": (
                    incumbent == TARGET_INCUMBENT
                    and order == TARGET_ORDER
                    and final_rule == TARGET_FINAL
                ),
            }
        )

    for budget in CROSS_VALIDATION_BUDGETS:
        selected = [row for row in summary_rows if row["total_budget"] == budget]
        for row in selected:
            value = float(row["average_gap"])
            row["mean_gap_rank"] = 1 + sum(
                float(other["average_gap"]) < value - 1e-15 for other in selected
            )
    summary_rows.sort(
        key=lambda row: (
            int(row["total_budget"]),
            int(row["mean_gap_rank"]),
            str(row["incumbent_rule"]),
            str(row["replication_order"]),
            str(row["final_recommendation_rule"]),
        )
    )

    comparisons: list[dict[str, object]] = []
    for budget in CROSS_VALIDATION_BUDGETS:
        target_key = (budget, TARGET_INCUMBENT, TARGET_ORDER, TARGET_FINAL)
        target = gap_vectors[target_key]
        budget_rows: list[dict[str, object]] = []
        for key, competitor in gap_vectors.items():
            if key[0] != budget or key == target_key:
                continue
            _, incumbent, order, final_rule = key
            difference = target - competitor
            lower, upper = _mean_ci(difference, bootstrap_by_budget[budget])
            nonzero = difference[np.abs(difference) > 1e-15]
            if nonzero.size == 0:
                p_value = 1.0
            else:
                try:
                    p_value = float(
                        wilcoxon(
                            difference,
                            alternative="less",
                            zero_method="pratt",
                            method="approx",
                        ).pvalue
                    )
                except ValueError:
                    p_value = 1.0
            changed = sum(
                (
                    incumbent != TARGET_INCUMBENT,
                    order != TARGET_ORDER,
                    final_rule != TARGET_FINAL,
                )
            )
            competitor_mean = float(np.mean(competitor))
            relative = (
                (competitor_mean - float(np.mean(target))) / competitor_mean
                if competitor_mean > 0.0
                else np.nan
            )
            budget_rows.append(
                {
                    "total_budget": budget,
                    "competitor_incumbent_rule": incumbent,
                    "competitor_incumbent_label": INCUMBENT_LABELS[incumbent],
                    "competitor_replication_order": order,
                    "competitor_replication_order_label": ORDER_LABELS[order],
                    "competitor_final_recommendation_rule": final_rule,
                    "competitor_final_recommendation_label": FINAL_LABELS[final_rule],
                    "is_one_factor_neighbor": changed == 1,
                    "target_average_gap": float(np.mean(target)),
                    "competitor_average_gap": competitor_mean,
                    "target_minus_competitor_mean_gap": float(np.mean(difference)),
                    "paired_mean_difference_ci95_lower": lower,
                    "paired_mean_difference_ci95_upper": upper,
                    "relative_improvement_of_target": relative,
                    "target_win_count": int(np.sum(difference < -1e-15)),
                    "tie_count": int(np.sum(np.abs(difference) <= 1e-15)),
                    "target_loss_count": int(np.sum(difference > 1e-15)),
                    "p_value": p_value,
                    "holm_adjusted_p_value": np.nan,
                    "strict_bootstrap_dominance": upper < 0.0,
                }
            )
        _holm_adjust(budget_rows)
        comparisons.extend(budget_rows)

    comparisons.sort(
        key=lambda row: (
            int(row["total_budget"]),
            not bool(row["is_one_factor_neighbor"]),
            float(row["competitor_average_gap"]),
        )
    )
    return summary_rows, comparisons


def plot_heatmaps(summary_rows: list[dict[str, object]]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    lookup = {
        (
            int(row["total_budget"]),
            str(row["incumbent_rule"]),
            str(row["replication_order"]),
            str(row["final_recommendation_rule"]),
        ): float(row["average_gap"])
        for row in summary_rows
    }
    positive = np.asarray([value for value in lookup.values() if value > 0.0])
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    image = None
    for row_index, budget in enumerate(CROSS_VALIDATION_BUDGETS):
        for column_index, final_rule in enumerate(FINAL_RECOMMENDATION_RULES):
            axis = axes[row_index, column_index]
            matrix = np.asarray(
                [
                    [
                        lookup[(budget, incumbent, order, final_rule)]
                        for order in REPLICATION_ORDERS
                    ]
                    for incumbent in INCUMBENT_RULES
                ]
            )
            image = axis.imshow(matrix, cmap="viridis_r", norm=norm, aspect="auto")
            axis.set_xticks(range(len(REPLICATION_ORDERS)))
            axis.set_xticklabels(
                [ORDER_LABELS[value] for value in REPLICATION_ORDERS],
                rotation=24,
                ha="right",
            )
            axis.set_yticks(range(len(INCUMBENT_RULES)))
            axis.set_yticklabels([INCUMBENT_LABELS[value] for value in INCUMBENT_RULES])
            axis.set_title(FINAL_LABELS[final_rule])
            if column_index == 0:
                axis.set_ylabel(
                    rf"$T=e^{{{budget_exponent(budget):g}}}$ ({budget:,})\nIncumbent"
                )
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if norm(matrix[i, j]) > 0.55 else "black",
                    )
            if final_rule == TARGET_FINAL:
                target_row = INCUMBENT_RULES.index(TARGET_INCUMBENT)
                target_column = REPLICATION_ORDERS.index(TARGET_ORDER)
                axis.add_patch(
                    Rectangle(
                        (target_column - 0.5, target_row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#D55E00",
                        linewidth=3,
                    )
                )
    if image is not None:
        figure.colorbar(image, ax=axes, label="Average raw GAP (log color scale)")
    figure.suptitle(
        "4D Matérn LQEI cross-validation: average raw GAP\n"
        "Orange outline = pre-registered target combination"
    )
    destination = FIGURE_DIR / "combination_average_gap_heatmaps.png"
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)
    return destination


def plot_target_neighbors(
    gap_vectors: dict[tuple[int, str, str, str], np.ndarray]
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target_suffix = (TARGET_INCUMBENT, TARGET_ORDER, TARGET_FINAL)
    neighbors = [
        (TARGET_INCUMBENT, TARGET_ORDER, "posterior_mean"),
        (TARGET_INCUMBENT, TARGET_ORDER, "posterior_mean_minus_0p2_lambda_sd"),
        (INCUMBENT_RULE_MEAN, TARGET_ORDER, TARGET_FINAL),
        (INCUMBENT_RULE_UPPER_ONE_SD, TARGET_ORDER, TARGET_FINAL),
        (TARGET_INCUMBENT, "constant", TARGET_FINAL),
        (TARGET_INCUMBENT, "log_log", TARGET_FINAL),
        (TARGET_INCUMBENT, "power_1p2", TARGET_FINAL),
    ]
    labels = [
        "Target",
        "Final: mu_N",
        "Final: mu_N-0.2LNs",
        "Inc: mu",
        "Inc: mu+sigma",
        "Order: O(1)",
        "Order: O(log log T)",
        "Order: O(T^1/2)",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, budget in zip(axes, CROSS_VALIDATION_BUDGETS):
        keys = [(budget, *target_suffix)] + [(budget, *item) for item in neighbors]
        values = [gap_vectors[key] for key in keys]
        box = axis.boxplot(
            values,
            labels=labels,
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
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor("#D55E00" if index == 0 else "#56B4E9")
            patch.set_alpha(0.75)
        if min(float(np.min(item)) for item in values) > 0.0:
            axis.set_yscale("log")
        axis.set_title(rf"$T=e^{{{budget_exponent(budget):g}}}$ ({budget:,})")
        axis.set_ylabel("Raw GAP")
        axis.tick_params(axis="x", labelrotation=32, labelsize=8)
        axis.grid(True, axis="y", which="both", alpha=0.22)
    figure.suptitle("Target combination versus seven one-factor alternatives")
    destination = FIGURE_DIR / "target_vs_one_factor_neighbors_boxplots.png"
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)
    return destination


def _fmt(value: object) -> str:
    return f"{float(value):.6g}"


def write_report(
    summary_rows: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    validation: dict[str, object],
) -> Path:
    lines = [
        "# 4D Matérn LQEI cross-validation",
        "",
        "This registered factorial study crosses three EI incumbents, three "
        "terminal sampled-point recommendation rules, and four replication-number "
        "orders at `T=e^10.5=36,316` and `T=e^11.5=98,716`.",
        "",
        "The pre-registered target is `mu_N + 0.2 Lambda_N sigma_N`, "
        "`mu_n - 0.2 Lambda_n sigma_n`, and `O(log T)` replication.",
        "",
        "Each cell is reported as mean raw GAP with a deterministic 10,000-resample "
        "nonparametric bootstrap 95% confidence interval, followed by median "
        "[Q1, Q3]. Pairwise comparisons resample paired macro differences.",
        "",
        "## Target combination",
        "",
        "| Budget | Rank / 36 | Mean GAP (95% CI) | Median [Q1, Q3] |",
        "|---:|---:|---:|---:|",
    ]
    for budget in CROSS_VALIDATION_BUDGETS:
        target = next(
            row
            for row in summary_rows
            if row["total_budget"] == budget and row["is_target_combination"]
        )
        lines.append(
            f"| e^{budget_exponent(budget):g} ({budget:,}) | "
            f"{target['mean_gap_rank']} / 36 | "
            f"{_fmt(target['average_gap'])} "
            f"[{_fmt(target['mean_gap_ci95_lower'])}, "
            f"{_fmt(target['mean_gap_ci95_upper'])}] | "
            f"{_fmt(target['median_gap'])} "
            f"[{_fmt(target['q25_gap'])}, {_fmt(target['q75_gap'])}] |"
        )

    for budget in CROSS_VALIDATION_BUDGETS:
        lines.extend(
            [
                "",
                f"## Top combinations at T=e^{budget_exponent(budget):g} ({budget:,})",
                "",
                "| Rank | Incumbent | Order | Final rule | Mean GAP (95% CI) | Median [Q1, Q3] |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        selected = [row for row in summary_rows if row["total_budget"] == budget][:10]
        for row in selected:
            marker = " **(target)**" if row["is_target_combination"] else ""
            lines.append(
                f"| {row['mean_gap_rank']} | {row['incumbent_label']} | "
                f"{row['replication_order_label']} | "
                f"{row['final_recommendation_label']}{marker} | "
                f"{_fmt(row['average_gap'])} "
                f"[{_fmt(row['mean_gap_ci95_lower'])}, "
                f"{_fmt(row['mean_gap_ci95_upper'])}] | "
                f"{_fmt(row['median_gap'])} "
                f"[{_fmt(row['q25_gap'])}, {_fmt(row['q75_gap'])}] |"
            )

        lines.extend(
            [
                "",
                "### Seven one-factor comparisons",
                "",
                "Negative paired difference favors the target combination.",
                "",
                "| Changed alternative | Competitor mean | Target - competitor (95% CI) | Win / tie / loss | Holm p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        neighbor_rows = [
            row
            for row in comparisons
            if row["total_budget"] == budget and row["is_one_factor_neighbor"]
        ]
        for row in neighbor_rows:
            changed = []
            if row["competitor_incumbent_rule"] != TARGET_INCUMBENT:
                changed.append(f"Incumbent: {row['competitor_incumbent_label']}")
            if row["competitor_replication_order"] != TARGET_ORDER:
                changed.append(f"Order: {row['competitor_replication_order_label']}")
            if row["competitor_final_recommendation_rule"] != TARGET_FINAL:
                changed.append(f"Final: {row['competitor_final_recommendation_label']}")
            lines.append(
                f"| {', '.join(changed)} | {_fmt(row['competitor_average_gap'])} | "
                f"{_fmt(row['target_minus_competitor_mean_gap'])} "
                f"[{_fmt(row['paired_mean_difference_ci95_lower'])}, "
                f"{_fmt(row['paired_mean_difference_ci95_upper'])}] | "
                f"{row['target_win_count']} / {row['tie_count']} / "
                f"{row['target_loss_count']} | "
                f"{_fmt(row['holm_adjusted_p_value'])} |"
            )

    strict_counts = {
        budget: sum(
            bool(row["strict_bootstrap_dominance"])
            for row in comparisons
            if row["total_budget"] == budget
        )
        for budget in CROSS_VALIDATION_BUDGETS
    }
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Validated {validation['validated_trajectory_count']:,} simulated "
            f"trajectories and {validation['validated_recommendation_count']:,} "
            "postprocessed recommendations.",
            "",
            "Strict bootstrap dominance counts (upper 95% paired-difference CI < 0):",
            "",
        ]
    )
    for budget in CROSS_VALIDATION_BUDGETS:
        lines.append(
            f"- `T=e^{budget_exponent(budget):g}`: {strict_counts[budget]} of 35 competitors."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `combination_summary.csv`: 72 cell summaries.",
            "- `all_macro_results.csv`: 7,200 macro-level final recommendations.",
            "- `target_pairwise_comparisons.csv`: target versus all 35 competitors per budget.",
            "- `figures/combination_average_gap_heatmaps.png` and `.pdf`.",
            "- `figures/target_vs_one_factor_neighbors_boxplots.png` and `.pdf`.",
            "- `validation_report.json`: machine-readable validation evidence.",
            "",
        ]
    )
    destination = ANALYSIS_DIR / "REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    gap_vectors, all_rows, validation = collect_and_validate()
    summary_rows, comparisons = summarize(gap_vectors)
    _write_csv(ANALYSIS_DIR / "all_macro_results.csv", all_rows)
    _write_csv(ANALYSIS_DIR / "combination_summary.csv", summary_rows)
    _write_csv(ANALYSIS_DIR / "target_pairwise_comparisons.csv", comparisons)
    with (ANALYSIS_DIR / "validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)
        stream.write("\n")
    heatmap_path = plot_heatmaps(summary_rows)
    neighbor_path = plot_target_neighbors(gap_vectors)
    report_path = write_report(summary_rows, comparisons, validation)

    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study": "4D Matern-5/2 LQEI factorial cross-validation",
        "all_validation_passed": bool(validation["passed"]),
        "trajectory_configuration_count": 24,
        "reported_combination_count": 72,
        "macro_count_per_trajectory_configuration": FORMAL_MACROS,
        "simulated_trajectory_count": validation["validated_trajectory_count"],
        "postprocessed_recommendation_count": validation[
            "validated_recommendation_count"
        ],
        "budgets": list(CROSS_VALIDATION_BUDGETS),
        "analysis_outputs": [
            str(report_path.relative_to(ROOT)),
            str(heatmap_path.relative_to(ROOT)),
            str(neighbor_path.relative_to(ROOT)),
        ],
    }
    with (ROOT / "suite_completion.json").open("w", encoding="utf-8") as stream:
        json.dump(completion, stream, indent=2)
        stream.write("\n")
    print(report_path)


if __name__ == "__main__":
    main()
