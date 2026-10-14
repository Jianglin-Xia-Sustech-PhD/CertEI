"""Validate and plot the requested 4D LQEI replication-order ablation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import FUNCTION_LABELS, _validate_lqei_spec
from experiment_runner import _initial_points, _macro_path
from run_experiments import ABLATION_ORDERS, FUNCTION_NAMES, ablation_specs
from study_config import (
    ABLATION_BUDGETS,
    ABLATION_N_INITIAL,
    BUDGET_EXPONENTS,
    FORMAL_MACROS,
    RESPONSE_CONSTANT_MEAN,
    RESPONSE_LENGTHSCALE,
    RESPONSE_PROCESS_VARIANCE,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "replication_order_ablation"
FIGURE_DIR = ANALYSIS_DIR / "figures"
ORDERS = tuple(ABLATION_ORDERS)
ORDER_LABELS = {
    "constant": r"$O(1)$",
    "log_log": r"$O(\log\log T)$",
    "log": r"$O(\log T)$",
    "power_1p2": r"$O(T^{1/2})$",
}
ORDER_SHORT_LABELS = {
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
    "constant": "#56B4E9",
    "log_log": "#E69F00",
    "log": "#0072B2",
    "power_1p2": "#D55E00",
}
ABLATION_EXPONENTS = BUDGET_EXPONENTS[-len(ABLATION_BUDGETS):]


def _budget_label(budget: int) -> str:
    exponent = ABLATION_EXPONENTS[ABLATION_BUDGETS.index(budget)]
    return f"e^{exponent:g} (T={budget:,})"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    function_name: str,
    order: str,
    budget: int,
    values: np.ndarray,
    spec,
) -> dict[str, object]:
    return {
        "function": function_name,
        "replication_order": order,
        "order_label": ORDER_SHORT_LABELS[order],
        "batch_formula": ORDER_FORMULAS[order],
        "rho": rho_for_function(function_name),
        "total_budget": budget,
        "initial_point_count": spec.initial_point_count,
        "nominal_replication_number": spec.nominal_batch_size,
        "minimum_actual_replication_number": int(np.min(spec.schedule)),
        "maximum_actual_replication_number": int(np.max(spec.schedule)),
        "number_of_batches": spec.schedule.size,
        "number_of_adaptive_batches": spec.schedule.size - spec.initial_point_count,
        "exact_budget_used": int(np.sum(spec.schedule)),
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


def collect_and_validate(function_names=FUNCTION_NAMES):
    configurations = {
        (spec.function_name, spec.total_budget, spec.replication_order): spec
        for spec in ablation_specs()
    }
    matrices = {
        (name, order): np.empty(
            (FORMAL_MACROS, len(ABLATION_BUDGETS)), dtype=float
        )
        for name in function_names
        for order in ORDERS
    }
    macro_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for name in function_names:
        for order in ORDERS:
            for column, budget in enumerate(ABLATION_BUDGETS):
                spec = configurations[(name, budget, order)]
                files = list(
                    _macro_path(spec, 1).parent.glob("macro_*.npz")
                )
                if len(files) != FORMAL_MACROS:
                    raise AssertionError(
                        f"{spec.slug}: expected exactly {FORMAL_MACROS} files, "
                        f"found {len(files)}"
                    )
                gaps, rows, _ = _validate_lqei_spec(spec, "order_ablation")
                for macro in range(1, FORMAL_MACROS + 1):
                    with np.load(_macro_path(spec, macro), allow_pickle=False) as data:
                        checks = (
                            str(data["replication_order"]) == order,
                            float(data["replication_multiplier"])
                            == spec.replication_multiplier,
                            bool(data["response_hyperparameters_fixed"]),
                            float(data["fitted_process_variance"])
                            == RESPONSE_PROCESS_VARIANCE,
                            float(data["fitted_constant_mean"])
                            == RESPONSE_CONSTANT_MEAN,
                            float(data["fitted_lengthscale"])
                            == RESPONSE_LENGTHSCALE,
                            np.array_equal(
                                data["initial_points"],
                                _initial_points(
                                    macro,
                                    spec.phase_code,
                                    spec.initial_point_count,
                                ),
                            ),
                        )
                        if not all(checks):
                            raise AssertionError(
                                "replication order, rho, GP, or paired initial "
                                f"design mismatch: {_macro_path(spec, macro)}"
                            )
                        log_history = np.asarray(
                            data["log_maximum_lqei_history"], dtype=float
                        )[spec.initial_point_count :]
                        if not np.all(np.isfinite(log_history)):
                            raise AssertionError(
                                "sequential log-EI history is not finite: "
                                f"{_macro_path(spec, macro)}"
                            )
                matrices[(name, order)][:, column] = gaps
                for row in rows:
                    row["replication_order"] = order
                    row["order_label"] = ORDER_SHORT_LABELS[order]
                    row["batch_formula"] = ORDER_FORMULAS[order]
                    row["rho"] = rho_for_function(name)
                    row["initial_point_count"] = spec.initial_point_count
                    row["nominal_replication_number"] = spec.nominal_batch_size
                    row["number_of_batches"] = spec.schedule.size
                    row["number_of_adaptive_batches"] = (
                        spec.schedule.size - spec.initial_point_count
                    )
                macro_rows.extend(rows)
                summary_rows.append(_summary(name, order, budget, gaps, spec))

    validation = {
        "passed": True,
        "macro_count": len(macro_rows),
        "new_ablation_macro_count": len(summary_rows) * FORMAL_MACROS,
        "all_four_orders_rerun_at_exponential_budgets": True,
        "replication_orders": [ORDER_SHORT_LABELS[order] for order in ORDERS],
        "budgets": list(ABLATION_BUDGETS),
        "initial_point_count": ABLATION_N_INITIAL,
        "rho_by_function": {
            name: rho_for_function(name) for name in function_names
        },
        "fixed_components": [
            "objective and heteroscedastic noise model",
            "response GP fixed hyperparameters",
            "equation-(20) minimum information-gain Lambda rule",
            "EI incumbent: min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            "EI formula, 8192-point Sobol screen, and continuous optimizer",
            "20-point initial designs and random-number construction",
            "final recommendation: mu_N + 0.2*Lambda_N*sigma_N",
            "exact total simulation budget",
        ],
        "checks": [
            "Every requested macro result exists exactly once",
            "Every schedule consumes its exact total budget",
            "Every order uses the same numerical function-specific rho",
            "Every run uses the registered equation-(20) minimum-bound Lambda",
            "Every sequential log-EI history is finite",
            "All four orders use the same 20-point initial design protocol",
            "Only the replication-number order changes",
        ],
        "numerical_implementation": [
            "Exact precision-weighted same-site Gaussian sufficient statistics",
            "Adaptive factor/candidate capacity avoids redundant rebuilds",
            "Batched GP queries for the same 10 continuous-optimizer starts",
            "Shared LAPACK-ready factor within each posterior round",
            "Reuse the already cached exact candidate triangular projections",
            "Log-EI fallback only if ordinary EI underflows",
        ],
        "limitations": [
            "Protocol validation does not certify confidence coverage or global acquisition optimality",
            "The kernel-order big-O multiplier remains one and is not finite-sample calibrated",
            "The existing nominal-batch information-gain rule is retained for cross-order comparability",
            "Tiny positive ordinary EI can stop polishing early; log-EI fallback occurs only after underflow",
        ],
    }
    return matrices, macro_rows, summary_rows, validation


def plot_boxplots(
    matrices: dict[tuple[str, str], np.ndarray], *, log_scale: bool
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(FUNCTION_NAMES),
        len(ABLATION_BUDGETS),
        figsize=(15.8, 8.8),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    for row, name in enumerate(FUNCTION_NAMES):
        for column, budget in enumerate(ABLATION_BUDGETS):
            axis = axes[row, column]
            values = [matrices[(name, order)][:, column] for order in ORDERS]
            means = [float(np.mean(item)) for item in values]
            if log_scale and any(np.any(item <= 0.0) for item in values):
                raise AssertionError("log-scale GAP plot requires positive values")
            box = axis.boxplot(
                values,
                labels=[ORDER_LABELS[order] for order in ORDERS],
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 5,
                },
            )
            for patch_box, order in zip(box["boxes"], ORDERS):
                patch_box.set_facecolor(COLORS[order])
                patch_box.set_alpha(0.72)
            if log_scale:
                axis.set_yscale("log")
            axis.set_title(_budget_label(budget))
            axis.tick_params(axis="x", labelrotation=22, labelsize=8)
            axis.grid(True, axis="y", alpha=0.22, which="both")
            if column == 0:
                scale_label = " (log scale)" if log_scale else ""
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP{scale_label}")
            text = "Average GAP\n" + "\n".join(
                f"{ORDER_SHORT_LABELS[order]}: {mean:.3g}"
                for order, mean in zip(ORDERS, means)
            )
            axis.text(
                0.035,
                0.965,
                text,
                transform=axis.transAxes,
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
            )
    scale_title = "logarithmic y-axis" if log_scale else "linear y-axis"
    figure.suptitle(
        "4D LQEI replication-number-order ablation "
        "(diamond = average GAP)\n"
        "Same numerical rho for all orders; common y-scale within each kernel row; "
        + scale_title
    )
    suffix = "log_scale" if log_scale else "linear_scale"
    destination = (
        FIGURE_DIR / f"lqei_replication_order_ablation_boxplots_{suffix}.png"
    )
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"), dpi=240)
    plt.close(figure)
    return destination


def write_report(
    summary_rows: list[dict[str, object]], validation: dict[str, object]
) -> None:
    budget_headers = " | ".join(_budget_label(budget) for budget in ABLATION_BUDGETS)
    separators = "|".join(["---:"] * len(ABLATION_BUDGETS))
    lines = [
        "# 4D LQEI replication-number-order ablation",
        "",
        "Only the replication-number order changes. The schedules are "
        "`rho`, `rho*log(log(T))`, `rho*log(T)`, and `rho*T^(1/2)`. The same numerical "
        "function-specific rho is used without anchor normalization, and every "
        "balanced schedule consumes the exact displayed budget.",
        "",
        f"Validated {int(validation['macro_count']):,} macro results. All four "
        f"orders are run with the same {ABLATION_N_INITIAL}-point initial-design "
        "protocol because the 40-point main design is infeasible for the "
        f"{_budget_label(ABLATION_BUDGETS[0])} square-root schedule.",
        "",
        "The four arms share rho=8 for Matern-5/2 and rho=6 for SE. The "
        "fixed response GP has mean=0, process variance=1, and lengthscale=0.2. "
        "The incumbent is `min(mu_n - 0.2*Lambda_n*sd_n)` and the final rule "
        "is `argmin(mu_N + 0.2*Lambda_N*sd_N)` over sampled points. The "
        "8192-point screen and 10-start L-BFGS-B settings are unchanged.",
        "",
        "All 2,400 macros across four orders were rerun at the exponential "
        "budgets. The constant rule has nominal batch size ten; exact-budget "
        "balancing makes individual batches ten or eleven replications.",
        "",
        "Lambda_n is recomputed for each replication schedule using the same "
        "equation-(20) minimum-bound rule; its numerical values are not frozen "
        "across orders because the batch size and number of posterior updates "
        "are different.",
        "",
        "Long paths use mathematically equivalent same-site observation "
        "compression, adaptive storage, batched GP queries, and a shared "
        "factor copy. These are numerical/implementation optimizations, not "
        "changes to the algorithm or experiment definition. A log-EI fallback "
        "prevents an underflowed acquisition from changing its maximizer.",
        "",
        "Interpretation limitations: this rerun retains the "
        "existing minimum-bound implementation and ordinary-EI polishing "
        "tolerances so all four orders remain comparable. The kernel-order "
        "big-O multiplier is fixed to one, and information-gain bounds use "
        "the nominal batch size rather than the slightly larger balanced "
        "batches where applicable. EI may stop polishing early when its "
        "positive ordinary-scale values/gradients are tiny; log-EI is used "
        "only after ordinary-EI underflow. Protocol validation is not a "
        "confidence-coverage or global-acquisition-optimality certificate. "
        "These are finite-budget results for the current implementation, "
        "not proof of asymptotic replication-order optimality.",
        "",
        "## Average raw GAP",
        "",
        "Boxplots show the median, interquartile range, and 1.5-IQR whiskers. "
        "Outlier markers are hidden for readability; each diamond and annotated "
        "average uses all 100 macro results, including outliers. Each kernel "
        "row uses a common y-axis across budgets.",
        "",
    ]
    for name in FUNCTION_NAMES:
        lines.extend(
            [
                f"### {FUNCTION_LABELS[name]}",
                "",
                f"| Order | rho | {budget_headers} |",
                f"|---|---:|{separators}|",
            ]
        )
        for order in ORDERS:
            selected = sorted(
                (
                    row
                    for row in summary_rows
                    if row["function"] == name
                    and row["replication_order"] == order
                ),
                key=lambda row: int(row["total_budget"]),
            )
            values = " | ".join(
                f"{float(row['average_gap']):.8f}" for row in selected
            )
            lines.append(
                f"| {ORDER_SHORT_LABELS[order]} | "
                f"{rho_for_function(name):g} | {values} |"
            )
        lines.extend(
            [
                "",
                "Nominal replication number / total batches / adaptive batches:",
                "",
                f"| Order | {budget_headers} |",
                f"|---|{separators}|",
            ]
        )
        for order in ORDERS:
            selected = sorted(
                (
                    row
                    for row in summary_rows
                    if row["function"] == name
                    and row["replication_order"] == order
                ),
                key=lambda row: int(row["total_budget"]),
            )
            values = " | ".join(
                f"{int(row['nominal_replication_number'])} / "
                f"{int(row['number_of_batches'])} / "
                f"{int(row['number_of_adaptive_batches'])}"
                for row in selected
            )
            lines.append(f"| {ORDER_SHORT_LABELS[order]} | {values} |")
        lines.append("")
    lines.extend(
        [
            f"At {_budget_label(ABLATION_BUDGETS[0])}, the Matern square-root "
            "schedule has only three adaptive batches after the common "
            "20-point initialization. This is an "
            "important exploration-budget limitation when interpreting that "
            "panel. These ablation runs use 20 initial points, not the "
            "40-point initialization of the main algorithm-comparison study.",
            "",
        ]
    )
    lines.extend(
        [
            "## Outputs",
            "",
            "- Linear and logarithmic boxplots in `figures/` (`.png` and `.pdf`)",
            "- `replication_order_ablation_summary.csv`",
            "- `all_macro_results.csv`",
            "- `gap_matrices.npz`",
            "- `validation_report.json`",
        ]
    )
    (ANALYSIS_DIR / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    matrices, macro_rows, summary_rows, validation = collect_and_validate()
    plot_boxplots(matrices, log_scale=False)
    plot_boxplots(matrices, log_scale=True)
    _write_csv(ANALYSIS_DIR / "all_macro_results.csv", macro_rows)
    _write_csv(
        ANALYSIS_DIR / "replication_order_ablation_summary.csv", summary_rows
    )
    payload = {
        f"gap__{name}__{order}": values
        for (name, order), values in matrices.items()
    }
    payload["budgets"] = np.asarray(ABLATION_BUDGETS, dtype=int)
    np.savez_compressed(ANALYSIS_DIR / "gap_matrices.npz", **payload)
    (ANALYSIS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary_rows, validation)
    (ANALYSIS_DIR / "run_completion.json").write_text(
        json.dumps(
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "passed_validation": True,
                "configuration_count": len(ablation_specs()),
                "macro_count_per_configuration": FORMAL_MACROS,
                "total_macro_count": validation["macro_count"],
                "budgets": list(ABLATION_BUDGETS),
                "orders": list(ORDERS),
                "initial_point_count": ABLATION_N_INITIAL,
                "rho_by_function": validation["rho_by_function"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {validation['macro_count']:,} replication-order macros "
        "and wrote the boxplots and tables.",
        flush=True,
    )


if __name__ == "__main__":
    main()
