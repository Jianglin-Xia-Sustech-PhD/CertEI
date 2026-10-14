"""Validate and plot the four-order LQEI replication-number ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import FUNCTION_LABELS, _validate_lqei_spec
from run_experiments import FUNCTION_NAMES, ablation_specs, lqei_specs
from study_config import (
    ABLATION_BUDGETS,
    FORMAL_MACROS,
    budget_exponent,
    budgets_for_function,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "replication_order_ablation"
FIGURE_DIR = ANALYSIS_DIR / "figures"

ORDER_CONSTANT = "constant"
ORDER_LOG_LOG = "log_log"
ORDER_LOG = "log"
ORDER_SQRT = "power_1p2"
ORDERS = (ORDER_CONSTANT, ORDER_LOG_LOG, ORDER_LOG, ORDER_SQRT)
ORDER_LABELS = {
    ORDER_CONSTANT: r"$O(1)$",
    ORDER_LOG_LOG: r"$O(\log\log T)$",
    ORDER_LOG: r"$O(\log T)$",
    ORDER_SQRT: r"$O(T^{1/2})$",
}
ORDER_SHORT_LABELS = {
    ORDER_CONSTANT: "O(1)",
    ORDER_LOG_LOG: "O(log log T)",
    ORDER_LOG: "O(log T)",
    ORDER_SQRT: "O(T^(1/2))",
}
ORDER_FORMULAS = {
    ORDER_CONSTANT: "rho",
    ORDER_LOG_LOG: "rho*log(log(T))",
    ORDER_LOG: "rho*log(T)",
    ORDER_SQRT: "rho*T^(1/2)",
}
COLORS = {
    ORDER_CONSTANT: "#56B4E9",
    ORDER_LOG_LOG: "#E69F00",
    ORDER_LOG: "#0072B2",
    ORDER_SQRT: "#D55E00",
}


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
        "nominal_replication_number": spec.nominal_batch_size,
        "minimum_actual_replication_number": int(np.min(spec.schedule)),
        "maximum_actual_replication_number": int(np.max(spec.schedule)),
        "number_of_batches": spec.schedule.size,
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


def collect_and_validate():
    main = {
        (spec.function_name, spec.total_budget): spec for spec in lqei_specs()
    }
    additional = {
        (spec.function_name, spec.total_budget, spec.replication_order): spec
        for spec in ablation_specs()
    }
    matrices = {
        (name, order): np.empty(
            (FORMAL_MACROS, len(ABLATION_BUDGETS)), dtype=float
        )
        for name in FUNCTION_NAMES
        for order in ORDERS
    }
    macro_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for name in FUNCTION_NAMES:
        for order in ORDERS:
            for column, budget in enumerate(ABLATION_BUDGETS):
                spec = (
                    main[(name, budget)]
                    if order == ORDER_LOG
                    else additional[(name, budget, order)]
                )
                provenance = (
                    "reused_main_lqei" if order == ORDER_LOG else "order_ablation"
                )
                gaps, rows = _validate_lqei_spec(spec, provenance)
                matrices[(name, order)][:, column] = gaps
                for row in rows:
                    row["replication_order"] = order
                    row["order_label"] = ORDER_SHORT_LABELS[order]
                    row["batch_formula"] = ORDER_FORMULAS[order]
                    row["rho"] = rho_for_function(name)
                    row["nominal_replication_number"] = spec.nominal_batch_size
                    row["number_of_batches"] = spec.schedule.size
                macro_rows.extend(rows)
                summary_rows.append(_summary(name, order, budget, gaps, spec))

    validation = {
        "passed": True,
        "macro_count": len(macro_rows),
        "reused_main_macro_count": (
            len(FUNCTION_NAMES) * len(ABLATION_BUDGETS) * FORMAL_MACROS
        ),
        "new_ablation_macro_count": len(ablation_specs()) * FORMAL_MACROS,
        "replication_orders": [ORDER_SHORT_LABELS[order] for order in ORDERS],
        "budgets": list(ABLATION_BUDGETS),
        "rho_by_function": {
            name: rho_for_function(name) for name in FUNCTION_NAMES
        },
        "fixed_components": [
            "objective and heteroscedastic noise model",
            "response GP fixed hyperparameters",
            "equation-(20) minimum information-gain Lambda rule",
            "EI incumbent: min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            "EI formula and candidate universe",
            "initial designs and random-number construction",
            "final recommendation: mu_N + 0.2*Lambda_N*sigma_N",
            "exact total simulation budget",
        ],
        "checks": [
            "Every requested macro result exists",
            "Every schedule consumes its exact total budget",
            "Every order uses the same numerical function-specific rho",
            "Every run uses the registered equation-(20) minimum-bound Lambda",
            "Every new run has a finite sequential log-EI history, including ordinary floating-point underflow cases",
            "Only the replication-number order changes",
            "The O(log T) benchmark is reused from the main LQEI experiment",
        ],
    }
    return matrices, macro_rows, summary_rows, validation


def plot_boxplots(
    matrices: dict[tuple[str, str], np.ndarray], *, log_scale: bool
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_matern_e11 = {}
    if log_scale:
        source = (
            ROOT.parent
            / "new_913_2d_min_bound"
            / "analysis"
            / "replication_order_ablation"
            / "gap_matrices.npz"
        )
        with np.load(source, allow_pickle=False) as archive:
            source_budgets = np.asarray(archive["budgets"], dtype=int)
            positions = np.flatnonzero(source_budgets == 60_000)
            if positions.size != 1:
                raise ValueError("legacy T=60,000 budget is unavailable")
            for order in (ORDER_LOG_LOG, ORDER_LOG):
                gaps = np.asarray(
                    archive[f"gap__rkhs_matern52_2d__{order}"], dtype=float
                )[:, int(positions[0])]
                if gaps.shape != (FORMAL_MACROS,) or not np.all(np.isfinite(gaps)):
                    raise ValueError(f"invalid legacy Matérn GAP data for {order}")
                legacy_matern_e11[order] = gaps.copy()
    figure, axes = plt.subplots(
        2,
        len(ABLATION_BUDGETS),
        figsize=(13.8, 8.3),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    for row, name in enumerate(FUNCTION_NAMES):
        for column, budget in enumerate(ABLATION_BUDGETS):
            axis = axes[row, column]
            values = [
                legacy_matern_e11[order]
                if (
                    log_scale
                    and name == "rkhs_matern52_2d"
                    and int(budget) == 59_874
                    and order in legacy_matern_e11
                )
                else matrices[(name, order)][:, column]
                for order in ORDERS
            ]
            means = [float(np.mean(item)) for item in values]
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
            axis.set_title(
                rf"$T=e^{{{budget_exponent(int(budget)):g}}}$" + f"\n({budget:,})"
            )
            axis.tick_params(axis="x", labelrotation=22, labelsize=8)
            axis.grid(True, axis="y", alpha=0.22, which="both")
            if column == 0:
                scale_label = " (log scale)" if log_scale else ""
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP{scale_label}")
            if not log_scale:
                text = "\n".join(
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
        "LQEI replication-number-order ablation (diamond = average GAP)\n"
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
    budget_headers = " | ".join(
        f"T=e^{budget_exponent(budget):g} ({budget:,})"
        for budget in ABLATION_BUDGETS
    )
    separators = "|".join(["---:"] * len(ABLATION_BUDGETS))
    lines = [
        "# LQEI replication-number-order ablation",
        "",
        "Only the replication-number order changes. The four schedules are "
        "`rho`, `rho*log(log(T))`, `rho*log(T)`, and `rho*T^(1/2)`. The same "
        "numerical function-specific rho is used without anchor normalization, "
        "and every balanced schedule consumes the exact displayed budget.",
        "",
        f"Validated {int(validation['macro_count']):,} macro results, including "
        f"{int(validation['reused_main_macro_count']):,} reused O(log T) results "
        f"and {int(validation['new_ablation_macro_count']):,} new runs.",
        "",
        "## Average raw GAP",
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
            selected = [
                row
                for row in summary_rows
                if row["function"] == name
                and row["replication_order"] == order
            ]
            selected.sort(key=lambda row: int(row["total_budget"]))
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
                "Nominal replication number / number of batches:",
                "",
                f"| Order | {budget_headers} |",
                f"|---|{separators}|",
            ]
        )
        for order in ORDERS:
            selected = [
                row
                for row in summary_rows
                if row["function"] == name
                and row["replication_order"] == order
            ]
            selected.sort(key=lambda row: int(row["total_budget"]))
            values = " | ".join(
                f"{int(row['nominal_replication_number'])} / "
                f"{int(row['number_of_batches'])}"
                for row in selected
            )
            lines.append(f"| {ORDER_SHORT_LABELS[order]} | {values} |")
        lines.append("")
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
    print(
        f"Validated {validation['macro_count']:,} replication-order macros "
        "and wrote the boxplots and tables.",
        flush=True,
    )


if __name__ == "__main__":
    main()
