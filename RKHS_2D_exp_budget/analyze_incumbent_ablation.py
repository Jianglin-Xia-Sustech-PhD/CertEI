"""Validate and plot the three-rule LQEI incumbent ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import FUNCTION_LABELS, _validate_lqei_spec
from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
from run_experiments import (
    FUNCTION_NAMES,
    incumbent_ablation_specs,
    lqei_specs,
)
from study_config import FORMAL_MACROS, budget_exponent, budgets_for_function


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "incumbent_ablation"
FIGURE_DIR = ANALYSIS_DIR / "figures"
RULES = (
    INCUMBENT_RULE_DYNAMIC,
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
)
RULE_LABELS = {
    INCUMBENT_RULE_DYNAMIC: r"$\mu-0.2\Lambda_n\sigma$",
    INCUMBENT_RULE_MEAN: r"$\mu$",
    INCUMBENT_RULE_UPPER_ONE_SD: r"$\mu+\sigma$",
}
RULE_SHORT_LABELS = {
    INCUMBENT_RULE_DYNAMIC: "-0.2Lambda",
    INCUMBENT_RULE_MEAN: "posterior mean",
    INCUMBENT_RULE_UPPER_ONE_SD: "mean + sd",
}
COLORS = {
    INCUMBENT_RULE_DYNAMIC: "#0072B2",
    INCUMBENT_RULE_MEAN: "#E69F00",
    INCUMBENT_RULE_UPPER_ONE_SD: "#009E73",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    function_name: str,
    rule: str,
    budget: int,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "function": function_name,
        "incumbent_rule": rule,
        "incumbent_label": RULE_SHORT_LABELS[rule],
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


def collect_and_validate():
    main = {
        (spec.function_name, spec.total_budget): spec for spec in lqei_specs()
    }
    additional = {
        (spec.function_name, spec.total_budget, spec.incumbent_rule): spec
        for spec in incumbent_ablation_specs()
    }
    matrices = {
        (name, rule): np.empty(
            (FORMAL_MACROS, len(budgets_for_function(name)))
        )
        for name in FUNCTION_NAMES
        for rule in RULES
    }
    macro_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for name in FUNCTION_NAMES:
        budgets = budgets_for_function(name)
        for rule in RULES:
            for column, budget in enumerate(budgets):
                spec = (
                    main[(name, budget)]
                    if rule == INCUMBENT_RULE_DYNAMIC
                    else additional[(name, budget, rule)]
                )
                provenance = (
                    "reused_main_lqei"
                    if rule == INCUMBENT_RULE_DYNAMIC
                    else "incumbent_ablation"
                )
                gaps, rows = _validate_lqei_spec(spec, provenance)
                matrices[(name, rule)][:, column] = gaps
                for row in rows:
                    row["incumbent_rule"] = rule
                    row["incumbent_label"] = RULE_SHORT_LABELS[rule]
                macro_rows.extend(rows)
                summary_rows.append(_summary(name, rule, budget, gaps))

    validation = {
        "passed": True,
        "macro_count": len(macro_rows),
        "reused_main_macro_count": (
            len(lqei_specs()) * FORMAL_MACROS
        ),
        "new_ablation_macro_count": (
            len(incumbent_ablation_specs()) * FORMAL_MACROS
        ),
        "incumbent_rules": [RULE_SHORT_LABELS[rule] for rule in RULES],
        "fixed_components": [
            "equation-(20) minimum information-gain Lambda schedule",
            "response GP parameters",
            "rho*log(T) exact-budget batch schedule",
            "initial designs, candidate set, and random-number streams",
            "EI formula after defining the incumbent",
            "final sampled-point recommendation: mu_N + 0.2*Lambda_N*sigma_N",
        ],
        "checks": [
            "Every macro result exists and has the registered incumbent rule",
            "Every run consumes its exact total budget",
            "Every run uses the registered equation-(20) minimum-bound Lambda",
            "Only the EI incumbent rule varies across the three arms",
            "Every final recommendation uses the unchanged 0.2*Lambda_N rule",
        ],
    }
    return matrices, macro_rows, summary_rows, validation


def plot_boxplots(matrices: dict[tuple[str, str], np.ndarray]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
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
        budgets = budgets_for_function(name)
        for column, budget in enumerate(budgets):
            axis = axes[row, column]
            values = [matrices[(name, rule)][:, column] for rule in RULES]
            means = [float(np.mean(item)) for item in values]
            box = axis.boxplot(
                values,
                labels=[RULE_LABELS[rule] for rule in RULES],
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
            for patch_box, rule in zip(box["boxes"], RULES):
                patch_box.set_facecolor(COLORS[rule])
                patch_box.set_alpha(0.72)
            axis.set_title(
                rf"$T=e^{{{budget_exponent(int(budget)):g}}}$" + f"\n({budget:,})"
            )
            axis.tick_params(axis="x", labelrotation=24, labelsize=7.5)
            axis.grid(True, axis="y", alpha=0.22)
            if column == 0:
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP")
            text = "\n".join(
                f"{RULE_SHORT_LABELS[rule]}: {mean:.3g}"
                for rule, mean in zip(RULES, means)
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
        "LQEI incumbent-rule ablation (diamond = average GAP)\n"
        "Only the EI incumbent varies; final recommendation remains "
        r"$\mu_N+0.2\Lambda_N\sigma_N$"
    )
    destination = FIGURE_DIR / "lqei_incumbent_rule_ablation_boxplots.png"
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"), dpi=240)
    plt.close(figure)
    return destination


def write_report(
    summary_rows: list[dict[str, object]], validation: dict[str, object]
) -> None:
    lines = [
        "# LQEI incumbent-rule ablation",
        "",
        "The three EI incumbents are the global minima of "
        "`mu - 0.2*Lambda_n*sd`, `mu`, and `mu + sd`. Only this incumbent "
        "definition changes. The final recommendation remains the sampled-point "
        "minimizer of `mu_N + 0.2*Lambda_N*sd_N`.",
        "",
        f"Validated {int(validation['macro_count']):,} macro results, including "
        f"{int(validation['reused_main_macro_count']):,} reused main LQEI results "
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
                "| Budget | -0.2 Lambda | Posterior mean | Mean + sd |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in budgets_for_function(name):
            selected = {
                str(row["incumbent_rule"]): float(row["average_gap"])
                for row in summary_rows
                if row["function"] == name and row["total_budget"] == budget
            }
            lines.append(
                f"| e^{budget_exponent(budget):g} ({budget:,}) | "
                f"{selected[INCUMBENT_RULE_DYNAMIC]:.8f} | "
                f"{selected[INCUMBENT_RULE_MEAN]:.8f} | "
                f"{selected[INCUMBENT_RULE_UPPER_ONE_SD]:.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "- `figures/lqei_incumbent_rule_ablation_boxplots.png` and `.pdf`",
            "- `incumbent_ablation_summary.csv`",
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
    plot_boxplots(matrices)
    _write_csv(ANALYSIS_DIR / "all_macro_results.csv", macro_rows)
    _write_csv(ANALYSIS_DIR / "incumbent_ablation_summary.csv", summary_rows)
    payload = {
        f"gap__{name}__{rule}": values
        for (name, rule), values in matrices.items()
    }
    for name in FUNCTION_NAMES:
        payload[f"budgets__{name}"] = np.asarray(
            budgets_for_function(name), dtype=int
        )
    np.savez_compressed(ANALYSIS_DIR / "gap_matrices.npz", **payload)
    (ANALYSIS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary_rows, validation)
    print(
        f"Validated {validation['macro_count']:,} incumbent-ablation macros "
        "and wrote the boxplots and tables.",
        flush=True,
    )


if __name__ == "__main__":
    main()
