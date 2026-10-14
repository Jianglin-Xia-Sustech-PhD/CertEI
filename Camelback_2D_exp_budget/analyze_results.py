"""Aggregate the six-budget Camelback comparison and draw raw-GAP boxplots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from experiment_runner import RESULT_DIR, ExperimentSpec, all_specs
from prepare_inputs import METADATA_PATH, load_function
from study_config import (
    ALGORITHMS,
    BUDGET_EXPONENTS,
    BUDGETS,
    FUNCTION_NAMES,
    NOISE_A_MULTIPLIER,
    NOISE_B_MULTIPLIER,
    REPLICATION_MULTIPLIER,
    RESULT_GROUP,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / RESULT_GROUP
FIGURE_DIR = ROOT / "figures" / RESULT_GROUP

COLORS = {
    "LQEI": "#4C78A8",
    "TSSO": "#F58518",
    "SKO": "#54A24B",
    "EQI": "#E45756",
}


def _read_rows(spec: ExperimentSpec) -> list[dict[str, str]]:
    path = RESULT_DIR / spec.slug / "macro_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != spec.macro_count:
        raise RuntimeError(f"{path} has {len(rows)} rows; expected {spec.macro_count}")
    return rows


def analyze(macro_count: int) -> tuple[Path, Path]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_gaps: dict[tuple[str, int, str], np.ndarray] = {}
    records: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []

    for spec in all_specs(macro_count):
        rows = _read_rows(spec)
        raw = np.asarray([float(row["raw_gap"]) for row in rows])
        normalized = np.asarray([float(row["normalized_gap"]) for row in rows])
        raw_gaps[(spec.function_name, spec.total_budget, spec.algorithm)] = raw
        for row in rows:
            records.append(
                {
                    "function": spec.function_name,
                    "budget": spec.total_budget,
                    "algorithm": spec.algorithm,
                    **row,
                }
            )
        table_rows.append(
            {
                "function": spec.function_name,
                "budget": spec.total_budget,
                "algorithm": spec.algorithm,
                "macros": raw.size,
                "mean_raw_gap": float(np.mean(raw)),
                "sd_raw_gap": float(np.std(raw, ddof=1)),
                "median_raw_gap": float(np.median(raw)),
                "q25_raw_gap": float(np.quantile(raw, 0.25)),
                "q75_raw_gap": float(np.quantile(raw, 0.75)),
                "mean_normalized_gap": float(np.mean(normalized)),
                "median_normalized_gap": float(np.median(normalized)),
            }
        )

    for function_name in FUNCTION_NAMES:
        for budget in BUDGETS:
            matrix = np.vstack(
                [
                    raw_gaps[(function_name, budget, algorithm)]
                    for algorithm in ALGORITHMS
                ]
            )
            ranks = np.column_stack(
                [rankdata(matrix[:, index], method="average") for index in range(macro_count)]
            )
            mean_ranks = np.mean(ranks, axis=1)
            for row in table_rows:
                if row["function"] == function_name and row["budget"] == budget:
                    row["mean_paired_rank"] = float(
                        mean_ranks[ALGORITHMS.index(str(row["algorithm"]))]
                    )

    combined_path = ANALYSIS_DIR / "combined_macro_results.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table_path = ANALYSIS_DIR / "algorithm_summary.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    budget_titles = {
        budget: rf"$T=\mathrm{{round}}(e^{{{exponent:g}}})={budget:,}$"
        for exponent, budget in zip(BUDGET_EXPONENTS, BUDGETS)
    }
    figure, axes = plt.subplots(
        1,
        len(BUDGETS),
        figsize=(30.0, 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, function_name in enumerate(FUNCTION_NAMES):
        function = load_function(function_name)
        for column_index, budget in enumerate(BUDGETS):
            axis = axes[row_index, column_index]
            values = [
                raw_gaps[(function_name, budget, algorithm)]
                for algorithm in ALGORITHMS
            ]
            boxes = axis.boxplot(
                values,
                labels=ALGORITHMS,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "black",
                    "markeredgecolor": "black",
                    "markersize": 4.5,
                },
                medianprops={"color": "black", "linewidth": 1.4},
            )
            for patch, algorithm in zip(boxes["boxes"], ALGORITHMS):
                patch.set_facecolor(COLORS[algorithm])
                patch.set_alpha(0.8)
            axis.set_title(f"{function.display_name} | {budget_titles[budget]}")
            axis.set_ylabel("Raw GAP")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Six-hump Camelback: negative response-linked noise and auxiliary noise GP\n"
        rf"$\sigma/R_f={NOISE_A_MULTIPLIER:.4g}+{NOISE_B_MULTIPLIER:.4g}z$, "
        rf"replications $={REPLICATION_MULTIPLIER:.4g}\log(T)$",
        fontsize=15,
    )
    png_path = FIGURE_DIR / "algorithm_comparison_boxplots_raw_gap.png"
    pdf_path = FIGURE_DIR / "algorithm_comparison_boxplots_raw_gap.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    report = {
        "macro_count_per_configuration": macro_count,
        "budgets": list(BUDGETS),
        "budget_definitions": {
            str(budget): f"round(exp({exponent}))"
            for exponent, budget in zip(BUDGET_EXPONENTS, BUDGETS)
        },
        "combined_results": str(combined_path.relative_to(ROOT)),
        "summary_table": str(table_path.relative_to(ROOT)),
        "boxplot_png": str(png_path.relative_to(ROOT)),
        "boxplot_pdf": str(pdf_path.relative_to(ROOT)),
        "configurations": table_rows,
    }
    (ANALYSIS_DIR / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown_report(table_rows)
    return png_path, table_path


def _write_markdown_report(table_rows: list[dict[str, object]]) -> Path:
    metadata = json.loads(METADATA_PATH.read_text())
    lines = [
        "# Six-hump Camelback negative linear-noise comparison",
        "",
        "SKO and EQI fit an auxiliary Matérn-5/2 GP to accumulated sample "
        "standard deviations and do not use the analytic noise-response relation.  "
        "The noise GP is dynamically re-estimated after every stage.  All GP "
        "outputs, intrinsic-noise variances, kernel variances and calibrated "
        "RKHS norms use the fixed scale `(f - f_min) / (f_max - f_min)`.  "
        "The dynamic GP weakly regularizes log process variance and both log "
        "lengthscales around each macro's initial MLE.  LQEI uses "
        "`0.2 * Lambda_n` for both acquisition and final sampled-point "
        "recommendation and has no dedicated identification stage.  The fixed "
        "candidate set contains 2,048 pure scrambled Sobol points and does not "
        "inject the analytic global minimizers.",
        f"The simulation SD is `R_f * ({NOISE_A_MULTIPLIER:.12g} + "
        f"{NOISE_B_MULTIPLIER:.12g}*z)`; it is used only to generate "
        "observations and its analytic form is not supplied to SKO or EQI.",
        f"The common replication rule is `{REPLICATION_MULTIPLIER:.12g} * log(T)` "
        "with balanced integer batch sizes that exhaust the exact budget.",
        "",
        "## Scale calibration",
        "",
        "| Function | Median initial lengthscales | Median initial kernel variance | B_f |",
        "|---|---:|---:|---:|",
    ]
    for function_name in FUNCTION_NAMES:
        item = metadata["functions"][function_name]
        lengths = ", ".join(f"{value:.4f}" for value in item["reference_lengthscales"])
        lines.append(
            f"| {item['display_name']} | ({lengths}) | "
            f"{item['reference_process_variance']:.6f} | {item['rkhs_norm']:.4f} |"
        )
    lines.extend(["", "## Raw-GAP results", ""])
    for function_name in FUNCTION_NAMES:
        display_name = metadata["functions"][function_name]["display_name"]
        lines.extend([f"### {display_name}", ""])
        for exponent, budget in zip(BUDGET_EXPONENTS, BUDGETS):
            selected = [
                row
                for row in table_rows
                if row["function"] == function_name and row["budget"] == budget
            ]
            selected.sort(key=lambda row: float(row["mean_raw_gap"]))
            lines.extend(
                [
                    f"Budget: `round(exp({exponent:g})) = {budget:,}`",
                    "",
                    "| Algorithm | Mean | Median | Q25 | Q75 | Mean paired rank |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in selected:
                lines.append(
                    f"| {row['algorithm']} | {row['mean_raw_gap']:.6g} | "
                    f"{row['median_raw_gap']:.6g} | {row['q25_raw_gap']:.6g} | "
                    f"{row['q75_raw_gap']:.6g} | {row['mean_paired_rank']:.3f} |"
                )
            lines.append("")
    path = ANALYSIS_DIR / "EXPERIMENT_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    analyze(100)
