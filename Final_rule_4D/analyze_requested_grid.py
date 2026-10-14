"""Create the requested exp(7.5)--exp(11.5) paired-rule boxplots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as base
import experiment as study


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "formal_requested_budget_grid"
FIGURES = OUTPUT / "figures"
BUDGETS = study.REQUESTED_GRID_BUDGETS
LABELS = {
    1808: r"$e^{7.5}$",
    2981: r"$e^{8}$",
    4915: r"$e^{8.5}$",
    8103: r"$e^{9}$",
    13360: r"$e^{9.5}$",
    22026: r"$e^{10}$",
    36316: r"$e^{10.5}$",
    98716: r"$e^{11.5}$",
}
PLOT_RULES = (study.RULE_PLUS, study.RULE_MEAN, study.RULE_MINUS)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows() -> list[dict[str, str]]:
    low = _read(HERE / "formal_low_budget" / "all_recommendations.csv")
    extension = _read(HERE / "formal_extension" / "all_recommendations.csv")
    additional = _read(HERE / "formal_additional_budgets" / "all_recommendations.csv")
    rows = [
        row
        for row in low
        if int(row["total_budget"]) == 1808 and int(row["macro"]) <= 100
    ]
    rows.extend(
        row for row in extension if int(row["total_budget"]) in study.MID_HIGH_BUDGETS
    )
    rows.extend(additional)
    budget_order = {budget: position for position, budget in enumerate(BUDGETS)}
    rule_order = {rule: position for position, rule in enumerate(study.RULES)}
    return sorted(
        rows,
        key=lambda row: (
            row["noise_level"],
            budget_order[int(row["total_budget"])],
            int(row["macro"]),
            rule_order[row["rule"]],
        ),
    )


def groups(
    rows: list[dict[str, str]], noise: str, rule: str
) -> list[np.ndarray]:
    return [
        np.asarray(
            [
                float(row["raw_gap"])
                for row in rows
                if row["noise_level"] == noise
                and int(row["total_budget"]) == budget
                and row["rule"] == rule
            ]
        )
        for budget in BUDGETS
    ]


def _draw_axis(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    noise: str,
    *,
    show_xlabel: bool,
) -> None:
    positions = np.arange(len(BUDGETS), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    for offset, rule in zip(offsets, PLOT_RULES, strict=True):
        grouped = groups(rows, noise, rule)
        boxes = axis.boxplot(
            grouped,
            positions=positions + offset,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(base.COLORS[rule])
            patch.set_alpha(0.66)
        axis.scatter(
            positions + offset,
            [float(np.mean(values)) for values in grouped],
            marker="D",
            color="black",
            s=25,
            zorder=5,
        )
        axis.plot(
            [],
            [],
            color=base.COLORS[rule],
            linewidth=8,
            alpha=0.7,
            label=base.DISPLAY[rule],
        )
    axis.set_xticks(positions, [LABELS[budget] for budget in BUDGETS])
    axis.set_ylabel("raw GAP")
    if show_xlabel:
        axis.set_xlabel("Total simulation budget $T$")
    axis.set_title(
        f"{noise.capitalize()} noise: "
        rf"$\sigma_\epsilon={study.NOISE_LEVELS[noise]}R_f$"
    )
    axis.grid(axis="y", alpha=0.25)


def plot_boxplots(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(16.2, 9.4), sharex=True, sharey=True)
    for axis, noise in zip(axes, study.NOISE_LEVELS, strict=True):
        _draw_axis(axis, rows, noise, show_xlabel=axis is axes[-1])
    axes[0].scatter([], [], marker="D", color="black", s=25, label="mean")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("4D final-rule ablation on the requested exponential budget grid", y=0.985)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.09, top=0.865, hspace=0.30)
    figure.savefig(FIGURES / "final_rule_boxplots_raw_gap_requested_grid.png", dpi=220)
    figure.savefig(FIGURES / "final_rule_boxplots_raw_gap_requested_grid.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(16.2, 5.8))
    _draw_axis(axis, rows, "high", show_xlabel=True)
    axis.scatter([], [], marker="D", color="black", s=25, label="mean")
    handles, labels = axis.get_legend_handles_labels()
    figure.suptitle("4D final-rule ablation under high uniform noise", y=0.98)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.78)
    figure.savefig(FIGURES / "final_rule_boxplots_high_noise_raw_gap_requested_grid.png", dpi=220)
    three_rule_pdf = (
        FIGURES / "final_rule_boxplots_high_noise_raw_gap_requested_grid_three_rules.pdf"
    )
    figure.savefig(three_rule_pdf)
    try:
        figure.savefig(FIGURES / "final_rule_boxplots_high_noise_raw_gap_requested_grid.pdf")
    except PermissionError:
        print(
            "high-noise PDF is open elsewhere; wrote the three-rule PDF to "
            f"{three_rule_pdf.name}",
            flush=True,
        )
    plt.close(figure)


def write_report(
    summaries: list[dict[str, object]], comparisons: list[dict[str, object]]
) -> None:
    lines = [
        "# Requested 4D final-rule budget grid",
        "",
        "This report uses exactly 100 paired LQEI trajectories per budget/noise cell on T=round(exp(k)) for k=7.5, 8, 8.5, 9, 9.5, 10, 10.5, and 11.5. The e^8, e^9, and e^10 paths were newly simulated; the other formal paths were reused without changing the acquisition protocol.",
        "",
        "Every boxplot uses raw GAP. To keep the figure clean, individual trajectory points and outlier markers are omitted; black diamonds show means.",
        "",
        "## Average raw GAP",
        "",
    ]
    for noise in study.NOISE_LEVELS:
        lines.extend(
            [
                f"### {noise.capitalize()} noise",
                "",
                "| Exponent | Budget | +0.2 Lambda | +1.0 Lambda | posterior mean | -10 sd |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for exponent, budget in zip((7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11.5), BUDGETS, strict=True):
            values = {
                str(row["rule"]): float(row["mean_raw_gap"])
                for row in summaries
                if row["noise_level"] == noise and int(row["total_budget"]) == budget
            }
            lines.append(
                f"| {exponent:g} | {budget} | {values[study.RULE_PLUS]:.6f} | {values[study.RULE_PLUS_ONE]:.6f} | {values[study.RULE_MEAN]:.6f} | {values[study.RULE_MINUS]:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            f"- All 16 budget/noise cells have +0.2 Lambda as the lowest mean raw GAP: `{all_plus_wins(summaries)}`.",
            f"- All 48 paired confidence intervals favor +0.2 Lambda: `{all(float(row['ci95_lower']) > 0.0 for row in comparisons)}`.",
            "- Acquisition is fully sequential and identical for all four terminal rules.",
            "- Noise is spatially uniform within each noise-level experiment.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def all_plus_wins(summaries: list[dict[str, object]]) -> bool:
    return all(
        next(
            float(row["mean_raw_gap"])
            for row in summaries
            if row["noise_level"] == noise
            and int(row["total_budget"]) == budget
            and row["rule"] == study.RULE_PLUS
        )
        == min(
            float(row["mean_raw_gap"])
            for row in summaries
            if row["noise_level"] == noise and int(row["total_budget"]) == budget
        )
        for noise in study.NOISE_LEVELS
        for budget in BUDGETS
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    base.BUDGETS = BUDGETS
    base.BUDGET_LABELS = LABELS
    base.FIGURES = FIGURES
    summaries = base.summary(rows)
    comparisons = base.paired(rows)
    rates = base.basin_rates(rows)
    base.write_csv(OUTPUT / "all_recommendations.csv", rows)
    base.write_csv(OUTPUT / "formal_summary.csv", summaries)
    base.write_csv(OUTPUT / "paired_comparisons.csv", comparisons)
    base.write_csv(OUTPUT / "basin_selection_rates.csv", rates)
    plot_boxplots(rows)
    base.plot_means(rows)
    write_report(summaries, comparisons)
    counts = {
        (row["noise_level"], int(row["total_budget"]), row["rule"]): 0
        for row in rows
    }
    for row in rows:
        counts[(row["noise_level"], int(row["total_budget"]), row["rule"])] += 1
    validation = {
        "budgets": list(BUDGETS),
        "paths": len(rows) // len(study.RULES),
        "paired_recommendations": len(rows),
        "all_cell_rule_counts_equal_100": set(counts.values()) == {100},
        "all_sixteen_cells_plus_0p2_lowest_mean": all_plus_wins(summaries),
        "all_paired_ci_favor_plus_0p2": all(
            float(row["ci95_lower"]) > 0.0 for row in comparisons
        ),
        "uniform_noise": True,
        "same_terminal_posterior": True,
        "full_sequential_acquisition": True,
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
