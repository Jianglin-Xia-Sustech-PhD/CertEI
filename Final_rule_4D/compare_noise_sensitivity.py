"""Compare the two fixed-budget noise-sensitivity studies."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as base
import experiment as study
import noise_sensitivity as sweep


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "noise_sensitivity_comparison"
FIGURES = OUTPUT / "figures"
STUDIES = (
    (HERE / "noise_sensitivity", 4915, 8.5),
    (HERE / "noise_sensitivity_budget_e10", 22026, 10.0),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_axis(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    *,
    budget: int,
    exponent: float,
) -> None:
    positions = np.arange(len(sweep.NOISE_GRID), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    for offset, rule in zip(offsets, sweep.PLOT_RULES, strict=True):
        groups = [
            np.asarray(
                [
                    float(row["raw_gap"])
                    for row in rows
                    if row["noise_level"] == label and row["rule"] == rule
                ]
            )
            for label, _ in sweep.NOISE_GRID
        ]
        boxes = axis.boxplot(
            groups,
            positions=positions + offset,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(base.COLORS[rule])
            patch.set_alpha(0.72)
        axis.scatter(
            positions + offset,
            [float(np.mean(group)) for group in groups],
            marker="D",
            color="black",
            s=26,
            zorder=5,
        )
        axis.plot(
            [],
            [],
            color=base.COLORS[rule],
            linewidth=9,
            alpha=0.72,
            label=base.DISPLAY[rule],
        )
    axis.set_xticks(
        positions, [f"{multiplier:g}" for _, multiplier in sweep.NOISE_GRID]
    )
    axis.set_ylabel("raw GAP")
    axis.set_title(rf"$T=e^{{{exponent:g}}}={budget}$")
    axis.set_ylim(-0.035, 1.04)
    axis.grid(axis="y", alpha=0.25)


def save_single_panel(
    folder: Path,
    *,
    budget: int,
    exponent: float,
    suffix: str,
) -> None:
    figure, axis = plt.subplots(figsize=(14.2, 5.8))
    draw_axis(
        axis,
        read_rows(folder / "all_recommendations.csv"),
        budget=budget,
        exponent=exponent,
    )
    axis.set_xlabel(r"Uniform noise standard deviation $\sigma_\epsilon/R_f$")
    axis.scatter([], [], marker="D", color="black", s=26, label="mean")
    handles, labels = axis.get_legend_handles_labels()
    figure.suptitle("4D final-rule noise sensitivity", y=0.985)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.75)
    figure.savefig(
        FIGURES / f"final_rule_boxplots_noise_levels_{suffix}.png", dpi=220
    )
    figure.savefig(FIGURES / f"final_rule_boxplots_noise_levels_{suffix}.pdf")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(14.2, 10.4), sharex=True, sharey=True)
    for axis, (folder, budget, exponent) in zip(
        axes, STUDIES, strict=True
    ):
        draw_axis(
            axis,
            read_rows(folder / "all_recommendations.csv"),
            budget=budget,
            exponent=exponent,
        )
    axes[-1].set_xlabel(r"Uniform noise standard deviation $\sigma_\epsilon/R_f$")
    axes[0].scatter([], [], marker="D", color="black", s=26, label="mean")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "4D final-rule noise sensitivity at two representative budgets", y=0.985
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.08, top=0.87, hspace=0.29
    )
    figure.savefig(
        FIGURES / "final_rule_boxplots_noise_levels_two_budgets.png", dpi=220
    )
    figure.savefig(FIGURES / "final_rule_boxplots_noise_levels_two_budgets.pdf")
    plt.close(figure)

    for folder, budget, exponent in STUDIES:
        suffix = "e8p5" if exponent == 8.5 else "e10"
        save_single_panel(
            folder,
            budget=budget,
            exponent=exponent,
            suffix=suffix,
        )


if __name__ == "__main__":
    main()
