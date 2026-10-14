"""Run and analyze a fixed-budget homoscedastic-noise sensitivity study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

import analyze as base
import experiment as study


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "noise_sensitivity"
CHECKPOINTS = OUTPUT / "checkpoints"
FIGURES = OUTPUT / "figures"
BUDGET = 4915
EXPONENT = 8.5
MACROS = 100
CANDIDATES = 2048
PHASE = 92510
NOISE_GRID = (
    ("noise_0p5", 0.5),
    ("noise_1p0", 1.0),
    ("noise_1p5", 1.5),
    ("noise_2p0", 2.0),
    ("noise_2p5", 2.5),
    ("noise_3p0", 3.0),
)
PLOT_RULES = (
    study.RULE_PLUS,
    study.RULE_MEAN,
    study.RULE_MINUS_SYMMETRIC,
)


def _checkpoint(label: str, macro: int) -> Path:
    return CHECKPOINTS / label / f"macro_{macro:03d}.json"


def _worker(payload: tuple[dict[str, object], str, float, int]) -> list[dict[str, object]]:
    function_data, label, multiplier, macro = payload
    rows = study.run_macro(
        study.objective_from_dict(function_data),
        label,
        BUDGET,
        CANDIDATES,
        macro,
        PHASE,
        continuous=True,
        noise_multiplier=multiplier,
        final_rules=PLOT_RULES,
    )
    return rows


def _checkpoint_is_current(label: str, macro: int) -> bool:
    path = _checkpoint(label, macro)
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return {str(row.get("rule")) for row in rows} == set(PLOT_RULES)


def run(workers: int) -> None:
    function_data = json.loads(
        (HERE / "formal_all_budgets" / "selected_function.json").read_text(
            encoding="utf-8"
        )
    )
    jobs: list[tuple[dict[str, object], str, float, int]] = []
    for label, multiplier in NOISE_GRID:
        for macro in range(1, MACROS + 1):
            if not _checkpoint_is_current(label, macro):
                jobs.append((function_data, label, multiplier, macro))

    print(f"noise-sensitivity paths remaining: {len(jobs)}", flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, payload): (payload[1], payload[3])
                for payload in jobs
            }
            completed = 0
            for future in as_completed(futures):
                label, macro = futures[future]
                rows = future.result()
                destination = _checkpoint(label, macro)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(rows, indent=2) + "\n", encoding="utf-8"
                )
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)}", flush=True)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, _ in NOISE_GRID:
        for macro in range(1, MACROS + 1):
            rows.extend(json.loads(_checkpoint(label, macro).read_text(encoding="utf-8")))
    order = {label: position for position, (label, _) in enumerate(NOISE_GRID)}
    rule_order = {rule: position for position, rule in enumerate(PLOT_RULES)}
    return sorted(
        rows,
        key=lambda row: (
            order[str(row["noise_level"])],
            int(row["macro"]),
            rule_order[str(row["rule"])],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for label, multiplier in NOISE_GRID:
        for rule in PLOT_RULES:
            values = np.asarray(
                [
                    float(row["raw_gap"])
                    for row in rows
                    if row["noise_level"] == label and row["rule"] == rule
                ]
            )
            half_width = float(
                t.ppf(0.975, values.size - 1)
                * np.std(values, ddof=1)
                / np.sqrt(values.size)
            )
            summaries.append(
                {
                    "noise_level": label,
                    "noise_sd_multiplier": multiplier,
                    "total_budget": BUDGET,
                    "rule": rule,
                    "count": values.size,
                    "mean_raw_gap": float(np.mean(values)),
                    "median_raw_gap": float(np.median(values)),
                    "q25_raw_gap": float(np.quantile(values, 0.25)),
                    "q75_raw_gap": float(np.quantile(values, 0.75)),
                    "ci95_lower": float(np.mean(values) - half_width),
                    "ci95_upper": float(np.mean(values) + half_width),
                }
            )
    return summaries


def paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for label, multiplier in NOISE_GRID:
        baseline = {
            int(row["macro"]): float(row["raw_gap"])
            for row in rows
            if row["noise_level"] == label and row["rule"] == study.RULE_PLUS
        }
        for comparison in (study.RULE_MEAN, study.RULE_MINUS_SYMMETRIC):
            other = {
                int(row["macro"]): float(row["raw_gap"])
                for row in rows
                if row["noise_level"] == label and row["rule"] == comparison
            }
            macros = sorted(set(baseline) & set(other))
            differences = np.asarray([other[macro] - baseline[macro] for macro in macros])
            standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
            half_width = float(t.ppf(0.975, differences.size - 1) * standard_error)
            result.append(
                {
                    "noise_level": label,
                    "noise_sd_multiplier": multiplier,
                    "total_budget": BUDGET,
                    "comparison_rule": comparison,
                    "paired_count": differences.size,
                    "mean_difference_comparison_minus_plus_0p2": float(
                        np.mean(differences)
                    ),
                    "ci95_lower": float(np.mean(differences) - half_width),
                    "ci95_upper": float(np.mean(differences) + half_width),
                    "plus_0p2_win_rate": float(np.mean(differences > 0.0)),
                }
            )
    return result


def plot_boxplots(rows: list[dict[str, object]]) -> None:
    positions = np.arange(len(NOISE_GRID), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    figure, axis = plt.subplots(figsize=(13.8, 6.4))
    for offset, rule in zip(offsets, PLOT_RULES, strict=True):
        groups = [
            np.asarray(
                [
                    float(row["raw_gap"])
                    for row in rows
                    if row["noise_level"] == label and row["rule"] == rule
                ]
            )
            for label, _ in NOISE_GRID
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
            s=28,
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
    axis.scatter([], [], marker="D", color="black", s=28, label="mean")
    axis.set_xticks(positions, [f"{multiplier:g}" for _, multiplier in NOISE_GRID])
    axis.set_xlabel(r"Uniform noise standard deviation $\sigma_\epsilon/R_f$")
    axis.set_ylabel("raw GAP")
    axis.set_title(
        rf"4D final-rule ablation across uniform noise levels ($T=e^{{{EXPONENT:g}}}={BUDGET}$)"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", ncol=4, frameon=True)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.90)
    figure.savefig(FIGURES / "final_rule_boxplots_across_noise_levels.png", dpi=220)
    figure.savefig(FIGURES / "final_rule_boxplots_across_noise_levels.pdf")
    plt.close(figure)


def write_report(
    summaries: list[dict[str, object]], comparisons: list[dict[str, object]]
) -> None:
    lines = [
        "# 4D final-rule sensitivity to uniform noise",
        "",
        f"This experiment fixes the total simulation budget at T=round(exp({EXPONENT}))={BUDGET} and varies the spatially uniform Gaussian noise standard deviation from 0.5 R_f to 3.0 R_f. Every noise level contains {MACROS} complete sequential LQEI paths. Only the three requested terminal rules are reported; all rules score the same terminal posterior within each path.",
        "",
        "## Average raw GAP",
        "",
        "| Noise sd / R_f | +0.2 Lambda | posterior mean | -0.2 Lambda |",
        "|---:|---:|---:|---:|",
    ]
    for label, multiplier in NOISE_GRID:
        values = {
            str(row["rule"]): float(row["mean_raw_gap"])
            for row in summaries
            if row["noise_level"] == label
        }
        lines.append(
            f"| {multiplier:g} | {values[study.RULE_PLUS]:.6f} | "
            f"{values[study.RULE_MEAN]:.6f} | "
            f"{values[study.RULE_MINUS_SYMMETRIC]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Complete LQEI paths: {len(NOISE_GRID) * MACROS}.",
            f"- Recommendations retained: {len(NOISE_GRID) * MACROS * len(PLOT_RULES)}.",
            "- Noise is spatially uniform within every path.",
            "- The three terminal rules use the same sampled points and terminal posterior within each path.",
            f"- All paired 95% intervals against +0.2 Lambda are positive: `{all(float(row['ci95_lower']) > 0.0 for row in comparisons)}`.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summaries = summarize(rows)
    comparisons = paired(rows)
    write_csv(OUTPUT / "all_recommendations.csv", rows)
    write_csv(OUTPUT / "summary.csv", summaries)
    write_csv(OUTPUT / "paired_comparisons.csv", comparisons)
    plot_boxplots(rows)
    write_report(summaries, comparisons)
    counts = {
        (label, rule): sum(
            row["noise_level"] == label and row["rule"] == rule for row in rows
        )
        for label, _ in NOISE_GRID
        for rule in PLOT_RULES
    }
    validation = {
        "budget": BUDGET,
        "noise_multipliers": [multiplier for _, multiplier in NOISE_GRID],
        "paths": len(rows) // len(PLOT_RULES),
        "paired_recommendations": len(rows),
        "all_cell_rule_counts_equal_100": set(counts.values()) == {MACROS},
        "uniform_noise": True,
        "same_terminal_posterior": True,
        "full_sequential_acquisition": True,
        "plotted_rules": list(PLOT_RULES),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("run", "analyze", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    if args.mode in ("run", "all"):
        run(args.workers)
    if args.mode in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
