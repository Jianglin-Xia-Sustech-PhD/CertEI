"""Re-score completed LQEI paths under three final recommendation rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import FUNCTION_LABELS, _validate_lqei_spec
from experiment_runner import INPUT_DIR, _macro_path
from rkhs_functions import load_fixed_function
from run_experiments import FUNCTION_NAMES, lqei_specs
from study_config import (
    FORMAL_MACROS,
    N_CANDIDATES,
    budget_exponent,
    budgets_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "final_recommendation_ablation"
FIGURE_DIR = ANALYSIS_DIR / "figures"

RULE_CURRENT = "posterior_mean_plus_0p2_lambda_sd"
RULE_MEAN = "posterior_mean"
RULE_MINUS_TEN_SD = "posterior_mean_minus_10_sd"
RULES = (RULE_CURRENT, RULE_MEAN, RULE_MINUS_TEN_SD)
RULE_LABELS = {
    RULE_CURRENT: r"$\mu+0.2\Lambda_{N_T}\sigma$",
    RULE_MEAN: r"$\mu$",
    RULE_MINUS_TEN_SD: r"$\mu-10\sigma$",
}
RULE_SHORT_LABELS = {
    RULE_CURRENT: "+0.2Lambda",
    RULE_MEAN: "posterior mean",
    RULE_MINUS_TEN_SD: "-10 sd",
}
COLORS = {
    RULE_CURRENT: "#0072B2",
    RULE_MEAN: "#E69F00",
    RULE_MINUS_TEN_SD: "#CC79A7",
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
    rule: str,
    budget: int,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "function": function_name,
        "final_recommendation_rule": rule,
        "final_recommendation_label": RULE_SHORT_LABELS[rule],
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


def _coefficient(rule: str, terminal_lambda: float) -> float:
    """Return signed c in the final score mu + c*sigma."""

    if rule == RULE_CURRENT:
        return float(0.2 * terminal_lambda)
    if rule == RULE_MEAN:
        return 0.0
    if rule == RULE_MINUS_TEN_SD:
        return -10.0
    raise ValueError(f"unknown final recommendation rule: {rule}")


def collect_and_validate():
    candidate_path = INPUT_DIR / f"sobol_candidates_2d_{N_CANDIDATES}.npy"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    candidate_pool = np.asarray(np.load(candidate_path), dtype=float)
    if candidate_pool.shape != (N_CANDIDATES, 2):
        raise AssertionError("candidate pool has the wrong shape")

    matrices = {
        (name, rule): np.empty(
            (FORMAL_MACROS, len(budgets_for_function(name))), dtype=float
        )
        for name in FUNCTION_NAMES
        for rule in RULES
    }
    macro_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in lqei_specs():
        stored_gaps, _ = _validate_lqei_spec(spec, "reused_main_lqei")
        function = load_fixed_function(
            ROOT / "fixed_functions" / f"{spec.function_name}.npz"
        )
        budget_column = budgets_for_function(spec.function_name).index(
            spec.total_budget
        )
        configuration_gaps = {
            rule: np.empty(FORMAL_MACROS, dtype=float) for rule in RULES
        }

        for macro in range(1, FORMAL_MACROS + 1):
            path = _macro_path(spec, macro)
            with np.load(path, allow_pickle=False) as data:
                sampled = np.asarray(data["sampled_unique_indices"], dtype=int)
                means = np.asarray(data["sampled_posterior_mean"], dtype=float)
                sds = np.asarray(data["sampled_posterior_sd"], dtype=float)
                initial_points = np.asarray(data["initial_points"], dtype=float)
                terminal_lambda = float(np.asarray(data["lambda_values"])[-1])
                if initial_points.shape != (20, 2):
                    raise AssertionError("initial design has the wrong shape")
                if not (sampled.size == means.size == sds.size):
                    raise AssertionError("saved sampled posterior arrays are misaligned")
                if not np.array_equal(sampled, np.unique(sampled)):
                    raise AssertionError("sampled indices are not sorted and unique")
                if np.any(~np.isfinite(means)) or np.any(~np.isfinite(sds)):
                    raise AssertionError("saved final posterior is not finite")
                if np.any(sds < 0.0):
                    raise AssertionError("saved posterior sd is negative")
                universe = np.vstack((initial_points, candidate_pool))
                if sampled[0] < 0 or sampled[-1] >= universe.shape[0]:
                    raise AssertionError("sampled index lies outside the candidate universe")
                sampled_points = universe[sampled]

                for rule in RULES:
                    coefficient = _coefficient(rule, terminal_lambda)
                    local_index = int(np.argmin(means + coefficient * sds))
                    recommendation_index = int(sampled[local_index])
                    recommendation_x = sampled_points[local_index]
                    recommendation_value = float(
                        function.evaluate(recommendation_x.reshape(1, 2))[0]
                    )
                    gap = max(recommendation_value - function.optimum_value, 0.0)
                    if rule == RULE_CURRENT:
                        if recommendation_index != int(data["recommendation_index"]):
                            raise AssertionError(
                                "recomputed current recommendation index does not match"
                            )
                        if not np.isclose(gap, stored_gaps[macro - 1]):
                            raise AssertionError(
                                "recomputed current recommendation GAP does not match"
                            )
                    configuration_gaps[rule][macro - 1] = gap
                    macro_rows.append(
                        {
                            "provenance": "postprocessed_main_lqei",
                            "function": spec.function_name,
                            "algorithm": "LQEI",
                            "incumbent_rule": "mu_n - 0.2*Lambda_n*sigma_n",
                            "final_recommendation_rule": rule,
                            "final_recommendation_label": RULE_SHORT_LABELS[rule],
                            "final_sd_coefficient": coefficient,
                            "terminal_lambda": terminal_lambda,
                            "total_budget": spec.total_budget,
                            "macro": macro,
                            "raw_gap": gap,
                            "recommendation_index": recommendation_index,
                            "recommendation_x1": float(recommendation_x[0]),
                            "recommendation_x2": float(recommendation_x[1]),
                            "recommendation_posterior_mean": float(means[local_index]),
                            "recommendation_posterior_sd": float(sds[local_index]),
                        }
                    )

        for rule in RULES:
            gaps = configuration_gaps[rule]
            matrices[(spec.function_name, rule)][:, budget_column] = gaps
            summary_rows.append(
                _summary(spec.function_name, rule, spec.total_budget, gaps)
            )

    validation = {
        "passed": True,
        "source_lqei_trajectory_count": len(lqei_specs()) * FORMAL_MACROS,
        "postprocessed_recommendation_count": len(macro_rows),
        "new_simulation_count": 0,
        "final_recommendation_rules": [
            RULE_SHORT_LABELS[rule] for rule in RULES
        ],
        "fixed_components": [
            "EI incumbent: min_x(mu_n - 0.2*Lambda_n*sigma_n)",
            "all LQEI sampling decisions and observations",
            "equation-(20) minimum information-gain Lambda schedule",
            "response GP parameters",
            "rho*log(T) exact-budget batch schedule",
            "initial designs, candidate set, and random-number streams",
            "sampled-point recommendation domain",
        ],
        "checks": [
            "Every main LQEI trajectory exists and uses its exact total budget",
            "Every trajectory uses the registered equation-(20) Lambda schedule",
            "The stored current-rule recommendation is reproduced exactly",
            "All three rules are evaluated on the same saved final posterior",
            "Only the final sampled-point scoring rule changes",
        ],
    }
    return matrices, macro_rows, summary_rows, validation


def plot_boxplots(
    matrices: dict[tuple[str, str], np.ndarray], *, log_scale: bool
) -> Path:
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
            if log_scale:
                axis.set_yscale("log")
            axis.set_title(
                rf"$T=e^{{{budget_exponent(int(budget)):g}}}$" + f"\n({budget:,})"
            )
            axis.tick_params(axis="x", labelrotation=24, labelsize=7.5)
            axis.grid(True, axis="y", alpha=0.22, which="both")
            if column == 0:
                scale_label = " (log scale)" if log_scale else ""
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP{scale_label}")
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
    scale_title = "; logarithmic y-axis" if log_scale else "; linear y-axis"
    figure.suptitle(
        "LQEI final-recommendation-rule ablation (diamond = average GAP)\n"
        r"Incumbent remains $\mu_n-0.2\Lambda_n\sigma_n$"
        + scale_title
    )
    suffix = "log_scale" if log_scale else "linear_scale"
    destination = (
        FIGURE_DIR
        / f"lqei_final_recommendation_rule_ablation_boxplots_{suffix}.png"
    )
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"), dpi=240)
    plt.close(figure)
    return destination


def write_report(
    summary_rows: list[dict[str, object]], validation: dict[str, object]
) -> None:
    lines = [
        "# LQEI final-recommendation-rule ablation",
        "",
        "The LQEI sampling path and EI incumbent remain unchanged, with the "
        "incumbent defined by `min_x(mu_n - 0.2*Lambda_n*sd_n)`. The same saved "
        "terminal posterior from each path is scored by `mu_N + "
        "0.2*Lambda_N*sd_N`, `mu_N`, and `mu_N - 10*sd_N` over sampled points.",
        "",
        f"Validated {int(validation['source_lqei_trajectory_count']):,} source "
        f"LQEI trajectories and {int(validation['postprocessed_recommendation_count']):,} "
        "paired final recommendations. No new simulation was required.",
        "",
        "## Average raw GAP",
        "",
    ]
    for name in FUNCTION_NAMES:
        lines.extend(
            [
                f"### {FUNCTION_LABELS[name]}",
                "",
                "| Budget | +0.2 Lambda | Posterior mean | -10 sd |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in budgets_for_function(name):
            selected = {
                str(row["final_recommendation_rule"]): float(row["average_gap"])
                for row in summary_rows
                if row["function"] == name and row["total_budget"] == budget
            }
            lines.append(
                f"| e^{budget_exponent(budget):g} ({budget:,}) | "
                f"{selected[RULE_CURRENT]:.8f} | "
                f"{selected[RULE_MEAN]:.8f} | "
                f"{selected[RULE_MINUS_TEN_SD]:.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "- Logarithmic and linear-scale boxplots in `figures/` (`.png` and `.pdf`)",
            "- `final_recommendation_ablation_summary.csv`",
            "- `all_recommendations.csv`",
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
    plot_boxplots(matrices, log_scale=True)
    plot_boxplots(matrices, log_scale=False)
    _write_csv(ANALYSIS_DIR / "all_recommendations.csv", macro_rows)
    _write_csv(
        ANALYSIS_DIR / "final_recommendation_ablation_summary.csv", summary_rows
    )
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
        f"Validated {validation['source_lqei_trajectory_count']:,} LQEI paths "
        f"and wrote {validation['postprocessed_recommendation_count']:,} paired "
        "final-rule results.",
        flush=True,
    )


if __name__ == "__main__":
    main()
