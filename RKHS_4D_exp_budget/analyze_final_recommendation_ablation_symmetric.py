"""Re-score saved LQEI paths with symmetric terminal uncertainty corrections.

This module deliberately writes to a new analysis directory so the registered
mu-10-sigma ablation and all of its figures remain untouched.
"""

from __future__ import annotations

import numpy as np

import analyze_final_recommendation_ablation as base


RULE_MINUS_SYMMETRIC = "posterior_mean_minus_0p2_lambda_sd"
ORIGINAL_COLLECT = base.collect_and_validate

base.ANALYSIS_DIR = (
    base.ROOT / "analysis" / "final_recommendation_ablation_symmetric"
)
base.FIGURE_DIR = base.ANALYSIS_DIR / "figures"
base.RULE_MINUS_TEN_SD = RULE_MINUS_SYMMETRIC
base.RULES = (base.RULE_CURRENT, base.RULE_MEAN, RULE_MINUS_SYMMETRIC)
base.RULE_LABELS = {
    base.RULE_CURRENT: r"$\mu+0.2\Lambda_{N_T}\sigma$",
    base.RULE_MEAN: r"$\mu$",
    RULE_MINUS_SYMMETRIC: r"$\mu-0.2\Lambda_{N_T}\sigma$",
}
base.RULE_SHORT_LABELS = {
    base.RULE_CURRENT: "+0.2Lambda",
    base.RULE_MEAN: "posterior mean",
    RULE_MINUS_SYMMETRIC: "-0.2Lambda",
}
base.COLORS = {
    base.RULE_CURRENT: "#0072B2",
    base.RULE_MEAN: "#E69F00",
    RULE_MINUS_SYMMETRIC: "#CC79A7",
}


def symmetric_coefficient(rule: str, terminal_lambda: float) -> float:
    """Return signed c in the final sampled-point score mu + c*sigma."""

    if rule == base.RULE_CURRENT:
        return float(0.2 * terminal_lambda)
    if rule == base.RULE_MEAN:
        return 0.0
    if rule == RULE_MINUS_SYMMETRIC:
        return float(-0.2 * terminal_lambda)
    raise ValueError(f"unknown final recommendation rule: {rule}")


base._coefficient = symmetric_coefficient


def collect_and_validate():
    matrices, macro_rows, summary_rows, validation = ORIGINAL_COLLECT()
    grouped: dict[tuple[str, int, int], dict[str, dict[str, object]]] = {}
    for row in macro_rows:
        key = (
            str(row["function"]),
            int(row["total_budget"]),
            int(row["macro"]),
        )
        grouped.setdefault(key, {})[
            str(row["final_recommendation_rule"])
        ] = row
    expected_paths = int(validation["source_lqei_trajectory_count"])
    if len(grouped) != expected_paths:
        raise AssertionError("the symmetric analysis does not contain every path")
    for recommendations in grouped.values():
        if set(recommendations) != set(base.RULES):
            raise AssertionError("a path does not contain all three final rules")
        plus = float(
            recommendations[base.RULE_CURRENT]["final_sd_coefficient"]
        )
        minus = float(
            recommendations[RULE_MINUS_SYMMETRIC]["final_sd_coefficient"]
        )
        if not np.isclose(plus, -minus, rtol=0.0, atol=1e-12):
            raise AssertionError("positive and negative coefficients are not symmetric")
    validation["symmetric_uncertainty_coefficients"] = True
    validation["original_noise_structure_and_observations_reused"] = True
    validation["original_acquisition_paths_reused"] = True
    validation["output_directory"] = str(base.ANALYSIS_DIR)
    validation["checks"].append(
        "Positive and negative terminal coefficients are pathwise symmetric"
    )
    return matrices, macro_rows, summary_rows, validation


base.collect_and_validate = collect_and_validate


def _budget_text(budget: int) -> str:
    if hasattr(base, "budget_exponent"):
        return f"e^{base.budget_exponent(budget):g} ({budget:,})"
    return str(base.budget_label(budget))


def write_report(
    summary_rows: list[dict[str, object]], validation: dict[str, object]
) -> None:
    dimension = "4D" if any(name.endswith("_4d") for name in base.FUNCTION_NAMES) else "2D"
    lines = [
        f"# {dimension} LQEI symmetric final-recommendation-rule ablation",
        "",
        "The registered LQEI acquisition paths, heterogeneous noise structure, "
        "observations, budgets, GP settings, and EI incumbent are unchanged. "
        "Each saved terminal posterior is rescored over sampled points by "
        "mu_N + 0.2*Lambda_N*sd_N, mu_N, and "
        "mu_N - 0.2*Lambda_N*sd_N.",
        "",
        f"Validated {int(validation['source_lqei_trajectory_count']):,} saved "
        f"LQEI trajectories and wrote "
        f"{int(validation['postprocessed_recommendation_count']):,} paired "
        "recommendations. No acquisition or simulation was rerun.",
        "",
        "## Average raw GAP",
        "",
    ]
    for name in base.FUNCTION_NAMES:
        lines.extend(
            [
                f"### {base.FUNCTION_LABELS[name]}",
                "",
                "| Budget | +0.2 Lambda | Posterior mean | -0.2 Lambda |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in base.budgets_for_function(name):
            selected = {
                str(row["final_recommendation_rule"]): float(row["average_gap"])
                for row in summary_rows
                if row["function"] == name and int(row["total_budget"]) == budget
            }
            lines.append(
                f"| {_budget_text(budget)} | "
                f"{selected[base.RULE_CURRENT]:.8f} | "
                f"{selected[base.RULE_MEAN]:.8f} | "
                f"{selected[RULE_MINUS_SYMMETRIC]:.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            f"- Pathwise symmetric coefficients: "
            f"{validation['symmetric_uncertainty_coefficients']}.",
            "- Original heteroscedastic noise structure and realized observations "
            "were reused unchanged.",
            "- Original acquisition decisions and budget grid were reused unchanged.",
            "- The original mu-10 sigma analysis directory and figures were not "
            "modified.",
            "",
            "## Outputs",
            "",
            "- Linear- and logarithmic-scale boxplots in figures/",
            "- final_recommendation_ablation_summary.csv",
            "- all_recommendations.csv",
            "- gap_matrices.npz",
            "- validation_report.json",
        ]
    )
    (base.ANALYSIS_DIR / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


base.write_report = write_report


if __name__ == "__main__":
    base.main()

