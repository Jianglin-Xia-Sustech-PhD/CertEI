"""Validate completed main/comparison runs and draw their figures.

This entry point deliberately excludes replication-order ablations so the
completed O(log T) study can be inspected while ablation workers keep running.
"""

from __future__ import annotations

import json

import analyze_results as analysis


def main() -> None:
    # collect_and_validate() is also used by the final all-results analysis.
    # Emptying only the ablation registry leaves its strict main/comparison
    # validation unchanged while avoiding reads of incomplete ablation runs.
    analysis.ablation_specs = lambda: []
    analysis.ABLATION_ORDERS = ()

    (
        algorithm_matrices,
        _,
        macro_rows,
        algorithm_summary,
        _,
        lambda_rows,
        validation,
    ) = analysis.collect_and_validate()
    slopes = analysis.plot_figures(algorithm_matrices)

    analysis._write_csv(
        analysis.ANALYSIS_DIR / "main_and_comparison_macro_results.csv",
        macro_rows,
    )
    analysis._write_csv(
        analysis.ANALYSIS_DIR / "algorithm_comparison_summary.csv",
        algorithm_summary,
    )
    analysis._write_csv(
        analysis.ANALYSIS_DIR / "main_lambda_schedule_summary.csv",
        lambda_rows,
    )
    analysis._write_csv(
        analysis.ANALYSIS_DIR / "convergence_slopes.csv",
        slopes,
    )
    validation_path = (
        analysis.ANALYSIS_DIR / "main_and_comparison_validation_report.json"
    )
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {validation['total_macro_count']:,} completed main/comparison "
        "macros and wrote both requested figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
