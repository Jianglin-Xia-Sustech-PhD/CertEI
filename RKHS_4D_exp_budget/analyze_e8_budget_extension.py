"""Validate the T=round(exp(8)) extension and create non-destructive figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_final_recommendation_ablation import (
    COLORS as FINAL_COLORS,
    RULES as FINAL_RULES,
    RULE_LABELS as FINAL_LABELS,
    RULE_SHORT_LABELS as FINAL_SHORT_LABELS,
    _coefficient as final_coefficient,
    terminal_posterior_on_sampled_points,
)
from analyze_incumbent_ablation import (
    COLORS as INCUMBENT_COLORS,
    RULES as INCUMBENT_RULES,
    RULE_LABELS as INCUMBENT_LABELS,
    RULE_SHORT_LABELS as INCUMBENT_SHORT_LABELS,
)
from analyze_results import (
    ALGORITHMS,
    COLORS as ALGORITHM_COLORS,
    FUNCTION_LABELS,
    _validate_lqei_spec,
)
from comparison_runner import _macro_path as comparison_macro_path
from e8_budget_extension import (
    ANALYSIS_DIR,
    E8_BUDGET,
    E8_BUDGET_EXPONENT,
    FIGURE_MANIFEST,
    ROOT,
    e8_comparison_specs,
    e8_incumbent_specs,
    e8_lqei_specs,
)
from experiment_runner import (
    INCUMBENT_RULE_DYNAMIC,
    N_CANDIDATES,
    _macro_path,
)
from rkhs_functions import load_fixed_function
from run_experiments import FUNCTION_NAMES
from study_config import (
    CONTINUOUS_POLISH_ENABLED,
    CONTINUOUS_POLISH_MAXITER,
    CONTINUOUS_POLISH_STARTS,
    FORMAL_MACROS,
    N_INITIAL,
    budgets_for_function,
)


FIGURE_DIR = ANALYSIS_DIR / "figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_preexisting_figures() -> int:
    if not FIGURE_MANIFEST.exists():
        raise FileNotFoundError(FIGURE_MANIFEST)
    manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
    figures = manifest["figures"]
    for relative, expected in figures.items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"pre-extension figure disappeared: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(f"pre-extension figure changed: {path}")
    return len(figures)


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
    factor_name: str,
    factor_value: str,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "function": function_name,
        factor_name: factor_value,
        "total_budget": E8_BUDGET,
        "budget_exponent": E8_BUDGET_EXPONENT,
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


def collect_algorithm_comparison():
    matrices: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    main_by_name = {spec.function_name: spec for spec in e8_lqei_specs()}

    for spec in e8_lqei_specs():
        gaps, spec_rows, _ = _validate_lqei_spec(spec, "e8_main")
        matrices[(spec.function_name, "LQEI")] = gaps[:, None]
        rows.extend(spec_rows)
        summaries.append(_summary(spec.function_name, "algorithm", "LQEI", gaps))

    for spec in e8_comparison_specs():
        gaps = np.empty(FORMAL_MACROS, dtype=float)
        reference = main_by_name[spec.function_name]
        initial_budget = int(np.sum(reference.schedule[:N_INITIAL]))
        for macro in range(1, FORMAL_MACROS + 1):
            path = comparison_macro_path(spec, macro)
            reference_path = _macro_path(reference, macro)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data, np.load(
                reference_path, allow_pickle=False
            ) as lqei:
                checks = (
                    str(data["algorithm"]) == spec.algorithm,
                    int(data["total_budget"]) == E8_BUDGET,
                    int(np.sum(data["replications"])) == E8_BUDGET,
                    np.asarray(data["raw_observations"]).size == E8_BUDGET,
                    np.array_equal(data["initial_points"], lqei["initial_points"]),
                    np.array_equal(
                        data["raw_observations"][:initial_budget],
                        lqei["raw_observations"][:initial_budget],
                    ),
                )
                if spec.algorithm == "TSSO":
                    checks += (
                        int(data["tsso_stage_count"])
                        == spec.schedule.size - N_INITIAL,
                        np.array_equal(
                            data["tsso_stage_budgets"], spec.schedule[N_INITIAL:]
                        ),
                    )
                if spec.algorithm == "SKO":
                    checks += (
                        bool(data["response_hyperparameters_fixed"]),
                        bool(data["auxiliary_noise_gp_initial_fit_only"]),
                        float(data["sko_incumbent_sd_multiplier"]) == 1.0,
                        bool(
                            data[
                                "sko_noise_penalty_uses_single_replication_variance"
                            ]
                        ),
                        bool(data["continuous_polish_enabled"])
                        == CONTINUOUS_POLISH_ENABLED,
                        int(data["continuous_polish_starts"])
                        == CONTINUOUS_POLISH_STARTS,
                        int(data["continuous_polish_maxiter"])
                        == CONTINUOUS_POLISH_MAXITER,
                    )
                if not all(checks):
                    raise AssertionError(f"invalid e^8 comparison macro: {path}")
                gap = float(data["raw_gap"])
                if not np.isfinite(gap) or gap < 0.0:
                    raise AssertionError(f"invalid e^8 comparison GAP: {path}")
                gaps[macro - 1] = gap
                rows.append(
                    {
                        "provenance": "e8_comparison",
                        "function": spec.function_name,
                        "algorithm": spec.algorithm,
                        "replication_order": "log",
                        "total_budget": E8_BUDGET,
                        "macro": macro,
                        "raw_gap": gap,
                        "recommendation_x1": float(data["recommendation_x"][0]),
                        "recommendation_x2": float(data["recommendation_x"][1]),
                        "recommendation_x3": float(data["recommendation_x"][2]),
                        "recommendation_x4": float(data["recommendation_x"][3]),
                    }
                )
        matrices[(spec.function_name, spec.algorithm)] = gaps[:, None]
        summaries.append(
            _summary(spec.function_name, "algorithm", spec.algorithm, gaps)
        )
    return matrices, rows, summaries


def collect_incumbent_ablation():
    matrices: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    additional = {
        (spec.function_name, spec.incumbent_rule): spec
        for spec in e8_incumbent_specs()
    }
    main = {spec.function_name: spec for spec in e8_lqei_specs()}
    for name in FUNCTION_NAMES:
        for rule in INCUMBENT_RULES:
            spec = (
                main[name]
                if rule == INCUMBENT_RULE_DYNAMIC
                else additional[(name, rule)]
            )
            provenance = (
                "e8_reused_main_lqei"
                if rule == INCUMBENT_RULE_DYNAMIC
                else "e8_incumbent_ablation"
            )
            gaps, spec_rows, _ = _validate_lqei_spec(spec, provenance)
            matrices[(name, rule)] = gaps[:, None]
            for row in spec_rows:
                row["incumbent_rule"] = rule
                row["incumbent_label"] = INCUMBENT_SHORT_LABELS[rule]
            rows.extend(spec_rows)
            summary = _summary(name, "incumbent_rule", rule, gaps)
            summary["incumbent_label"] = INCUMBENT_SHORT_LABELS[rule]
            summaries.append(summary)
    return matrices, rows, summaries


def collect_final_rule_ablation():
    matrices: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for spec in e8_lqei_specs():
        stored_gaps, _, _ = _validate_lqei_spec(spec, "e8_reused_main_lqei")
        function = load_fixed_function(
            ROOT / "fixed_functions" / f"{spec.function_name}.npz"
        )
        configuration_gaps = {
            rule: np.empty(FORMAL_MACROS, dtype=float) for rule in FINAL_RULES
        }
        for macro in range(1, FORMAL_MACROS + 1):
            path = _macro_path(spec, macro)
            with np.load(path, allow_pickle=False) as data:
                universe = np.asarray(data["universe_points"], dtype=float)
                selected_indices = np.asarray(data["selected_indices"], dtype=int)
                observed_means = np.asarray(data["sample_means"], dtype=float)
                mean_noise_variances = np.asarray(
                    data["clipped_variances"], dtype=float
                ) / np.asarray(data["replications"], dtype=float)
                terminal_lambda = float(np.asarray(data["lambda_values"])[-1])
                sampled, means, sds = terminal_posterior_on_sampled_points(
                    universe=universe,
                    selected_indices=selected_indices,
                    observed_means=observed_means,
                    mean_noise_variances=mean_noise_variances,
                    kernel=function.kernel,
                    lengthscale=float(data["fitted_lengthscale"]),
                    process_variance=float(data["fitted_process_variance"]),
                    constant_mean=float(data["fitted_constant_mean"]),
                )
                sampled_points = universe[sampled]
                for rule in FINAL_RULES:
                    coefficient = final_coefficient(rule, terminal_lambda)
                    local_index = int(np.argmin(means + coefficient * sds))
                    recommendation_index = int(sampled[local_index])
                    recommendation_x = sampled_points[local_index]
                    recommendation_value = float(
                        function.evaluate(recommendation_x.reshape(1, 4))[0]
                    )
                    gap = max(recommendation_value - function.optimum_value, 0.0)
                    if rule == FINAL_RULES[0]:
                        checks = (
                            recommendation_index == int(data["recommendation_index"]),
                            np.isclose(
                                means[local_index],
                                float(data["recommendation_posterior_mean"]),
                                rtol=2e-9,
                                atol=2e-10,
                            ),
                            np.isclose(
                                sds[local_index],
                                float(data["recommendation_posterior_sd"]),
                                rtol=2e-8,
                                atol=2e-10,
                            ),
                            np.isclose(gap, stored_gaps[macro - 1]),
                        )
                        if not all(checks):
                            raise AssertionError(
                                f"e^8 terminal-posterior recovery mismatch: {path}"
                            )
                    configuration_gaps[rule][macro - 1] = gap
                    rows.append(
                        {
                            "provenance": "e8_postprocessed_main_lqei",
                            "function": spec.function_name,
                            "algorithm": "LQEI",
                            "incumbent_rule": "mu_n - 0.2*Lambda_n*sigma_n",
                            "final_recommendation_rule": rule,
                            "final_recommendation_label": FINAL_SHORT_LABELS[rule],
                            "final_sd_coefficient": coefficient,
                            "terminal_lambda": terminal_lambda,
                            "total_budget": E8_BUDGET,
                            "macro": macro,
                            "raw_gap": gap,
                            "recommendation_index": recommendation_index,
                            "recommendation_x1": float(recommendation_x[0]),
                            "recommendation_x2": float(recommendation_x[1]),
                            "recommendation_x3": float(recommendation_x[2]),
                            "recommendation_x4": float(recommendation_x[3]),
                            "recommendation_posterior_mean": float(
                                means[local_index]
                            ),
                            "recommendation_posterior_sd": float(sds[local_index]),
                        }
                    )
        for rule, gaps in configuration_gaps.items():
            matrices[(spec.function_name, rule)] = gaps[:, None]
            summary = _summary(
                spec.function_name, "final_recommendation_rule", rule, gaps
            )
            summary["final_recommendation_label"] = FINAL_SHORT_LABELS[rule]
            summaries.append(summary)
    return matrices, rows, summaries


def _load_original_algorithm_matrices():
    path = ROOT / "analysis" / "main_and_comparison_macro_results.csv"
    original_budgets = tuple(budgets_for_function(FUNCTION_NAMES[0]))
    grouped = {
        (name, algorithm, budget): []
        for name in FUNCTION_NAMES
        for algorithm in ALGORITHMS
        for budget in original_budgets
    }
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["function"],
                row["algorithm"],
                int(row["total_budget"]),
            )
            if key in grouped:
                grouped[key].append(float(row["raw_gap"]))
    matrices = {}
    for name in FUNCTION_NAMES:
        for algorithm in ALGORITHMS:
            columns = []
            for budget in original_budgets:
                values = np.asarray(grouped[(name, algorithm, budget)], dtype=float)
                if values.size != FORMAL_MACROS:
                    raise AssertionError(
                        f"expected {FORMAL_MACROS} original rows for "
                        f"{name}/{algorithm}/T={budget}; found {values.size}"
                    )
                columns.append(values)
            matrices[(name, algorithm)] = np.column_stack(columns)
    return matrices


def _load_original_npz_matrices(
    relative: str, rules: tuple[str, ...]
) -> dict[tuple[str, str], np.ndarray]:
    matrices = {}
    with np.load(ROOT / relative, allow_pickle=False) as data:
        for name in FUNCTION_NAMES:
            expected_budgets = np.asarray(budgets_for_function(name), dtype=int)
            np.testing.assert_array_equal(data[f"budgets__{name}"], expected_budgets)
            for rule in rules:
                values = np.asarray(data[f"gap__{name}__{rule}"], dtype=float)
                if values.shape != (FORMAL_MACROS, expected_budgets.size):
                    raise AssertionError(
                        f"unexpected original matrix shape for {name}/{rule}"
                    )
                matrices[(name, rule)] = values
    return matrices


def _prepend_extension(original, extension, factors):
    return {
        (name, factor): np.column_stack(
            [extension[(name, factor)], original[(name, factor)]]
        )
        for name in FUNCTION_NAMES
        for factor in factors
    }


def plot_augmented_boxplots(
    matrices: dict[tuple[str, str], np.ndarray],
    factors: tuple[str, ...],
    labels: dict[str, str] | dict[str, object],
    colors: dict[str, str],
    title: str,
    stem: str,
    *,
    log_scale: bool = False,
    annotation_labels: dict[str, str] | None = None,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    budgets = (E8_BUDGET, *budgets_for_function(FUNCTION_NAMES[0]))
    exponents = (E8_BUDGET_EXPONENT, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5)
    figure, axes = plt.subplots(
        len(FUNCTION_NAMES),
        len(budgets),
        figsize=(3.05 * len(budgets), 8.1),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    for row_index, name in enumerate(FUNCTION_NAMES):
        for column, (budget, exponent) in enumerate(zip(budgets, exponents)):
            axis = axes[row_index, column]
            values = [matrices[(name, factor)][:, column] for factor in factors]
            box = axis.boxplot(
                values,
                labels=[labels[factor] for factor in factors],
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
            for patch, factor in zip(box["boxes"], factors):
                patch.set_facecolor(colors[factor])
                patch.set_alpha(0.72)
            if log_scale:
                if any(np.any(item <= 0.0) for item in values):
                    axis.set_yscale("symlog", linthresh=1e-6)
                else:
                    axis.set_yscale("log")
            axis.set_title(f"e^{exponent:g} (T={budget:,})")
            axis.tick_params(axis="x", labelrotation=26, labelsize=7.2)
            axis.grid(True, axis="y", alpha=0.22, which="both")
            if column == 0:
                scale = " (log scale)" if log_scale else ""
                axis.set_ylabel(f"{FUNCTION_LABELS[name]}\nRaw GAP{scale}")
            means = [float(np.mean(item)) for item in values]
            shown_labels = annotation_labels or {
                factor: factor for factor in factors
            }
            axis.text(
                0.035,
                0.965,
                "\n".join(
                    f"{shown_labels[factor]}: {mean:.3g}"
                    for factor, mean in zip(factors, means)
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=6.6,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    figure.suptitle(title)
    destination = FIGURE_DIR / f"{stem}.png"
    figure.savefig(destination, dpi=240)
    figure.savefig(destination.with_suffix(".pdf"), dpi=240)
    plt.close(figure)
    return destination


def write_report(
    algorithm_summary: list[dict[str, object]],
    incumbent_summary: list[dict[str, object]],
    final_summary: list[dict[str, object]],
    protected_figure_count: int,
) -> None:
    def average(rows, name, key, value):
        return next(
            float(row["average_gap"])
            for row in rows
            if row["function"] == name and row[key] == value
        )

    lines = [
        "# 4D e^8 budget extension",
        "",
        f"Added `T=round(exp(8))={E8_BUDGET:,}` with {FORMAL_MACROS} macros per "
        "configuration. Existing registered figures were not overwritten.",
        "",
        f"Verified unchanged SHA-256 hashes for {protected_figure_count} pre-existing "
        "PNG/PDF figures.",
        "",
        "## Average raw GAP at e^8",
        "",
    ]
    for name in FUNCTION_NAMES:
        lines.extend(
            [
                f"### {FUNCTION_LABELS[name]}",
                "",
                "| Experiment | Arm | Average GAP |",
                "|---|---|---:|",
            ]
        )
        for algorithm in ALGORITHMS:
            lines.append(
                f"| Algorithm | {algorithm} | "
                f"{average(algorithm_summary, name, 'algorithm', algorithm):.8f} |"
            )
        for rule in INCUMBENT_RULES:
            lines.append(
                f"| EI incumbent | {INCUMBENT_SHORT_LABELS[rule]} | "
                f"{average(incumbent_summary, name, 'incumbent_rule', rule):.8f} |"
            )
        for rule in FINAL_RULES:
            lines.append(
                f"| Final rule | {FINAL_SHORT_LABELS[rule]} | "
                f"{average(final_summary, name, 'final_recommendation_rule', rule):.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "- `figures/`: new 2x8 augmented subplot grids in PNG and PDF",
            "- `algorithm_comparison_*_e8.csv`: four-algorithm e^8 data",
            "- `incumbent_ablation_*_e8.csv`: three-incumbent e^8 data",
            "- `final_rule_ablation_*_e8.csv`: paired final-rule e^8 data",
            "- `gap_matrices_e8.npz`: all new GAP matrices",
            "- `validation_report.json`: completeness and protocol checks",
            "- `preexisting_figure_sha256.json`: immutability manifest",
        ]
    )
    (ANALYSIS_DIR / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    protected_figure_count = validate_preexisting_figures()
    algorithm, algorithm_rows, algorithm_summary = collect_algorithm_comparison()
    incumbent, incumbent_rows, incumbent_summary = collect_incumbent_ablation()
    final, final_rows, final_summary = collect_final_rule_ablation()

    original_algorithm = _load_original_algorithm_matrices()
    original_incumbent = _load_original_npz_matrices(
        "analysis/incumbent_ablation/gap_matrices.npz", INCUMBENT_RULES
    )
    original_final = _load_original_npz_matrices(
        "analysis/final_recommendation_ablation/gap_matrices.npz", FINAL_RULES
    )
    augmented_algorithm = _prepend_extension(
        original_algorithm, algorithm, ALGORITHMS
    )
    augmented_incumbent = _prepend_extension(
        original_incumbent, incumbent, INCUMBENT_RULES
    )
    augmented_final = _prepend_extension(original_final, final, FINAL_RULES)

    algorithm_labels = {algorithm_name: algorithm_name for algorithm_name in ALGORITHMS}
    plot_augmented_boxplots(
        augmented_algorithm,
        ALGORITHMS,
        algorithm_labels,
        ALGORITHM_COLORS,
        "Same-budget 4D algorithm comparison with the e^8 extension\n"
        "Original seven-budget figures remain unchanged",
        "algorithm_comparison_boxplots_with_e8",
    )
    plot_augmented_boxplots(
        augmented_incumbent,
        INCUMBENT_RULES,
        INCUMBENT_LABELS,
        INCUMBENT_COLORS,
        "4D LQEI incumbent-rule ablation with the e^8 extension\n"
        "Only the EI incumbent varies",
        "incumbent_rule_ablation_boxplots_with_e8",
        annotation_labels=INCUMBENT_SHORT_LABELS,
    )
    plot_augmented_boxplots(
        augmented_final,
        FINAL_RULES,
        FINAL_LABELS,
        FINAL_COLORS,
        "4D LQEI final-recommendation-rule ablation with the e^8 extension\n"
        "All acquisition paths are fixed",
        "final_rule_ablation_boxplots_with_e8_linear_scale",
        annotation_labels=FINAL_SHORT_LABELS,
    )
    plot_augmented_boxplots(
        augmented_final,
        FINAL_RULES,
        FINAL_LABELS,
        FINAL_COLORS,
        "4D LQEI final-recommendation-rule ablation with the e^8 extension\n"
        "Logarithmic y-scale; all acquisition paths are fixed",
        "final_rule_ablation_boxplots_with_e8_log_scale",
        log_scale=True,
        annotation_labels=FINAL_SHORT_LABELS,
    )

    _write_csv(ANALYSIS_DIR / "algorithm_comparison_macro_results_e8.csv", algorithm_rows)
    _write_csv(ANALYSIS_DIR / "algorithm_comparison_summary_e8.csv", algorithm_summary)
    _write_csv(ANALYSIS_DIR / "incumbent_ablation_macro_results_e8.csv", incumbent_rows)
    _write_csv(ANALYSIS_DIR / "incumbent_ablation_summary_e8.csv", incumbent_summary)
    _write_csv(ANALYSIS_DIR / "final_rule_ablation_recommendations_e8.csv", final_rows)
    _write_csv(ANALYSIS_DIR / "final_rule_ablation_summary_e8.csv", final_summary)

    payload = {"budget": np.asarray(E8_BUDGET, dtype=int)}
    for prefix, matrices in (
        ("algorithm", algorithm),
        ("incumbent", incumbent),
        ("final_rule", final),
    ):
        for (name, factor), values in matrices.items():
            payload[f"gap__{prefix}__{name}__{factor}"] = values
    np.savez_compressed(ANALYSIS_DIR / "gap_matrices_e8.npz", **payload)

    validation = {
        "passed": True,
        "budget_exponent": E8_BUDGET_EXPONENT,
        "total_budget": E8_BUDGET,
        "macro_count_per_configuration": FORMAL_MACROS,
        "main_lqei_macro_count": len(e8_lqei_specs()) * FORMAL_MACROS,
        "comparison_macro_count": len(e8_comparison_specs()) * FORMAL_MACROS,
        "incumbent_additional_macro_count": (
            len(e8_incumbent_specs()) * FORMAL_MACROS
        ),
        "final_rule_postprocessed_recommendation_count": len(final_rows),
        "final_rule_new_simulation_count": 0,
        "preexisting_figure_count_verified_unchanged": protected_figure_count,
        "new_augmented_figure_count": 8,
        "checks": [
            "Every requested e^8 macro exists exactly once",
            "Every run consumes exactly 2,981 simulation replications",
            "All four algorithms share paired initial observations and the same O(log T) schedule",
            "Only the EI incumbent varies in the incumbent ablation",
            "Final-rule alternatives reuse each completed main LQEI terminal posterior",
            "All pre-existing PNG/PDF figure hashes remain unchanged",
        ],
    }
    (ANALYSIS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        algorithm_summary,
        incumbent_summary,
        final_summary,
        protected_figure_count,
    )
    print(
        f"Validated the e^8 extension and protected {protected_figure_count} "
        "pre-existing figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
