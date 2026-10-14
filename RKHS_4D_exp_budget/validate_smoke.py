"""Validate the pre-registered SE stability smoke run."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from confidence import INFORMATION_GAIN_METHOD
from experiment_runner import ROOT, _macro_path
from run_experiments import smoke_specs
from study_config import (
    LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
    LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
)


OUTPUT_DIR = ROOT / "analysis" / "smoke_validation"


def _validate_spec(spec: object) -> dict[str, object]:
    paths = [_macro_path(spec, macro) for macro in range(1, spec.macro_count + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} smoke macro files")

    total_sequential = 0
    exact_zero = 0
    nonpositive = 0
    tiny = 0
    final_initial = 0
    final_index_zero = 0
    raw_gaps: list[float] = []
    terminal_coefficients: list[float] = []
    selected_unique_counts: list[int] = []
    terminal_active_bounds: list[str] = []

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["confidence_method"]) != INFORMATION_GAIN_METHOD:
                raise AssertionError(f"unexpected confidence method: {path}")
            if not np.isclose(
                float(data["acquisition_lambda_multiplier"]),
                LQEI_ACQUISITION_LAMBDA_MULTIPLIER,
            ):
                raise AssertionError(f"unexpected acquisition multiplier: {path}")
            if not np.isclose(
                float(data["recommendation_lambda_multiplier"]),
                LQEI_RECOMMENDATION_LAMBDA_MULTIPLIER,
            ):
                raise AssertionError(f"unexpected recommendation multiplier: {path}")
            expected = np.minimum(
                np.asarray(data["kernel_order_upper_bound"], dtype=float),
                np.asarray(data["finite_batch_upper_bound"], dtype=float),
            )
            np.testing.assert_allclose(
                np.asarray(data["information_gain_upper_bound"], dtype=float),
                expected,
            )
            acquisition = np.asarray(data["maximum_lqei_history"], dtype=float)
            acquisition = acquisition[np.isfinite(acquisition)]
            total_sequential += acquisition.size
            exact_zero += int(np.count_nonzero(acquisition == 0.0))
            nonpositive += int(np.count_nonzero(acquisition <= 0.0))
            tiny += int(np.count_nonzero(acquisition <= np.finfo(float).tiny))
            recommendation_index = int(data["recommendation_index"])
            initial_count = int(data["initial_point_count"])
            final_initial += int(recommendation_index < initial_count)
            final_index_zero += int(recommendation_index == 0)
            raw_gaps.append(float(data["raw_gap"]))
            terminal_coefficients.append(float(data["recommendation_coefficient"]))
            selected_unique_counts.append(
                int(np.unique(np.asarray(data["selected_indices"], dtype=int)).size)
            )
            terminal_active_bounds.append(
                "kernel_order"
                if float(data["kernel_order_upper_bound"][-1])
                <= float(data["finite_batch_upper_bound"][-1])
                else "finite_batch"
            )

    macros = len(paths)
    return {
        "function": spec.function_name,
        "total_budget": spec.total_budget,
        "macro_count": macros,
        "batch_count": int(spec.schedule.size),
        "nominal_batch_size": spec.nominal_batch_size,
        "sequential_acquisition_count": total_sequential,
        "exact_zero_acquisition_count": exact_zero,
        "exact_zero_acquisition_fraction": exact_zero / total_sequential,
        "nonpositive_acquisition_count": nonpositive,
        "tiny_or_zero_acquisition_count": tiny,
        "final_recommendation_in_initial_count": final_initial,
        "final_recommendation_in_initial_fraction": final_initial / macros,
        "final_recommendation_index_zero_count": final_index_zero,
        "mean_raw_gap": float(np.mean(raw_gaps)),
        "median_raw_gap": float(np.median(raw_gaps)),
        "mean_unique_selected_points": float(np.mean(selected_unique_counts)),
        "terminal_0p2_lambda": float(np.mean(terminal_coefficients)),
        "terminal_active_bound": sorted(set(terminal_active_bounds)),
        "passes_no_exact_zero_collapse": exact_zero == 0,
        "passes_no_systematic_initial_lock": final_initial < macros,
    }


def main() -> None:
    rows = [_validate_spec(spec) for spec in smoke_specs()]
    passed = all(
        bool(row["passes_no_exact_zero_collapse"])
        and bool(row["passes_no_systematic_initial_lock"])
        for row in rows
    )
    payload = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "information_gain_method": INFORMATION_GAIN_METHOD,
        "decision_rule": (
            "pass only if every budget has zero exact-zero acquisitions and "
            "not all terminal recommendations are initial-design points"
        ),
        "passed": passed,
        "results": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "smoke_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "smoke_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit("SE min-bound smoke validation failed")


if __name__ == "__main__":
    main()
