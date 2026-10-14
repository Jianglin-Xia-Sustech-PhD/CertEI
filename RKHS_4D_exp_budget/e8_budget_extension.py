"""Run the registered 4D study at the additional T=round(exp(8)) budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from comparison_runner import ComparisonSpec, run_comparison_specs
from experiment_runner import (
    INCUMBENT_RULE_MEAN,
    INCUMBENT_RULE_UPPER_ONE_SD,
    ExperimentSpec,
    run_specs,
)
from run_experiments import (
    COMPARISON_ALGORITHMS,
    COMPARISON_RESULT_GROUP,
    FUNCTION_NAMES,
    INCUMBENT_ABLATION_RESULT_GROUP,
    LQEI_RESULT_GROUP,
)
from study_config import (
    CONFIDENCE_DELTA,
    FORMAL_MACROS,
    FORMAL_PHASE_CODE,
    N_CANDIDATES,
    N_INITIAL,
    rho_for_function,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "e8_budget_extension"
FIGURE_MANIFEST = ANALYSIS_DIR / "preexisting_figure_sha256.json"
PROTOCOL = ANALYSIS_DIR / "protocol.json"
COMPLETION = ANALYSIS_DIR / "completion.json"

E8_BUDGET_EXPONENT = 8.0
E8_BUDGET = round(math.exp(E8_BUDGET_EXPONENT))


def e8_lqei_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            phase=LQEI_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=E8_BUDGET,
            replication_order="log",
            replication_multiplier=rho_for_function(name),
            anchor_budget=E8_BUDGET,
            macro_count=FORMAL_MACROS,
            initial_point_count=N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
        )
        for name in FUNCTION_NAMES
    ]


def e8_comparison_specs() -> list[ComparisonSpec]:
    return [
        ComparisonSpec(
            algorithm=algorithm,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=E8_BUDGET,
            replication_multiplier=rho_for_function(name),
            anchor_budget=E8_BUDGET,
            macro_count=FORMAL_MACROS,
            candidate_count=N_CANDIDATES,
            initial_replications=None,
            result_group=COMPARISON_RESULT_GROUP,
        )
        for name in FUNCTION_NAMES
        for algorithm in COMPARISON_ALGORITHMS
    ]


def e8_incumbent_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            phase=INCUMBENT_ABLATION_RESULT_GROUP,
            phase_code=FORMAL_PHASE_CODE,
            function_name=name,
            total_budget=E8_BUDGET,
            replication_order="log",
            replication_multiplier=rho_for_function(name),
            anchor_budget=E8_BUDGET,
            macro_count=FORMAL_MACROS,
            initial_point_count=N_INITIAL,
            candidate_count=N_CANDIDATES,
            confidence_delta=CONFIDENCE_DELTA,
            incumbent_rule=rule,
        )
        for name in FUNCTION_NAMES
        for rule in (INCUMBENT_RULE_MEAN, INCUMBENT_RULE_UPPER_ONE_SD)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_preexisting_figure_manifest() -> None:
    """Freeze hashes of every pre-extension PNG/PDF so analysis can protect them."""

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if FIGURE_MANIFEST.exists():
        return
    figures = {}
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {".png", ".pdf"}
            and ANALYSIS_DIR not in path.parents
        ):
            figures[str(path.relative_to(ROOT)).replace("\\", "/")] = _sha256(path)
    FIGURE_MANIFEST.write_text(
        json.dumps({"captured_utc": _stamp(), "figures": figures}, indent=2) + "\n",
        encoding="utf-8",
    )


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_extension_protocol(workers: int) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": _stamp(),
        "purpose": "Add T=round(exp(8)) without overwriting the registered seven-budget figures",
        "budget_exponent": E8_BUDGET_EXPONENT,
        "total_budget": E8_BUDGET,
        "macro_count_per_configuration": FORMAL_MACROS,
        "workers": workers,
        "lqei_specifications": [asdict(spec) for spec in e8_lqei_specs()],
        "comparison_specifications": [
            asdict(spec) for spec in e8_comparison_specs()
        ],
        "incumbent_ablation_specifications": [
            asdict(spec) for spec in e8_incumbent_specs()
        ],
        "final_recommendation_ablation": (
            "Paired terminal-posterior rescoring of the 200 new main LQEI paths; "
            "no additional simulation"
        ),
        "output_policy": (
            "T_2981 data are added to the existing canonical result groups; all "
            "new tables and figures are isolated under analysis/e8_budget_extension"
        ),
    }
    PROTOCOL.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=("lqei", "comparisons", "incumbent", "analyze", "all"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    capture_preexisting_figure_manifest()
    write_extension_protocol(args.workers)

    if args.phase in ("lqei", "all"):
        run_specs(
            e8_lqei_specs(), workers=args.workers, overwrite=args.overwrite
        )
    if args.phase in ("comparisons", "all"):
        run_comparison_specs(
            e8_comparison_specs(),
            workers=args.workers,
            overwrite=args.overwrite,
            write_protocol_file=False,
        )
    if args.phase in ("incumbent", "all"):
        run_specs(
            e8_incumbent_specs(), workers=args.workers, overwrite=args.overwrite
        )
    if args.phase in ("analyze", "all"):
        from analyze_e8_budget_extension import main as analyze_extension

        analyze_extension()

    elapsed = time.perf_counter() - started
    if args.phase == "all":
        payload = {
            "completed_utc": _stamp(),
            "passed": True,
            "elapsed_seconds": elapsed,
            "total_budget": E8_BUDGET,
            "lqei_macro_count": len(e8_lqei_specs()) * FORMAL_MACROS,
            "comparison_macro_count": len(e8_comparison_specs()) * FORMAL_MACROS,
            "incumbent_additional_macro_count": (
                len(e8_incumbent_specs()) * FORMAL_MACROS
            ),
            "final_rule_new_simulation_count": 0,
        }
        COMPLETION.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Finished e^8 extension phase {args.phase} in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
