"""Run the second fixed-budget noise study at T=round(exp(10))."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import experiment as study
import noise_sensitivity as base


BUDGET = 22026
EXPONENT = 10.0
PHASE = 92520


def configure() -> None:
    base.OUTPUT = base.HERE / "noise_sensitivity_budget_e10"
    base.CHECKPOINTS = base.OUTPUT / "checkpoints"
    base.FIGURES = base.OUTPUT / "figures"
    base.BUDGET = BUDGET
    base.EXPONENT = EXPONENT
    base.PHASE = PHASE


def _worker(payload: tuple[dict[str, object], str, float, int]) -> list[dict[str, object]]:
    function_data, label, multiplier, macro = payload
    rows = study.run_macro(
        study.objective_from_dict(function_data),
        label,
        BUDGET,
        base.CANDIDATES,
        macro,
        PHASE,
        continuous=True,
        noise_multiplier=multiplier,
        final_rules=base.PLOT_RULES,
    )
    return rows


def run(workers: int) -> None:
    function_data = json.loads(
        (base.HERE / "formal_all_budgets" / "selected_function.json").read_text(
            encoding="utf-8"
        )
    )
    jobs: list[tuple[dict[str, object], str, float, int]] = []
    for label, multiplier in base.NOISE_GRID:
        for macro in range(1, base.MACROS + 1):
            if not base._checkpoint_is_current(label, macro):
                jobs.append((function_data, label, multiplier, macro))

    print(f"e^10 noise-sensitivity paths remaining: {len(jobs)}", flush=True)
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
                destination = base._checkpoint(label, macro)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(rows, indent=2) + "\n", encoding="utf-8"
                )
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("run", "analyze", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    configure()
    if args.mode in ("run", "all"):
        run(args.workers)
    if args.mode in ("analyze", "all"):
        base.analyze()


if __name__ == "__main__":
    main()
