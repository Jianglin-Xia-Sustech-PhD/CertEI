"""Command-line entry point for preparation, execution and analysis."""

from __future__ import annotations

import argparse
import os

# Prevent each worker from creating its own multithreaded BLAS pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from analyze_results import analyze
from experiment_runner import all_specs, run_experiments
from prepare_inputs import prepare_inputs
from study_config import FORMAL_MACROS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "smoke", "formal", "analyze", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.phase in ("prepare", "smoke", "formal", "all"):
        print(f"Input metadata: {prepare_inputs(overwrite=args.overwrite)}", flush=True)
    if args.phase == "prepare":
        return
    if args.phase == "smoke":
        smoke_specs = all_specs(macro_count=1)
        run_experiments(smoke_specs, workers=args.workers, overwrite=args.overwrite)
        return
    if args.phase in ("formal", "all"):
        run_experiments(
            all_specs(macro_count=FORMAL_MACROS),
            workers=args.workers,
            overwrite=args.overwrite,
        )
    if args.phase in ("analyze", "all"):
        png, table = analyze(FORMAL_MACROS)
        print(f"Boxplot: {png}", flush=True)
        print(f"Summary: {table}", flush=True)


if __name__ == "__main__":
    main()
