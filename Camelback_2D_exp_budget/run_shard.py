"""Robust direct-process shard runner for the formal macroreplications."""

from __future__ import annotations

import argparse
import os

# Match the formal entry point: each shard is a process, so numerical kernels
# must stay single-threaded to avoid nested BLAS thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiment_runner import _macro_path, all_specs, run_one_macro
from study_config import FORMAL_MACROS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_index", type=int)
    parser.add_argument("shard_count", type=int)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require 0 <= shard_index < shard_count")

    assigned = []
    job_index = 0
    for spec in all_specs(FORMAL_MACROS):
        for macro in range(1, FORMAL_MACROS + 1):
            if job_index % args.shard_count == args.shard_index:
                assigned.append((spec, macro))
            job_index += 1

    pending = [
        (spec, macro)
        for spec, macro in assigned
        if not _macro_path(spec, macro).exists()
    ]
    print(
        f"Shard {args.shard_index + 1}/{args.shard_count}: "
        f"{len(pending)} pending jobs",
        flush=True,
    )
    for completed, (spec, macro) in enumerate(pending, start=1):
        result = run_one_macro(spec, macro, overwrite=False)
        if completed % 10 == 0 or completed == len(pending):
            print(
                f"Shard {args.shard_index + 1}/{args.shard_count}: "
                f"completed {completed}/{len(pending)}; "
                f"{spec.algorithm}/T={spec.total_budget}/macro={macro}; "
                f"gap={float(result['normalized_gap']):.6g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
