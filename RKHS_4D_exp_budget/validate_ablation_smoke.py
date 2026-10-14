"""Check saved trajectories from the optional replication-order smoke run."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from experiment_runner import ROOT, _macro_path
from run_experiments import ABLATION_SMOKE_MACROS, ablation_smoke_specs


OUTPUT_DIR = ROOT / "analysis" / "ablation_smoke_validation"


def main() -> None:
    checked: list[dict[str, object]] = []
    for spec in ablation_smoke_specs():
        for macro in range(1, ABLATION_SMOKE_MACROS + 1):
            path = _macro_path(spec, macro)
            if not path.is_file():
                raise FileNotFoundError(f"missing ablation smoke trajectory: {path}")
            with np.load(path, allow_pickle=False) as data:
                replications = np.asarray(data["replications"], dtype=int)
                observations = np.asarray(data["raw_observations"], dtype=float)
                selected = np.asarray(data["selected_indices"], dtype=int)
                valid = (
                    int(data["macro"]) == macro
                    and str(data["function_name"]) == spec.function_name
                    and int(data["total_budget"]) == spec.total_budget
                    and str(data["replication_order"]) == spec.replication_order
                    and int(data["initial_point_count"]) == spec.initial_point_count
                    and replications.size == spec.schedule.size
                    and selected.size == replications.size
                    and int(replications.sum()) == spec.total_budget
                    and observations.size == spec.total_budget
                    and np.all(np.isfinite(observations))
                    and np.isfinite(float(data["raw_gap"]))
                )
                if not valid:
                    raise AssertionError(f"invalid ablation smoke trajectory: {path}")
            checked.append(
                {
                    "function": spec.function_name,
                    "replication_order": spec.replication_order,
                    "total_budget": spec.total_budget,
                    "macro": macro,
                }
            )

    payload = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "checked_trajectories": checked,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
