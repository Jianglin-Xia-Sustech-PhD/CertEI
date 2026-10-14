"""Resume and supervise the complete exponential-budget experiment in order."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from run_experiments import comparison_specs, incumbent_ablation_specs, lqei_specs, ablation_specs
from study_config import FORMAL_MACROS


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
STATUS = ROOT / "suite_status.json"
COMPLETE = ROOT / "suite_completion.json"
WORKERS = 8


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(phase: str, state: str, **extra: object) -> None:
    payload = {"updated_utc": stamp(), "phase": phase, "state": state, **extra}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def formal_complete() -> bool:
    path = ROOT / "run_completion.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            data["lqei_macro_count"] == len(lqei_specs()) * FORMAL_MACROS
            and data["comparison_macro_count"] == len(comparison_specs()) * FORMAL_MACROS
            and (ROOT / "analysis" / "figures" / "lqei_convergence_rate.png").is_file()
        )
    except (OSError, ValueError, KeyError):
        return False


def formal_process_running() -> bool:
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (
                "run_experiments.py" in command
                and "formal" in command
                and Path(process.cwd()).resolve() == ROOT
            ):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return False


def run_phase(phase: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    write_status(phase, "running")
    command = [sys.executable, "-u", "run_experiments.py", phase]
    if phase in {"formal", "incumbent_ablation", "ablation"}:
        command += ["--workers", str(WORKERS)]
    with (LOG_DIR / f"{phase}.stdout.log").open("a", encoding="utf-8") as output, \
         (LOG_DIR / f"{phase}.stderr.log").open("a", encoding="utf-8") as errors:
        output.write(f"\n[{stamp()}] Starting {' '.join(command)}\n")
        output.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=errors, check=False)
    if result.returncode:
        write_status(phase, "failed", exit_code=result.returncode)
        raise SystemExit(result.returncode)
    write_status(phase, "completed")


def main() -> None:
    write_status("formal", "waiting_for_current_run")
    while not formal_complete():
        if not formal_process_running():
            run_phase("formal")
        else:
            time.sleep(30)
    for phase in ("incumbent_ablation", "final_recommendation_ablation", "ablation", "analyze"):
        run_phase(phase)
    required = (
        ROOT / "analysis" / "validation_report.json",
        ROOT / "analysis" / "incumbent_ablation" / "validation_report.json",
        ROOT / "analysis" / "final_recommendation_ablation" / "validation_report.json",
        ROOT / "analysis" / "replication_order_ablation" / "validation_report.json",
    )
    for path in required:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("passed", False):
            raise RuntimeError(f"Validation did not pass: {path}")
    payload = {
        "completed_utc": stamp(),
        "passed": True,
        "main_lqei_macros": len(lqei_specs()) * FORMAL_MACROS,
        "comparison_macros": len(comparison_specs()) * FORMAL_MACROS,
        "incumbent_additional_macros": len(incumbent_ablation_specs()) * FORMAL_MACROS,
        "replication_order_macros": len(ablation_specs()) * FORMAL_MACROS,
        "final_rule": "paired postprocessing of main LQEI paths",
    }
    COMPLETE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_status("full_suite", "completed", **payload)


if __name__ == "__main__":
    main()
