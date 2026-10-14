# 4D LQEI final recommendation rule experiment

This is a small, self-contained code and input snapshot of the accepted four-dimensional Matérn-5/2 RKHS experiment. It compares four final recommendation rules on the same LQEI trajectories. The selected objective is `three_tier_r11_o14p0_d6p0`. The formal study uses uniform noise levels `2.5 R_f` and `3.0 R_f`, seven budgets, 250 macroreplications per low-budget/noise cell, and 100 per medium/high-budget/noise cell.

## Setup

Use Python 3.11. From this folder:

```powershell
python -m pip install -r requirements.txt
python experiment.py --help
```

`experiment.py` is the main experiment program. Its core support modules (`confidence.py`, `continuous_acquisition.py`, `experiment_runner.py`, `rkhs_functions.py`, `sequential_gp.py`, and `study_config.py`) are included here, so a separate `new_921_4d_exp_budget` folder is not needed. The small JSON and CSV files in `screening_round2/`, `confirmation_round1/`, `formal_low_budget/`, and `formal_all_budgets/` fix the accepted objective and permit the later phases to start without the large screening and formal outputs.

## Run the formal experiment

```powershell
python experiment.py formal --workers 8
python experiment.py extension --workers 8
python analyze.py
```

The formal phase uses the included confirmation records; it runs the low budgets. The extension runs the medium and high budgets. These are long runs and create output in `formal_low_budget/`, `formal_extension/`, and `formal_all_budgets/`.

For the additional requested budget grid and noise studies:

```powershell
python experiment.py additional --workers 8
python analyze_requested_grid.py
python noise_sensitivity.py all --workers 8
python noise_sensitivity_e10.py all --workers 8
python compare_noise_sensitivity.py
python noise_3Rf_budget_sweep.py all --workers 8
```

The `screen` and `confirm` phases are also available in `experiment.py`; `confirm` uses the included second-round screening records. `analyze.py` and the other analysis programs require their corresponding simulation outputs. Raw trajectories, analysis tables, figures, caches, and logs are excluded from this upload folder.
