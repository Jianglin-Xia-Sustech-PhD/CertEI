# Hartmann 6D experiment with a fixed budget

This folder contains the core code and fixed inputs for the selected Hartmann 6D experiment. Its default configuration uses 10,000 Sobol candidate points, 28 initial points, noise `sigma/R_f = 7.2 - 3.5 z`, replication multiplier 8, the 0.75 RKHS calibration quantile with 0.5 shrinkage, six budgets, four algorithms, and 100 formal macroreplications. The result group is `formal_six_budget_10000_candidates_cand_sobol_ninit_28_a_7p2_b_m3p5_rho_8_bf_q75_x0p5_100_macros`.

## Run

Use Python 3.11. From this folder:

```bash
python -m pip install -r requirements.txt
python run_experiments.py prepare
python run_experiments.py smoke --workers 1
python run_experiments.py formal --workers 8
python run_experiments.py analyze
python validate_results.py
```

`prepare` checks the supplied fixed candidates and calibration metadata. `smoke` runs one macroreplication per configuration. `formal` runs all 100 and may take a long time. Set `--workers` according to the machine. Analysis and validation require complete formal outputs. If you change the configuration, regenerate its inputs with `python run_experiments.py prepare --overwrite`; the 6D calibration can also take substantial time.

`run_experiments.py` is the main entry point. `run_shard.py` runs a subset of work. `experiment_runner.py` implements the experiments; `dynamic_gp.py`, `test_functions.py`, and `study_config.py` define the model, function, and settings. `prepare_inputs.py`, `analyze_results.py`, and `validate_results.py` handle preparation and postrun checks. All three files in `inputs/` are required for the selected configuration.

Generated `results/`, `analysis/`, and `figures/` are deliberately excluded. Run the included component checks with `python -m pytest -q tests`.
