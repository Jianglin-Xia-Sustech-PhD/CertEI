# Hills 2D experiment with a fixed budget

This folder contains the code and fixed inputs for the accepted Hills `-g(x)` experiment. The default configuration uses `sigma/R_f = 8 - 1.5 z`, replication multiplier 8, 2,048 Sobol candidates, six budgets, four algorithms, and 100 formal macroreplications. The corresponding result group is `formal_six_budget_2048_candidates_a_8_b_m1p5_rho_8_100_macros`.

## Run

Use Python 3.11. Install dependencies from this folder:

```bash
python -m pip install -r requirements.txt
python run_experiments.py prepare
python run_experiments.py smoke --workers 1
python run_experiments.py formal --workers 8
python run_experiments.py analyze
python validate_results.py
```

`prepare` checks the supplied candidate points and calibration metadata. `smoke` runs one macroreplication for each configuration. `formal` runs all 100; it can take a long time and produces many files. Use `--workers` to fit the machine. `analyze` and `validate_results.py` require the complete formal outputs. To regenerate inputs after changing the configuration, use `python run_experiments.py prepare --overwrite`.

The main entry point is `run_experiments.py`. `run_shard.py` is for running a subset of configurations. `experiment_runner.py` implements the experiments; `dynamic_gp.py`, `test_functions.py`, and `study_config.py` contain the model, objective, and settings. `prepare_inputs.py`, `analyze_results.py`, and `validate_results.py` handle inputs and postrun checks. The three files in `inputs/a_8_b_m1p5_rho_8/` are required fixed inputs for the accepted configuration.

Generated `results/`, `analysis/`, and `figures/` are intentionally excluded from this upload folder. The included component tests can be run with `python -m pytest -q tests`.
