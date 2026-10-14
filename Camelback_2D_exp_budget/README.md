# Six-hump Camelback: 2D budget comparison

This folder contains the core code and small, fixed inputs for the accepted
negative-slope noise experiment. Raw simulation trajectories, generated plots,
and analysis outputs are intentionally omitted.

The default configuration is the accepted formal study: noise standard
deviation `sigma(x)/R_f = 4 - 1.75 z(x)`, replication multiplier `rho = 8`, six
budgets from `round(exp(7.5))` through `round(exp(10))`, and 100 independent
macroreplications per configuration. The environment variables in
`study_config.py` can override these defaults. The two files under
`inputs/a_4_b_m1p75_rho_8/` contain the fixed calibration metadata and Sobol
candidate set for these defaults.
If you change the noise parameters, regenerate the matching inputs with
`python run_experiments.py prepare --overwrite` before running simulations.

## Run

Use Python 3.11 from this folder:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_experiments.py prepare
python run_experiments.py smoke --workers 1
python run_experiments.py formal --workers 8
python run_experiments.py analyze
python validate_results.py
```

`smoke` runs one macroreplication per configuration. The formal run is much
larger and generates `results/`, `analysis/`, and `figures/` locally. The
`run_shard.py` entry point can distribute formal macroreplications across
separate processes.

## Main files

- `run_experiments.py`: preparation, simulation, and analysis entry point.
- `experiment_runner.py`: LQEI, TSSO, SKO, and EQI experiment loops.
- `dynamic_gp.py`: fitted Gaussian-process surrogate.
- `prepare_inputs.py` and `test_functions.py`: fixed candidates, calibration,
  objective, and noise model.
- `study_config.py`: accepted parameters, budgets, and random seeds.
- `analyze_results.py` and `validate_results.py`: summaries and checks after
  the complete formal run.
