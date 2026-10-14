# Rescaled Branin negative-noise-slope study

## Code snapshot

This folder contains the core implementation and the small fixed inputs for
the selected formal `sigma(x)/R_f = 4 - 3z(x)` experiment. Raw simulation
trajectories, generated figures, and analysis outputs are not included. The
defaults use `a=4`, `b=-3`, `rho=8`, six budgets from `round(exp(7.5))` through
`round(exp(10))`, and 100 independent macroreplications per configuration.

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

The formal run generates `results/`, `analysis/`, and `figures/` locally and
requires substantially more time and storage than the checks. If changing
the noise coefficients through environment variables, regenerate matching
inputs with `python run_experiments.py prepare --overwrite` first.

This directory compares LQEI, TSSO, SKO, and EQI on the standardized
two-dimensional Branin function under

`sigma(x) / R_f = a + b z(x),  b < 0`,

where `z(x)` is the standardized true function value in `[0, 1]`. Thus the
near-optimal region has larger simulation noise than high-function-value
regions. The formula is used only by the simulator. SKO and EQI infer noise
with an auxiliary Matérn-5/2 GP and are not given `a`, `b`, or `z(x)`.

Unchanged settings include 2,048 Sobol candidates without exact-optimum
injection, 20 initial LHS points, Matérn-5/2 response GPs, weakly regularized
MLE updates, identical final-identification rules, and budgets exp(7.5)
through exp(10).

The study uses 30 independent macroreplications for screening and 100 new
macroreplications (with disjoint seeds) for confirmation. Each coefficient
pair has separate input, result, analysis, and figure subdirectories.

`run_experiments.py` is the main entry point. `experiment_runner.py` runs LQEI,
TSSO, SKO, and EQI; `dynamic_gp.py` fits the surrogate; `prepare_inputs.py`
and `test_functions.py` define the fixed inputs and function; and
`analyze_results.py` and `validate_results.py` process a completed formal run.
