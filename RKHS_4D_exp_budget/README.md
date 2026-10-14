# 4D LQEI with exponential simulation budgets

## Code snapshot

This folder contains the core 4D experiment, its analysis code, the optional
`e^8` budget extension, fixed test functions, candidate points, and tests.
Raw simulation results, generated figures, logs, and cached confidence
schedules are not included. Use Python 3.11 and run these checks from this
folder before starting the computationally expensive experiments:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_experiments.py --help
```

The optional `ablation_smoke` phase has a structural validator for its two
saved trajectories. It checks metadata and exact budget accounting; it does
not establish statistical performance.

This is a fresh rerun of `new_914_4d_min_bound`. The only scientific-design
change is the total-budget grid: `T=round(exp(k))` for
`k=8.5, 9, 9.5, 10, 10.5, 11, 11.5`, yielding
`4,915, 8,103, 13,360, 22,026, 36,316, 59,874, 98,716`.
All other objectives, noise, algorithms, seeds, macro counts, GP parameters,
initial design sizes, batch-order definitions, and analyses remain as in the
source experiment.

This study is based on `93restart_4d_trial1(continuous_0p1_accept)` and keeps
its fixed four-dimensional RKHS functions, heteroscedastic noise, 8,192-point
Sobol screen, continuous L-BFGS-B acquisition polishing, and fixed response-GP
hyperparameters.

The LQEI confidence sequence uses the pointwise minimum
`min{G_k(U_n), n/2*log(1+b_T/nu_min)}`, where `U_n=n*b_T/nu_min` and the
four-dimensional theoretical-order terms retain a unit big-O multiplier:

- squared exponential: `G(U)=log(1+U)^5`;
- Matérn-5/2: `G(U)=U^(4/9)*log(1+U)^(5/9)`.

The search incumbent is `min_x [mu_n(x)-0.2*Lambda_n*sigma_n(x)]`. The final
recommendation minimizes `mu_N+0.2*Lambda_N*sigma_N` over sampled points.

Main LQEI, SKO, EQI, and TSSO runs use 40 initial points and the same exact balanced
`rho*log(T)` schedule at all seven exponential budgets. EQI and TSSO logic is
unchanged.

## SKO comparison

`sko.py` implements the acquisition registered as `SKO`. Its response GP uses the same fixed four-dimensional
kernel parameters as every comparator. The auxiliary noise GP fits its
constant mean and process variance from the 40-point initial design once and
does not re-estimate them during the sequential run. Each SKO iteration uses
one complete common `rho*log(T)` batch. Acquisition maximization uses the same
8,192-point Sobol screen and 10-start bounded L-BFGS-B polishing policy used by
the other continuously optimized four-dimensional methods. The final SKO
recommendation minimizes `posterior_mean + posterior_sd` over sampled points.

## Incumbent-rule ablation

The first 4D LQEI ablation compares the three global EI incumbents
`min(mu - 0.2*Lambda_n*sd)`, `min(mu)`, and `min(mu + sd)`. Only the incumbent
definition changes. The fixed GP parameters, 40-point initial designs, common
random numbers, exact `rho*log(T)` schedules, continuous acquisition optimizer,
and final recommendation `argmin(mu_N + 0.2*Lambda_N*sd_N)` remain unchanged.
The dynamic `-0.2*Lambda_n` arm reuses the completed main LQEI results; the two
additional arms are stored under `results/incumbent_rule_ablation_min_bound`.

Run or resume this ablation and generate its boxplots with:

```powershell
python run_experiments.py incumbent_ablation --workers 8
```

## Final-recommendation-rule ablation

The `final_recommendation_ablation` phase keeps the original LQEI acquisition
path and its incumbent `min(mu_n - 0.2*Lambda_n*sd_n)` fixed. Over the same
sampled points and recovered terminal posterior, it compares
`min(mu_N + 0.2*Lambda_N*sd_N)`, `min(mu_N)`, and
`min(mu_N - 10*sd_N)`. The final rule cannot affect earlier samples, so this is
an exact paired postprocessing comparison of the existing 1,400 main LQEI
paths and requires no new simulation.

Generate the linear- and logarithmic-scale boxplots and tables with:

```powershell
python run_experiments.py final_recommendation_ablation
```

The symmetric update keeps the registered heterogeneous noise, full budget
grid, and every saved LQEI path unchanged, but replaces the negative control
with `min(mu_N - 0.2*Lambda_N*sd_N)`. It writes to a separate directory, so
the original `mu_N - 10*sd_N` tables and figures remain available:

```powershell
python analyze_final_recommendation_ablation_symmetric.py
```

Updated outputs are under
`analysis/final_recommendation_ablation_symmetric/`.

The replication-number-order ablation separately uses 20 initial points for
every order so the direct same-rho `rho*T^(1/2)` schedule remains feasible at
the lowest ablation budget. At budgets 36,316, 59,874, and 98,716 it compares
`rho`, `rho*log(log(T))`, `rho*log(T)`, and `rho*T^(1/2)` without anchor
normalization. Every schedule exhausts its exact total budget. The response GP,
equation-(20) Lambda rule, EI incumbent, final recommendation, acquisition
optimizer, seeds, and noise model remain fixed.

The O(1) arm uses the same rounding rule as the other arms: rho is rounded up
to a multiple of five. With rho=8 for Matérn and rho=6 for SE, this means ten
nominal replications per batch for both kernels. Exact-budget balancing makes
individual batches 10 or 11. All 2,400 ablation macros across four orders are
rerun for the new grid. The run caps concurrent SE workers at four for
T=59,874 and two for T=98,716 to bound memory consumption on long paths.
This only changes execution concurrency, not the experimental protocol.

This rerun retains the current confidence-bound proxy and
continuous-optimization settings. The unit big-O multiplier has not been
finite-sample calibrated, and positive but tiny EI values can terminate
ordinary-scale polishing early. Completion validation checks the protocol and
data, not confidence coverage or exact global acquisition maximization.

Long constant and log-log paths use exact same-site Gaussian sufficient-statistic
compression, adaptive storage, batched queries for the same continuous
optimizer starts, and a shared LAPACK-ready factor. These are equivalent
implementation optimizations; none changes the GP, acquisition, or sampling
protocol. Ordinary EI underflow is handled with a mathematically identical
log-EI maximization fallback.

Run or resume this ablation and generate its boxplots with:

```powershell
python run_experiments.py ablation --workers 8
```

Run the full suite in the requested order with:

```powershell
python run_experiments.py smoke --workers 8
python run_experiments.py formal --workers 8
python run_experiments.py incumbent_ablation --workers 8
python run_experiments.py final_recommendation_ablation
python run_experiments.py ablation --workers 8
python run_experiments.py analyze
```

Individual phases are `confidence`, `lqei`, `comparisons`,
`incumbent_ablation`, `final_recommendation_ablation`, `ablation`,
`constant_ablation`, and `analyze`; `smoke` runs a 10-macro SE stability check
at budgets 22,026 and 98,716. Results and figures are written below this
directory. The replication-order phase is intentionally last.

To run only the four-algorithm comparison and LQEI convergence experiment,
without any replication-order ablation, use:

```powershell
python run_experiments.py formal --workers 8
```

## Additional e^8 budget

The later `e8_budget_extension.py` entry point adds
`T=round(exp(8))=2,981` without modifying the registered seven-budget grid or
overwriting any existing figure. It runs 100 macros for main LQEI, SKO, EQI,
TSSO, and the two additional EI-incumbent rules. The three final recommendation
rules are evaluated by paired postprocessing of the new main LQEI trajectories.

Run or resume the complete extension with:

```powershell
python e8_budget_extension.py all --workers 8
```

The raw `T_2981` results are added to their canonical result groups. New tables,
validation files, and 2-by-8 augmented subplot grids are isolated under
`analysis/e8_budget_extension/`; SHA-256 checks verify that all pre-existing
PNG/PDF figures remain byte-for-byte unchanged.
