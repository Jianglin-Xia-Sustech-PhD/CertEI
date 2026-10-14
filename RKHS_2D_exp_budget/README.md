# 2D LQEI with exponential simulation budgets

This experiment is an exact-parameter rerun of `new_913_2d_min_bound`, except
that both kernels now use the common exponential budget grid
`e^7.5, e^8, ..., e^11.5`, rounded to the authoritative integers
`1,808, 2,981, 4,915, 8,103, 13,360, 22,026, 36,316, 59,874, 98,716`.
The functions, noise model, fixed GP parameters, seeds, 100 macro-replications,
candidate set, batch rule, acquisition rule, final rule, and comparison
algorithms are unchanged. The replication-order ablation is included in the
complete rerun.

For every batch index `n = 0, ..., N_T`, define

```text
U_n = n*b_T/nu_min

kernel-order bound:
  SE, d=2:       G_k(U_n) = log(1+U_n)^3
  Matern-5/2:    G_k(U_n) = U_n^(2/7)*log(1+U_n)^(5/7)

finite-batch bound:
  F_n(b_T) = n/2 * log(1+b_T/nu_min)

equation-(20) bound:
  gamma_bar_n^k(b_T) = min{G_k(U_n), F_n(b_T)}
```

The implicit big-O multiplier in `G_k` remains fixed to one. At `n=N_T`,
the definition above is equation (20) in `reference/paper.md`; applying the
same bound at every `n` supplies the dynamic sequence required by `Lambda_n`:

```text
Lambda_n = B_f + sqrt(
    2*kappa_T*(gamma_bar_n^k(b_T) + log(3/(2*delta)))
)
```

LQEI acquisition uses the incumbent
`min_x [mu_n(x) - 0.2*Lambda_n*sigma_n(x)]`. The final recommendation minimizes
`mu_N + 0.2*Lambda_N*sigma_N` over sampled points.

The Matérn-5/2 and squared-exponential budgets are both `e^7.5, e^8, ...,
e^11.5` in increments of 0.5.
Every main LQEI, SKO, EQI, and TSSO configuration uses the same exact balanced
`rho*log(T)` batch schedule. The LQEI replication-order ablation covers
`e^10.5=36,316`, `e^11=59,874`, and `e^11.5=98,716` and compares `rho`,
`rho*log(log(T))`, `rho*log(T)`, and `rho*T^(1/2)` using the same numerical
kernel-specific rho for every order.

## SKO comparison

`sko.py` adapts the acquisition implemented in
`KrigingBasedHetero_sustech_python/AEI_python/aei/acquisition.py`; the formal
algorithm name and all saved metadata use `SKO`. Its effective incumbent is
the posterior mean at the sampled point minimizing `mean + posterior_sd`, and
ordinary EI is multiplied by

```text
1 - sqrt(single_replication_noise_variance)
      / sqrt(posterior_variance + single_replication_noise_variance).
```

The response GP uses the same fixed mean 0, process variance 1, lengthscale
0.2, and function-matched kernel as the other algorithms. The auxiliary noise
GP is initialized from the first 20 designs; its mean, process variance,
lengthscale, and kernel are then held fixed. Each SKO iteration receives one
complete batch from the common schedule, so its per-iteration and total budgets
match LQEI and EQI exactly.

With 100 macros per configuration, the formal comparison study contains 1,800
main LQEI runs and 5,400 comparison runs (SKO, EQI, and TSSO), for 7,200 macros
across 72 configurations. The incumbent ablation adds 3,600 runs for its two
alternative incumbents; the dynamic incumbent arm reuses the 1,800 main LQEI
paths. The final-rule ablation is paired postprocessing of those same paths.

## Incumbent-rule ablation

The `incumbent_ablation` phase compares three global EI incumbents:
`min(mu - 0.2*Lambda_n*sd)`, `min(mu)`, and `min(mu + sd)`. Only the incumbent
changes. The response GP, equation-(20) confidence schedule, exact
`rho*log(T)` batch schedule, random-number streams, EI calculation, and final
sampled-point recommendation `min(mu_N + 0.2*Lambda_N*sd_N)` remain fixed. The
dynamic `-0.2*Lambda_n` arm reuses the main LQEI results; the other two arms
are stored separately under `results/incumbent_rule_ablation_min_bound`.

Run or resume this ablation and create its boxplots with:

```powershell
python run_experiments.py incumbent_ablation --workers 8
```

## Final-recommendation-rule ablation

The `final_recommendation_ablation` phase keeps the LQEI acquisition path and
its incumbent `min(mu_n - 0.2*Lambda_n*sd_n)` fixed. Over the same sampled
points and terminal posterior, it compares `min(mu_N +
0.2*Lambda_N*sd_N)`, `min(mu_N)`, and `min(mu_N - 10*sd_N)`. Because the final
recommendation does not affect earlier samples, this is an exact paired
postprocessing comparison of the existing 1,600 main LQEI paths and requires
no new simulation.

Generate the boxplots and tables with:

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

## Replication-number-order ablation

The `ablation` phase keeps the LQEI GP, equation-(20) Lambda rule, EI
incumbent, final recommendation, seeds, and exact total budget fixed. It varies
only the replication number among `rho`, `rho*log(log(T))`, `rho*log(T)`, and
`rho*T^(1/2)` at budgets 30,000, 60,000, and 100,000. All orders use the same
numerical function-specific rho; the `rho*log(T)` arm reuses the main LQEI
results.

For the very long `O(1)` paths, independent Gaussian batch means observed at
the same design are combined through their exact precision-weighted sufficient
statistics and the fixed-hyperparameter posterior is periodically rebuilt.
If ordinary EI numerically underflows to zero, its unchanged mathematical
argmax is evaluated in the log domain.

Run or resume the additional configurations and create the boxplots with:

```powershell
python run_experiments.py ablation --workers 8
```

Run or resume the requested formal study without ablations with:

```powershell
python run_experiments.py formal --workers 8
```

The complete requested suite is run in this order:

```powershell
python run_experiments.py formal --workers 16
python run_experiments.py incumbent_ablation --workers 16
python run_experiments.py final_recommendation_ablation
python run_experiments.py ablation --workers 16
```

Individual phases are `confidence`, `lqei`, `comparison`, `ablation`,
`incumbent_ablation`, `final_recommendation_ablation`, and `analyze`. The two
rule ablations are run only when selected explicitly; they are not included
in `all`. This rerun explicitly executes both rule ablations and the
replication-order ablation after the formal comparison phase.
New confidence schedules, raw results, and analysis outputs are created locally
in this folder; no outputs from the baseline experiment were copied here.
