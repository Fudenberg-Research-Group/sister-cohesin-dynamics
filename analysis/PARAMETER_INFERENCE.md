# Parameter inference from cis and trans P(s)

This document records the goodness-of-fit calculation used to compare
experimental and simulated cis and trans P(s) curves. Contact extraction and
P(s) calculation are described in the main [analysis README](README.md).

## Combined log-space R-squared

The same scoring procedure is used for the G2-0 h initial-configuration sweep
and the WT residence-time sweep:

1. Normalize each experimental data set by its cis P(s) at 40 kb. Apply this
   same cis normalization factor to its trans P(s).
2. Normalize each simulation in the same way, using that simulation's cis
   P(s) at 40 kb for both its cis and trans curves.
3. Generate 500 logarithmically spaced genomic distances over the scoring
   interval. Average experimental P(s) within bins around this grid and
   linearly interpolate the untransformed simulated P(s) onto the grid.
4. Transform the valid cis and trans values with log10 and concatenate them.
5. Calculate one global R-squared value,

   ```text
   R2 = 1 - sum((log10(Pexp) - log10(Psim))^2)
            / sum((log10(Pexp) - mean(log10(Pexp)))^2).
   ```

Unless otherwise stated, the scoring interval is 40 kb to 5 Mb. A combined
score is used rather than averaging separate cis and trans scores, so the
reported value describes their joint agreement with the experiment.

The implementation is in `combined_log_r2()` in
[`inference_helpers.py`](inference_helpers.py). `score_two_parameter_grid()`
applies the calculation to any two-dimensional parameter sweep. Missing
parameter combinations are retained as `NaN` rather than silently omitted.

## G2-0 h initial sister configuration (Figure S2)

The G2-0 h sweep varied the input width of the sister-offset distribution and
the mean separation between neighboring sister linkers:

| Parameter | Sampled values |
|---|---|
| Input offset width, sigma | 50, 75, 100, 125, 150, 175 kb |
| Mean linker separation | 200 to 400 kb in 25 kb increments |

The combined log-space R-squared was evaluated from 40 kb to 5 Mb. Several
parameter combinations formed a broad near-optimal region with R-squared
greater than 0.996. Across the sampled grid, this region spanned input sigma
values of 100–175 kb and mean separations of 275–300 kb.

The anchors used for production simulations were generated with an input
sigma of 175 kb and a mean separation of 275 kb (2,909 linkers per replica),
giving a realized sister-offset standard deviation of approximately 141 kb.
The input sigma and the realized standard deviation are distinct quantities.


## WT linker residence time

The WT linker residence-time sweep was evaluated with the same combined
log-space R-squared definition over 40 kb to 5 Mb. At alpha = 50,000, the
finite residence-time grid sampled tau = 5–20 h in 1 h increments. An
additional tau = 50,000 h simulation represented the effectively infinite
residence-time limit. The finite grid reached its maximum at tau = 11 h
(R-squared = 0.989564), which was used for the WT production simulations.

## Extruder diffusion rate

The extruder diffusion-rate sweep was fitted to the delta-WAPL cis P(s) curve.
Experimental and simulated cis curves were normalized by their respective cis
P(s) at 40 kb and compared over 40 kb to 10 Mb on 1,000 logarithmically spaced
distances. Experimental P(s) was averaged around this grid, simulated P(s) was
interpolated onto it, and R-squared was calculated after log10 transformation.

The sampled `LEF_diffusion_rate` values were 0.01, 0.05, 0.10, 0.15, 0.20,
0.25, 0.30, 0.40, and 0.50 per second.

```text
k = LEF_diffusion_rate
```

The swept quantity is the per-leg step rate `k = LEF_diffusion_rate`
(s⁻¹), which corresponds to the diffusion coefficient only through the
lattice spacing. The selected value was
`k = 0.25 s^-1`. At the 0.5 kb lattice spacing used for the WAPL-depletion
simulations, the manuscript reporting convention gives

```text
2D = k * dx^2 = 0.25 s^-1 * (0.5 kb)^2 = 0.0625 kb^2/s.
```

Across the sampled diffusion rates, the cis log-space R-squared
was maximal at k = 0.25 s^-1.

The delta-SORORIN plus delta-WAPL condition was not fitted independently; it
used the `k = 0.25 s^-1` value inferred from delta-WAPL. The implementation is
provided by `cis_log_r2()` and `score_diffusion_rate_grid()` in
`inference_helpers.py`.

## WAPL-depletion encounter parameters

The encounter-parameter sweep was fitted to the delta-WAPL data only. Unlike
the preceding analyses, this fit used trans P(s) alone because the trans curve
was the target of the encounter-parameter inference. The experimental and
simulated trans curves were normalized by their corresponding cis P(s) values
at 40 kb. Experimental trans P(s) was averaged and simulated trans P(s) was
interpolated onto 1,000 logarithmically spaced distances from 40 kb to 10 Mb;
R-squared was then calculated from the log10-transformed values.

The reported grid varied the following encounter parameters while holding the
extruder diffusion rate fixed at 0.25 per second:

| Parameter | Sampled values |
|---|---|
| Sister damping factor, alpha | 10, 100, 1,000 |
| Collision-release probability, beta | `1e-7`, `1e-6`, `1e-5` per step |
| Bypass probability, gamma | `1e-1`, `1e-3`, `1e-5` per step |

The highest trans-only R-squared in the reported grid occurred at alpha = 100,
beta = `1e-6`, and gamma = `1e-3` (R-squared = 0.984), as marked by the star
in the manuscript heatmap. This parameter combination was used for production
simulations. More generally, beta and gamma were weakly identifiable: several
combinations across the sampled encounter regimes produced similar scores.
The numerical maximum is therefore not interpreted as a precise or unique
estimate of either probability.

The delta-SORORIN plus delta-WAPL condition was not fitted independently. It
used the same diffusion and encounter parameters selected from the delta-WAPL
sweep; only the condition-specific biochemical rates and linker lifetime were
changed.

The WAPL-depletion parameters were selected sequentially. First,
`k` was selected from the delta-WAPL cis P(s) log-space R-squared sweep described above.
With `k` fixed at 0.25 per second, alpha, beta, and gamma were then evaluated
using delta-WAPL trans P(s).

The implementation is provided by `trans_log_r2()` and
`score_trans_parameter_grid()` in `inference_helpers.py`. The latter accepts
an arbitrary number of parameter axes, including the three-dimensional alpha,
beta, and gamma grid used here.

The original sweep files encoded beta as the indices 0, 1, and 2 and omitted
the minus sign from gamma directory labels. Convert these storage labels to
physical parameter values before calling the reusable helper:

```python
from inference_helpers import score_trans_parameter_grid

beta_from_index = {0: 1e-7, 1: 1e-6, 2: 1e-5}
gamma_from_label = {
    "1e1": 1e-1,
    "1e3": 1e-3,
    "1e5": 1e-5,
}

ps_data_physical = {
    (alpha, beta_from_index[beta], gamma_from_label[gamma]): data
    for (alpha, beta, gamma), data in ps_data.items()
    if gamma in gamma_from_label
}

results = score_trans_parameter_grid(
    ps_data=ps_data_physical,
    parameter_axes={
        "alpha": [10, 100, 1000],
        "beta": [1e-7, 1e-6, 1e-5],
        "gamma": [1e-1, 1e-3, 1e-5],
    },
    dist_trans_exp=cvd_trans_dW["dist_bp"],
    ps_trans_exp=cvd_trans_dW["count.avg.smoothed.agg"],
    exp_cis_norm_factor=exp_cis_norm_factor_dW,
    start_dist=40_000,
    end_dist=10_000_000,
    n_log_points=1_000,
)
```

## Expected simulation data structure

The grid helper expects a dictionary keyed by parameter pairs. Each entry
contains the simulated cis and trans P(s) curves and the simulation's cis
normalization factor:

```python
ps_data[(row_value, column_value)] = {
    "dist_cis": dist_cis,
    "ps_cis": ps_cis,
    "dist_trans": dist_trans,
    "ps_trans": ps_trans,
    "norm_factor": cis_ps_at_40_kb,
}
```

The curves come from the contact maps written by
`contacts_vs_distance_from_trajectories.py` (run from `analysis/`). For
example, for the WT residence-time sweep, keyed by `(alpha, tau)`:

```python
from contacts_vs_distance_from_trajectories import (
    load_exp_ps, load_sim_ps, norm_factor, trajectory_dir)

NORM_DIST_BP = 40_000

ps_data = {}
for alpha in alphas:
    for tau in residence_times:
        path = trajectory_dir("WT12h", alpha, tau)
        dist_cis, ps_cis = load_sim_ps(path, "cis")
        dist_trans, ps_trans = load_sim_ps(path, "trans")
        ps_data[(alpha, tau)] = {
            "dist_cis": dist_cis,
            "ps_cis": ps_cis,
            "dist_trans": dist_trans,
            "ps_trans": ps_trans,
            "norm_factor": norm_factor(dist_cis, ps_cis, NORM_DIST_BP),
        }

dist_cis_exp, ps_cis_exp = load_exp_ps(EXP_CIS_COOL)
dist_trans_exp, ps_trans_exp = load_exp_ps(EXP_TRANS_COOL)
exp_cis_norm_factor = norm_factor(dist_cis_exp, ps_cis_exp, NORM_DIST_BP)
```

Simulated maps are balanced (`balanced.avg.smoothed.agg`) and experimental
maps are used unbalanced (`count.avg.smoothed.agg`). Both are normalized by
the corresponding cis P(s) at 40 kb; this processing difference had negligible
effect on parameter selection.

For the G2-0 h sweep, the two keys are input sigma and mean separation:

```python
from inference_helpers import score_two_parameter_grid

results = score_two_parameter_grid(
    ps_data=ps_data,
    row_values=sigma_list,
    column_values=separation_list,
    row_name="sigma_kb",
    column_name="separation_kb",
    dist_cis_exp=cvd_cis_S8h["dist_bp"],
    ps_cis_exp=cvd_cis_S8h["count.avg.smoothed.agg"],
    dist_trans_exp=cvd_trans_S8h["dist_bp"],
    ps_trans_exp=cvd_trans_S8h["count.avg.smoothed.agg"],
    exp_cis_norm_factor=exp_cis_norm_factor_S8h,
    start_dist=40_000,
    end_dist=5_000_000,
    n_log_points=500,
)

print(results["best_parameters"])
print(results["best_r2"])
```

For compatibility with the original WT analysis, `score_parameter_grid()` is
a WT-specific wrapper that retains the `best_alpha` and
`best_residence_time` result keys.
