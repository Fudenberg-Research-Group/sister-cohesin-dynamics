# Cohesin biochemical kinetics

<p align="center">
  <img src="../docs/Biochemistry_Schematic.png"
       alt="Cohesin biochemical kinetics workflow"
       width="1000">
</p>


Three-step pipeline that derives the kinetic parameters for the loop-extruding and sister-linking cohesins
used by the lattice simulation in `../discrete_time_extrusion_sister/`.

```
chemistry/
├── Step1_Centroid/               # find the (WAPL, PDS5) fraction-bound at linkers centroid
├── Step2_Kinetics/               # validate RAD21 residence-time kinetics at that centroid
└── Step3_Extrusion_Parameters/   # generate the final per-condition extrusion-rate JSONs
```

## Data flow

```
Step1_Centroid/CRN_centroid.ipynb
    → centroid (x, y) ≈ (0.12028, 0.30274)
        → Step2_Kinetics/bounded_linker_over_time_tau10h.ipynb
        → Step3_Extrusion_Parameters/network_parameters_{condition}.json
            → {condition}_rates.py
                → Step3_Extrusion_Parameters/{condition}/{condition}_alpha..._tau..h_beta.._gamma..json
                    → ../discrete_time_extrusion_sister/WT_dS.py, dW_dSdW.py
```

## Step1_Centroid/ — `CRN_centroid.ipynb`

Solves two chemical reaction networks symbolically (sympy): the **linker /
cohesive network** (PDS5, WAPL, SORORIN binding + RAD21 unloading) and the
**extrusive network** (NIPBL, PDS5, WAPL binding + RAD21 unloading). Both are
solved for steady-state rate expressions, then evaluated over a grid of
`(x, y)` = (fraction WAPL bound, fraction PDS5 bound at the linkers) to find
the region where every solved rate is positive (the physically valid
"feasible region"). The centroid of that region
`(x, y) ≈ (0.12028, 0.30274)` is selected as the parameter point for downstream
calculations.

- **Input:** `parameters_10h.json` — base network parameters. `Step2_Kinetics/` reads it from here too.
- **Output:** centroid of the feasible region.

## Step2_Kinetics/ — `bounded_linker_over_time_tau10h.ipynb`

Takes the Step1 centroid, builds the linker/cohesive reaction network, and simulates the fraction of
RAD21-bound cohesin over time (spanning G2) for WT and several
depletion conditions: dSORORIN, dWAPL, dSORORIN+dWAPL, dPDS5, and dRAD21.

- **Input:** `../Step1_Centroid/parameters_10h.json`.

## Condition naming

The simulated timeline starts at S-phase entry (S0h) and ends at G2-16h, 24 h
in total (8 h of S phase followed by 16 h of G2).

| Name | Perturbation | Lattice resolution | Used by |
|---|---|---|---|
| `WT12h` | none | 2500 bp/site | `WT_dS.py`, single phase |
| `dS12h` | SORORIN depleted | 2500 bp/site | `WT_dS.py`, single phase |
| `WT7h` | none | 500 bp/site | `dW_dSdW.py`, phase 1 |
| `dS7h` | SORORIN depleted | 500 bp/site | `dW_dSdW.py`, phase 1 |
| `dW` | WAPL depleted | 500 bp/site | `dW_dSdW.py`, phase 2 |
| `dSdW` | SORORIN + WAPL depleted | 500 bp/site | `dW_dSdW.py`, phase 2 |

Each `{condition}_rates.py` integrates the network from its own origin and
reads the rates out at the midpoint of that integration window
(`analysis_timepoint_hours` in `network_parameters_{condition}.json`; for
`dW` and `dSdW`, the midpoint of the post-depletion window). SORORIN-depleted
conditions start at S0h, because SORORIN loss already affects linkers during
S phase; WT and dW start at G2 entry (S8h) with linkers fully established.
The `12h` / `7h` in a name labels the run family (single-phase 16 h G2 run
vs phase 1 of the two-phase run).

| Name | ODE origin | Integration | Readout (local) | Readout (timeline) |
|---|---|---|---|---|
| `WT12h` | S8h | 16 h (to S24h) | 8 h | S16h (G2-8h) |
| `dS12h` | S0h | 24 h (to S24h) | 12 h | S12h (G2-4h) |
| `WT7h` | S8h | 6 h (to S14h) | 3 h | S11h (G2-3h) |
| `dS7h` | S0h | 14 h (to S14h) | 7 h | S7h |
| `dW` | S8h; WAPL depleted at S14h | 6 h + 10 h | 5 h after depletion | S19h (G2-11h) |
| `dSdW` | S0h; WAPL depleted at S14h | 14 h + 10 h | 5 h after depletion | S19h (G2-11h) |

The `tau11h` that also appears in the parameter filenames is a different
quantity: it is the RAD21 residence time (`RESIDENCE_TIMES_HOURS = [11]`), not
a timepoint.

## Step3_Extrusion_Parameters/ — `{condition}_rates.py`

This is what generates the parameter JSONs read by the lattice simulations in
`../discrete_time_extrusion_sister/`

- **`crn_core.py`** — shared module. Solves the cohesive and extrusive
  networks symbolically for their rate constants once at import
  (`sol_rates_coh`, `sol_rates_ext`), defines the mass-action ODEs of the
  combined network (`_network_rhs`), and integrates them with SciPy
  `solve_ivp` (LSODA, rtol = atol = 1e-8) in `simulate_network`. It also
  provides `calculate_cohesive_parameters`, `calculate_extrusive_parameters`
  and `calculate_sister_RAD21_bound_time`.
- **One script per condition** — `WT12h_rates.py`, `WT7h_rates.py`,
  `dS12h_rates.py`, `dS7h_rates.py`, `dW_rates.py`, `dSdW_rates.py`. Each
  loads its own `network_parameters_{condition}.json`, integrates the
  combined network with `simulate_network`, and derives the LEF on/off/
  transition rates from the simulated state at a chosen analysis timepoint.
- **`extrusion_dict_RN_RB_RP_RW_HBD_WT.json`** — base template. Each
  condition script loads it and overwrites the rate / species-count fields
  with its own computed values before saving.
- **Output:**
  `{condition}/{condition}_alpha{sister_damping}_tau{residence_time}h_beta{k}_gamma{q}.json`
  — this is the file `../discrete_time_extrusion_sister/WT_dS.py` and
  `dW_dSdW.py` load directly.

Run with, e.g.:

```bash
python WT12h_rates.py
```

No CLI args — the condition, its `network_parameters_*.json`, and the sweep
ranges (residence time / sister damping / collision-release / bypass
probabilities) are set as constants near the top of each script.

Conditions currently generated, and how the run scripts in
`discrete_time_extrusion_sister/` consume them:

| Condition | Used as |
|---|---|
| `WT12h` | `WT_dS.py`, `CONDITIONS["WT"]` |
| `dS12h` | `WT_dS.py`, `CONDITIONS["dS"]` |
| `WT7h` | `dW_dSdW.py`, phase 1 of `CONDITIONS["WT_dW"]` |
| `dW` | `dW_dSdW.py`, phase 2 of `CONDITIONS["WT_dW"]` |
| `dS7h` | `dW_dSdW.py`, phase 1 of `CONDITIONS["dS_dSdW"]` |
| `dSdW` | `dW_dSdW.py`, phase 2 of `CONDITIONS["dS_dSdW"]` |

## Environment

Beyond the usual numpy/scipy/matplotlib stack, this pipeline additionally
needs:

- `sympy` (symbolic CRN solving, all three steps)
