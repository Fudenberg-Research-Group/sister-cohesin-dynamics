# discrete_time_extrusion (sister-aware fork)

Sister-chromatid-aware loop-extrusion simulation.  1D lattice LEF model with "sister" anchor per pair of
chromatids, so that cohesin loop extrusion and sister-chromatid cohesion
can be simulated together.

## Repo layout

```
discrete_time_extrusion_code/
├── WT_dS.py                  # single-phase run: WT alone, or dSORORIN alone
├── dW_dSdW.py                # two-phase run: WT->dW, or dS->dSdW
├── input/
│   ├── anchor_2909_{i}_sister1.txt   # initial sister positions, chromatid 1
│   └── anchor_2909_{i}_sister2.txt   # initial sister positions, chromatid 2
├── output/                   # created automatically when a run script executes
└── discrete_time_extrusion/  # the model package
    ├── Translocator_Sister.py        # one chromatid's translocator
    ├── sister_lifecycle.py           # sister decay/collision + synchronized run loop
    ├── arrays.py                     # shared CTCF/LEF rate-array helpers
    ├── boundaries/                   # CTCF barrier engines
    └── extruders/                    # LEF + sister extruder engines
```

The parameter JSON files are in one level above this folder, in
`../chemistry/Step2_Generate_dicts/{condition}/{condition}_alpha{...}_tau{...}h_beta{...}_gamma{...}.json`
(e.g. `../chemistry/Step2_Generate_dicts/WT3h/WT3h_alpha100_tau11h_beta0_gamma0.json`).
These are produced upstream (Step 1/Step 2 of the parameter-fitting
pipeline, not part of this repo) and are not regenerated here.


## Environment

- Python (conda env `discrete-extrusion`)
- numpy, numba (CPU backend); cupy is optional or a GPU backend

```bash
conda activate discrete-extrusion
```

## Input data

**Config JSON** (one per condition, e.g. `WT3h`, `dS7h`, `dW`, `dSdW`):
holds the physical/simulation parameters read via `**paramdict` in
`Translocator_Sister`, including region size, LEF on/off/diffusion rates,
CTCF stall probabilities, sister lifetime/damping/bypass probability,
number of sisters, and step counts. See `discrete_time_extrusion/arrays.py`
and `Translocator_Sister.py` for the exact keys consumed.

**Anchor files** (`input/anchor_2909_{iteration}_sister1.txt` /
`sister2.txt`): initial sister-cohesin positions per replica/iteration.
These come from a separate sweeping of misalignment and linker
density parameters and identifying the best fit to expermental scsHi-C at G2-0h—
not generated inside this repo. If `num_of_sisters` in the condition's
JSON is smaller than the number of positions in the anchor file (the dS/
dSdW case), the scripts randomly subsample `num_of_sisters` of them
(same indices for both sister files, seeded by `iteration` for
reproducibility).

## Running a simulation

Both scripts run one replica at a time, selected via `process_id`
(hardcoded to `0` for local testing; set from `sys.argv[1]` for a cluster
job array — see the commented-out line near the top of each script).

**Single-phase (`WT_dS.py`)** — runs one condition once, no parameter
switch mid-run:

```python
CONDITIONS = {
    "WT": "WT12h",
    "dS": "dS12h",
}
condition_name = "WT"   # or "dS"
```

**Two-phase (`dW_dSdW.py`)** — runs a "phase 1" condition, then switches
parameters (`update()`) and continues as "phase 2":

```python
CONDITIONS = {
    "WT_dW": ("WT3h", "dW"),
    "dS_dSdW": ("dS7h", "dSdW"),
}
condition_name = "WT_dW"   # or "dS_dSdW"
```

Both scripts also expose `tau`, `sister_dampings` (alpha),
`collision_prob_index` (beta), and `bypass_index` (gamma) — these select
which JSON file gets loaded and are encoded into the output folder name.

```bash
python WT_dS.py
python dW_dSdW.py
```

`steps` is currently hardcoded to `10` in both scripts for local
smoke-testing. For a full run, see the derivation comments above each
`run_synchronized_trajectories_continue(...)` call (production values:
86400 / 144000 depending on phase).

## Output

Each run writes to `output/tau{tau}h/{condition}_alpha_{...}_tau_{...}h_beta{...}_gamma{...}/`:

- `{condition}_trajectory1_{iteration}.pkl`, `{condition}_trajectory2_{iteration}.pkl`
  — one per chromatid, each a dict with keys `sister`, `lef`, `ctcf`,
  `state`, `coupling` (trajectories sampled every `sample_interval` steps)
- `ctcf_left_positions_full.npy`, `ctcf_right_positions_full.npy`

