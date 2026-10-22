

# polymer_sister_chromatids/ — 3D sister-chromatid polymer simulations

Takes the linker and extruder configurations from the lattice stage in
`../discrete_time_extrusion_sister/` and places them on a two-chromatid polymer
model, then runs molecular dynamics using polychrom-hoomd.
Sister-linking cohesins become inter-chromatid bonds (`Sister`), loop-extruding
cohesins become bonds between the two extruder legs (`LEF`).

```
polymer_sister_chromatids/
├── WT_dS_polymer.py          # WT12h, dS12h
├── dW_dSdW_polymer.py        # dW, dSdW
├── data/
│   └── force_dict_sister_extruder.json   # force parameters
├── input/                    # pre-equilibrated configurations (not in repo)
├── output/                   # created per run
└── Sweep_S8h/                # standalone density x misalignment sweep
    ├── S8h_lattice.py
    ├── S8h_polymer.py
    └── WT_extrusion_rates.json
```

Two chromatids of 320,000 monomers each at 2.5 kb/monomer (800 Mb per
chromatid), monomer density 0.4, in a spherical confinement. Chromatid 2 is
offset by `+320000`, so an aligned linker is the bond `(i, i + 320000)`.

## Running

```bash
python WT_dS_polymer.py $SLURM_ARRAY_TASK_ID        # --array=1-5
```

The condition is set at the top of the script (`condition_name`); the array
index selects the replica. Each run reads the last frame of the pre-equilibrated
GSD, strips its existing `LEF` / `Sister` bonds, adds the bonds taken from the
final frame of the lattice trajectories (`-1` entries dropped), FIRE-relaxes,
equilibrates, then records a production trajectory.

Reads `../discrete_time_extrusion_sister/output/tau11h/{condition}_alpha_100_tau_11h_beta0_gamma0/{condition}_trajectory{1,2}_{i}.pkl`,
writes `output/{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0/` containing
the production `.gsd` plus the anchor and bond lists as text.

`dW_dSdW_polymer.py` divides lattice positions by 5 — see *Lattice resolution
and the `factor5` output path* in `../discrete_time_extrusion_sister/README.md`.

## Sweep_S8h/

A standalone sweep of linker density against misalignment, independent of the
biochemical pipeline: linkers are placed synthetically and extrusion runs
independently on each chromatid.

| Parameter | Values |
|---|---|
| Misalignment sigma | 50, 75, 100, 125, 150, 175 kb |
| Linker separation | 200-400 kb in 25 kb steps |
| Replicas | 5 |

```bash
python S8h_lattice.py $SLURM_ARRAY_TASK_ID        # --array=1-270, run lattice simulations first
python S8h_polymer.py $SLURM_ARRAY_TASK_ID        # --array=1-270, run polymer simulations then
```

Stage 1 writes the linker and LEF positions as text files; stage 2 reads them
and runs the polymer simulation. Both derive `(sigma, separation, replica)` from
the same array index, so they must be run over the same range and their
parameter blocks kept identical.


## Environment

[`HOOMD-blue`](https://hoomd-blue.readthedocs.io/),
[`polychrom-hoomd`](https://github.com/open2c/polychrom-hoomd), `gsd`, `numpy`.
GPU is used when available, otherwise CPU.

`Sweep_S8h/S8h_lattice.py` additionally needs the **non-sister** package
[discrete-time-extrusion](https://github.com/Fudenberg-Research-Group/discrete-time-extrusion)
(`discrete_time_extrusion.Translocator`,
`discrete_time_extrusion.extruders.MultistateExtruder`) — not the sister-aware
classes in `../discrete_time_extrusion_sister/`, installed separately.
