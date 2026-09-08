# sister_discrete_translocator

![workflow overview](docs/Cartoon_Schematic.png)

**Illustration**: Kinetic parameters fit to *in vivo* cohesin/regulator binding data (`chemistry/`) are used to drive a 1D lattice model of loop extrusion and cohesion maintainance (`discrete_time_extrusion_sister/`), whose trajectories are then placed onto a 3D HOOMD polymer model to simulate large-scale chromatid organization (`polymer_sister_chromatids/`). A separate side sweep (`Sweep_S8h/`) reuses the same polymer stage to explore sister-crosslink density and misalignment without sister-lifecycle coupling.

### Description

This repository simulates cohesin loop extrusion together with sister-chromatid cohesion, to ask whether maintaining sister cohesion through G2 requires modeling cohesin (de)acetylation / the WAPL–PDS5–Sororin regulatory cycle, or whether a simpler static-cohesion model is sufficient. It combines a chemical-reaction-network fit of cohesin regulator kinetics, a 1D lattice loop-extrusion simulation with an explicit sister-cohesin lifecycle (loading, decay, collision release, bypass), and 3D molecular-dynamics simulation of the resulting chromatid conformations.

Manuscript: in preparation.

### Requirements

- **chemistry/**: `sympy`, `tellurium`, `pandas` (symbolic CRN solving + ODE simulation)
- **discrete_time_extrusion_sister/**: `numpy`, `numba` (CPU backend); `cupy` optional for a GPU backend
- **polymer_sister_chromatids/**: [`HOOMD-blue`](https://hoomd-blue.readthedocs.io/), [`polychrom-hoomd`](https://github.com/open2c/polychrom-hoomd), `gsd`

### Workflow

The pipeline runs in three stages, condition by condition (`WT`, `dS`, `dW`, `dSdW`, ...):

1. **`chemistry/`** — fits the (WAPL, PDS5) operating point and derives per-condition LEF on/off/transition rates from the cohesin regulator kinetics. See [`chemistry/README.md`](chemistry/README.md).
2. **`discrete_time_extrusion_sister/`** — runs the paired-chromatid 1D lattice extrusion model (`WT_dS.py`, `dW_dSdW.py`), coupling loop extrusion to an explicit sister-cohesin lifecycle (`sister_lifecycle.py`). See [`discrete_time_extrusion_sister/README.md`](discrete_time_extrusion_sister/README.md).
3. **`polymer_sister_chromatids/`** — reads the lattice-stage trajectories and places the corresponding crosslink/LEF bonds onto a pre-equilibrated two-chromatid HOOMD polymer system (`WT_dS_polymer.py`, `dW_dSdW_polymer.py`), then runs FIRE relaxation followed by recorded Langevin production.

**`polymer_sister_chromatids/Sweep_S8h/`** is a separate, self-contained sweep: it assigns sister crosslinks at a fixed density/misalignment and runs each chromatid's extrusion independently (no sister-lifecycle coupling), using the crosslinks as static boundary stalls for the extruder (`S8hsig50_lattice.py` → `S8hsig50_polymer.py`).

### Attribution

The plain (non-sister) `Translocator` / `MultistateExtruder` classes used by `Sweep_S8h/` come from the Fudenberg Research Group's original 1D lattice loop-extrusion code: [Fudenberg-Research-Group/discrete-time-extrusion](https://github.com/Fudenberg-Research-Group/discrete-time-extrusion), associated with Tortora et al., "The physical chemistry of interphase loop extrusion" ([bioRxiv 2024.08.23.609419](https://www.biorxiv.org/content/10.1101/2024.08.23.609419v1)). This repository extends that base code with sister-chromatid-aware classes (`Translocator_Sister`, `MultistateExtruder_Sister`, `sister_lifecycle.py`) to simulate cohesin loop extrusion together with sister-chromatid cohesion.
