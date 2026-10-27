# Sister cohesin dynamics

A multiscale model linking cohesin regulation and dynamics to sister-chromatid folding.

![Workflow overview](docs/Model_Schematic.png)

## Overview

This repository models sister-chromatid cohesion and loop extrusion during G2. Biochemical reaction networks determine cohesin kinetics, paired one-dimensional lattices simulate cohesin transport and encounters, and 3D polymer simulations translate the resulting cohesin configurations into chromosome conformations. Simulated folding can be compared with scsHi-C and imaging measurements.

## Workflow

1. **Biochemical kinetics — `chemistry/`**

   Uses cohesin–regulator binding measurements to parameterize reaction networks and derive cohesin transition kinetics for WT and perturbations.
   See [`chemistry/README.md`](chemistry/README.md).

2. **Lattice dynamics — `discrete_time_extrusion_sister/`**

   Simulates loop-extruding and sister-linking cohesins on paired sister-chromatid lattices. The model combines condition-specific biochemical kinetics with loop extrusion, extruder diffusion, and linker–extruder encounters, including pushing, collision-mediated removal, and bypass.
   Condition-specific simulations are implemented in `WT_dS.py` and `dW_dSdW.py`; sister-linking cohesin dynamics are implemented in `sister_lifecycle.py`.
   See [`discrete_time_extrusion_sister/README.md`](discrete_time_extrusion_sister/README.md).

3. **3D chromosome folding — `polymer_sister_chromatids/`**

   Converts lattice configurations into bonds in a two-sister-chromatid polymer model. Sister linkers form inter-sister bonds, and extruders form bonds between their two legs. Simulations use HOOMD-blue and polychrom-hoomd, with FIRE relaxation followed by Langevin molecular dynamics.

## Additional analyses

`polymer_sister_chromatids/Sweep_S8h/` contains a separate parameter sweep examining how sister-linker density and misalignment affect chromosome folding. Sister linkers are initialized at specified densities and misalignments, while loop extrusion proceeds independently on each sister chromatid.

## Requirements

- **`chemistry/`**: `sympy`, `tellurium`, `pandas`
- **`discrete_time_extrusion_sister/`**: `numpy`, `numba`; `cupy` is optional for GPU acceleration
- **`polymer_sister_chromatids/`**: [`HOOMD-blue`](https://hoomd-blue.readthedocs.io/), [`polychrom-hoomd`](https://github.com/open2c/polychrom-hoomd), `gsd`

## Attribution

The loop-extrusion implementation builds on the [discrete-time extrusion framework](https://github.com/Fudenberg-Research-Group/discrete-time-extrusion) described in Tortora et al., *The physical chemistry of interphase loop extrusion* ([Cell Genomics, 2025](https://doi.org/10.1016/j.xgen.2025.101098)).

## Associated publication

Yan, X., Tortora, M., Goloborodko, A., Fudenberg, G.
*Chromosome folding encodes mechanisms of cohesion maintenance.*
Manuscript in preparation (2026).
