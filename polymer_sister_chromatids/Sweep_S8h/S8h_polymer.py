"""
S8h sweep, polymer simulation. Reads the pre-equilibrated two-chromatid HOOMD
system and the lattice-stage sister/LEF positions for each (sigma, sep) point
of the sweep, places those crosslinks/LEF bonds onto the system, then
FIRE-relaxes and runs recorded Langevin production.
"""

import codecs
import itertools
import json
import os
import sys
import time

import hoomd
import numpy as np
import polychrom_hoomd.build as build
import polychrom_hoomd.extrude as extrude
import polychrom_hoomd.forces as forces
import polychrom_hoomd.log as log

# --- sweep definition; must stay identical to S8h_lattice.py ---
NUMBER_OF_MONOMERS = 320000
KB_PER_MONOMER = 2.5                               # 320000 monomers = 800 Mb
SIGMA_KB_LIST = [50, 75, 100, 125, 150, 175]       # misalignment sigma, in kb
SISTER_SEP = [200, 225, 250, 275, 300, 325, 350, 375, 400]
NUM_SISTER_LIST = [round(800000 / s) for s in SISTER_SEP]
REPLICA_IDS = np.arange(5)

DENSITY = 0.4


def main():
    process_id = int(sys.argv[1]) - 1

    parameters = list(itertools.product(
        range(len(SIGMA_KB_LIST)), range(len(SISTER_SEP)), REPLICA_IDS))
    sigma_idx, sep_idx, iteration = parameters[process_id]

    sigma_kb = SIGMA_KB_LIST[sigma_idx]
    sep_value = SISTER_SEP[sep_idx]
    num_sister = NUM_SISTER_LIST[sep_idx]

    print("sigma_kb:", sigma_kb, "sep_value:", sep_value,
          "num_sister:", num_sister, "replica:", iteration)

    readpath = '../input'
    savepath = f'output/sigma{sigma_kb}/sep{sep_value}'
    os.makedirs(savepath, exist_ok=True)

    # Initialise HooMD on the CPU or GPU, based on availability
    hoomd_device = build.get_hoomd_device()

    rng_seed = int(codecs.encode(os.urandom(2), 'hex'), 16)
    print("Using entropy-harvested random seed: %d" % rng_seed)

    sim = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
    sim.create_state_from_gsd(
        f"{readpath}/trajectory_4000_{iteration}.gsd", frame=-1)

    # Remove the pre-equilibrated system's LEF and Sister bonds
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        bond_types = snap.bonds.types
        bond_typeid = snap.bonds.typeid
        bond_group = snap.bonds.group

        remove_types = [bond_types.index(name)
                        for name in ("LEF", "LEF_dummy", "Sister")
                        if name in bond_types]

        if len(remove_types) > 0:
            mask = ~np.isin(bond_typeid, remove_types)
            snap.bonds.N = np.count_nonzero(mask)
            snap.bonds.group[:] = bond_group[mask]
            snap.bonds.typeid[:] = bond_typeid[mask]

    sim.state.set_snapshot(snap)

    number_of_monomers_whole = 2 * NUMBER_OF_MONOMERS
    confinement_radius = (
        3 * number_of_monomers_whole / (4 * np.pi * DENSITY))**(1 / 3.)

    nl = hoomd.md.nlist.Tree(buffer=0.2)

    with open("../data/force_dict_sister_extruder.json", 'r') as dict_file:
        force_dict = json.load(dict_file)

    repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
    bonded_forces = forces.get_bonded_forces(**force_dict)
    angular_forces = forces.get_angular_forces(**force_dict)

    force_dict['External forces']['Confinement']['Spherical'] = dict(
        R=confinement_radius)
    confinement_forces = forces.get_confinement_forces(**force_dict)

    relax_field = repulsion_forces + bonded_forces + confinement_forces
    force_field = relax_field + angular_forces

    # Sister crosslinks from the lattice stage
    sister1 = np.loadtxt(
        f'{savepath}/anchor_{num_sister}_{iteration}_sister1.txt').astype(np.int64)
    sister2 = np.loadtxt(
        f'{savepath}/anchor_{num_sister}_{iteration}_sister2.txt').astype(np.int64)
    bond_array = np.column_stack((sister1, sister2))
    extrude.update_topology(sim, bond_array, bond_name='Sister')

    # Bound LEFs from the lattice stage
    extruders1 = np.genfromtxt(f'{savepath}/lef_traj1_{num_sister}_{iteration}.txt')
    extruders2 = np.genfromtxt(f'{savepath}/lef_traj2_{num_sister}_{iteration}.txt')
    extruders = np.r_[extruders1, extruders2]
    extrude.update_topology(sim, extruders, bond_name='LEF')

    # FIRE relaxation
    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    fire = hoomd.md.minimize.FIRE(dt=5e-3, methods=[nve], forces=relax_field,
                                  force_tol=5e-2, angmom_tol=5e-2,
                                  energy_tol=5e-2)

    # HooMD to openMM time conversion factor
    t_conv = (1.67377e-27 / (1.380649e-23 * 300))**0.5
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    integrator = hoomd.md.Integrator(dt=15 * t_conv, methods=[langevin],
                                     forces=force_field)

    logger = log.get_logger(sim)
    sim.operations.integrator = fire
    sim.operations.writers.append(log.table_formatter(logger, period=1e5))

    fire_start = time.time()
    for _ in range(10):
        sim.run(50_000)
    print(f"Fire runtime: {(time.time() - fire_start) / 3600:.2f} hours")

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    sim.operations.integrator = integrator

    prod_start = time.time()

    # Equilibration, unrecorded
    sim.run(5e6)

    # Production, recorded every 25000 steps
    gsd_writer = hoomd.write.GSD(
        filename=f'{savepath}/trajectory_S8h_sep{sep_value}_sigma{sigma_kb}_{iteration}.gsd',
        trigger=hoomd.trigger.Periodic(25000),
        dynamic=['topology', 'momentum'],
        mode='wb')
    sim.operations.writers.append(gsd_writer)
    sim.run(1e6)

    print(f"PROD runtime: {(time.time() - prod_start) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
