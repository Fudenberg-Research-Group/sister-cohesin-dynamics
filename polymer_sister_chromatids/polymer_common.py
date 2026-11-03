"""Shared setup for the polymer run scripts (WT_dS_polymer.py, dW_dSdW_polymer.py).

Everything identical between the two conditions families lives here; the run
scripts only express which conditions they run and the lattice-to-monomer
scale factor.
"""

import codecs
import json
import os
import pickle
import time

import numpy as np
import hoomd
import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces
import polychrom_hoomd.extrude as extrude

# Bond types carried over from the pre-equilibrated system that must be
# cleared before the current configuration is applied.
STRIPPED_BOND_TYPES = ("LEF", "LEF_dummy", "Sister")


class TranslocatorFromTrajectory:
    """Minimal stand-in exposing the trajectories stored in a lattice pickle."""

    def __init__(self, traj):
        self.sister_trajectory = traj["sister"]
        self.lef_trajectory = traj["lef"]
        self.ctcf_trajectory = traj["ctcf"]


def init_simulation(readpath, iteration):
    """Load the pre-equilibrated system and strip its existing LEF/Sister bonds."""
    hoomd_device = build.get_hoomd_device()

    rng_seed = int(codecs.encode(os.urandom(2), 'hex'), 16)
    print("Using entropy-harvested random seed: %d" % rng_seed)

    sim = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
    sim.create_state_from_gsd(f"{readpath}/trajectory_4000_{iteration}.gsd", frame=-1)

    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        bond_types = snap.bonds.types
        bond_typeid = snap.bonds.typeid
        bond_group = snap.bonds.group

        remove_types = [bond_types.index(name)
                        for name in STRIPPED_BOND_TYPES if name in bond_types]

        if len(remove_types) > 0:
            mask = ~np.isin(bond_typeid, remove_types)
            snap.bonds.N = np.count_nonzero(mask)
            snap.bonds.group[:] = bond_group[mask]
            snap.bonds.typeid[:] = bond_typeid[mask]

    sim.state.set_snapshot(snap)
    return sim


def system_geometry(chromosome_sizes=(320000,), density=0.4):
    """Monomers per chromatid and the confining radius enclosing both.

    320000 monomers * 2.5 kb/monomer = 800 Mb per chromatid.
    """
    number_of_monomers = sum(chromosome_sizes)
    number_of_monomers_whole = 2 * number_of_monomers
    confinement_radius = (
        3 * number_of_monomers_whole / (4 * np.pi * density))**(1 / 3.)
    return number_of_monomers, confinement_radius


def build_force_field(force_dict_path, confinement_radius):
    """Assemble the relaxation and production force fields.

    Returns (relax_field, force_field). FIRE uses relax_field, which omits the
    angular terms; production uses force_field, which includes them.
    """
    nl = hoomd.md.nlist.Tree(buffer=0.2)

    with open(force_dict_path, 'r') as dict_file:
        force_dict = json.load(dict_file)

    repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
    bonded_forces = forces.get_bonded_forces(**force_dict)
    angular_forces = forces.get_angular_forces(**force_dict)

    force_dict['External forces']['Confinement']['Spherical'] = dict(
        R=confinement_radius)
    confinement_forces = forces.get_confinement_forces(**force_dict)

    relax_field = repulsion_forces + bonded_forces + confinement_forces
    force_field = relax_field + angular_forces

    return relax_field, force_field


def load_lattice_trajectories(read_traj_path, condition, iteration):
    """Load both chromatids' lattice trajectory pickles."""
    translocators = []
    for index in (1, 2):
        filename = f"{read_traj_path}/{condition}_trajectory{index}_{iteration}.pkl"
        with open(filename, "rb") as f:
            translocators.append(TranslocatorFromTrajectory(pickle.load(f)))
    return translocators[0], translocators[1]


def add_sister_bonds(sim, translocator1, translocator2,
                     savepath, iteration, N, scale=1):
    """Add inter-chromatid Sister bonds from the final lattice frame.

    scale > 1 maps a finer lattice onto polymer monomers. Floor division is
    required: -1 // 5 == -1, so the sentinel survives and is filtered out
    below. Truncating division would turn it into a valid position 0 and
    create spurious bonds.
    """
    arr1 = np.array(translocator1.sister_trajectory[-1], dtype=int) // scale
    arr2 = np.array(translocator2.sister_trajectory[-1], dtype=int) // scale

    anchors1 = arr1[arr1 != -1]
    anchors2 = arr2[arr2 != -1] + N

    bond_array = np.stack((anchors1, anchors2), axis=-1)
    bond_unique = np.unique(bond_array, axis=0)

    np.savetxt(f'{savepath}/anchor1_{iteration}.txt', anchors1)
    np.savetxt(f'{savepath}/anchor2_{iteration}.txt', anchors2)
    np.savetxt(f'{savepath}/bond_{iteration}.txt', bond_array)
    np.savetxt(f'{savepath}/bond_unique_{iteration}.txt', bond_unique)

    extrude.update_topology(sim, bond_unique, bond_name='Sister')
    print(f"Sister bonds: {len(bond_array)} ({len(bond_unique)} unique)")

    return bond_array


def add_lef_bonds(sim, translocator1, translocator2, N, scale=1):
    """Add LEF leg-pair bonds from the final lattice frame.

    Rows whose both legs are -1 (unbound LEFs) are dropped; see add_sister_bonds
    for why the division must floor.
    """
    lef1_all = np.array(translocator1.lef_trajectory[-1], dtype=int) // scale
    lef2_all = np.array(translocator2.lef_trajectory[-1], dtype=int) // scale

    lef1 = lef1_all[~np.all(lef1_all == -1, axis=1)]
    lef2 = lef2_all[~np.all(lef2_all == -1, axis=1)] + N

    extruders = np.r_[lef1, lef2]
    extrude.update_topology(sim, extruders, bond_name='LEF')
    print(f"LEF bonds: {len(extruders)}")

    return extruders


def relax_and_run(sim, relax_field, force_field, gsd_filename,
                  fire_blocks=10, fire_steps=50_000,
                  equil_steps=3e7, prod_steps=1e6, gsd_period=25000):
    """FIRE-relax, equilibrate unrecorded, then run a recorded production."""
    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    fire = hoomd.md.minimize.FIRE(dt=5e-3, methods=[nve], forces=relax_field,
                                  force_tol=5e-2, angmom_tol=5e-2,
                                  energy_tol=5e-2)

    # HOOMD to OpenMM time conversion factor
    t_conv = (1.67377e-27 / (1.380649e-23 * 300))**0.5
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    integrator = hoomd.md.Integrator(dt=15 * t_conv, methods=[langevin],
                                     forces=force_field)

    logger = log.get_logger(sim)
    sim.operations.integrator = fire
    sim.operations.writers.append(log.table_formatter(logger, period=1e3))

    fire_start = time.time()
    for _ in range(fire_blocks):
        sim.run(fire_steps)
    print(f"FIRE runtime: {(time.time() - fire_start) / 3600:.2f} hours")

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    sim.operations.integrator = integrator

    prod_start = time.time()
    sim.run(equil_steps)

    gsd_writer = hoomd.write.GSD(filename=gsd_filename,
                                 trigger=hoomd.trigger.Periodic(gsd_period),
                                 dynamic=['topology', 'momentum'],
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)
    sim.run(prod_steps)

    print(f"Production runtime: {(time.time() - prod_start) / 3600:.2f} hours")
