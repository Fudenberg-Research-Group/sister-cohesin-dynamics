"""Prepare the pre-equilibrated two-chromatid conformation read by every
polymer run (input/trajectory_4000_{replica}.gsd).

One chain of 320,000 monomers is grown on a cubic lattice and duplicated into
two sister chromatids, which are joined by 4000 aligned, randomly placed
linkers to keep them paired during preparation. The system is equilibrated
with Langevin dynamics for 1e7 steps and then sampled for 5e6 steps. The
downstream scripts read the last frame, remove these linkers and add the
lattice-stage linkers and extruders. Each replica uses an independent random
seed.

Usage (from polymer_sister_chromatids/, GPU recommended):
    python prepare_initial_conformation.py --replica 0
"""
import argparse
import codecs
import json
import os
import time

import gsd.hoomd
import hoomd
import numpy as np
import polychrom_hoomd.build as build
import polychrom_hoomd.extrude as extrude
import polychrom_hoomd.forces as forces
import polychrom_hoomd.log as log
from polykit.generators.initial_conformations import grow_cubic


def crosslink_chains_random(system, N_crosslinks, iteration, path):
    """Add N_crosslinks aligned sister links (i, i + N) at random positions."""
    N_monomers = system.state.N_particles // 2
    anchor_positions = np.random.choice(np.arange(N_monomers), size = N_crosslinks, replace=False)
    bond_array = np.stack((anchor_positions, anchor_positions + N_monomers), axis=-1)
    np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}.txt', anchor_positions)
    extrude.update_topology(system, bond_array, bond_name='Sister')


def replicate_chain(system):
    ## Function to make a same copy of the sister chromatid
    ## copy the exsiting data into snapshots
    snap = system.state.get_snapshot()
    ## bring into a new snapshot, creating an empty snapshot
    snap_gsd = gsd.hoomd.Frame()

    N_bonds = snap.bonds.N
    N_angles = snap.angles.N
    N_monomers = snap.particles.N

    ## group is indexes of particles taking part in either bonds or angles or xxx
    ## 2 particles
    bond_groups = np.zeros((2*N_bonds, 2), dtype=np.uint32)
    ## 3 particles per angle
    angle_groups = np.zeros((2*N_angles, 3), dtype=np.uint32)

    ## types of the bonds
    bond_typeids = np.zeros(2*N_bonds, dtype=np.uint32)
    angle_typeids = np.zeros(2*N_angles, dtype=np.uint32)
    particle_typeids = np.zeros(2*N_monomers, dtype=np.uint32)

    # Bond resizing in HOOMD v3 requires full array reassignment
    snap_gsd.bonds.N = 2*N_bonds
    snap_gsd.angles.N = 2*N_angles
    snap_gsd.particles.N = 2*N_monomers

    masses = np.tile(snap.particles.mass, (2, 1))
    diameters = np.tile(snap.particles.diameter, (2, 1))
    positions = np.tile(snap.particles.position, (2, 1))

    bond_groups[:N_bonds] = snap.bonds.group
    bond_groups[N_bonds:] = snap.bonds.group + N_monomers

    angle_groups[:N_angles] = snap.angles.group
    angle_groups[N_angles:] = snap.angles.group + N_monomers

    particle_typeids[:N_monomers] = snap.particles.typeid
    particle_typeids[N_monomers:] = snap.particles.typeid

    bond_typeids[:N_bonds] = snap.bonds.typeid
    bond_typeids[N_bonds:] = snap.bonds.typeid

    angle_typeids[:N_angles] = snap.angles.typeid
    angle_typeids[N_angles:] = snap.angles.typeid

    snap_gsd.bonds.types = snap.bonds.types
    snap_gsd.angles.types = snap.angles.types
    snap_gsd.particles.types = snap.particles.types

    snap_gsd.particles.mass = masses
    snap_gsd.particles.diameter = diameters

    snap_gsd.particles.position = positions
    snap_gsd.particles.typeid = particle_typeids

    snap_gsd.bonds.group = bond_groups
    snap_gsd.bonds.typeid = bond_typeids

    snap_gsd.angles.group = angle_groups
    snap_gsd.angles.typeid = angle_typeids

    # Configuration step/box data requires manual setting as of gsd v2.8
    snap_gsd.configuration.step = system.timestep
    snap_gsd.configuration.box = snap.configuration.box

    # Load snapshot and re-thermalize, if required
    snap = hoomd.Snapshot.from_gsd_snapshot(snap_gsd, snap.communicator)

    system.state.set_snapshot(snap)


# Initialise HooMD on the CPU or GPU, based on availability
hoomd_device = build.get_hoomd_device()

parser = argparse.ArgumentParser(
    description="Prepare a pre-equilibrated two-chromatid conformation.")
parser.add_argument("--replica", type=int, choices=range(5), default=0,
                    help="Zero-based replica index (0-4).")
iteration = parser.parse_args().replica

NUM_PREP_LINKS = 4000      # aligned sister links used only during preparation
print(f"Preparing replica {iteration}")


savepath = 'input'
os.makedirs(savepath, exist_ok=True)

rng_seed = os.urandom(2)
rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)
print("Using entropy-harvested random seed: %d" % rng_seed)
# Initialize empty simulation object
sim = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
# Two chromatids of 320,000 monomers each (2.5 kb/monomer, 800 Mb) at number
# density 0.4 in a spherical confinement.
density = 0.4
chromosome_sizes = [320000]
# Initialize simulation with the appropriate box size
number_of_monomers = sum(chromosome_sizes)

number_of_monomers_whole = 2*number_of_monomers
confinement_radius = (3*number_of_monomers_whole / (4*np.pi*density))**(1/3.)

snapshot = build.get_simulation_box(box_length = 2*confinement_radius, pad=1.)

# Build random, dense initial conformations
inscribed_cube_length = confinement_radius*2/3**.5
monomer_positions = grow_cubic(N=number_of_monomers, boxSize=int(inscribed_cube_length))
# Populate snapshot with the generated chains
build.set_chromosomes(snapshot, monomer_positions, chromosome_sizes, monomer_type_list=['A'], bond_type_list=['Backbone', 'LEF', 'LEF_dummy', 'Sister', 'Sister_dummy'], center= True)

sim.create_state_from_snapshot(snapshot)
# Setup neighbor list
nl = hoomd.md.nlist.Cell(buffer=0.2)
# Read input force parameters
with open("data/force_dict_sister_extruder.json", 'r') as dict_file:
    force_dict = json.load(dict_file)

# Set chromosome excluded volume
repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
# Set bonded/angular potentials
bonded_forces = forces.get_bonded_forces(**force_dict)
angular_forces = forces.get_angular_forces(**force_dict)
# Set spherical confining wall
force_dict['External forces']['Confinement']['Spherical'] = dict(R=confinement_radius)
confinement_forces = forces.get_confinement_forces(**force_dict)


force_field_langevin = repulsion_forces + bonded_forces + angular_forces + confinement_forces

# HooMD to openMM time conversion factor
t_conv = (1.67377*10**-27/(1.380649*10**-23*300))**0.5

# Initialize integrators and Langevin thermostat
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = 1.0)
integrator_langevin = hoomd.md.Integrator(dt = 15*t_conv, methods=[langevin], forces = force_field_langevin)

# Setup log and simulation engines
logger = log.get_logger(sim)

sim.operations.integrator = integrator_langevin
sim.operations.writers.append(log.table_formatter(logger, period=1e4))

replicate_chain(sim)

# Join the sisters with aligned, randomly placed linkers so they stay paired
# during preparation. Downstream runs strip these and add the lattice-stage
# linkers and extruders instead.
crosslink_chains_random(sim, NUM_PREP_LINKS, iteration, path = savepath)

# Start timing equilibration
eq_start_time = time.time()
# Equilibrate for 1e7 steps.
for _ in range(100):
    sim.run(100_000)

# Calculate equilibration time
eq_end_time = time.time()
eq_runtime_hours = (eq_end_time - eq_start_time) / 3600
print(f"Equilibration runtime: {eq_runtime_hours:.2f} hours")

# Start timing production
prod_start_time = time.time()

gsd_writer = hoomd.write.GSD(filename=f'{savepath}/trajectory_{NUM_PREP_LINKS}_{iteration}.gsd',
                            trigger=hoomd.trigger.Periodic(100000),
                            dynamic=['topology', 'momentum'],
                            mode='wb')
sim.operations.writers.append(gsd_writer)

# Production: 5e6 steps, one frame every 1e5 steps.
sim.run(5e6)

# Calculate production time
prod_end_time = time.time()
prod_runtime_hours = (prod_end_time - prod_start_time) / 3600
print(f"Sampling runtime: {prod_runtime_hours:.2f} hours")

# Calculate total time
total_runtime_hours = eq_runtime_hours + prod_runtime_hours
print(f"Total runtime: {total_runtime_hours:.2f} hours")
