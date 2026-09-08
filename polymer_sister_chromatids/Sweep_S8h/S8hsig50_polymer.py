
"""
S8h sweep, polymer simulation. Reads the pre-equilibrated two-chromatid HOOMD
system and the lattice-stage sister/LEF trajectories for each (sep, sigma)
point of the sweep, places those crosslinks/LEF bonds onto the system,
then FIRE-relaxes and runs recorded Langevin production.
"""

import sys
import os
import json
import hoomd
import codecs
import time
import numpy as np

import gsd.hoomd
import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces
import polychrom_hoomd.extrude as extrude

# Initialise HooMD on the CPU or GPU, based on availability
hoomd_device = build.get_hoomd_device()

process_id = int(sys.argv[1])-1

sister_sep = [200, 225, 250, 275, 300, 325, 350, 375, 400]
num_sister_RAD21_list = [round(800000 / s) for s in sister_sep]

replica_ids = np.arange(5)

# Use index instead of value
sep_indices = np.arange(len(sister_sep))

x, y = np.meshgrid(sep_indices, replica_ids)
parameters = list(zip(x.flatten(), y.flatten()))

sep_idx, iteration = parameters[process_id]

# Now everything is recoverable
sep_value = sister_sep[sep_idx]
num_sister = num_sister_RAD21_list[sep_idx]

print("sep_value:", sep_value)
print("num_sister:", num_sister)
print("replica:", iteration)


sigma_offset = 50          # misalignment sigma, in monomers
kb_per_monomer = 2.5       # same lattice resolution as polymer stage (320000 monomers = 800 Mb)
sigma_kb = int(sigma_offset * kb_per_monomer)   # -> 125, used for output folder naming

readpath = '../input/Sister_4000_extrude_0_S1'
savepath = f'output/sigma{sigma_kb}/sep{sep_value}'

os.makedirs(savepath, exist_ok=True)

rng_seed = os.urandom(2)
rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)
print("Using entropy-harvested random seed: %d" % rng_seed)
# Initialize empty simulation object
sim = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
fname = f'trajectory_4000_{iteration}.gsd'
sim.create_state_from_gsd(f"{readpath}/{fname}", frame=-1)

## Remove LEF and Sister bonds from the snapshot
# snapshot-based bond deletion 
snap = sim.state.get_snapshot()
if snap.communicator.rank == 0:
    bond_types = snap.bonds.types
    bond_typeid = snap.bonds.typeid
    bond_group = snap.bonds.group

    # Find type indices
    remove_types = []
    for name in ["LEF", "LEF_dummy", "Sister"]:
        if name in bond_types:
            remove_types.append(bond_types.index(name))

    if len(remove_types) > 0:
        mask = ~np.isin(bond_typeid, remove_types)

        snap.bonds.N = np.count_nonzero(mask)
        snap.bonds.group[:] = bond_group[mask]
        snap.bonds.typeid[:] = bond_typeid[mask]
        
sim.state.set_snapshot(snap)

density = 0.4
# single chromsome size is 320000 monomer
# 320000*2.5kb/Monomer = 800Mb 
chromosome_sizes = [320000]
number_of_monomers = sum(chromosome_sizes)
number_of_monomers_whole = 2*number_of_monomers
confinement_radius = (3*number_of_monomers_whole / (4*np.pi*density))**(1/3.)

# Setup neighbor list
nl = hoomd.md.nlist.Tree(buffer=0.2)

# Read input force parameters
with open("../data/force_dict_sister_extruder.json", 'r') as dict_file:
    force_dict = json.load(dict_file)

# print(json.dumps(force_dict, indent=4, sort_keys=True))

# Set chromosome excluded volume
repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
# Set bonded/angular potentials
bonded_forces = forces.get_bonded_forces(**force_dict)
angular_forces = forces.get_angular_forces(**force_dict)

# Set spherical confining wall
force_dict['External forces']['Confinement']['Spherical'] = dict(R=confinement_radius)
confinement_forces = forces.get_confinement_forces(**force_dict)

# Define full force_field
relax_field = repulsion_forces + bonded_forces + confinement_forces 
force_field = relax_field + angular_forces

# load the sisterCs 
sister1 = np.loadtxt(f'{savepath}/anchor_{num_sister}_{iteration}_sister1.txt').astype(np.int64)
sister2 = np.loadtxt(f'{savepath}/anchor_{num_sister}_{iteration}_sister2.txt').astype(np.int64)

bond_array = np.column_stack((sister1, sister2))
extrude.update_topology(sim, bond_array, bond_name='Sister')


extruders1 = np.genfromtxt(f'{savepath}/lef_traj1_{num_sister}_{iteration}.txt')
extruders2 = np.genfromtxt(f'{savepath}/lef_traj2_{num_sister}_{iteration}.txt')
extruders = np.r_[extruders1, extruders2]
extrude.update_topology(sim, extruders, bond_name = 'LEF')


# Setup FIRE Steps
nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
fire = hoomd.md.minimize.FIRE(dt=5e-3, methods=[nve], forces=relax_field,
                              force_tol=5e-2, angmom_tol=5e-2, energy_tol=5e-2)

# Set up langevin integrator 
# HooMD to openMM time conversion factor
t_conv = (1.67377*10**-27/(1.380649*10**-23*300))**0.5
# Initialize integrators and Langevin thermostat
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
integrator = hoomd.md.Integrator(dt=15*t_conv, methods=[langevin], forces=force_field)

# Setup log and simulation engines
logger = log.get_logger(sim)

sim.operations.integrator = fire
sim.operations.writers.append(log.table_formatter(logger, period=1e5))

# Start timing production
fire_start_time = time.time()
print("Fire started at:", fire_start_time)

for _ in range(10):
    sim.run(50_000)

fire_end_time = time.time()
print("Fire ended at:", fire_end_time)


sim.state.thermalize_particle_momenta(
    filter = hoomd.filter.All(),
    kT = 1.0
)

sim.operations.integrator = integrator

# Start timing production
prod_start_time = time.time()
print("Production started at:", prod_start_time)

# Run equilibration, 5e6 steps
sim.run(5e6)

# Production phase - record as regular trajectory, 25000
gsd_writer = hoomd.write.GSD(filename=f'{savepath}/trajectory_S8h_sep{sep_value}_sigma{sigma_kb}_{iteration}.gsd',
                            trigger=hoomd.trigger.Periodic(25000),
                            dynamic=['topology', 'momentum'],
                            mode='wb')
sim.operations.writers.append(gsd_writer)

# Run production, 1e6 steps 
sim.run(1e6)

# Start timing production
prod_end_time = time.time()
print("Production ended at:", prod_end_time)

# fire time 
fire_runtime_hours = (fire_end_time - fire_start_time) / 3600
print(f"Fire runtime: {fire_runtime_hours:.2f} hours")

# prod time 
prod_runtime_hours = (prod_end_time - prod_start_time) / 3600
print(f"PROD runtime: {prod_runtime_hours:.2f} hours")


