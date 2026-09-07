
"""
Reads the last frame of a pre-computed lattice-extrusion sister/LEF
trajectory (from dW_dSdW.py) and places those crosslinks/LEF bonds onto a
pre-equilibrated two-chromatid HOOMD polymer system, then FIRE-relaxes,
equilibrates, and runs recorded Langevin production.
"""

import os
import sys
import time
import json
import pickle
import codecs
import numpy as np
import hoomd
import gsd.hoomd
import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces
import polychrom_hoomd.extrude as extrude 

CONDITIONS = {
    "dW": "dW",
    "dSdW": "dSdW",
}
condition_name = "dSdW"
condition = CONDITIONS[condition_name]


process_id = int(sys.argv[1])-1

replica_ids = np.arange(5)

alpha = 100
tau = 11
iteration = replica_ids[process_id] 

readpath = 'input/Sister_4000_extrude_0_S1'

savepath = f'output/{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0'

read_traj_path = f'../discrete_time_extrusion_sister/output/tau11h/{condition}_alpha_100_tau_11h_beta0_gamma0'

class TranslocatorFromTrajectory:
    def __init__(self, traj):
        self.sister_trajectory = traj["sister"]
        self.lef_trajectory = traj["lef"]
        self.ctcf_trajectory = traj["ctcf"]

def crosslink_from_translocators_5(iteration, system, translocator1, translocator2, path, N):
    """
    Add sister crosslinks from existing translocator objects.
    """

    arr1 = np.array(translocator1.sister_trajectory[-1]//5, dtype=int)
    anchors1 = arr1[arr1 != -1]
    arr2 = np.array(translocator2.sister_trajectory[-1]//5, dtype=int)
    anchors2 = arr2[arr2 != -1] + N

    bond_array = np.stack((anchors1, anchors2), axis=-1)

    bond_unique = np.unique(bond_array, axis=0)

    if iteration is not None:
        np.savetxt(f'{path}/anchor1_{iteration}.txt', anchors1)
        np.savetxt(f'{path}/anchor2_{iteration}.txt', anchors2)
        np.savetxt(f'{path}/bond_{iteration}.txt', bond_array)
        np.savetxt(f'{path}/bond_unique_{iteration}.txt', bond_unique)

    extrude.update_topology(system, bond_unique, bond_name='Sister')

    return bond_array
        
# Initialise HooMD on the CPU or GPU, based on availability
hoomd_device = build.get_hoomd_device()

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
# Initialize simulation with the appropriate box size 
number_of_monomers = sum(chromosome_sizes)
number_of_monomers_whole = 2*number_of_monomers
confinement_radius = (3*number_of_monomers_whole / (4*np.pi*density))**(1/3.)

# Setup neighbor list
nl = hoomd.md.nlist.Tree(buffer=0.2)

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

# Define full force_field
relax_field = repulsion_forces + bonded_forces + confinement_forces 
force_field = relax_field + angular_forces

# Add extruders 
with open(f"{read_traj_path}/{condition}_trajectory1_{iteration}.pkl", "rb") as f:
    traj1 = pickle.load(f)

with open(f"{read_traj_path}/{condition}_trajectory2_{iteration}.pkl", "rb") as f:
   traj2 = pickle.load(f)

translocator1 = TranslocatorFromTrajectory(traj1)
translocator2 = TranslocatorFromTrajectory(traj2)

# Add sister links
crosslink_from_translocators_5(iteration, sim, translocator1, translocator2, path = savepath, N = number_of_monomers)

# Add extruders, for the lattice 5 times finer resolution so divide by 5 to get the monomer resolution in polymer
lef1_all = [[x // 5 for x in pair] for pair in translocator1.lef_trajectory[-1]]
lef2_all = [[x // 5 for x in pair] for pair in translocator2.lef_trajectory[-1]]

lef1_all = np.array(lef1_all, dtype=int)
lef2_all = np.array(lef2_all, dtype=int)

valid_rows_lef1 = lef1_all[~np.all(lef1_all == -1, axis=1)]
valid_rows_lef2 = lef2_all[~np.all(lef2_all == -1, axis=1)]
lef1 = np.array(valid_rows_lef1, dtype=int)
lef2 = np.array(valid_rows_lef2, dtype=int) + number_of_monomers
# Combine them like in the old extruders1 + extruders2
extruders = np.r_[lef1, lef2]
# Update HOOMD topology
extrude.update_topology(sim, extruders, bond_name='LEF')

del translocator1
del translocator2

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
sim.operations.writers.append(log.table_formatter(logger, period=1e3))

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

# Run equlibrium, 3e7 steps
sim.run(3e7)

# Production phase - record as regular trajectory, 25000
gsd_writer = hoomd.write.GSD(filename=f'{savepath}/trajectory_{condition}_bypass_tau{tau}_{iteration}.gsd',
                            trigger=hoomd.trigger.Periodic(25000),
                            dynamic=['topology', 'momentum'],
                            mode='wb')
sim.operations.writers.append(gsd_writer)

# Run production, 1e6 steps 
sim.run(1e6)

# Calculate production time
prod_end_time = time.time()

# fire time 
fire_runtime_hours = (fire_end_time - fire_start_time) / 3600
print(f"Fire runtime: {fire_runtime_hours:.2f} hours")

# prod time 
prod_runtime_hours = (prod_end_time - prod_start_time) / 3600
print(f"PROD runtime: {prod_runtime_hours:.2f} hours")









