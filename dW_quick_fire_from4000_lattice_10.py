
import os
import sys
import time
import json
import pickle
import codecs

import numpy as np
import matplotlib.pyplot as plt


import discrete_time_extrusion.arrays as arrays 
from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder_Sister import MultistateExtruder_Sister


# process_id = int(sys.argv[1])-1
process_id = 0

# alpha_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 50000]

alpha_list = [90]

tau = 8

base_savepath = f'tau{tau}h'

for alpha in alpha_list:
    savepath = f'{base_savepath}/dW_alpha_{alpha}_tau_{tau}h_10'
    os.makedirs(savepath, exist_ok=True)
    print(f'Created (or exists): {savepath}')

replica_ids = np.arange(5)

x, y = np.meshgrid(alpha_list, replica_ids)
parameters = list(zip(x.flatten(), y.flatten()))

alpha, iteration = parameters[process_id] 
print(alpha, iteration)


savepath = f'tau{tau}h/dW_alpha_{alpha}_tau_{tau}h_10'


with open(
    f"WT4h_alpha{alpha}_tau{tau}h_10.json",
    "r",
) as dict_file:
    paramdict_WT = json.load(dict_file)


with open(
    f"dW_alpha{alpha}_tau{tau}h_10.json",
    "r",
) as dict_file:
    paramdict_dW = json.load(dict_file)

monomers_per_replica = paramdict_WT['region_size_bp']//paramdict_WT['bp_per_monomer']
sites_per_monomer = paramdict_WT['sites_per_monomer']
num_of_sisters = paramdict_WT['num_of_sisters']
# Work with a single type of monomers (A, assigned to type index 0)
type_list = ['A']
monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
site_types = np.repeat(monomer_types, sites_per_monomer)
    
# LEF/CTCF properties in type A monomers may be obtained from the paramdict as follows
LEF_off_rate = paramdict_WT['LEF_off_rate']
CTCF_facestall = paramdict_WT['CTCF_facestall']
print(LEF_off_rate['A'], CTCF_facestall['A'])

anchor_positions = []
ctcf_left_positions = []
ctcf_right_positions = []

class TranslocatorFromTrajectory:
    def __init__(self, traj):
        self.sister_trajectory = traj["sister"]
        self.lef_trajectory = traj["lef"]
        self.ctcf_trajectory = traj["ctcf"]

def update(translocator, param_dict):
    lef_arrays = arrays.make_LEF_arrays(np, type_list, site_types, **param_dict)
    lef_transition_dict = arrays.make_LEF_transition_dict(np, type_list, site_types, **param_dict)
    ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(np, type_list, site_types, **param_dict)

    translocator.extrusion_engine.birth_prob = lef_arrays[0]
    translocator.extrusion_engine.death_prob = lef_arrays[1]
    translocator.extrusion_engine.stalled_death_prob = lef_arrays[2]

    translocator.barrier_engine.birth_prob = ctcf_dynamic_arrays[0]
    translocator.barrier_engine.death_prob = ctcf_dynamic_arrays[1]

    translocator.extrusion_engine.transition_dict = lef_transition_dict['LEF_transitions']
        
def shared_sister_update(translocator1, translocator2, step_number, shared_decay_decisions):
    """
    Update sister states for both translocators using shared decisions
    """
    if step_number >= len(shared_decay_decisions):
        return
    # Get the decay mask for this step
    fall_off_mask = shared_decay_decisions[step_number]
    # Apply to translocator1
    if translocator1.extrusion_engine.sister_positions is not None:
        active_mask_1 = translocator1.extrusion_engine.sister_positions != -1
        final_mask_1 = fall_off_mask & active_mask_1
        translocator1.extrusion_engine.sister_positions[final_mask_1] = -1
    # Apply to translocator2 
    if translocator2.extrusion_engine.sister_positions is not None:
        active_mask_2 = translocator2.extrusion_engine.sister_positions != -1
        final_mask_2 = fall_off_mask & active_mask_2
        translocator2.extrusion_engine.sister_positions[final_mask_2] = -1

def run_synchronized_trajectories_continue(translocator1, translocator2, sister_lifetime, 
                        period=None, steps=None, 
                        prune_unbound_LEFs=True, track_sisters=False, 
                        sample_interval=1, seed=42, clear_trajectories=False, **kwargs):
    """
    Modified version that optionally preserves existing trajectory data
    """
    # Pre-generate shared decay decisions
    num_sisters = translocator1.extrusion_engine.num_sisters
    decay_prob = 1 - np.exp(-1 / sister_lifetime / 10)
        
    # Calculate total steps needed
    steps = int(steps) if steps else translocator1.params['steps']
    period = int(period) if period else translocator1.params['sites_per_monomer']
    dummy_steps = translocator1.params['dummy_steps'] * period
    total_steps = steps * period + dummy_steps
    
    np.random.seed(seed)
    shared_decay_decisions = np.random.random((total_steps, num_sisters)) < decay_prob
    
    print(f"Generated shared decay schedule for {total_steps} total steps")
    
    # Initialize trajectory lists only if clearing or they don't exist
    if clear_trajectories or not hasattr(translocator1, 'state_trajectory'):
        translocator1.state_trajectory = []
        translocator1.lef_trajectory = []
        translocator1.coupling_trajectory = []
        translocator1.ctcf_trajectory = []
        translocator1.sister_trajectory = []
    
    if clear_trajectories or not hasattr(translocator2, 'state_trajectory'):
        translocator2.state_trajectory = []
        translocator2.lef_trajectory = []
        translocator2.coupling_trajectory = []
        translocator2.ctcf_trajectory = []
        translocator2.sister_trajectory = []
       
    step_counter = 0
    # Dummy steps
    translocator1.run(dummy_steps*period, **kwargs)
    translocator2.run(dummy_steps*period, **kwargs)
    print(f"Running {dummy_steps} dummy steps...")

    for step_idx in range(steps):
        # Run 'period' number of steps
        translocator1.run(1)
        translocator2.run(1)
        if step_idx % 50000 == 0:
            print(f"Step {step_idx} completed")
        # Update sisters after each individual step
        shared_sister_update(translocator1, translocator2, step_counter, shared_decay_decisions)
        step_counter += 1
        # Data collection at sampling intervals or last step 
        if step_idx % sample_interval == 0 or step_idx == steps - 1:
            # Collect data for translocator1
            LEF_states_1 = translocator1.extrusion_engine.get_states()
            CTCF_positions_1 = translocator1.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_1 = translocator1.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_1 = translocator1.extrusion_engine.get_positions()
            
            translocator1.state_trajectory.append(LEF_states_1)
            translocator1.lef_trajectory.append(LEF_positions_1)
            translocator1.ctcf_trajectory.append(CTCF_positions_1)
            
            # Collect data for translocator2
            LEF_states_2 = translocator2.extrusion_engine.get_states()
            CTCF_positions_2 = translocator2.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_2 = translocator2.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_2 = translocator2.extrusion_engine.get_positions()
            
            translocator2.state_trajectory.append(LEF_states_2)
            translocator2.lef_trajectory.append(LEF_positions_2)
            translocator2.ctcf_trajectory.append(CTCF_positions_2)
            
            # Track sisters
            if track_sisters:
                if hasattr(translocator1.extrusion_engine, 'sister_positions'):
                    sister_positions_1 = translocator1.extrusion_engine.sister_positions.copy()
                    coupling_status_1 = translocator1.extrusion_engine.get_coupling_status()
                    translocator1.sister_trajectory.append(sister_positions_1)
                    translocator1.coupling_trajectory.append(coupling_status_1)
                
                if hasattr(translocator2.extrusion_engine, 'sister_positions'):
                    sister_positions_2 = translocator2.extrusion_engine.sister_positions.copy()
                    coupling_status_2 = translocator2.extrusion_engine.get_coupling_status()
                    translocator2.sister_trajectory.append(sister_positions_2)
                    translocator2.coupling_trajectory.append(coupling_status_2)

## used to generate random walk initial condition for PBC conditions
def _random_points_sphere(N):
    theta = np.random.uniform(0.0, 1.0, N)
    theta = 2.0 * np.pi * theta
    u = np.random.uniform(0.0, 1.0, N)
    u = 2.0 * u - 1.0

    return np.vstack([theta, u]).T

def create_random_walk_PBC(step_size, N, L):
    """
    Creates a freely joined chain of length N with step step_size
    """
    theta, u = _random_points_sphere(N).T
    dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
    dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
    dz = step_size * u

    x, y, z = np.cumsum(dx), np.cumsum(dy), np.cumsum(dz)

    positions = np.zeros((N, 3))
    
    positions[:,0] = x
    positions[:,1] = y
    positions[:,2] = z

    # positions -= positions.mean(axis=0, keepdims=True)
    positions = np.mod(positions+L/2., L) - L/2.
    
    return positions

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


def crosslink_from_translocators(iteration, system, translocator1, translocator2, path, N):
    """
    Add sister crosslinks from existing translocator objects.
    """

    arr1 = np.array(translocator1.sister_trajectory[-1], dtype=int)
    anchors1 = arr1[arr1 != -1]
    arr2 = np.array(translocator2.sister_trajectory[-1], dtype=int)
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


np.save(f"{savepath}/ctcf_left_positions_full.npy", ctcf_left_positions)
np.save(f"{savepath}/ctcf_right_positions_full.npy", ctcf_left_positions)


start = time.time()

common_sisters = np.random.choice(np.arange(monomers_per_replica), size = num_of_sisters, replace=False)

translocator1 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            initial_sister_positions = common_sisters, 
                            **paramdict_WT)

translocator2 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            initial_sister_positions = common_sisters, 
                            **paramdict_WT)

# 28800

run_synchronized_trajectories_continue(
    translocator1, translocator2,
    paramdict_WT['sister_lifetime'],
    steps = 288000, 
    prune_unbound_LEFs = True,
    track_sisters = True,
    sample_interval = 100,
    seed = 42,
    clear_trajectories=True  # Start fresh
)

update(translocator1, paramdict_dW)
update(translocator2, paramdict_dW)

# 36000
# Second run - continue from where we left off
run_synchronized_trajectories_continue(
   translocator1, translocator2, 
   paramdict_dW['sister_lifetime'],
   steps = 360000,    # Different parameters
   prune_unbound_LEFs = False,
   track_sisters = True,
   sample_interval = 100, 
   seed = 123,
   clear_trajectories = False  # Keep existing data
)


end = time.time()
print(f"Run time: {end - start:.2f} seconds")
print(f"Before manual init: num_sisters = {translocator1.extrusion_engine}")


with open(f"{savepath}/dW_trajectory1_{iteration}.pkl", 'wb') as f:
    pickle.dump({
        "sister": translocator1.sister_trajectory,
        "lef": translocator1.lef_trajectory,
        "ctcf": translocator1.ctcf_trajectory,
        "state": translocator1.state_trajectory, 
        "coupling": translocator1.coupling_trajectory, 
    }, f)

with open(f"{savepath}/dW_trajectory2_{iteration}.pkl", 'wb') as f:
    pickle.dump({
        "sister": translocator2.sister_trajectory,
        "lef": translocator2.lef_trajectory,
        "ctcf": translocator2.ctcf_trajectory,
        "state": translocator2.state_trajectory, 
        "coupling": translocator2.coupling_trajectory, 
    }, f)
print("Sister, LEF, CTCF, state, coupling trajectory saved!")





