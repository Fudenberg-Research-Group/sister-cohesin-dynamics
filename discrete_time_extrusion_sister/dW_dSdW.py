
import os
import sys
import time
import json
import pickle
import numpy as np

from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder_Sister import MultistateExtruder_Sister

from discrete_time_extrusion.sister_lifecycle import (
    update,
    run_synchronized_trajectories_continue,
)

CONDITIONS = {
    "WT_dW": ("WT7h", "dW"),
    "dS_dSdW": ("dS7h", "dSdW"),
}

condition_name = "WT_dW" 
condition_phase1, condition_phase2 = CONDITIONS[condition_name]

# Local test: process_id set manually.
# For cluster job array, use: process_id = int(sys.argv[1]) - 1
process_id = 0 

tau = 11

sister_dampings = 100

collision_prob_index = 0

bypass_index = 0 

base_savepath = f'output/tau{tau}h'

replica_ids = np.arange(5)

iteration = replica_ids[process_id]

savepath = f'output/tau{tau}h/{condition_phase2}_alpha_{sister_dampings}_tau_{tau}h_beta{collision_prob_index}_gamma{bypass_index}/factor5'
os.makedirs(savepath, exist_ok=True)
print(f'Created (or exists): {savepath}')

with open(
    f"../chemistry/Step3_Extrusion_Parameters/{condition_phase1}/{condition_phase1}_alpha{sister_dampings}_tau{tau}h_beta{collision_prob_index}_gamma{bypass_index}.json",
    "r",
) as dict_file:
    paramdict_phase1 = json.load(dict_file)

with open(
    f"../chemistry/Step3_Extrusion_Parameters/{condition_phase2}/{condition_phase2}_alpha{sister_dampings}_tau{tau}h_beta{collision_prob_index}_gamma{bypass_index}.json",
    "r",
) as dict_file:
    paramdict_phase2 = json.load(dict_file)


monomers_per_replica = paramdict_phase1['region_size_bp']//paramdict_phase1['bp_per_site']
num_of_sisters = paramdict_phase1['num_of_sisters']
# Work with a single type of monomers (A, assigned to type index 0)
type_list = ['A']
monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
site_types = np.repeat(monomer_types, 1)

# LEF/CTCF properties in type A monomers may be obtained from the paramdict as follows
LEF_off_rate = paramdict_phase1['LEF_off_rate']
CTCF_facestall = paramdict_phase1['CTCF_facestall']

print(LEF_off_rate['A'], CTCF_facestall['A'])

## Can also specify CTCF positions here if desired; left and rights are for two anchor sites. 
ctcf_left_positions = []
ctcf_right_positions = []

np.save(f"{savepath}/ctcf_left_positions_full.npy", ctcf_left_positions)
np.save(f"{savepath}/ctcf_right_positions_full.npy", ctcf_right_positions)

start = time.time()

# common_sisters = np.random.choice(np.arange(monomers_per_replica), size = num_of_sisters, replace=False)
common_sisters1 = np.loadtxt(f"input/anchor_2909_{iteration}_sister1.txt")
common_sisters2 = np.loadtxt(f"input/anchor_2909_{iteration}_sister2.txt")

if num_of_sisters < len(common_sisters1):
    np.random.seed(iteration)  # reproducible per-replica subsample
    sample_idx = np.random.choice(len(common_sisters1), size=num_of_sisters, replace=False)
    common_sisters1 = common_sisters1[sample_idx]
    common_sisters2 = common_sisters2[sample_idx]

common_sisters1 *= 5
common_sisters2 *= 5


translocator1 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            initial_sister_positions = common_sisters1, 
                            **paramdict_phase1)

translocator2 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            initial_sister_positions = common_sisters2 - monomers_per_replica, 
                            **paramdict_phase1)


paramdict_phase1['time_unit'] = translocator1.params['time_unit']
paramdict_phase2['time_unit'] = translocator1.params['time_unit']

# Full run should use steps = 86400 (6h * 3600s / 0.25s per step).

run_synchronized_trajectories_continue(
    translocator1, translocator2,
    paramdict_phase1['sister_lifetime'],
    steps = 86400, 
    prune_unbound_LEFs = True,
    track_sisters = True,
    sample_interval = 100,
    seed = 42,
    clear_trajectories = True  # Start fresh
)

update(translocator1, paramdict_phase2, type_list, site_types)
update(translocator2, paramdict_phase2, type_list, site_types)

# Second run - continue from where we left off
# Full run should use steps = 144000 (10h*3600 / 0.25s per step).

run_synchronized_trajectories_continue(
   translocator1, translocator2, 
   paramdict_phase2['sister_lifetime'],
   steps = 144000,   
   prune_unbound_LEFs = False,
   track_sisters = True,
   sample_interval = 100, 
   seed = 123,
   clear_trajectories = False  # Keep existing data
)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")

with open(f"{savepath}/{condition_phase2}_trajectory1_{iteration}.pkl", 'wb') as f:
    pickle.dump({
        "sister": translocator1.sister_trajectory,
        "lef": translocator1.lef_trajectory,
        "ctcf": translocator1.ctcf_trajectory,
        "state": translocator1.state_trajectory, 
        "coupling": translocator1.coupling_trajectory, 
    }, f)

with open(f"{savepath}/{condition_phase2}_trajectory2_{iteration}.pkl", 'wb') as f:
    pickle.dump({
        "sister": translocator2.sister_trajectory,
        "lef": translocator2.lef_trajectory,
        "ctcf": translocator2.ctcf_trajectory,
        "state": translocator2.state_trajectory, 
        "coupling": translocator2.coupling_trajectory, 
    }, f)

print("Sister, LEF, CTCF, state, coupling trajectory saved!")





