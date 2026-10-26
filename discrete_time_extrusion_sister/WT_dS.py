"""Run lattice simulations for WT or dSORORIN.

The script loads condition-specific biochemical parameters and paired linker
anchors, then writes synchronized extrusion trajectories for both chromatids.
"""

import json
import os
import pickle
import sys
import time

import numpy as np

from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder_Sister import (
    MultistateExtruder_Sister
)
from discrete_time_extrusion.sister_lifecycle import (
    run_synchronized_trajectories_continue,
)

CONDITIONS = {
    "WT": "WT12h",
    "dS": "dS12h",
}


def main():
    condition_name = "dS"
    condition_phase = CONDITIONS[condition_name]
    # Local test: process_id set manually.
    # For cluster job array, use: process_id = int(sys.argv[1]) - 1
    process_id = 0
    tau = 11
    sister_dampings = 100
    collision_prob_index = 0
    bypass_index = 0

    replica_ids = np.arange(5)

    iteration = replica_ids[process_id]

    savepath = (
        f"output/tau{tau}h/"
        f"{condition_phase}_alpha_{sister_dampings}_tau_{tau}h_"
        f"beta{collision_prob_index}_gamma{bypass_index}/"
    )
    os.makedirs(savepath, exist_ok=True)
    print(f'Created (or exists): {savepath}')

    parameter_path = (
        "../chemistry/Step3_Extrusion_Parameters/"
        f"{condition_phase}/"
        f"{condition_phase}_alpha{sister_dampings}_tau{tau}h_"
        f"beta{collision_prob_index}_gamma{bypass_index}.json"
    )

    with open(parameter_path, "r") as dict_file:
        paramdict_phase = json.load(dict_file)

    monomers_per_replica = paramdict_phase['region_size_bp']//paramdict_phase['bp_per_site']
    num_of_sisters = paramdict_phase['num_of_sisters']
    # Work with a single type of monomers (A, assigned to type index 0)
    type_list = ['A']
    monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
    site_types = np.repeat(monomer_types, 1)

    # No CTCF boundary sites are included in the current simulations.
    ctcf_left_positions = []
    ctcf_right_positions = []

    np.save(f"{savepath}/ctcf_left_positions_full.npy", ctcf_left_positions)
    np.save(f"{savepath}/ctcf_right_positions_full.npy", ctcf_right_positions)

    start = time.time()

    common_sisters1 = np.loadtxt(f"input/anchor_2909_{iteration}_sister1.txt")
    common_sisters2 = np.loadtxt(f"input/anchor_2909_{iteration}_sister2.txt")

    if num_of_sisters < len(common_sisters1):
        np.random.seed(iteration)  # reproducible per-replica subsample
        sample_idx = np.random.choice(len(common_sisters1), size=num_of_sisters, replace=False)
        common_sisters1 = common_sisters1[sample_idx]
        common_sisters2 = common_sisters2[sample_idx]

    translocator1 = Translocator_Sister(
        MultistateExtruder_Sister,
        StaticBoundary,
        type_list,
        site_types,
        ctcf_left_positions,
        ctcf_right_positions,
        initial_sister_positions=common_sisters1,
        **paramdict_phase,
    )

    translocator2 = Translocator_Sister(
        MultistateExtruder_Sister,
        StaticBoundary,
        type_list,
        site_types,
        ctcf_left_positions,
        ctcf_right_positions,
        initial_sister_positions=common_sisters2 - monomers_per_replica,
        **paramdict_phase,
    )

    paramdict_phase['time_unit'] = translocator1.params['time_unit']

    # Full run should use steps = 46080 (16h * 3600s / 1.25s per step).
    run_synchronized_trajectories_continue(
        translocator1, translocator2,
        paramdict_phase['sister_lifetime'],
        steps=46080,
        prune_unbound_LEFs=True,
        track_sisters=True,
        sample_interval=100,
        seed=42,
        clear_trajectories=True,
    )

    end = time.time()
    print(f"Run time: {end - start:.2f} seconds")

    with open(f"{savepath}/{condition_phase}_trajectory1_{iteration}.pkl", 'wb') as f:
        pickle.dump({
            "sister": translocator1.sister_trajectory,
            "lef": translocator1.lef_trajectory,
            "ctcf": translocator1.ctcf_trajectory,
            "state": translocator1.state_trajectory,
            "coupling": translocator1.coupling_trajectory,
        }, f)

    with open(f"{savepath}/{condition_phase}_trajectory2_{iteration}.pkl", 'wb') as f:
        pickle.dump({
            "sister": translocator2.sister_trajectory,
            "lef": translocator2.lef_trajectory,
            "ctcf": translocator2.ctcf_trajectory,
            "state": translocator2.state_trajectory,
            "coupling": translocator2.coupling_trajectory,
        }, f)

    print("Sister, LEF, CTCF, state, coupling trajectory saved!")


if __name__ == "__main__":
    main()
