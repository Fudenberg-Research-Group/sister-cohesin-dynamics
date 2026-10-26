"""Run two-phase lattice simulations for dWAPL or dSORORIN+dWAPL.

Phase 1 runs the pre-depletion condition; parameters are then switched with
update() and the simulation continues as the post-depletion condition on the
same translocators.
"""

import time

import numpy as np

from discrete_time_extrusion.sister_lifecycle import (
    update,
    run_synchronized_trajectories_continue,
)

import simulation_setup as setup

CONDITIONS = {
    "WT_dW": ("WT7h", "dW"),
    "dS_dSdW": ("dS7h", "dSdW"),
}


def main():
    condition_name = "WT_dW"
    condition_phase1, condition_phase2 = CONDITIONS[condition_name]

    # Local test: process_id set manually.
    # For cluster job array, use: process_id = int(sys.argv[1]) - 1
    process_id = 0
    tau = 11
    sister_dampings = 100
    collision_prob_index = 0
    bypass_index = 0

    replica_ids = np.arange(5)
    iteration = replica_ids[process_id]

    paramdict_phase1 = setup.load_paramdict(
        condition_phase1, tau, sister_dampings, collision_prob_index, bypass_index)
    paramdict_phase2 = setup.load_paramdict(
        condition_phase2, tau, sister_dampings, collision_prob_index, bypass_index)

    scale = setup.anchor_scale(paramdict_phase1)         # 5 for the 7h conditions

    # Output is named after the post-depletion condition.
    savepath, ctcf_left, ctcf_right = setup.prepare_output(
        condition_phase2, tau, sister_dampings,
        collision_prob_index, bypass_index, scale=scale)

    monomers_per_replica, type_list, site_types, num_of_sisters = \
        setup.lattice_geometry(paramdict_phase1)

    common_sisters1, common_sisters2 = setup.load_anchor_pair(
        iteration, num_of_sisters, scale=scale)

    start = time.time()

    # Both chromatids start from phase 1's parameters.
    translocator1, translocator2 = setup.build_translocator_pair(
        paramdict_phase1, type_list, site_types, ctcf_left, ctcf_right,
        common_sisters1, common_sisters2, monomers_per_replica)

    paramdict_phase1['time_unit'] = translocator1.params['time_unit']
    paramdict_phase2['time_unit'] = translocator1.params['time_unit']

    # Phase 1: 86400 steps = 6 h at 0.25 s per lattice step (bp_per_site = 500)
    run_synchronized_trajectories_continue(
        translocator1, translocator2,
        paramdict_phase1['sister_lifetime'],
        steps=86400,
        prune_unbound_LEFs=True,
        track_sisters=True,
        sample_interval=100,
        seed=42,
        clear_trajectories=True,
    )

    # Switch to the post-depletion condition on the same translocators.
    update(translocator1, paramdict_phase2, type_list, site_types)
    update(translocator2, paramdict_phase2, type_list, site_types)

    # Phase 2: 144000 steps = 10 h, continuing from phase 1's state
    run_synchronized_trajectories_continue(
        translocator1, translocator2,
        paramdict_phase2['sister_lifetime'],
        steps=144000,
        prune_unbound_LEFs=False,
        track_sisters=True,
        sample_interval=100,
        seed=123,
        clear_trajectories=False,
    )

    print(f"Run time: {time.time() - start:.2f} seconds")

    setup.save_trajectory_pair(
        savepath, condition_phase2, iteration, translocator1, translocator2)


if __name__ == "__main__":
    main()
