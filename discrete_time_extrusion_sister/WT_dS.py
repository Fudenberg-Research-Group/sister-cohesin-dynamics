"""Run lattice simulations for WT or dSORORIN.

Single-phase run: one condition's parameters are used throughout. The script
loads condition-specific biochemical parameters and paired linker anchors,
then writes synchronized extrusion trajectories for both chromatids.
"""

import argparse
import time

from discrete_time_extrusion.sister_lifecycle import (
    run_synchronized_trajectories_continue,
)

import simulation_setup as setup

CONDITIONS = {
    "WT": "WT12h",
    "dS": "dS12h",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a WT or dSORORIN lattice simulation."
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        required=True,
        help="Biological condition to simulate.",
    )
    parser.add_argument(
        "--replica",
        type=int,
        choices=range(5),
        default=0,
        help="Zero-based replica index (0-4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    condition_phase = CONDITIONS[args.condition]
    iteration = args.replica

    tau = 11
    sister_dampings = 50000
    collision_prob_index = 0
    bypass_index = 0

    print(f"Running simulation for condition: {condition_phase}, replica: {iteration}")
    paramdict_phase = setup.load_paramdict(
        condition_phase, tau, sister_dampings, collision_prob_index, bypass_index)

    scale = setup.anchor_scale(paramdict_phase)          # 1 for the 12h conditions

    savepath, ctcf_left, ctcf_right = setup.prepare_output(
        condition_phase, tau, sister_dampings,
        collision_prob_index, bypass_index, scale=scale)

    monomers_per_replica, type_list, site_types, num_of_sisters = \
        setup.lattice_geometry(paramdict_phase)

    common_sisters1, common_sisters2 = setup.load_anchor_pair(
        iteration, num_of_sisters, scale=scale)

    start = time.time()

    translocator1, translocator2 = setup.build_translocator_pair(
        paramdict_phase, type_list, site_types, ctcf_left, ctcf_right,
        common_sisters1, common_sisters2, monomers_per_replica)

    paramdict_phase['time_unit'] = translocator1.params['time_unit']

    # 46080 steps = 16 h at 1.25 s per lattice step (bp_per_site = 2500)
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

    print(f"Run time: {time.time() - start:.2f} seconds")

    setup.save_trajectory_pair(
        savepath, condition_phase, iteration, translocator1, translocator2)


if __name__ == "__main__":
    main()
