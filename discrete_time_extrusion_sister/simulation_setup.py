"""Shared setup helpers for the lattice run scripts (WT_dS.py, dW_dSdW.py).

Everything that is identical between the single-phase and two-phase runs lives
here; the scripts themselves only express what differs (which conditions, how
many phases, and the per-phase run arguments).
"""

import json
import os
import pickle

import numpy as np

from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder_Sister import (
    MultistateExtruder_Sister
)

# Anchor files are tabulated in 2.5 kb monomer units; a condition whose lattice
# is finer than that needs its anchors scaled up by this factor.
BP_PER_ANCHOR_UNIT = 2500

def load_paramdict(condition_phase, tau, sister_dampings,
                   collision_prob_index, bypass_index):
    """Load one condition's extrusion-parameter JSON."""
    path = (
        "../chemistry/Step3_Extrusion_Parameters/"
        f"{condition_phase}/"
        f"{condition_phase}_alpha{sister_dampings}_tau{tau}h_"
        f"beta{collision_prob_index}_gamma{bypass_index}.json"
    )
    with open(path, "r") as dict_file:
        return json.load(dict_file)

def anchor_scale(paramdict):
    """How many lattice sites one anchor-file unit corresponds to (1 or 5)."""
    return BP_PER_ANCHOR_UNIT // paramdict['bp_per_site']

def prepare_output(condition_phase, tau, sister_dampings,
                   collision_prob_index, bypass_index, scale=1):
    """Create the output directory and write the (empty) CTCF position files.

    Returns (savepath, ctcf_left_positions, ctcf_right_positions).
    """
    savepath = (
        f"output/tau{tau}h/"
        f"{condition_phase}_alpha_{sister_dampings}_tau_{tau}h_"
        f"beta{collision_prob_index}_gamma{bypass_index}"
    )
    if scale != 1:
        # finer lattice than the polymer stage -- see README, "Lattice resolution"
        savepath = f"{savepath}/factor{scale}"

    os.makedirs(savepath, exist_ok=True)
    print(f'Created (or exists): {savepath}')

    # No CTCF boundary sites are included in the current simulations.
    ctcf_left_positions = []
    ctcf_right_positions = []
    np.save(f"{savepath}/ctcf_left_positions_full.npy", ctcf_left_positions)
    np.save(f"{savepath}/ctcf_right_positions_full.npy", ctcf_right_positions)

    return savepath, ctcf_left_positions, ctcf_right_positions

def lattice_geometry(paramdict):
    """Derive lattice size and site types from a condition's paramdict.

    Returns (monomers_per_replica, type_list, site_types, num_of_sisters).
    """
    monomers_per_replica = paramdict['region_size_bp'] // paramdict['bp_per_site']
    num_of_sisters = paramdict['num_of_sisters']

    # Work with a single type of monomers (A, assigned to type index 0)
    type_list = ['A']
    monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
    site_types = np.repeat(monomer_types, 1)

    return monomers_per_replica, type_list, site_types, num_of_sisters

def load_anchor_pair(iteration, num_of_sisters, scale=1):
    """Load the paired linker anchors for one replica.

    Subsamples down to num_of_sisters (dS / dSdW case), using the same indices
    for both chromatids and seeding by iteration for reproducibility.
    Positions are then scaled to lattice sites.
    """
    common_sisters1 = np.loadtxt(f"input/anchor_2909_{iteration}_sister1.txt")
    common_sisters2 = np.loadtxt(f"input/anchor_2909_{iteration}_sister2.txt")

    if num_of_sisters < len(common_sisters1):
        np.random.seed(iteration)  # reproducible per-replica subsample
        sample_idx = np.random.choice(
            len(common_sisters1), size=num_of_sisters, replace=False)
        common_sisters1 = common_sisters1[sample_idx]
        common_sisters2 = common_sisters2[sample_idx]

    if scale != 1:
        common_sisters1 = common_sisters1 * scale
        common_sisters2 = common_sisters2 * scale

    return common_sisters1, common_sisters2

def build_translocator_pair(paramdict, type_list, site_types,
                            ctcf_left_positions, ctcf_right_positions,
                            common_sisters1, common_sisters2,
                            monomers_per_replica):
    """Construct the two chromatids' translocators from the same paramdict.

    Chromatid 2's anchors are shifted by -monomers_per_replica so that both
    translocators index their own lattice from zero.
    """
    translocator1 = Translocator_Sister(
        MultistateExtruder_Sister,
        StaticBoundary,
        type_list,
        site_types,
        ctcf_left_positions,
        ctcf_right_positions,
        initial_sister_positions=common_sisters1,
        **paramdict,
    )

    translocator2 = Translocator_Sister(
        MultistateExtruder_Sister,
        StaticBoundary,
        type_list,
        site_types,
        ctcf_left_positions,
        ctcf_right_positions,
        initial_sister_positions=common_sisters2 - monomers_per_replica,
        **paramdict,
    )

    return translocator1, translocator2

def save_trajectory_pair(savepath, label, iteration, translocator1, translocator2):
    """Pickle both chromatids' sampled trajectories."""
    for index, translocator in ((1, translocator1), (2, translocator2)):
        filename = f"{savepath}/{label}_trajectory{index}_{iteration}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({
                "sister": translocator.sister_trajectory,
                "lef": translocator.lef_trajectory,
                "ctcf": translocator.ctcf_trajectory,
                "state": translocator.state_trajectory,
                "coupling": translocator.coupling_trajectory,
            }, f)

    print("Sister, LEF, CTCF, state, coupling trajectory saved!")
