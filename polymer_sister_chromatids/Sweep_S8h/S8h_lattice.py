
"""
S8h sweep, lattice simulations. Sweeps sister-linker separation and misalignment:assigns sister anchors with Gaussian misalignment,
then runs each chromatid's lattice extrusion independently, using those anchors as static boundary stalls.
Output the trajectories.
"""

import sys
import os
import json
import numpy as np
import itertools

from discrete_time_extrusion.Translocator import Translocator
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder import MultistateExtruder

RATES_JSON = "WT_extrusion_rates.json"

# --- sweep definition; must stay identical to S8h_polymer.py ---
NUMBER_OF_MONOMERS = 320000
KB_PER_MONOMER = 2.5                               # 320000 monomers = 800 Mb
SIGMA_KB_LIST = [50, 75, 100, 125, 150, 175]       # misalignment sigma, in kb
SISTER_SEP = [200, 225, 250, 275, 300, 325, 350, 375, 400]
NUM_SISTER_LIST = [round(800000 / s) for s in SISTER_SEP]
REPLICA_IDS = np.arange(5)

def extruder_pos_gaussian_misalign(N_sister_RAD21, iteration, chromosome_size, path):

    with open(RATES_JSON, 'r') as dict_file:
        paramdict = json.load(dict_file)

    paramdict['monomers_per_replica'] = chromosome_size
    sites_per_monomer = paramdict['sites_per_monomer']

    # Work with a single type of monomers (A, assigned to type index 0)
    type_list = ['A']
    monomer_types = type_list.index('A') * np.ones(chromosome_size, dtype=int)
    site_types = np.repeat(monomer_types, sites_per_monomer)

    anchor_positions_sister1 = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}_sister1.txt')
    anchor_positions_sister2 = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}_sister2.txt')

    # Create CTCF boundary sites for sister1
    ctcf_left_positions_sister1 = anchor_positions_sister1
    ctcf_right_positions_sister1 = anchor_positions_sister1

    # Create CTCF boundary sites for sister2
    ctcf_left_positions_sister2 = anchor_positions_sister2
    ctcf_right_positions_sister2 = anchor_positions_sister2

    translocator1 = Translocator(MultistateExtruder,
                            StaticBoundary,
                            type_list,
                            site_types,
                            ctcf_left_positions_sister1,
                            ctcf_right_positions_sister1,
                            **paramdict)
    translocator1.run(paramdict['steps'])
    positions1 = np.asarray(translocator1.extrusion_engine.get_bound_positions())

    np.savetxt(f'{path}/lef_traj1_{N_sister_RAD21}_{iteration}.txt', positions1)

    translocator2 = Translocator(MultistateExtruder,
                            StaticBoundary,
                            type_list,
                            site_types,
                            ctcf_left_positions_sister2 - chromosome_size,
                            ctcf_right_positions_sister2 - chromosome_size,
                            **paramdict)

    translocator2.run(paramdict['steps'])
    positions2 = np.asarray(translocator2.extrusion_engine.get_bound_positions()) + chromosome_size

    np.savetxt(f'{path}/lef_traj2_{N_sister_RAD21}_{iteration}.txt', positions2)


def crosslink_chains_gaussian_misalign(N_monomers, iteration, N_crosslinks, sigma_offset, path, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Sample anchor positions on first chain (without replacement)
    anchor_positions = rng.choice(np.arange(N_monomers), size = N_crosslinks, replace=False)
    # Generate Gaussian misalignments and round to lattice positions
    raw_misalignment = rng.normal(0, sigma_offset, size = N_crosslinks)
    lattice_misalignment = np.round(raw_misalignment).astype(int)
    # For each crosslink, randomly choose which leg to move (50/50 chance)
    move_first_chain = rng.random(N_crosslinks) < 0.5
    # Initialize bond array: [chain1_pos, chain2_pos]
    bond_array = np.zeros((N_crosslinks, 2), dtype=int)
    bond_array[:, 0] = anchor_positions  # First chain positions
    bond_array[:, 1] = anchor_positions + N_monomers  # Second chain positions (initially aligned)

    # Apply misalignments
    for i in range(N_crosslinks):
        if move_first_chain[i]:
            # Move first chain leg
            new_pos = anchor_positions[i] + lattice_misalignment[i]
            bond_array[i, 0] = np.clip(new_pos, 0, N_monomers - 1)
        else:
            # Move second chain leg
            new_pos = anchor_positions[i] + N_monomers + lattice_misalignment[i]
            bond_array[i, 1] = np.clip(new_pos, N_monomers, 2 * N_monomers - 1)

    ## sort the array
    bond_array[:, 0] = np.sort(bond_array[:, 0])
    bond_array[:, 1] = np.sort(bond_array[:, 1])

    np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}.txt', anchor_positions)

    # Save files
    np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}_sister1.txt', bond_array[:, 0])
    np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}_sister2.txt', bond_array[:, 1])

def main():
    process_id = int(sys.argv[1]) - 1

    parameters = list(itertools.product(
        range(len(SIGMA_KB_LIST)), range(len(SISTER_SEP)), REPLICA_IDS))
    sigma_idx, sep_idx, iteration = parameters[process_id]

    sigma_kb = SIGMA_KB_LIST[sigma_idx]
    sigma_offset = sigma_kb / KB_PER_MONOMER
    sep_value = SISTER_SEP[sep_idx]
    num_sister = NUM_SISTER_LIST[sep_idx]

    print("sigma_kb:", sigma_kb, "sep_value:", sep_value,
          "num_sister:", num_sister, "replica:", iteration)

    savepath = f'output/sigma{sigma_kb}/sep{sep_value}'
    os.makedirs(savepath, exist_ok=True)

    crosslink_chains_gaussian_misalign(
        NUMBER_OF_MONOMERS, iteration, num_sister,
        sigma_offset=sigma_offset, path=savepath,
        rng=np.random.default_rng((sigma_idx, sep_idx, iteration)),
    )

    # Add extruding cohesins and run extrusion simulations
    extruder_pos_gaussian_misalign(
        num_sister, iteration,
        chromosome_size=NUMBER_OF_MONOMERS, path=savepath)


if __name__ == "__main__":
    main()
