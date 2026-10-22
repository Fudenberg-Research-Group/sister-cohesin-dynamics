
"""
S8h sweep, lattice simulations. Sweeps sister-linker separation and misalignment:assigns sister anchors with Gaussian misalignment, 
then runs each chromatid's lattice extrusion independently, using those anchors as static boundary stalls. 
Output the trajectories. 
"""

import sys
import os
import json
import codecs
import numpy as np
import itertools

from discrete_time_extrusion.Translocator import Translocator
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion.extruders.MultistateExtruder import MultistateExtruder

def extruder_pos_gaussian_misalign(N_sister_RAD21, iteration, chromosome_size, path):

    path_name = f"WT_extrusion_rates.json"

    with open(path_name, 'r') as dict_file:
        paramdict = json.load(dict_file)
        
    monomers_per_replica = paramdict['monomers_per_replica'] 
    sites_per_monomer = paramdict['sites_per_monomer']

    # sites_per_replica = monomers_per_replica*sites_per_monomer

    # Work with a single type of monomers (A, assigned to type index 0)
    type_list = ['A']
    monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
    site_types = np.repeat(monomer_types, sites_per_monomer)

    # LEF/CTCF properties in type A monomers may be obtained from the paramdict as follows
    LEF_off_rate = paramdict['LEF_off_rate']
    CTCF_facestall = paramdict['CTCF_facestall']
    print(LEF_off_rate['A'], CTCF_facestall['A'])
    
    anchor_positions_sister1 = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}_sister1.txt')
    anchor_positions_sister2 = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}_sister2.txt')

    # Create CTCF boundary sites for sister1
    ctcf_left_positions_sister1 = anchor_positions_sister1
    ctcf_right_positions_sister1 = anchor_positions_sister1

    # Create CTCF boundary sites for sister2
    ctcf_left_positions_sister2 = anchor_positions_sister2
    ctcf_right_positions_sister2 = anchor_positions_sister2

    paramdict['monomers_per_replica'] = chromosome_size

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
    
    # Update topology
    # extrude.update_topology(system, bond_array, bond_name='Sister')
    

process_id = int(sys.argv[1])-1

number_of_monomers = 320000
kb_per_monomer = 2.5       # 320000 monomers = 800 Mb

sigma_kb_list = [50, 75, 100, 125, 150, 175]      # misalignment sigma, in kb
sister_sep = [200, 225, 250, 275, 300, 325, 350, 375, 400]
num_sister_RAD21_list = [round(800000 / s) for s in sister_sep]
replica_ids = np.arange(5)

parameters = list(itertools.product(
    range(len(sigma_kb_list)), range(len(sister_sep)), replica_ids))
sigma_idx, sep_idx, iteration = parameters[process_id]

sigma_kb = sigma_kb_list[sigma_idx]
sigma_offset = sigma_kb / kb_per_monomer 

sep_value = sister_sep[sep_idx]
num_sister = num_sister_RAD21_list[sep_idx]

print("sigma_kb:", sigma_kb, "sep_value:", sep_value, "num_sister:", num_sister, "replica:", iteration)

savepath = f'output/sigma{sigma_kb}/sep{sep_value}'
os.makedirs(savepath, exist_ok=True)

rng_seed = os.urandom(2)
rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)
print("Using entropy-harvested random seed: %d" % rng_seed)

crosslink_chains_gaussian_misalign(number_of_monomers, iteration, num_sister, sigma_offset = sigma_offset, path = savepath, rng=None)

### Add extruding cohesins and run extrusion simulations  
extruder_pos_gaussian_misalign(num_sister, iteration, chromosome_size = number_of_monomers, path = savepath)


