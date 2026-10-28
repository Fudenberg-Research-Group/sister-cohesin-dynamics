"""Place a dWAPL or dSORORIN+dWAPL lattice configuration onto the polymer model.

As WT_dS_polymer.py, but these conditions run on a lattice five times finer
than the polymer monomer spacing, so positions are divided by SCALE and the
lattice output lives one level deeper (factor5/).
"""

import os

import argparse

import polymer_common as pc

CONDITIONS = {
    "dW": "dW",
    "dSdW": "dSdW",
}

# Lattice sites per polymer monomer for these conditions (bp_per_site = 500).
SCALE = 5

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a two-phase dWAPL or dSORORIN+dWAPL lattice simulation."
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        required=True,
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
    condition = CONDITIONS[args.condition]
    iteration = args.replica

    print(f"Running simulation for condition: {condition}, replica: {iteration}")

    alpha = 100
    tau = 11

    readpath = 'input'

    savepath = f'output/{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0'
    os.makedirs(savepath, exist_ok=True)

    # The finer-lattice runs write into a factor{SCALE}/ subdirectory.
    read_traj_path = (
        f'../discrete_time_extrusion_sister/output/tau{tau}h/'
        f'{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0/factor{SCALE}'
    )

    sim = pc.init_simulation(readpath, iteration)
    number_of_monomers, confinement_radius = pc.system_geometry()
    relax_field, force_field = pc.build_force_field(
        "data/force_dict_sister_extruder.json", confinement_radius)

    translocator1, translocator2 = pc.load_lattice_trajectories(
        read_traj_path, condition, iteration)

    pc.add_sister_bonds(sim, translocator1, translocator2,
                        savepath, iteration, number_of_monomers, scale=SCALE)
    pc.add_lef_bonds(sim, translocator1, translocator2,
                     number_of_monomers, scale=SCALE)

    del translocator1, translocator2

    gsd_filename = (
        f'{savepath}/trajectory_{condition}_alpha{alpha}_tau{tau}_{iteration}.gsd')
    pc.relax_and_run(sim, relax_field, force_field, gsd_filename)


if __name__ == "__main__":
    main()
