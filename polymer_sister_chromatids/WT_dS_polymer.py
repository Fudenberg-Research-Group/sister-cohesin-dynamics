"""Place a WT or dSORORIN lattice configuration onto the polymer model.

Reads the last frame of the lattice trajectories produced by
../discrete_time_extrusion_sister/WT_dS.py, applies those crosslinks and LEF
bonds to a pre-equilibrated two-chromatid HOOMD system, then relaxes and runs
recorded Langevin production.
"""

import argparse
import os

import polymer_common as pc

CONDITIONS = {
    "WT": "WT12h",
    "dS": "dS12h",
}

# Lattice sites per polymer monomer for these conditions (bp_per_site = 2500).
SCALE = 1


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
    condition = CONDITIONS[args.condition]
    iteration = args.replica

    print(f"Running simulation for condition: {condition}, replica: {iteration}")

    alpha = 100
    tau = 11

    readpath = 'input'

    savepath = f'output/{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0'
    os.makedirs(savepath, exist_ok=True)

    read_traj_path = (
        f'../discrete_time_extrusion_sister/output/tau{tau}h/'
        f'{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0'
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

    equil_steps = 5e6 if condition == "WT12h" else 3e7
    pc.relax_and_run(
        sim,
        relax_field,
        force_field,
        gsd_filename,
        equil_steps=equil_steps
    )


if __name__ == "__main__":
    main()
