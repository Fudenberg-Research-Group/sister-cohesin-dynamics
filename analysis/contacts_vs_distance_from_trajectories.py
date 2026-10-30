"""
Contact maps from the polymer trajectories produced by
../polymer_sister_chromatids/{WT_dS,dW_dSdW}_polymer.py.

Builds two binned maps per condition:

  cis   -- contacts within each chromatid, summed over both chromatids
  trans -- contacts between the two chromatids

Both are accumulated over every frame of every replica and written as HDF5
next to the trajectories.

Usage:
    python contact_map.py $SLURM_ARRAY_TASK_ID
"""

import multiprocessing as mp
import os
import sys
import time
from functools import partial

import gsd.hoomd
import h5py
import numpy as np
import polychrom_hoomd.utils as utils
from polykit.analysis import polymer_analyses
from scipy.spatial import cKDTree

# --- what to analyse -----------------------------------------------------
CONDITION = "WT12h"
ALPHA_LIST = [100]
TAU_LIST = [11]
NUMBER_TRIALS = 5          # replicas, 0-indexed

# --- analysis parameters -------------------------------------------------
NUMBER_OF_MAPS = 8         # monomers per bin
CONTACT_RADIUS = 2.3       # contact cutoff, in monomer diameters

# Where the polymer stage wrote its trajectories, relative to this script.
OUTPUT_ROOT = "../polymer_sister_chromatids/output"


def trajectory_dir(condition, alpha, tau):
    """Directory holding one condition's replicas."""
    return (f"{OUTPUT_ROOT}/"
            f"{condition}_alpha_{alpha}_tau_{tau}h_beta0_gamma0")


def trajectory_file(path_name, condition, alpha, tau, rng):
    """One replica's recorded production trajectory."""
    return f"{path_name}/trajectory_{condition}_alpha{alpha}_tau{tau}_{rng}.gsd"


def existing_replicas(path_name, condition, alpha, tau, number_trials):
    """Replica indices whose trajectory file is actually present.

    Replicas are 0-indexed, matching `replica_ids = np.arange(5)` in the run
    scripts and the `_{iteration}.gsd` suffix they write.
    """
    existing = []
    for rng in range(number_trials):
        file_path = trajectory_file(path_name, condition, alpha, tau, rng)
        if os.path.exists(file_path):
            existing.append(rng)
        else:
            print(f"[Skip] Missing file: {file_path}")
    return existing


def calculate_trans_contacts(data, sister_size, cutoff):
    """Contacts between the two chromatids, within `cutoff`.

    Parameters
    ----------
    data : (2*sister_size, 3) array
        Unwrapped coordinates, chromatid 1 first.
    sister_size : int
        Monomers per chromatid.
    cutoff : float
        Contact radius.

    Returns
    -------
    (k, 2) array of (chromatid-1 index, chromatid-2 index) pairs.

    No periodic shift is applied: the chromatids are held in a single
    spherical confinement, so after `utils.unwrap_coordinates` they cannot sit
    in different periodic images.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")
    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    sister1 = data[:sister_size]
    sister2 = data[sister_size:]

    tree1 = cKDTree(sister1)
    tree2 = cKDTree(sister2)
    indexes = tree1.query_ball_tree(tree2, cutoff)

    pairs = [(row_idx, item)
             for row_idx, row in enumerate(indexes) for item in row]

    return np.array(pairs)


def _accumulate(matrix, contacts, number_of_maps):
    """Bin a (k, 2) contact list and add it symmetrically into `matrix`."""
    contact_array = np.asarray(contacts, dtype=np.int32)
    if len(contact_array) == 0:
        return
    bin_x = contact_array[:, 0] // number_of_maps
    bin_y = contact_array[:, 1] // number_of_maps
    np.add.at(matrix, (bin_x, bin_y), 1)
    np.add.at(matrix, (bin_y, bin_x), 1)


def process_trajectory_cis(rng, path_name, condition, alpha, tau,
                           number_of_monomers, number_of_maps, cutoff):
    """Per-chromatid cis contact maps for one replica."""
    num_bins = number_of_monomers // number_of_maps
    local_cis_matrix1 = np.zeros((num_bins, num_bins), dtype='int32')
    local_cis_matrix2 = np.zeros((num_bins, num_bins), dtype='int32')

    file_path = trajectory_file(path_name, condition, alpha, tau, rng)
    traj = gsd.hoomd.open(file_path, 'r')
    total_frames = len(traj)

    for k, snap in enumerate(traj):
        if k % 100 == 0:
            print(f"RNG {rng}: cis frame {k}/{total_frames}", flush=True)

        unfolded = utils.unwrap_coordinates(snap)

        contacts1 = polymer_analyses.calculate_contacts(
            unfolded[0:number_of_monomers], cutoff=cutoff)
        contacts2 = polymer_analyses.calculate_contacts(
            unfolded[number_of_monomers:], cutoff=cutoff)

        _accumulate(local_cis_matrix1, contacts1, number_of_maps)
        _accumulate(local_cis_matrix2, contacts2, number_of_maps)

    traj.close()
    return local_cis_matrix1, local_cis_matrix2


def process_trajectory_trans(rng, path_name, condition, alpha, tau,
                             number_of_monomers, number_of_maps, cutoff):
    """Inter-chromatid trans contact map for one replica."""
    num_bins = number_of_monomers // number_of_maps
    local_trans_matrix = np.zeros((num_bins, num_bins), dtype='int32')

    file_path = trajectory_file(path_name, condition, alpha, tau, rng)
    traj = gsd.hoomd.open(file_path, 'r')
    total_frames = len(traj)

    for k, snap in enumerate(traj):
        if k % 100 == 0:
            print(f"RNG {rng}: trans frame {k}/{total_frames}", flush=True)

        unfolded = utils.unwrap_coordinates(snap)
        contacts = calculate_trans_contacts(
            unfolded, sister_size=number_of_monomers, cutoff=cutoff)

        _accumulate(local_trans_matrix, contacts, number_of_maps)

    traj.close()
    return local_trans_matrix


def _worker_count(n_tasks):
    """Cores to use, respecting a SLURM allocation when there is one."""
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    available = int(allocated) if allocated else mp.cpu_count() - 1
    return max(1, min(available, n_tasks))


def _run_pool(process_func, rng_list, label):
    """Map `process_func` over the replicas and report timing."""
    num_cores = _worker_count(len(rng_list))
    print(f"{label}: {len(rng_list)} replicas on {num_cores} cores")

    start_time = time.time()
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_func, rng_list)
    print(f"{label}: processing time {time.time() - start_time:.2f} s")

    return results


def cis_count(path_name, condition, alpha, tau,
              number_of_monomers, number_of_maps, rng_list, cutoff):
    """Cis maps per chromatid and their sum, accumulated over all replicas."""
    process_func = partial(
        process_trajectory_cis,
        path_name=path_name, condition=condition, alpha=alpha, tau=tau,
        number_of_monomers=number_of_monomers,
        number_of_maps=number_of_maps, cutoff=cutoff,
    )
    results = _run_pool(process_func, rng_list, "cis")

    num_bins = number_of_monomers // number_of_maps
    cis_matrix1 = np.zeros((num_bins, num_bins), dtype='int32')
    cis_matrix2 = np.zeros((num_bins, num_bins), dtype='int32')

    for rng, (local1, local2) in zip(rng_list, results):
        cis_matrix1 += local1
        cis_matrix2 += local2
        print(f"cis: added replica {rng}")

    return cis_matrix1, cis_matrix2, cis_matrix1 + cis_matrix2


def trans_count(path_name, condition, alpha, tau,
                number_of_monomers, number_of_maps, rng_list, cutoff):
    """Trans map accumulated over all replicas."""
    process_func = partial(
        process_trajectory_trans,
        path_name=path_name, condition=condition, alpha=alpha, tau=tau,
        number_of_monomers=number_of_monomers,
        number_of_maps=number_of_maps, cutoff=cutoff,
    )
    results = _run_pool(process_func, rng_list, "trans")

    num_bins = number_of_monomers // number_of_maps
    trans_matrix = np.zeros((num_bins, num_bins), dtype='int32')

    for rng, local_matrix in zip(rng_list, results):
        trans_matrix += local_matrix
        print(f"trans: added replica {rng}")

    return trans_matrix


def main():
    process_id = int(sys.argv[1]) - 1

    x, y = np.meshgrid(ALPHA_LIST, TAU_LIST)
    parameters = list(zip(x.flatten(), y.flatten()))
    alpha, tau = parameters[process_id]
    print(f"condition={CONDITION}, alpha={alpha}, tau={tau}")

    path_name = trajectory_dir(CONDITION, alpha, tau)
    rng_list = existing_replicas(path_name, CONDITION, alpha, tau, NUMBER_TRIALS)
    if not rng_list:
        raise FileNotFoundError(f"No trajectories found under {path_name}")

    # Monomers per chromatid, taken from the trajectory itself
    with gsd.hoomd.open(
            trajectory_file(path_name, CONDITION, alpha, tau, rng_list[0]),
            'r') as traj:
        number_of_monomers = traj[0].particles.N // 2

    num_bins = number_of_monomers // NUMBER_OF_MAPS
    print(f"{number_of_monomers} monomers per chromatid, "
          f"{NUMBER_OF_MAPS} monomers per bin -> {num_bins} bins")

    savedir = os.path.join(path_name, f"cutoff{CONTACT_RADIUS}")
    os.makedirs(savedir, exist_ok=True)

    _, _, cis_matrix = cis_count(
        path_name, CONDITION, alpha, tau,
        number_of_monomers, NUMBER_OF_MAPS, rng_list, cutoff=CONTACT_RADIUS)

    with h5py.File(f"{savedir}/cis_matrix.h5", 'w') as f:
        f.create_dataset('cis_matrix', data=cis_matrix)

    trans_matrix = trans_count(
        path_name, CONDITION, alpha, tau,
        number_of_monomers, NUMBER_OF_MAPS, rng_list, cutoff=CONTACT_RADIUS)

    with h5py.File(f"{savedir}/trans_matrix.h5", 'w') as f:
        f.create_dataset('trans_matrix', data=trans_matrix)

    print(f"Wrote cis_matrix.h5 and trans_matrix.h5 to {savedir}")


if __name__ == "__main__":
    main()
