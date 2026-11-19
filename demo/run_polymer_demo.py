"""Run a reduced-size sister-polymer demonstration.

This demo maps the final frame of the reduced lattice demo onto two short
chromatid polymers and exercises the same named-topology updates, force-field
construction, FIRE relaxation, and Langevin production used by the
manuscript-scale polymer simulations. It validates execution and output
formats; it is not intended to reproduce manuscript results.
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np
import polychrom_hoomd.build as build
import polychrom_hoomd.extrude as extrude


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLYMER_ROOT = REPOSITORY_ROOT / "polymer_sister_chromatids"
sys.path.insert(0, str(POLYMER_ROOT))

import polymer_common as pc  # noqa: E402


BOND_TYPES = ["Backbone", "LEF", "LEF_dummy", "Sister", "Sister_dummy"]


def require_named_topology_support():
    """Fail clearly when the unpatched upstream dependency is installed."""
    parameters = inspect.signature(extrude.update_topology).parameters
    if "bond_name" not in parameters:
        raise RuntimeError(
            "The polymer demo requires the polychrom-hoomd commit pinned in "
            "environment.yml, which supports named topology updates."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reduced-size sister-polymer demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "polymer_demo_config.json",
        help="Demo configuration JSON (default: demo/polymer_demo_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "demo_output" / "polymer",
        help="Output directory (default: demo/demo_output/polymer).",
    )
    return parser.parse_args()


def load_config(path):
    with path.open() as stream:
        config = json.load(stream)

    positive_integer_keys = (
        "monomers_per_chromatid",
        "fire_blocks",
        "fire_steps",
        "equilibration_steps",
        "production_steps",
        "gsd_period",
    )
    for key in positive_integer_keys:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or not 1 <= config["seed"] <= 65535:
        raise ValueError("seed must be an integer between 1 and 65535")
    if config["production_steps"] < config["gsd_period"]:
        raise ValueError("production_steps must be at least gsd_period")
    if config["density"] <= 0:
        raise ValueError("density must be positive")
    if config["sister_separation"] <= 0:
        raise ValueError("sister_separation must be positive")

    return config


def cubic_walk(length):
    """Return a deterministic self-avoiding walk on an even cubic grid."""
    side = int(np.ceil(length ** (1 / 3)))
    if side % 2:
        side += 1

    positions = []
    for z_index in range(side):
        y_values = (
            range(side)
            if z_index % 2 == 0
            else range(side - 1, -1, -1)
        )
        for row_index, y_index in enumerate(y_values):
            x_values = (
                range(side)
                if row_index % 2 == 0
                else range(side - 1, -1, -1)
            )
            for x_index in x_values:
                positions.append((x_index, y_index, z_index))
                if len(positions) == length:
                    return np.asarray(positions, dtype=float)

    raise RuntimeError("Cubic grid did not contain enough sites")


def paired_chromatid_positions(length, separation):
    """Create two aligned copies of one cubic walk."""
    base = cubic_walk(length)
    shift = np.array([separation / 2, 0.0, 0.0])
    return np.vstack((base - shift, base + shift))


def initialize_simulation(length, density, separation, seed):
    """Create a small two-chromatid HOOMD system in spherical confinement."""
    number_of_monomers, confinement_radius = pc.system_geometry(
        chromosome_sizes=(length,), density=density
    )
    positions = paired_chromatid_positions(number_of_monomers, separation)

    snapshot = build.get_simulation_box(
        box_length=2 * confinement_radius, pad=1.0
    )
    build.set_chromosomes(
        snapshot,
        positions,
        [number_of_monomers, number_of_monomers],
        monomer_type_list=["A"],
        bond_type_list=BOND_TYPES,
        center=True,
    )
    simulation = hoomd.Simulation(
        device=build.get_hoomd_device(), seed=seed
    )
    simulation.create_state_from_snapshot(snapshot)
    return simulation, number_of_monomers, confinement_radius


def require_lattice_output(lattice_output, expected_monomers):
    paths = [
        lattice_output / "lattice_demo_trajectory1_0.pkl",
        lattice_output / "lattice_demo_trajectory2_0.pkl",
        lattice_output / "summary.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run 'python demo/run_lattice_demo.py' first; missing: "
            + ", ".join(missing)
        )

    with (lattice_output / "summary.json").open() as stream:
        summary = json.load(stream)
    actual_monomers = summary["lattice_sites_per_chromatid"]
    if actual_monomers != expected_monomers:
        raise ValueError(
            "Polymer and lattice demos must use the same chromatid size: "
            f"{expected_monomers} != {actual_monomers}"
        )


def main():
    args = parse_args()
    config = load_config(args.config.resolve())
    require_named_topology_support()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lattice_output = (REPOSITORY_ROOT / config["lattice_output"]).resolve()
    require_lattice_output(
        lattice_output, config["monomers_per_chromatid"]
    )

    translocator1, translocator2 = pc.load_lattice_trajectories(
        str(lattice_output), "lattice_demo", 0
    )

    start = time.perf_counter()
    simulation, monomers, confinement_radius = initialize_simulation(
        config["monomers_per_chromatid"],
        config["density"],
        config["sister_separation"],
        config["seed"],
    )
    relax_field, force_field = pc.build_force_field(
        POLYMER_ROOT / "data" / "force_dict_sister_extruder.json",
        confinement_radius,
    )
    sister_bonds = pc.add_sister_bonds(
        simulation,
        translocator1,
        translocator2,
        str(output_dir),
        0,
        monomers,
    )
    lef_bonds = pc.add_lef_bonds(
        simulation,
        translocator1,
        translocator2,
        monomers,
    )

    trajectory_path = output_dir / "polymer_demo_trajectory.gsd"
    pc.relax_and_run(
        simulation,
        relax_field,
        force_field,
        str(trajectory_path),
        fire_blocks=config["fire_blocks"],
        fire_steps=config["fire_steps"],
        equil_steps=config["equilibration_steps"],
        prod_steps=config["production_steps"],
        gsd_period=config["gsd_period"],
    )

    writer = simulation.operations.writers[-1]
    if hasattr(writer, "flush"):
        writer.flush()

    with gsd.hoomd.open(str(trajectory_path), "r") as trajectory:
        frames = len(trajectory)
        particles = int(trajectory[0].particles.N) if frames else 0

    if frames == 0 or particles != 2 * monomers:
        raise RuntimeError("Polymer demo did not write the expected trajectory")

    summary = {
        "monomers_per_chromatid": monomers,
        "particles": particles,
        "sister_bonds": len(np.unique(sister_bonds, axis=0)),
        "lef_bonds": len(lef_bonds),
        "fire_steps": config["fire_blocks"] * config["fire_steps"],
        "equilibration_steps": config["equilibration_steps"],
        "production_steps": config["production_steps"],
        "saved_frames": frames,
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    elapsed = time.perf_counter() - start
    print(f"Polymer demo complete in {elapsed:.2f} seconds")
    print(f"Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
