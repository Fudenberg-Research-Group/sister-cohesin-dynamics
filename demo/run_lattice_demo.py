"""Run a reduced-size, CPU-only sister-lattice demonstration.

This test exercises the same translocator construction and synchronized
trajectory runner as the manuscript-scale simulations, but it uses a 1 Mb
lattice, four sister linkers, and 1,000 steps. It validates installation and
output formats; it is not intended to reproduce manuscript results.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LATTICE_ROOT = REPOSITORY_ROOT / "discrete_time_extrusion_sister"
sys.path.insert(0, str(LATTICE_ROOT))

import simulation_setup as setup  # noqa: E402
from discrete_time_extrusion.sister_lifecycle import (  # noqa: E402
    run_synchronized_trajectories_continue,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reduced-size CPU lattice demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "lattice_demo_config.json",
        help="Demo configuration JSON (default: demo/lattice_demo_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "demo_output" / "lattice",
        help="Output directory (default: demo/demo_output/lattice).",
    )
    return parser.parse_args()


def load_demo(config_path):
    with config_path.open() as stream:
        demo = json.load(stream)

    parameter_path = REPOSITORY_ROOT / demo["base_parameters"]
    with parameter_path.open() as stream:
        parameters = json.load(stream)
    parameters.update(demo["parameter_overrides"])

    return demo, parameters


def validate_anchors(demo, monomers_per_replica, num_of_sisters):
    anchors_local = np.asarray(
        demo["paired_anchor_positions_local"], dtype=int
    )

    if len(anchors_local) != num_of_sisters:
        raise ValueError(
            "paired_anchor_positions_local must contain num_of_sisters entries"
        )
    if np.any((anchors_local < 0) | (anchors_local >= monomers_per_replica)):
        raise ValueError("Local anchors must lie within one chromatid lattice")

    # build_translocator_pair expects chromatid 2 in concatenated coordinates
    # and converts it back to local coordinates by subtracting this offset.
    anchors1 = anchors_local
    anchors2 = anchors_local + monomers_per_replica
    return anchors1, anchors2


def main():
    args = parse_args()
    demo, parameters = load_demo(args.config.resolve())

    monomers_per_replica, type_list, site_types, num_of_sisters = (
        setup.lattice_geometry(parameters)
    )
    anchors1, anchors2 = validate_anchors(
        demo, monomers_per_replica, num_of_sisters
    )

    np.random.seed(demo["seed"])
    translocator1, translocator2 = setup.build_translocator_pair(
        parameters,
        type_list,
        site_types,
        [],
        [],
        anchors1,
        anchors2,
        monomers_per_replica,
    )

    start = time.perf_counter()
    run_synchronized_trajectories_continue(
        translocator1,
        translocator2,
        parameters["sister_lifetime"],
        steps=demo["steps"],
        prune_unbound_LEFs=True,
        track_sisters=True,
        sample_interval=demo["sample_interval"],
        seed=demo["seed"],
        clear_trajectories=True,
    )
    elapsed = time.perf_counter() - start

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup.save_trajectory_pair(
        str(output_dir),
        "lattice_demo",
        0,
        translocator1,
        translocator2,
    )

    summary = {
        "lattice_sites_per_chromatid": monomers_per_replica,
        "extruders_per_chromatid": int(
            parameters["number_of_replica"]
            * parameters["region_size_bp"]
            * parameters["Total_LEF_per_bp"]
        ),
        "sister_linkers": num_of_sisters,
        "steps": demo["steps"],
        "sample_interval": demo["sample_interval"],
        "samples_per_chromatid": len(translocator1.lef_trajectory),
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    print(f"Lattice demo complete in {elapsed:.2f} seconds")
    print(f"Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
