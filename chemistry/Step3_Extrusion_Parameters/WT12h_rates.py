"""Generate lattice extrusion parameters for WT for tau = 11 h.

Solves the cohesin reaction network for the WT condition,
reads out extruder state-transition rates at the analysis timepoint, and
writes one JSON per (alpha, tau, beta, gamma) point into the output
directory named in network_parameters_WT12h.json.
"""

import json
from pathlib import Path

import pandas as pd
import sympy as sym
import tellurium as te

from crn_core import (
    MODEL_EXT_COH_TEMPLATE,
    N_R,
    N_S,
    build_model,
    calculate_cohesive_parameters,
    calculate_extrusive_parameters,
    paras_coh,
    paras_ext,
    sol_rates_coh,
    sol_rates_ext,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(filename):
    """Load configuration from JSON file"""
    with open(SCRIPT_DIR / filename, 'r') as f:
        return json.load(f)


# RAD21 residence time, in hours (chemistry input).
RESIDENCE_TIMES_HOURS = [11]

# Downstream lattice/polymer parameters
SISTER_DAMPING_FACTORS = [100]
COLLISION_RELEASE_PROBS = [1e-6]
BYPASS_PROBS = [1e-4]

# Cohesin copy numbers
NUM_EXTRUDERS = 7764
GENOMIC_REGION_SIZE_BP = 800_000_000
BP_PER_SITE = 2500
NUM_LATTICE_SITES = GENOMIC_REGION_SIZE_BP // BP_PER_SITE


def run_simulation(config, residence_time, sister_damping, collision_prob, bypass_prob):
    """Solve the reaction network and assemble one parameter set.

    residence_time is in hours; sister_damping (alpha), collision_prob (beta)
    and bypass_prob (gamma) are passed through to the lattice simulation
    unchanged. Returns the dict that gets written as JSON.
    """
    # Calculate parameter values
    paras_values_coh = calculate_cohesive_parameters(config, residence_time)
    paras_dict_coh_local = dict(zip(paras_coh, paras_values_coh))

    # Solve for cohesive rates
    for rate_symbol, rate_expr in sol_rates_coh.items():
        rate = rate_expr.evalf(subs=paras_dict_coh_local)
        paras_dict_coh_local[rate_symbol] = rate

    # Calculate extrusive parameters
    paras_values_ext = calculate_extrusive_parameters(config)
    paras_dict_ext_local = dict(zip(paras_ext, paras_values_ext))

    # Solve for extrusive rates
    for rate_symbol, rate_expr in sol_rates_ext.items():
        rate = rate_expr.evalf(subs=paras_dict_ext_local)
        paras_dict_ext_local[rate_symbol] = rate

    # Define symbolic initial conditions
    Rac_free_init, Rac_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init = sym.symbols(
        "Rac_free_init, Rac_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init")

    init_conditions_ext_coh = {
       Rac_free_init: 0,
       Rac_init: paras_dict_coh_local[N_R]*1/2,
       RacP_init: 0,
       RacPW_init: 0,
       RacPS_init: 0,

       R_free_init: paras_dict_coh_local[N_R],
       RN_init: 0,
       R_init: 0,
       RP_init: 0,
       RPW_init: 0,

       N_init: config['base_parameters']['N_N'],
       S_init: paras_dict_coh_local[N_S],
       W_init: config['base_parameters']['N_W'],
       P_init: config['base_parameters']['N_P'],
    }
    # Update combined dictionary
    paras_dict_local = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh
    paras_dict_ext_coh_local = {str(key): value for key, value in paras_dict_local.items()}

    # Simulate model 16h
    Model_ext_coh = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_WT = r_ext_coh.simulate(0, 3600*16, 3600*16)

    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW',
           'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']

    df_WT = pd.DataFrame(Model_ext_coh_WT, columns=columns)

    start_time = 0
    end_time = 3600 * 16 - 1
    bound_sisterC_start = (df_WT['Rac'][start_time] +  df_WT['RacPS'][start_time] +
                       df_WT['RacP'][start_time] + df_WT['RacPW'][start_time])
    bound_sisterC_end = (df_WT['Rac'][end_time] +  df_WT['RacPS'][end_time] +
                       df_WT['RacP'][end_time] + df_WT['RacPW'][end_time])
    num_used_in_bypass = bound_sisterC_end/bound_sisterC_start*2909

    analysis_hours = config['simulation_parameters']['analysis_timepoint_hours']
    index = 3600 * analysis_hours - 1
    total_lef_per_bp = int(NUM_EXTRUDERS) / GENOMIC_REGION_SIZE_BP
    # Calculate transition rates
    rates = {
        'R_free_to_RN': paras_dict_ext_coh_local['Kext_R_free_RN'] * df_WT['N'][index],
        'RPW_R_free': paras_dict_ext_coh_local['Kext_RPW_R_free'],
        'RN_R': paras_dict_ext_coh_local['Kext_RN_R'],
        'R_RN': paras_dict_ext_coh_local['Kext_R_RN'] * df_WT['N'][index],
        'R_RP': paras_dict_ext_coh_local['Kext_R_RP'] * df_WT['P'][index],
        'RP_R': paras_dict_ext_coh_local['Kext_RP_R'],
        'RP_RPW': paras_dict_ext_coh_local['Kext_RP_RPW'] * df_WT['W'][index],
        'RPW_RP': paras_dict_ext_coh_local['Kext_RPW_RP'],
    }

    # Load base parameters and update
    with open(SCRIPT_DIR / "extrusion_dict_RN_RB_RP_RW_HBD_WT.json") as f:
        output_params = json.load(f)

    output_params["LEF_on_rate"]["A"] = float(rates['R_free_to_RN'])
    output_params["LEF_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_stalled_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_transition_rates"]["21"]["A"] = float(rates['R_RN'])
    output_params["LEF_transition_rates"]["23"]["A"] = float(rates['R_RP'])
    output_params["LEF_transition_rates"]["12"]["A"] = float(rates['RN_R'])
    output_params["LEF_transition_rates"]["32"]["A"] = float(rates['RP_R'])
    output_params["LEF_transition_rates"]["34"]["A"] = float(rates['RP_RPW'])
    output_params["LEF_transition_rates"]["43"]["A"] = float(rates['RPW_RP'])

    output_params["Total_LEF_per_bp"] = total_lef_per_bp
    output_params["bp_per_site"] = BP_PER_SITE
    output_params["region_size_bp"] = GENOMIC_REGION_SIZE_BP
    output_params["steps"] = 16 * 3600
    output_params["num_of_sisters"] = 2909
    output_params["sister_damping_factor"] = sister_damping
    output_params["sister_lifetime"] = residence_time * 3600
    output_params["collision_release_prob"] = collision_prob
    output_params["bypass_prob"] = bypass_prob

    return output_params


def main():
    """Sweep all parameter combinations and write one JSON per point."""
    # Load configuration
    config = load_config('network_parameters_WT12h.json')

    # Create output directory
    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for residence_time in RESIDENCE_TIMES_HOURS:
            for sister_damping in SISTER_DAMPING_FACTORS:
                for k, collision_prob in enumerate(COLLISION_RELEASE_PROBS):
                    for q, bypass_prob in enumerate(BYPASS_PROBS):
                        print(f"\nRunning: residence_time={residence_time}h, sister_damping={sister_damping}, collision_release_prob={collision_prob}, bypass_prob={bypass_prob}")
                        # Run simulation
                        params = run_simulation(config, residence_time, sister_damping, collision_prob, bypass_prob)

                        # Save to file
                        filename = (f"{config['output_prefix']}_"
                            f"alpha{sister_damping}_tau{residence_time}h_beta{k}_gamma{q}.json")
                        filepath = output_dir / filename

                        with open(filepath, "w") as f:
                            json.dump(params, f, indent=4)

                        print(f"Saved: {filepath}")

    print("\nAll simulations complete!")


if __name__ == "__main__":
    main()
