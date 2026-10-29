"""Generate lattice extrusion parameters for SORORIN depletion, 7 h readout.

As dS12h_rates.py, but the rates are read out at (6 h + 8 h) / 2 = 7 h and the
lattice stage runs for 6 h. Used as phase 1 of the two-phase dSORORIN+dWAPL
runs.
"""
import json
from pathlib import Path

import numpy as np
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
    calculate_sister_RAD21_bound_time,
    paras_coh,
    paras_ext,
    sol_rates_coh,
    sol_rates_ext,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(filename):
    """Load a JSON configuration stored next to this script."""
    with open(SCRIPT_DIR / filename, "r") as f:
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
BP_PER_SITE = 500
NUM_LATTICE_SITES = GENOMIC_REGION_SIZE_BP // BP_PER_SITE

# run simulations for given parameters


def run_simulation(config, residence_time, sister_damping, collision_prob, bypass_prob):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
    """
    initial_num_sister_cohesins = int(2909 * np.exp(8/residence_time))
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

    depletion_level_sororin = config['simulation_parameters']['depletion_level']
    remaining_level_sororin = 1 - depletion_level_sororin

    # whether to use the new number of sisterCs:
    init_conditions_ext_coh = {
       Rac_free_init: 0,
       Rac_init: initial_num_sister_cohesins,
       RacP_init: 0,
       RacPW_init: 0,
       RacPS_init: 0,

       R_free_init: paras_dict_coh_local[N_R],
       RN_init: 0,
       R_init: 0,
       RP_init: 0,
       RPW_init: 0,

       N_init: config['base_parameters']['N_N'],
       S_init: paras_dict_coh_local[N_S] * remaining_level_sororin,
       W_init: config['base_parameters']['N_W'],
       P_init: config['base_parameters']['N_P'],
    }
    # Update combined dictionary
    paras_dict_local = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh
    paras_dict_ext_coh_local = {str(key): value for key, value in paras_dict_local.items()}

    # Simulate model for 8h + 6h = 14h
    Model_ext_coh = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_dS = r_ext_coh.simulate(0, 3600*18, 3600*18)

    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW',
           'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']

    df_dS = pd.DataFrame(Model_ext_coh_dS, columns=columns)

    analysis_time_hours = config['simulation_parameters']['analysis_timepoint_hours']

    analysis_index = 3600 * analysis_time_hours - 1

    total_lef_per_bp = NUM_EXTRUDERS / (NUM_LATTICE_SITES * BP_PER_SITE)

    total_sister_rad21 = (df_dS['RacPS'][analysis_index] + df_dS['RacPW'][analysis_index] +
                          df_dS['RacP'][analysis_index] + df_dS['Rac'][analysis_index])

    sister_rad21_lifetime_at_analysis = calculate_sister_RAD21_bound_time(
        K_RacPW_Rac_free=paras_dict_ext_coh_local['K_RacPW_Rac_free'],
        B_W_sister=df_dS['RacPW'][analysis_index],
        B_R_sister=total_sister_rad21)

    # Calculate transition rates
    rates = {
        'R_free_to_RN': paras_dict_ext_coh_local['Kext_R_free_RN'] * df_dS['N'][analysis_index],
        'RPW_R_free': paras_dict_ext_coh_local['Kext_RPW_R_free'],
        'RN_R': paras_dict_ext_coh_local['Kext_RN_R'],
        'R_RN': paras_dict_ext_coh_local['Kext_R_RN'] * df_dS['N'][analysis_index],
        'R_RP': paras_dict_ext_coh_local['Kext_R_RP'] * df_dS['P'][analysis_index],
        'RP_R': paras_dict_ext_coh_local['Kext_RP_R'],
        'RP_RPW': paras_dict_ext_coh_local['Kext_RP_RPW'] * df_dS['W'][analysis_index],
        'RPW_RP': paras_dict_ext_coh_local['Kext_RPW_RP'],
    }

    # Load base parameters and update
    with open(SCRIPT_DIR / config["output_template_file"], "r") as f:
        output_params = json.load(f)

    # establishment, 2909 is the number of sistercs at S8h for 800mb
    gamma_constant = 2909/residence_time/(1-np.exp(-8/residence_time))
    sister_rad21_lifetime_hours = float(sister_rad21_lifetime_at_analysis / 3600)

    # Calculate the number of sisterCs at S8h using the birth-death model
    num_sister_cohesins_s8h = gamma_constant * sister_rad21_lifetime_hours * (1 - np.exp(-8 / sister_rad21_lifetime_hours))

    output_params["LEF_on_rate"]["A"] = float(rates['R_free_to_RN'])
    output_params["LEF_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_diffusion_rate"]["A"] = 0

    output_params["LEF_stalled_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_transition_rates"]["21"]["A"] = float(rates['R_RN'])
    output_params["LEF_transition_rates"]["23"]["A"] = float(rates['R_RP'])
    output_params["LEF_transition_rates"]["12"]["A"] = float(rates['RN_R'])
    output_params["LEF_transition_rates"]["32"]["A"] = float(rates['RP_R'])
    output_params["LEF_transition_rates"]["34"]["A"] = float(rates['RP_RPW'])
    output_params["LEF_transition_rates"]["43"]["A"] = float(rates['RPW_RP'])

    output_params["Total_LEF_per_bp"] = total_lef_per_bp
    output_params["bp_per_site"] = BP_PER_SITE
    output_params["region_size_bp"] = NUM_LATTICE_SITES * BP_PER_SITE
    output_params["steps"] = 6 * 3600
    output_params["num_of_sisters"] = int(num_sister_cohesins_s8h)
    output_params["sister_damping_factor"] = sister_damping
    output_params["sister_lifetime"] = int(sister_rad21_lifetime_at_analysis)
    output_params["collision_release_prob"] = collision_prob
    output_params["bypass_prob"] = bypass_prob

    return output_params


def main():
    """Sweep all parameter combinations and write one JSON per point."""
    # Load configuration
    config = load_config('network_parameters_dS7h.json')
    # Create output directory
    output_dir = SCRIPT_DIR / config['output_directory']
    output_dir.mkdir(parents=True, exist_ok=True)

    for residence_time in RESIDENCE_TIMES_HOURS:
        for sister_damping in SISTER_DAMPING_FACTORS:
            for k, collision_prob in enumerate(COLLISION_RELEASE_PROBS):
                for q, bypass_prob in enumerate(BYPASS_PROBS):
                    print(f"\nRunning: residence_time={residence_time}h, sister_damping={sister_damping}, collision_prob={collision_prob}, bypass_prob={bypass_prob}")
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
