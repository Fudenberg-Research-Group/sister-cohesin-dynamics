

###
### This script generates the parameters for lattice simulations for dSORORIN at 12h (WT 10h, dWAPL 14h)
import json
import numpy as np
import sympy as sym
import tellurium as te
import pandas as pd
from pathlib import Path
from itertools import product

from crn_core import (
    MODEL_EXT_COH_TEMPLATE,
    N_R,
    N_S,
    paras_coh,
    paras_ext,
    sol_rates_coh,
    sol_rates_ext,
    build_model,
    calculate_sister_RAD21_bound_time,
    calculate_cohesive_parameters,
    calculate_extrusive_parameters,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Load configuration files
def load_config(filename):
    """Load configuration from JSON file"""
    with open(SCRIPT_DIR / filename, 'r') as f:
        return json.load(f)


# Chemistry parameter
# Define the ranges for residence time and damping values
RESIDENCE_TIMES_HOURS = [11]

# Downstream lattice simulation parameters
SISTER_DAMPING_FACTORS = [100]
COLLISION_RELEASE_PROBS = [1e-6]
BYPASS_PROBS = [1e-3]
# COLLISION_RELEASE_PROBS = np.logspace(-6, -4, 20)
# pairs = list(product(RESIDENCE_TIMES_HOURS, SISTER_DAMPING_FACTORS))

# Physical constants
NUM_SISTERCS = 7764
NUM_EXTRUDERS = 7764
GENOMIC_REGION_SIZE_BP = 800_000_000
BP_PER_SITE = 2500
NUM_LATTICE_SITES = GENOMIC_REGION_SIZE_BP // BP_PER_SITE


# run simulations for given parameters
def run_simulation(config, residence_time, sister_damping, collision_prob, bypass_prob):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
    """
    # back 8h to make the Sororin effect over replication
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

    # Simulate model for 24h since S0h
    Model_ext_coh = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_dS = r_ext_coh.simulate(0, 3600*24, 3600*24)

    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW',
           'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']

    df_dS = pd.DataFrame(Model_ext_coh_dS, columns=columns)

    # index_start_S8h = 3600*8 - 1
    # NUM_SISTERCS_start_S8h = int(df_dS['Rac'][index_start_S8h]  + df_dS['RacPS'][index_start_S8h] +
    #                   df_dS['RacP'][index_start_S8h] + df_dS['RacPW'][index_start_S8h])
    '''
    start_time_0 = 0
    start_time_S8h = 3600 * 8 - 1
    end_time_S24h = 3600 * 24 - 1
    bound_sisterC_0 = (df_dS['Rac'][start_time_0] +  df_dS['RacPS'][start_time_0] +
                       df_dS['RacP'][start_time_0] + df_dS['RacPW'][start_time_0])
    bound_sisterC_S8h = (df_dS['Rac'][start_time_S8h] +  df_dS['RacPS'][start_time_S8h] +
                       df_dS['RacP'][start_time_S8h] + df_dS['RacPW'][start_time_S8h])
    bound_sisterC_S24h = (df_dS['Rac'][end_time_S24h] +  df_dS['RacPS'][end_time_S24h] +
                       df_dS['RacP'][end_time_S24h] + df_dS['RacPW'][end_time_S24h])
    '''
    # num_used_in_bypass = bound_sisterC_S24h/bound_sisterC_S8h*1352
    # num_used_in_bypass1 = bound_sisterC_S24h/bound_sisterC_0*initial_num_sister_cohesins

    # Calculate LEF_sep and velocity from 8h to 24h, using mean 16h (8h since S8h)
    # the rate comes from center of the process 12h
    analysis_time_hours = config['simulation_parameters']['analysis_timepoint_hours']
    analysis_index = 3600 * analysis_time_hours - 1
    total_bound_ext = (df_dS['R'][analysis_index] + df_dS['RN'][analysis_index] +
                       df_dS['RP'][analysis_index] + df_dS['RPW'][analysis_index])
    bound_extC_ratio = total_bound_ext / (paras_dict_coh_local[N_R]*0.5)

    # extC_bound_frac = total_bound_ext / (total_bound_ext + df_dS['R_free'][analysis_index])
    # extC_value = int(NUM_EXTRUDERS * bound_extC_ratio)
    # velocity_9h = 1/5 * total_bound_ext / df_dS['RN'][analysis_index]
    # LEF_sep_9h = int(NUM_LATTICE_SITES * extC_bound_frac / (extC_value / 2))
    total_lef_per_bp = int(NUM_EXTRUDERS) / (NUM_LATTICE_SITES * BP_PER_SITE)

    total_sister_rad21 = (df_dS['RacPS'][analysis_index] + df_dS['RacPW'][analysis_index] +
                          df_dS['RacP'][analysis_index] + df_dS['Rac'][analysis_index])

    sister_rad21_lifetime_at_analysis = calculate_sister_RAD21_bound_time(
        K_RacPW_Rac_free = paras_dict_ext_coh_local['K_RacPW_Rac_free'], \
        B_W_sister = df_dS['RacPW'][analysis_index], \
        B_R_sister = total_sister_rad21)

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

    # establiment, 2909 is the number of sistercs at S8h for 800mb
    gamma_constant = 2909/residence_time/(1-np.exp(-8/residence_time))
    sister_rad21_lifetime_hours = float(sister_rad21_lifetime_at_analysis / 3600)

    '''
    N_sister_WT_2h = gamma_constant * residence_time * (1 - np.exp(-2/residence_time))
    N_sister_WT_4h = gamma_constant * residence_time * (1 - np.exp(-4/residence_time))
    N_sister_WT_6h = gamma_constant * residence_time * (1 - np.exp(-6/residence_time))
    N_sister_WT_8h = gamma_constant * residence_time * (1 - np.exp(-8/residence_time))
    N_sister_WT_10h = 2909 * np.exp(-2/residence_time)
    N_sister_WT_12h = 2909 * np.exp(-4/residence_time)
    '''
    num_sister_cohesins_s8h = gamma_constant * sister_rad21_lifetime_hours * (1 - np.exp(-8 / sister_rad21_lifetime_hours))
    # N_sister_S24h = num_sister_cohesins_s8h * np.exp(-16 / sister_rad21_lifetime_hours)
    # Load base parameters and update
    with open(SCRIPT_DIR / config["output_template_file"], "r") as f:
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
    output_params["region_size_bp"] = NUM_LATTICE_SITES * BP_PER_SITE
    output_params["steps"] = 16 * 3600
    output_params["num_of_sisters"] = int(num_sister_cohesins_s8h)
    output_params["sister_damping_factor"] = sister_damping
    output_params["sister_lifetime"] = int(sister_rad21_lifetime_at_analysis)

    output_params["collision_release_prob"] = collision_prob
    output_params["bypass_prob"] = bypass_prob

    return output_params


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('network_parameters_dS12h.json')

    # Create output directory
    output_dir = SCRIPT_DIR / config['output_directory']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run parameter sweep
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
