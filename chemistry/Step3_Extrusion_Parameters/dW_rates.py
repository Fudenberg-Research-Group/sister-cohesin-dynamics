"""Generate lattice extrusion parameters for WAPL depletion (dWAPL).

Two-stage solve: the WT network runs for 6 h from S8h, then WAPL is depleted
and the network continues for 10 h with WAPL-dependent species scaled by
remaining_level. Extruder rates and the sister RAD21 lifetime are read out at
the analysis timepoint of the second stage.
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
    calculate_sister_RAD21_bound_time,
    paras_coh,
    paras_ext,
    sol_rates_coh,
    sol_rates_ext,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# Load configuration files
def load_config(filename):
    """Load configuration from JSON file"""
    with open(SCRIPT_DIR / filename, 'r') as f:
        return json.load(f)

# RAD21 residence time, in hours (chemistry input).
RESIDENCE_TIMES_HOURS = [11]

# Downstream lattice/polymer parameters
SISTER_DAMPING_FACTORS = [100]
COLLISION_RELEASE_PROBS = [1e-7]
BYPASS_PROBS = [1e-5]

# Cohesin copy numbers
NUM_EXTRUDERS = 7764
GENOMIC_REGION_SIZE_BP = 800_000_000
BP_PER_SITE = 500
NUM_LATTICE_SITES = GENOMIC_REGION_SIZE_BP // BP_PER_SITE


def run_simulation(config, residence_time, sister_damping, collision_prob, bypass_prob):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
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

    # Rebuild and simulate model
    Model_ext_coh = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_WT = r_ext_coh.simulate(0, 3600*18, 3600*18)

    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW',
               'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']
    df_WT = pd.DataFrame(Model_ext_coh_WT, columns=columns)

    ## Apply Wapl depletion at 6h since S8h (14h since start of S phase)
    time_6h = 3600 * 6 - 1
    depletion_level = config['simulation_parameters']['depletion_level']
    remaining_level = 1 - depletion_level

    init_conditions_ext_coh_dWapl_6h = {
        Rac_free_init: df_WT['Rac_free'][time_6h],
        Rac_init: df_WT['Rac'][time_6h],
        RacP_init: df_WT['RacP'][time_6h] + df_WT['RacPW'][time_6h] * depletion_level,
        RacPW_init: df_WT['RacPW'][time_6h] * remaining_level,
        RacPS_init: df_WT['RacPS'][time_6h],
        R_free_init: df_WT['R_free'][time_6h],
        RN_init: df_WT['RN'][time_6h] ,
        R_init: df_WT['R'][time_6h]  ,
        RP_init: df_WT['RP'][time_6h] + df_WT['RPW'][time_6h] * depletion_level,
        RPW_init: df_WT['RPW'][time_6h] * remaining_level,

        N_init: df_WT['N'][time_6h],
        S_init: df_WT['S'][time_6h],
        W_init: df_WT['W'][time_6h] * remaining_level,
        P_init: df_WT['P'][time_6h],
       }

    paras_dict_since_6h_dW = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh_dWapl_6h
    paras_dict_ext_coh_since_6h_dW = {str(key): value for key, value in paras_dict_since_6h_dW.items()}
    Model_ext_coh_since_6h_dW = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_since_6h_dW)

    r_ext_coh_since_6h_dW = te.loada(Model_ext_coh_since_6h_dW)
    # Simulate the model for another 10h
    Model_ext_coh_since_6h_dW = r_ext_coh_since_6h_dW.simulate(0, 3600*10, 3600*10)

    df_since_6h = pd.DataFrame(Model_ext_coh_since_6h_dW, columns=columns)

    analysis_time_hours = config['simulation_parameters']['analysis_timepoint_hours']
    analysis_index = 3600 * analysis_time_hours - 1

    total_lef_per_bp = NUM_EXTRUDERS / GENOMIC_REGION_SIZE_BP

    total_sister_rad21 = (df_since_6h['RacPS'][analysis_index] + df_since_6h['RacPW'][analysis_index] +
                          df_since_6h['RacP'][analysis_index] + df_since_6h['Rac'][analysis_index])

    sister_rad21_lifetime_at_analysis = calculate_sister_RAD21_bound_time(
        K_RacPW_Rac_free=paras_dict_ext_coh_since_6h_dW['K_RacPW_Rac_free'],
        B_W_sister=df_since_6h['RacPW'][analysis_index],
        B_R_sister=total_sister_rad21)

    # Calculate transition rates
    rates = {
        'R_free_to_RN': paras_dict_ext_coh_since_6h_dW['Kext_R_free_RN']*df_since_6h['N'][analysis_index],
        'RPW_R_free': paras_dict_ext_coh_since_6h_dW['Kext_RPW_R_free'],
        'RN_R': paras_dict_ext_coh_since_6h_dW['Kext_RN_R'],
        'R_RN': paras_dict_ext_coh_since_6h_dW['Kext_R_RN']*df_since_6h['N'][analysis_index],
        'R_RP': paras_dict_ext_coh_since_6h_dW['Kext_R_RP']*df_since_6h['P'][analysis_index],
        'RP_R': paras_dict_ext_coh_since_6h_dW['Kext_RP_R'],
        'RP_RPW': paras_dict_ext_coh_since_6h_dW['Kext_RP_RPW']*df_since_6h['W'][analysis_index],
        'RPW_RP': paras_dict_ext_coh_since_6h_dW['Kext_RPW_RP'],
    }

    # Load base parameters and update
    with open(SCRIPT_DIR / config["output_template_file"], "r") as f:
        output_params = json.load(f)

    output_params["LEF_on_rate"]["A"] = float(rates['R_free_to_RN'])
    output_params["LEF_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_diffusion_rate"]["A"] = 0.25

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
    output_params["steps"] = 10 * 3600
    output_params["sister_damping_factor"] = sister_damping
    output_params["sister_lifetime"] = int(sister_rad21_lifetime_at_analysis)
    output_params["collision_release_prob"] = collision_prob
    output_params["bypass_prob"] = bypass_prob

    return output_params


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('network_parameters_dW.json')

    # Create output directory
    output_dir = SCRIPT_DIR / config['output_directory']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run parameter sweep
    # for residence_time in RESIDENCE_TIMES_HOURS:
    #     for sister_damping in SISTER_DAMPING_FACTORS:
    for residence_time in RESIDENCE_TIMES_HOURS:
        for sister_damping in SISTER_DAMPING_FACTORS:
            for k, collision_prob in enumerate(COLLISION_RELEASE_PROBS):
                for q, bypass_prob in enumerate(BYPASS_PROBS):
                    print(f"\nRunning: residence_time={residence_time}h, sister_damping={sister_damping}, collision_prob={collision_prob}")
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
