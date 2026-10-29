"""Cohesin reaction networks: symbolic rate solutions and model template.

Two coupled networks are defined here:

  cohesive  -- sister-linking cohesin (Rac states), parameterised by the
               measured bound fractions and residence times of SORORIN,
               WAPL, PDS5 and RAD21
  extrusive -- loop-extruding cohesin (R states), with NIPBL in place of
               SORORIN

For each network mass-action equations (rhs_coh, rhs_ext) are
solved symbolically for the rate constants, once at import time. The six
*_rates.py scripts then substitute condition-specific numbers into those
solutions and integrate the network with simulate_network.
"""

import numpy as np
import pandas as pd
import sympy as sym
from scipy.integrate import solve_ivp

## Rates for cohesive network
rates_coh = sym.symbols("K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP,  K_RacPW_Rac_free")
K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP, K_RacPW_Rac_free = rates_coh

paras_coh = sym.symbols("tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_R, F_R_sister, N_R")
tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_R, F_R_sister, N_R = paras_coh

## Equations for cohesive network
rhs_coh = [K_RacPW_Rac_free * N_W * F_W - F_R_sister*N_R/tau_R,                              # RAD21 unbinding
           F_W/(tau_W * (1-F_W)) - K_RacP_RacPW * (F_P * N_P - F_S * N_S - F_W * N_W),       # Wapl binding
           K_RacPW_RacP + K_RacPW_Rac_free - 1/tau_W,                                        # Wapl unbinding
           F_S/(tau_S * (1-F_S)) - K_RacP_RacPS * (F_P * N_P - F_S * N_S - F_W * N_W),       # Sororin binding
           K_RacPS_RacP - 1/tau_S,                                                           # Sororin unbinding
           F_P/(tau_P * (1-F_P)) - K_Rac_RacP * (F_R_sister * N_R - F_P * N_P),                  # PDS5 binding
           K_RacPW_Rac_free * F_W *N_W + K_RacP_Rac*(F_P * N_P - F_S * N_S - F_W * N_W) - F_P * N_P/tau_P,   # PDS5 unbinding
]

## solutions cohesive network
sol_rates_coh = sym.solve(rhs_coh, rates_coh)

## Rates for extrusive network
rates_ext = sym.symbols("Kext_R_free_RN, Kext_RN_R, Kext_R_RN, Kext_R_RP, Kext_RP_R, Kext_RP_RPW, Kext_RPW_RP, Kext_RPW_R_free")
Kext_R_free_RN, Kext_RN_R, Kext_R_RN, Kext_R_RP, Kext_RP_R, Kext_RP_RPW, Kext_RPW_RP, Kext_RPW_R_free = rates_ext

paras_ext = sym.symbols("tau_N_ext, F_N_ext, N_N_ext, tau_W_ext, F_W_ext, N_W_ext, tau_P_ext, F_P_ext, N_P_ext, tau_R_ext, F_R_ext, N_R_ext")
tau_N_ext, F_N_ext, N_N_ext, tau_W_ext, F_W_ext, N_W_ext, tau_P_ext, F_P_ext, N_P_ext, tau_R_ext, F_R_ext, N_R_ext = paras_ext

## Equations for extrusive network
rhs_ext = [ Kext_RN_R - 1 / tau_N_ext,  # NIPBL unbinding kinetics
        Kext_R_free_RN * N_R_ext *(1-F_R_ext) + Kext_R_RN * (N_R_ext*F_R_ext - N_N_ext*F_N_ext - N_P_ext*F_P_ext) -  F_N_ext / tau_N_ext/ (1-F_N_ext), # NIPBL binding equilibrium
        Kext_RP_R * (N_P_ext * F_P_ext - N_W_ext * F_W_ext) + Kext_RPW_R_free * N_W_ext * F_W_ext  - N_P_ext * F_P_ext/tau_P_ext, # PDS5 unbinding kinetics
        Kext_R_RP * (N_R_ext * F_R_ext - N_N_ext * F_N_ext - N_P_ext * F_P_ext)  - F_P_ext / tau_P_ext /(1-F_P_ext), # PDS5 binding equilibrium,
        Kext_RPW_RP + Kext_RPW_R_free - 1 / tau_W_ext, # WAPL unbinding kinetics
        Kext_RP_RPW * (N_P_ext * F_P_ext - N_W_ext* F_W_ext)  - F_W_ext / tau_W_ext / (1-F_W_ext), # WAPL binding equilibrium
        Kext_RPW_R_free * N_W_ext * F_W_ext - F_R_ext * N_R_ext / tau_R_ext,  # RAD21 unbinding kinetics
        Kext_R_free_RN * N_N_ext * (1-F_N_ext) - F_R_ext / tau_R_ext / (1-F_R_ext) # RAD21 binding equilibrium,
]

sol_rates_ext = sym.solve(rhs_ext, rates_ext)

## Initial-condition species of the combined network. These names must match
## the species declared in MODEL_EXT_COH_TEMPLATE below.
INIT_SYMBOL_NAMES = (
    "Rac_free_init Rac_init RacP_init RacPW_init RacPS_init "
    "R_free_init RN_init R_init RP_init RPW_init "
    "N_init S_init W_init P_init"
)

# ---- ODE integration  ---------------------------------

# Column order of the DataFrame returned by simulate_network.
SPECIES = ("Rac", "P", "RacP", "S", "RacPS", "W", "RacPW",
           "Rac_free", "R_free", "N", "RN", "R", "RP", "RPW")


def _network_rhs(t, y, k):
    """Mass-action right-hand side of the reactions in MODEL_EXT_COH_TEMPLATE."""
    Rac, P, RacP, S, RacPS, W, RacPW, Rac_free, R_free, N, RN, R, RP, RPW = y

    # cohesive network
    v1 = k["K_Rac_RacP"] * Rac * P - k["K_RacP_Rac"] * RacP          # Rac + P <-> RacP
    v2 = k["K_RacP_RacPS"] * RacP * S - k["K_RacPS_RacP"] * RacPS    # RacP + S <-> RacPS
    v3 = k["K_RacP_RacPW"] * RacP * W - k["K_RacPW_RacP"] * RacPW    # RacP + W <-> RacPW
    v4 = k["K_RacPW_Rac_free"] * RacPW                               # RacPW -> Rac_free + P + W

    # extrusive network
    v5 = k["Kext_R_free_RN"] * R_free * N                            # R_free + N -> RN
    v6 = k["Kext_RN_R"] * RN - k["Kext_R_RN"] * R * N                # RN <-> R + N
    v7 = k["Kext_R_RP"] * R * P - k["Kext_RP_R"] * RP                # R + P <-> RP
    v8 = k["Kext_RP_RPW"] * RP * W - k["Kext_RPW_RP"] * RPW          # RP + W <-> RPW
    v9 = k["Kext_RPW_R_free"] * RPW                                  # RPW -> R_free + P + W

    return [
        -v1,                    # Rac
        -v1 + v4 - v7 + v9,     # P
        v1 - v2 - v3,           # RacP
        -v2,                    # S
        v2,                     # RacPS
        -v3 + v4 - v8 + v9,     # W
        v3 - v4,                # RacPW
        v4,                     # Rac_free
        -v5 + v9,               # R_free
        -v5 + v6,               # N
        v5 - v6,                # RN
        v6 - v7,                # R
        v7 - v8,                # RP
        v8 - v9,                # RPW
    ]


def simulate_network(params, t_end, n_points):
    """Integrate the combined network from the *_init values in params.

    Replaces te.loada(build_model(MODEL_EXT_COH_TEMPLATE, params)).simulate(
    0, t_end, n_points): the same time grid (n_points evenly spaced, both
    endpoints included, so the spacing is slightly above 1 s) and the same
    columns, returned as a DataFrame.
    """
    k = {str(key): float(value) for key, value in params.items()}
    y0 = [k[f"{s}_init"] for s in SPECIES]
    t = np.linspace(0, t_end, n_points)
    sol = solve_ivp(_network_rhs, (0, t_end), y0, method="LSODA", t_eval=t,
                    args=(k,), rtol=1e-8, atol=1e-8)
    return pd.DataFrame(np.column_stack([t, sol.y.T]), columns=("time",) + SPECIES)


def calculate_sister_RAD21_bound_time(K_RacPW_Rac_free, B_W_sister, B_R_sister):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl
    return B_R_sister/(K_RacPW_Rac_free * B_W_sister)

def calculate_extrusive_RAD21_bound_time(K_RPW_R_free, B_W_extruder, B_R_extruder):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl
    return B_R_extruder/(K_RPW_R_free * B_W_extruder)

def calculate_cohesive_parameters(config, residence_time):
    """
    Calculate cohesive network parameters from configuration

    Args:
        config: Dictionary containing base parameters and modifiers
        residence_time: Residence time in hours

    Returns:
        List of cohesive parameter values
    """
    base = config['base_parameters']
    mod = config['sister_networks']

    return [
        base['tau_S'],
        base['F_S'],
        base['N_S'],
        base['tau_W'],
        mod['F_W_sister'] / (1 - base['F_W'] + mod['F_W_sister']),
        base['N_W'] * (1 - base['F_W'] + mod['F_W_sister']),
        base['tau_P'],
        mod['F_P_sister'] / (1 - base['F_P'] + mod['F_P_sister']),
        base['N_P'] * (1 - base['F_P'] + mod['F_P_sister']),
        3600 * residence_time,  # tau_R_sister
        base['F_R_sister'],
        base['N_R'] * base['N_R_fraction'],
    ]

def calculate_extrusive_parameters(config):
    """
    Calculate extrusive network parameters from configuration

    Args:
        config: Dictionary containing base parameters and modifiers

    Returns:
        List of extrusive parameter values
    """
    base = config['base_parameters']
    mod = config['sister_networks']

    return [
        base['tau_N'],
        base['F_N'],
        base['N_N'],
        base['tau_W'],
        (base['F_W'] - mod['F_W_sister'])/(1 - mod['F_W_sister']),
        base['N_W']*(1 - mod['F_W_sister']),
        base['tau_P'],
        (base['F_P'] - mod['F_P_sister'])/(1 - mod['F_P_sister']),
        base['N_P']*(1 - mod['F_P_sister']),
        base["tau_R_extrusive"],
        base['F_R_sister'],
        base['N_R'] * base['N_R_fraction'],
    ]






