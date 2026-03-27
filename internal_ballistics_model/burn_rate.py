from .config import *

def lenoir_robillard(base_burn_rate, config: SimulationConfig, state: FlowState):

    alpha = 0.92E-5
    beta = 53.0

    u = np.maximum(abs(state.u), 1e-10)

    Dh = 4 * state.A / state.P_wetted
    G = state.rho * u
    burn_rate = base_burn_rate + alpha * ((G ** 0.8) / (Dh ** 0.2)) * np.exp(-(beta * base_burn_rate * config.rho_p) / G)

    return burn_rate

def mukunda_paul(base_burn_rate, config: SimulationConfig, state: FlowState):

    rho_p = config.rho_p
    rho = state.rho
    u = np.maximum(abs(state.u), 1e-10)

    mf_port = rho * u * state.A

    mu = 8.85E-5
    K1 = 0.023
    K2 = 2.3714 # DV (2.820054) # Correct (2.3714)
    m = -0.125
    gth = 35.0

    Dh = 4 * state.A / state.P_wetted
    Re0 = rho_p * base_burn_rate * Dh / mu

    G = abs(mf_port) / state.A
    g0 = G / (rho_p * base_burn_rate)
    g = g0 * K2 * Re0 ** m
    H = np.where(g > gth, 1.0, 0.0)  # Heaviside function

    eta = 1 + K1 * (g ** 0.8 - gth ** 0.8) * H

    burn_rate = base_burn_rate * eta

    return burn_rate



def burn_rate_model(config: SimulationConfig, state: FlowState, model="MP"):

    coefficient = config.a_coef
    exponent = config.n_exp

    base_burn_rate = coefficient * state.p ** exponent

    if model=="override":
        burn_rate = config.br_initial * np.ones_like(state.p)

    elif model=="LR":
        burn_rate = lenoir_robillard(base_burn_rate, config, state)

    elif model=="MP" or model=="universal":
        burn_rate = mukunda_paul(base_burn_rate, config, state)

    else:
        burn_rate=base_burn_rate


    return burn_rate