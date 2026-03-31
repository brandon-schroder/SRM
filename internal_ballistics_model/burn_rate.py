from numba import njit
from enum import IntEnum

from .config import *


def lenoir_robillard(base_burn_rate, config: SimulationConfig, state: FlowState):
    alpha = 0.92E-5
    beta = 53.0

    u = np.maximum(abs(state.u), 1e-7)

    Dh = 4 * state.A / state.P_wetted
    G = state.rho * u
    burn_rate = base_burn_rate + alpha * ((G ** 0.8) / (Dh ** 0.2)) * np.exp(-(beta * base_burn_rate * config.rho_p) / G)

    return burn_rate


def mukunda_paul(base_burn_rate, config: SimulationConfig, state: FlowState):
    rho_p = config.rho_p
    rho = state.rho
    u = np.maximum(abs(state.u), 1e-7)

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

class BurnModel(IntEnum):
    NONE = 0
    OVERRIDE = 1
    LR = 2
    MP = 3

def burn_rate_model(config: SimulationConfig, state: FlowState, model_flag):
    coefficient = config.a_coef
    exponent = config.n_exp

    base_burn_rate = coefficient * state.p ** exponent

    if model_flag==BurnModel.OVERRIDE.value:
        burn_rate = config.br_initial * np.ones_like(state.p)

    elif model_flag==BurnModel.LR.value:
        burn_rate = lenoir_robillard(base_burn_rate, config, state)

    elif model_flag==BurnModel.MP.value:
        burn_rate = mukunda_paul(base_burn_rate, config, state)

    else:
        burn_rate=base_burn_rate

    return burn_rate