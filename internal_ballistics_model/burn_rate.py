from numba import njit
from enum import IntEnum

from .config import *

@njit(cache=True)
def st_roberts(pressure, coefficient, exponent):
    base_burn_rate = coefficient * pressure ** exponent
    return base_burn_rate

@njit(cache=True)
def override(base_burn_rate, br_initial):
    burn_rate = br_initial * np.ones_like(base_burn_rate)
    return burn_rate

@njit(cache=True)
def lenoir_robillard(base_burn_rate, alpha, beta, u, rho, A, P_wetted, rho_p):
    u = np.maximum(np.abs(u), 1e-7)

    Dh = 4 * A / P_wetted
    G = np.abs(rho * u)

    burn_rate = base_burn_rate + alpha * ((G ** 0.8) / (Dh ** 0.2)) * np.exp(-(beta * base_burn_rate * rho_p) / G)

    return burn_rate

@njit(cache=True)
def mukunda_paul(base_burn_rate, mu, K1, K2, m, gth, u, rho, A, P_wetted, rho_p):
    u = np.maximum(np.abs(u), 1e-7)

    Dh = 4 * A / P_wetted
    G = np.abs(rho * u)

    Re0 = rho_p * base_burn_rate * Dh / mu
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

    base_burn_rate = st_roberts(state.p, config.a_coef, config.n_exp)

    if model_flag==BurnModel.OVERRIDE.value:
        burn_rate = override(base_burn_rate, config.br_initial)

    elif model_flag==BurnModel.LR.value:
        alpha = getattr(config, "lr_alpha", 1E-5)
        beta = getattr(config, "lr_beta", 53)
        burn_rate = lenoir_robillard(base_burn_rate,
                                     alpha, beta,
                                     state.u, state.rho, state.A, state.P_wetted,
                                     config.rho_p)

    elif model_flag==BurnModel.MP.value:
        mu =  getattr(config, "mp_mu",     8.85E-5)
        K1 =  getattr(config, "mp_k1",  0.023)
        K2 =  getattr(config, "mp_k2",  2.3714)
        m  =  getattr(config, "mp_m",   -0.125)
        gth = getattr(config, "mp_gth", 35.0)

        burn_rate = mukunda_paul(base_burn_rate,
                                 mu, K1, K2, m, gth,
                                 state.u, state.rho, state.A, state.P_wetted,
                                 config.rho_p)

    else:
        burn_rate=base_burn_rate

    return burn_rate