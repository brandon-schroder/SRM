from ._structure import *

def erosive_models(base_burn_rate, config: SimulationConfig, state: FlowState, model="MP"):

    port_area = state.A[-config.ng]
    port_perimeter = state.P.mean()

    rho_p = config.rho_p
    rho = state.rho.mean()
    mf_port = state.rho[-config.ng] * state.u[-config.ng] * port_area


    if model=="MP":

        mu = config.erosive_mu
        K1 = config.erosive_K1
        K2 = config.erosive_K2
        m = config.erosive_m
        gth = config.erosive_gth

        Dh = 4 * port_area / port_perimeter

        Re0 = rho * base_burn_rate * Dh / mu

        G = abs(mf_port) / port_area
        g0 = G / (rho_p * base_burn_rate)
        g = g0 * K2 * Re0 ** m

        H = 1.0 if g > gth else 0.0  # Heaviside function

        eta = 1 + K1 * (g ** 0.8 - gth ** 0.8) * H

    else:
        eta = 1

    return eta


def burn_rate(config: SimulationConfig, state: FlowState, model="MP"):

    coefficient = config.a_coef
    exponent = config.n_exp

    pressure = state.p.max()

    base_burn_rate = coefficient * pressure ** exponent

    eta = erosive_models(base_burn_rate, config, state, model)

    burn_rate = eta * base_burn_rate

    return burn_rate, eta