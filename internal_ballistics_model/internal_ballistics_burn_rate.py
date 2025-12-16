

def erosive_models(params, state, model="MP"):
    if model=="MP":
        mu = 8.85E-5

        K1 = 0.023
        K2 = 2.3714
        m = -0.125
        gth = 35

        Dh = 4 * state.A.mean() / state.P.mean()

        Re0 = state.rho.mean() * state.br * Dh / mu

        A_port = state.A[-1]
        mf_port = state.rho[-1] * state.u[-1] * A_port

        G = mf_port / A_port
        g0 = G / (params.rho_p * state.br)
        g = g0 * K2 * Re0 ** m

        H = 1.0 if g > gth else 0.0  # Heaviside function

        eta = 1 + K1 * (g ** 0.8 - gth ** 0.8) * H

    else:
        eta = 1



    return eta


def burn_rate(params, states):
    coefficient = 0.000035
    exponent = 0.36

    pressure = states.p.max()

    br = coefficient * pressure ** exponent

    eta = erosive_models(params, states)

    states.br = eta * br

    return states