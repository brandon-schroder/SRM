import numpy as np
from numba import njit

from core.wenos import weno3_left as weno_left, weno3_right as weno_right


"""
Optimized WENO3 Quasi-1D Nozzle Solver with Numba JIT compilation
(fastmath=True, cache=True, and primitive pre-calculation)

Conserved Vector: U
U(1) = rho * A
U(2) = rho * u * A
U(3) = rho * e_T * A

Flux Vector: F
F(1) = rho * u * A
F(2) = (rho * u ** 2 + p) * A
F(3) = (rho * e_T + p) * u * A

Source Vector: S
S(1) = rho_p * br * P
S(2) = p * dAdx
S(3) = rho_p * br * P * (hf + 1/2 * uf^2)
"""

"""
hf -> static enthalpy of combustion products
uf -> injection velocity of combustion products
P -> perimeter of burn surface

uf = (rho_p * br * R * Tf) / p
hf = gamma / (gamma - 1) * R * Tf
"""

import numpy as np
from numba import njit


@njit(cache=True)
def primitive_to_conserved(rho, u, p, A, gamma):
    U = np.zeros((3, len(rho)))
    U[0, :] = rho * A
    U[1, :] = rho * u * A
    U[2, :] = (p / (gamma - 1) + 0.5 * rho * u ** 2) * A
    return U


@njit(cache=True)
def compute_primitives_jit(U, A, gamma):
    rho = U[0, :] / A
    u = U[1, :] / U[0, :]
    p = (U[2, :] / A - 0.5 * rho * u ** 2) * (gamma - 1)
    c = np.sqrt(gamma * p / rho)
    return rho, u, p, c


@njit(cache=True)
def compute_numerical_flux_jit(U, A, rho, u, p, alpha, ng):
    """Local Lax-Friedrichs Flux"""
    n_interfaces = U.shape[1] - 2 * ng + 1
    F_hat = np.zeros((3, n_interfaces))

    # Pre-calculate physical fluxes
    F = np.zeros_like(U)
    F[0, :] = rho * u * A
    F[1, :] = (rho * u ** 2 + p) * A
    F[2, :] = (U[2, :] / A + p) * u * A

    for i in range(n_interfaces):
        idx_l = i + ng - 1
        idx_r = i + ng
        a_max = max(alpha[idx_l], alpha[idx_r])
        F_hat[:, i] = 0.5 * (F[:, idx_l] + F[:, idx_r]) - 0.5 * a_max * (U[:, idx_r] - U[:, idx_l])

    return F_hat


@njit(cache=True)
def source_jit(rho_p, Tf, br, R, gamma, p, P_propellant, A_interfaces, dz):
    """
    Well-balanced source term.
    Uses interface area differences to balance the pressure flux.
    """
    n = p.shape[0]
    S = np.zeros((3, n))

    # Mass Source
    mdot_gen = rho_p * br * P_propellant
    S[0, :] = mdot_gen

    # Momentum Source: p * (A_right - A_left) / dz
    # This ensures that if p is constant, the net force is zero.
    dA = A_interfaces[1:] - A_interfaces[:-1]
    S[1, :] = p * (dA / dz)

    # Energy Source
    cp = (gamma * R) / (gamma - 1)
    S[2, :] = mdot_gen * cp * Tf

    return S


@njit(cache=True)
def adaptive_timestep(CFL, U, A, gamma, dz, ng, t, t_end):
    # Safety: ensure we don't divide by zero if U is small
    rho = U[0, ng:-ng] / A[ng:-ng]
    u = U[1, ng:-ng] / (U[0, ng:-ng] + 1e-12)
    p = (U[2, ng:-ng] / A[ng:-ng] - 0.5 * rho * u ** 2) * (gamma - 1)
    p = np.maximum(p, 1e-5)  # Pressure floor for stability
    c = np.sqrt(gamma * p / rho)

    smax = np.max(np.abs(u) + c)
    dt_stable = CFL * dz / (smax + 1e-12)

    if t + dt_stable > t_end:
        dt_stable = max(0.0, t_end - t)
    return dt_stable