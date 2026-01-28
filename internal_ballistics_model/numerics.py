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


@njit(fastmath=True, cache=True)
def flux(U, A, rho, u, p):
    """
    JIT-compiled, array-wise flux calculation.
    Uses pre-computed primitive variables.
    """
    F = np.zeros_like(U)

    F[0, :] = rho * u * A
    F[1, :] = (rho * u ** 2 + p) * A
    F[2, :] = (U[2, :] / A + p) * u * A

    return F


@njit(cache=True)
def source_jit(rho_p, Tf, br, R, gamma, p, P_propellant, A_interfaces, dz):
    """
    Well-balanced source term.
    Uses interface area differences to balance the pressure flux.
    """

    S = np.zeros((3, p.shape[0]))

    uf = (rho_p * br * R * Tf) / p
    hf = gamma / (gamma - 1) * R * Tf

    S[0, :] = rho_p * br * P_propellant
    S[1, :] = p * ((A_interfaces[1:] - A_interfaces[:-1]) / dz)
    S[2, :] = rho_p * br * P_propellant * (hf + 1/2 * uf ** 2)

    return S


@njit(fastmath=True, cache=True)
def compute_numerical_flux_jit(U, A, rho, u, p, c, ng):
    """Jiang-Shu numerical flux computation"""

    num_comp, n_tot = U.shape
    nc = n_tot - 2 * ng

    alpha = np.abs(u) + c
    alpha = np.nan_to_num(alpha, nan=1000.0)

    # Compute physical flux using pre-computed primitives
    F = flux(U, A, rho, u, p)

    # Lax-Friedrichs splitting
    fp = 0.5 * (F + alpha * U)
    fm = 0.5 * (F - alpha * U)

    F_hat = np.zeros((num_comp, nc+1))
    for i in range(nc + 1):
        ii = i + ng - 1

        # For each component
        for m in range(num_comp):
            # Left-biased reconstruction (f+)
            F_hat_p = weno_left(fp[m, ii - 2], fp[m, ii - 1], fp[m, ii], fp[m, ii + 1], fp[m, ii + 2], fp[m, ii + 3])

            # Right-biased reconstruction (f-)
            F_hat_m = weno_right(fm[m, ii - 2], fm[m, ii - 1], fm[m, ii], fm[m, ii + 1], fm[m, ii + 2], fm[m, ii + 3])

            F_hat[m, i] = F_hat_p + F_hat_m

    return F_hat



@njit(cache=True)
def adaptive_timestep(CFL, U, A, gamma, dz, ng, t, t_end):

    precision = U.dtype.type
    eps = precision(1e-12)

    rho, u, p, c = compute_primitives_jit(U, A, gamma)

    # rho = U[0, ng:-ng] / A[ng:-ng]
    # u = U[1, ng:-ng] / (U[0, ng:-ng] + eps)
    # p = (U[2, ng:-ng] / A[ng:-ng] - 0.5 * rho * u ** 2) * (gamma - 1)
    # p = np.maximum(p, 1e-5)
    # c = np.sqrt(gamma * p / rho)

    smax = np.max(np.abs(u) + c)
    dt_stable = CFL * dz / (smax + eps)

    if t + dt_stable > t_end:
        dt_stable = max(0.0, t_end - t)
    return dt_stable