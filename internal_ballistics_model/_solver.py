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

# ==============================================================

@njit(fastmath=True, cache=True)
def compute_primitives_jit(U, A, gamma):
    """
    JIT-compiled, array-wise primitive variable extraction.
    This computes primitives for the *entire* domain (including ghosts).
    """
    eps = 1e-12  # for stability
    rho = U[0] / (A + eps)
    u = U[1] / (U[0] + eps)
    rho_eT = U[2] / A  # Total energy per volume
    p = (gamma - 1.0) * (rho_eT - 0.5 * rho * u ** 2)

    # Floor for stability
    rho = np.maximum(rho, 1e-10)
    p = np.maximum(p, 1e-10)

    c = np.sqrt(gamma * p / rho)

    return rho, u, p, c

@njit(fastmath=True, cache=True)
def primitive_to_conserved(rho, u, p, A, gamma):
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack((A * rho,
                     A * rho * u,
                     A * E))

# ==============================================================

@njit(fastmath=True, cache=True)
def flux_jit(U, A, rho, u, p):
    """
    JIT-compiled, array-wise flux calculation.
    Uses pre-computed primitive variables.
    """
    F = np.zeros_like(U)
    # F[0] = rho * u * A, which is already U[1]
    F[0] = U[1]
    # F[1] = (rho * u**2 + p) * A
    F[1] = rho * u ** 2 * A + p * A
    # F[2] = (rho * e_T + p) * u * A
    # U[2] = rho * e_T * A
    F[2] = u * (U[2] + p * A)
    return F


def source_jit(rho_p, Tf, br, R, gamma, p, P_propellant, dAdx):
    """
    JIT-compiled, array-wise source term.
    Uses pre-computed interior pressure.
    """
    # rho_p, R, Tf, br, R, gamma, p

    S = np.zeros((3, p.shape[0]))

    uf = (rho_p * br * R * Tf) / p
    hf = gamma / (gamma - 1) * R * Tf

    S[0, :] = rho_p * br * P_propellant
    S[1, :] = p * dAdx
    S[2, :] = rho_p * br * P_propellant * (hf + 1/2 * uf ** 2)
    return S

# ==============================================================

@njit(fastmath=True, cache=True)
def compute_numerical_flux_jit(U, A, rho, u, p, alpha, ng):
    """Fully JIT-compiled numerical flux computation"""
    num_comp, n_tot = U.shape
    nc = n_tot - 2 * ng

    # Compute physical flux using pre-computed primitives
    F = flux_jit(U, A, rho, u, p)

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


def adaptive_timestep(CFL, U, A, gamma, dx, ng):

    rho, u, p, c = compute_primitives_jit(U, A, gamma)

    c = np.sqrt(np.maximum(gamma * p / rho, 1e-10))

    smax = np.max(np.abs(u[ng:-ng]) + c[ng:-ng])

    dt = CFL * dx / (smax + 1e-16)

    return dt

