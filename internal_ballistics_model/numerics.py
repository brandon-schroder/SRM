import numpy as np
from numba import njit, prange

from schemes.spatial_reconstruction import weno3_left as left_biased, weno3_right as right_biased
from schemes.flux_splitting import hllc_flux as flux_splitting
from .boundary import apply_boundary_jit


"""
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


@njit(cache=True, parallel=True)
def primitives_to_conserved(rho, u, p, A, gamma, U):
    for i in prange(len(rho)):
        U[0, i] = rho[i] * A[i]
        U[1, i] = rho[i] * u[i] * A[i]
        U[2, i] = (p[i] / (gamma - 1) + 0.5 * rho[i] * u[i] ** 2) * A[i]
    return U


@njit(fastmath=True, cache=True, parallel=True)
def conserved_to_primitives(U, A, gamma, rho, u, p, c):
    for i in prange(U.shape[1]):
        rho[i] = U[0, i] / A[i]
        u[i]   = U[1, i] / U[0, i]
        p[i]   = (U[2, i] / A[i] - 0.5 * rho[i] * u[i] ** 2) * (gamma - 1)
        c[i]   = np.sqrt(gamma * p[i] / rho[i])
    return rho, u, p, c


@njit(fastmath=True, cache=True, parallel=True)
def flux(U, A, rho, u, p, F):
    for i in prange(U.shape[1]):
        F[0, i] = rho[i] * u[i] * A[i]
        F[1, i] = (rho[i] * u[i] ** 2 + p[i]) * A[i]
        F[2, i] = (U[2, i] / A[i] + p[i]) * u[i] * A[i]
    return F


@njit(fastmath=True, cache=True, parallel=True)
def source(rho_p, Tf, br, R, gamma, p, P_propellant, A_interfaces, dz, S):
    hf = gamma / (gamma - 1) * R * Tf
    for i in prange(p.shape[0]):
        uf = (rho_p * br[i] * R * Tf) / p[i]
        S[0, i] = rho_p * br[i] * P_propellant[i]
        S[1, i] = p[i] * ((A_interfaces[i+1] - A_interfaces[i]) / dz)
        S[2, i] = rho_p * br[i] * P_propellant[i] * (hf + 0.5 * uf ** 2)
    return S


@njit(fastmath=True, cache=True, parallel=True)
def compute_numerical_flux(U, A, rho, u, p, c, gamma, ng, F_hat):
    num_comp, n_tot = U.shape
    nc = n_tot - 2 * ng

    for i in prange(nc + 1):
        ii = i + ng - 1

        A_int = A[i]

        rho_L = left_biased(rho[ii - 2], rho[ii - 1], rho[ii], rho[ii + 1], rho[ii + 2], rho[ii + 3])
        u_L   = left_biased(u[ii - 2],   u[ii - 1],   u[ii],   u[ii + 1],   u[ii + 2],   u[ii + 3])
        p_L   = left_biased(p[ii - 2],   p[ii - 1],   p[ii],   p[ii + 1],   p[ii + 2],   p[ii + 3])

        rho_R = right_biased(rho[ii - 2], rho[ii - 1], rho[ii], rho[ii + 1], rho[ii + 2], rho[ii + 3])
        u_R   = right_biased(u[ii - 2],   u[ii - 1],   u[ii],   u[ii + 1],   u[ii + 2],   u[ii + 3])
        p_R   = right_biased(p[ii - 2],   p[ii - 1],   p[ii],   p[ii + 1],   p[ii + 2],   p[ii + 3])

        rho_L, rho_R = max(rho_L, 1e-6), max(rho_R, 1e-6)
        p_L, p_R = max(p_L, 1e-6), max(p_R, 1e-6)

        f0, f1, f2 = flux_splitting(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)

        F_hat[0, i] = f0 * A_int
        F_hat[1, i] = f1 * A_int
        F_hat[2, i] = f2 * A_int

    return F_hat


@njit(fastmath=True, cache=True)
def adaptive_timestep(CFL, u, c, dz, ng, t, t_end):
    smax = np.max(np.abs(u[ng:-ng]) + c[ng:-ng])
    dt_stable = CFL * dz / (smax + 1e-12)

    if t + dt_stable > t_end:
        dt_stable = max(0.0, t_end - t)

    return dt_stable


@njit(fastmath=True, cache=True)
def rhs_numerics(U_interior, U_full, A, gamma, R, p0_inlet, t0_inlet, p_inf, ng,
                 inlet_bc, outlet_bc, rho_p, Tf, br, P_propellant, dz,
                 rho_out, u_out, p_out, c_out, A_interfaces, F_hat, S, rhs_out):
    nc = U_interior.shape[1]

    for i in range(3):
        for j in range(nc):
            U_full[i, j + ng] = U_interior[i, j]

    for i in range(nc + 1):
        A_interfaces[i] = 0.5 * (A[ng - 1 + i] + A[ng + i])

    U_full = apply_boundary_jit(U_full, A, gamma, R, p0_inlet, t0_inlet, p_inf, ng, inlet_bc, outlet_bc)

    rho_out, u_out, p_out, c_out = conserved_to_primitives(U_full, A, gamma, rho_out, u_out, p_out, c_out)

    F_hat = compute_numerical_flux(U_full, A_interfaces, rho_out, u_out, p_out, c_out, gamma, ng, F_hat)
    S = source(rho_p, Tf, br[ng:-ng], R, gamma, p_out[ng:-ng], P_propellant[ng:-ng], A_interfaces, dz, S)

    for i in range(3):
        for j in range(nc):
            rhs_out[i, j] = S[i, j] - (F_hat[i, j + 1] - F_hat[i, j]) / dz

    return rhs_out