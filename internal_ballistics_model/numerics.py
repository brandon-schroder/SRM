import numpy as np
from numba import njit, prange

from core.wenos import weno3_left as weno_left, weno3_right as weno_right
from .boundary import apply_boundary_jit


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
def primitives_to_conserved(rho, u, p, A, gamma):

    U = np.zeros((3, len(rho)))

    U[0, :] = rho * A
    U[1, :] = rho * u * A
    U[2, :] = (p / (gamma - 1) + 0.5 * rho * u ** 2) * A

    return U


@njit(cache=True)
def conserved_to_primitives(U, A, gamma):

    rho = U[0, :] / A
    u = U[1, :] / U[0, :]
    p = (U[2, :] / A - 0.5 * rho * u ** 2) * (gamma - 1)
    c = np.sqrt(gamma * p / rho)

    return rho, u, p, c


@njit(fastmath=True, cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
def source(rho_p, Tf, br, R, gamma, p, P_propellant, A_interfaces, dz):
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


@njit(fastmath=True, cache=True, parallel=True)
def compute_numerical_flux(U, A, rho, u, p, c, gamma, ng):
    """
    WENO-HLLC numerical flux with Correct Area Scaling.
    """

    num_comp, n_tot = U.shape
    nc = n_tot - 2 * ng

    F_hat = np.zeros((num_comp, nc + 1))

    for i in prange(nc + 1):
        ii = i + ng - 1

        # 1. Calculate the EXACT same interface area used in the source term
        A_int = A[i]

        # 2. Reconstruct Primitives
        rho_L = weno_left(rho[ii - 2], rho[ii - 1], rho[ii], rho[ii + 1], rho[ii + 2], rho[ii + 3])
        u_L   = weno_left(u[ii - 2],   u[ii - 1],   u[ii],   u[ii + 1],   u[ii + 2],   u[ii + 3])
        p_L   = weno_left(p[ii - 2],   p[ii - 1],   p[ii],   p[ii + 1],   p[ii + 2],   p[ii + 3])

        rho_R = weno_right(rho[ii - 2], rho[ii - 1], rho[ii], rho[ii + 1], rho[ii + 2], rho[ii + 3])
        u_R   = weno_right(u[ii - 2],   u[ii - 1],   u[ii],   u[ii + 1],   u[ii + 2],   u[ii + 3])
        p_R   = weno_right(p[ii - 2],   p[ii - 1],   p[ii],   p[ii + 1],   p[ii + 2],   p[ii + 3])

        # Safety constraints
        rho_L, rho_R = max(rho_L, 1e-6), max(rho_R, 1e-6)
        p_L, p_R = max(p_L, 1e-6), max(p_R, 1e-6)

        # 3. Compute HLLC Flux (per unit area)
        F_hllc = hllc_flux(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)

        # 4. Apply Interface Area HERE
        F_hat[0, i] = F_hllc[0] * A_int
        F_hat[1, i] = F_hllc[1] * A_int
        F_hat[2, i] = F_hllc[2] * A_int

    return F_hat


@njit(fastmath=True, cache=True)
def hllc_flux(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    """
    Computes HLLC flux for the Quasi-1D Euler equations.
    Flux is scaled by the interface Area (A_int).
    """
    # 1. Compute Enthalpy and Sound Speed
    e_L = p_L / ((gamma - 1) * rho_L)
    E_L = e_L + 0.5 * u_L ** 2
    H_L = E_L + p_L / rho_L
    c_L = np.sqrt(gamma * p_L / rho_L)

    e_R = p_R / ((gamma - 1) * rho_R)
    E_R = e_R + 0.5 * u_R ** 2
    H_R = E_R + p_R / rho_R
    c_R = np.sqrt(gamma * p_R / rho_R)

    # 2. Estimate Wave Speeds (Roe-Einfeldt like estimates for robustness)
    #    Simple estimates often suffice: S_L = min(u_L - c_L, u_R - c_R)
    S_L = min(u_L - c_L, u_R - c_R)
    S_R = max(u_L + c_L, u_R + c_R)

    # 3. Compute Contact Wave Speed (S_star)
    #    (See Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics)
    num = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / (den + 1e-16)  # Avoid div by zero

    # 4. Select Flux Region
    F = np.zeros(3)

    if 0 <= S_L:
        # Left State Flux
        F[0] = rho_L * u_L
        F[1] = rho_L * u_L ** 2 + p_L
        F[2] = rho_L * u_L * H_L

    elif S_L <= 0 <= S_star:
        # Star Left Flux
        # F*_L = F_L + S_L * (U*_L - U_L)
        # U*_L = rho_L * ((S_L - u_L) / (S_L - S_star)) * [1, S_star, E_L + (S_star - u_L)*(S_star + p_L/(rho_L*(S_L - u_L)))]

        factor = rho_L * (S_L - u_L) / (S_L - S_star)
        U_star_0 = factor
        U_star_1 = factor * S_star
        U_star_2 = factor * (E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))

        F[0] = rho_L * u_L + S_L * (U_star_0 - rho_L)
        F[1] = (rho_L * u_L ** 2 + p_L) + S_L * (U_star_1 - rho_L * u_L)
        F[2] = (rho_L * u_L * H_L) + S_L * (U_star_2 - rho_L * E_L)

    elif S_star <= 0 <= S_R:
        # Star Right Flux
        factor = rho_R * (S_R - u_R) / (S_R - S_star)
        U_star_0 = factor
        U_star_1 = factor * S_star
        U_star_2 = factor * (E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))

        F[0] = rho_R * u_R + S_R * (U_star_0 - rho_R)
        F[1] = (rho_R * u_R ** 2 + p_R) + S_R * (U_star_1 - rho_R * u_R)
        F[2] = (rho_R * u_R * H_R) + S_R * (U_star_2 - rho_R * E_R)

    else:  # S_R <= 0
        # Right State Flux
        F[0] = rho_R * u_R
        F[1] = rho_R * u_R ** 2 + p_R
        F[2] = rho_R * u_R * H_R

    return F


@njit(cache=True)
def adaptive_timestep(CFL, U, A, gamma, dz, ng, t, t_end):

    precision = U.dtype.type
    eps = precision(1e-12)

    rho, u, p, c = conserved_to_primitives(U, A, gamma)

    smax = np.max(np.abs(u) + c)
    dt_stable = CFL * dz / (smax + eps)

    if t + dt_stable > t_end:
        dt_stable = max(0.0, t_end - t)

    return dt_stable


@njit(cache=True)
def rhs_numerics(U_interior, U_full, A, gamma, R, p0_inlet, t0_inlet, p_inf, ng,
                         inlet_bc, outlet_bc, rho_p, Tf, br, P_propellant, dz,
                         rho_out, u_out, p_out, c_out):
    nc = U_interior.shape[1]

    # 1. Update interior
    for i in range(3):
        for j in range(nc):
            U_full[i, j + ng] = U_interior[i, j]

    # 2. Compute interface areas internally
    A_interfaces = np.zeros(nc + 1)
    for i in range(nc + 1):
        A_interfaces[i] = 0.5 * (A[ng - 1 + i] + A[ng + i])

    # 3. Apply Boundaries (Write back to U_full in-place)
    U_bound = apply_boundary_jit(U_full, A, gamma, R, p0_inlet, t0_inlet, p_inf, ng, inlet_bc, outlet_bc)
    for i in range(3):
        for j in range(U_full.shape[1]):
            U_full[i, j] = U_bound[i, j]

    # 4. Primitives (Write directly to Python's memory in-place!)
    rho_temp, u_temp, p_temp, c_temp = conserved_to_primitives(U_full, A, gamma)
    for j in range(rho_temp.shape[0]):
        rho_out[j] = rho_temp[j]
        u_out[j] = u_temp[j]
        p_out[j] = p_temp[j]
        c_out[j] = c_temp[j]

    # 5. Compute Fluxes and Sources
    F_hat = compute_numerical_flux(U_full, A_interfaces, rho_out, u_out, p_out, c_out, gamma, ng)
    S = source(rho_p, Tf, br[ng:-ng], R, gamma, p_out[ng:-ng], P_propellant[ng:-ng], A_interfaces, dz)

    # 6. Final Assembly
    rhs_out = np.zeros((3, nc))
    for i in range(3):
        for j in range(nc):
            rhs_out[i, j] = S[i, j] - (F_hat[i, j + 1] - F_hat[i, j]) / dz

    return rhs_out