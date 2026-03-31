import numpy as np
from numba import njit
from enum import IntEnum

@njit(fastmath=True, cache=True)
def boundary_inlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng):
    i0 = ng

    rho_int = U[0, i0] / A[i0]
    u_int = U[1, i0] / U[0, i0]
    p_int = (gamma - 1.0) * (U[2, i0] / A[i0] - 0.5 * rho_int * u_int * u_int)
    p_int = max(p_int, 1e-8)
    c_int = np.sqrt(gamma * p_int / rho_int)

    M_guess = 0.3

    T_static = t0 / (1.0 + 0.5 * (gamma - 1.0) * M_guess ** 2)
    p_static = p0 * (T_static / t0) ** (gamma / (gamma - 1.0))
    rho_static = p_static / (R * T_static)
    c_static = np.sqrt(gamma * p_static / rho_static)
    u_static = M_guess * c_static

    for i in range(ng):

        weight = float(ng - i) / float(ng + 1)

        rho_blend = weight * rho_static + (1.0 - weight) * rho_int
        u_blend = weight * u_static + (1.0 - weight) * u_int
        p_blend = weight * p_static + (1.0 - weight) * p_int

        e_blend = p_blend / (gamma - 1.0) + 0.5 * rho_blend * u_blend ** 2

        U[0, i] = rho_blend * A[i]
        U[1, i] = rho_blend * u_blend * A[i]
        U[2, i] = e_blend * A[i]

    return U


@njit(fastmath=True, cache=True)
def boundary_outlet_characteristic(U, A, gamma, R, p0, t0, p_back, ng):
    N = U.shape[1]
    im1 = N - ng - 1

    rho_int = U[0, im1] / A[im1]

    if rho_int < 1e-9:
        rho_int = 1e-9

    u_int = U[1, im1] / U[0, im1]

    p_int = (gamma - 1.0) * (U[2, im1] / A[im1] - 0.5 * rho_int * u_int ** 2)
    p_int = max(p_int, 1e-9)

    c_int = np.sqrt(gamma * p_int / rho_int)
    M_int = abs(u_int) / c_int

    rho_g, u_g, p_g = rho_int, u_int, p_int

    if M_int < 1.0:

        J_plus = u_int + 2.0 * c_int / (gamma - 1.0)

        s_int = p_int / (rho_int ** gamma)

        p_g = p_back

        rho_g = (p_g / s_int) ** (1.0 / gamma)
        rho_g = max(rho_g, 1e-9)

        c_g = np.sqrt(gamma * p_g / rho_g)

        u_g = J_plus - 2.0 * c_g / (gamma - 1.0)

        u_g = max(u_g, 0.0)

    else:
        pass

    e_g = p_g / (gamma - 1.0) + 0.5 * rho_g * u_g ** 2

    for i in range(N - ng, N):

        U[0, i] = rho_g * A[i]
        U[1, i] = rho_g * u_g * A[i]
        U[2, i] = e_g * A[i]

    return U


@njit(fastmath=True, cache=True)
def boundary_inlet_reflective(U, A, gamma, R, p0, t0, p_back, ng):
    i0 = ng

    for i in range(ng):
        i_mirror = i0 + (ng - 1 - i)

        U[0, i] = U[0, i_mirror] * A[i] / A[i_mirror]  # Mass
        U[1, i] = -U[1, i_mirror] * A[i] / A[i_mirror]  # Momentum (reversed)
        U[2, i] = U[2, i_mirror] * A[i] / A[i_mirror]  # Energy

    return U


@njit
def boundary_outlet_reflective(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        ghost_idx = -ng + i
        interior_idx = -ng - 1 - i

        U[0, ghost_idx] = U[0, interior_idx]  # Mass (symmetric)
        U[1, ghost_idx] = -U[1, interior_idx]  # Momentum (anti-symmetric)
        U[2, ghost_idx] = U[2, interior_idx]  # Energy (symmetric)

    return U


@njit(fastmath=True, cache=True)
def boundary_inlet_transmissive(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        U[0, i] = U[0, ng]
        U[1, i] = U[1, ng]
        U[2, i] = U[2, ng]
    return U


@njit(fastmath=True, cache=True)
def boundary_outlet_transmissive(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        U[0, -1 - i] = U[0, -ng - 1]
        U[1, -1 - i] = U[1, -ng - 1]
        U[2, -1 - i] = U[2, -ng - 1]
    return U


class BCType(IntEnum):
    REFLECTIVE = 0
    CHARACTERISTIC = 1
    TRANSMISSIVE = 2

@njit(fastmath=True, cache=True)
def apply_boundary_jit(U, A, gamma, R, p0, t0, p_inf, ng, inlet_type, outlet_type):

    # Route Inlet Boundary Condition
    if inlet_type == BCType.REFLECTIVE.value:
        U = boundary_inlet_reflective(U, A, gamma, R, p0, t0, p_inf, ng)
    elif inlet_type == BCType.CHARACTERISTIC.value:
        U = boundary_inlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng)
    elif inlet_type == BCType.TRANSMISSIVE.value:
        U = boundary_inlet_transmissive(U, A, gamma, R, p0, t0, p_inf, ng)

    # Route Outlet Boundary Condition
    if outlet_type == BCType.REFLECTIVE.value:
        U = boundary_outlet_reflective(U, A, gamma, R, p0, t0, p_inf, ng)
    elif outlet_type == BCType.CHARACTERISTIC.value:
        U = boundary_outlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng)
    elif outlet_type == BCType.TRANSMISSIVE.value:
        U = boundary_outlet_transmissive(U, A, gamma, R, p0, t0, p_inf, ng)

    return U


