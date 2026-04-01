import numpy as np
from numba import njit
from enum import IntEnum


@njit(fastmath=True, cache=True)
def boundary_inlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng):
    i_int = ng

    rho_int = U[0, i_int] / A[i_int]
    u_int = U[1, i_int] / U[0, i_int]
    p_int = (gamma - 1.0) * (U[2, i_int] / A[i_int] - 0.5 * rho_int * u_int ** 2)
    p_int = max(p_int, 1e-8)
    c_int = np.sqrt(gamma * p_int / rho_int)

    J_minus = u_int - 2.0 * c_int / (gamma - 1.0)

    c0 = np.sqrt(gamma * R * t0)

    a_quad = (gamma + 1.0) / 4.0
    b_quad = -J_minus * (gamma - 1.0) / 2.0
    c_quad = ((gamma - 1.0) / 4.0) * (J_minus ** 2) - c0 ** 2

    discriminant = b_quad ** 2 - 4.0 * a_quad * c_quad

    u_b = (-b_quad + np.sqrt(max(0.0, discriminant))) / (2.0 * a_quad)
    u_b = max(0.0, u_b)

    c_b = (gamma - 1.0) / 2.0 * (u_b - J_minus)

    t_b = (c_b ** 2) / (gamma * R)
    p_b = p0 * (t_b / t0) ** (gamma / (gamma - 1.0))
    rho_b = p_b / (R * t_b)
    e_b = p_b / (gamma - 1.0) + 0.5 * rho_b * u_b ** 2

    for i in range(ng):
        U[0, i] = rho_b * A[i]
        U[1, i] = rho_b * u_b * A[i]
        U[2, i] = e_b * A[i]

    return U


@njit(fastmath=True, cache=True)
def boundary_outlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng):
    N = U.shape[1]
    i_int = N - ng - 1

    rho_int = U[0, i_int] / A[i_int]
    u_int = U[1, i_int] / U[0, i_int]
    p_int = (gamma - 1.0) * (U[2, i_int] / A[i_int] - 0.5 * rho_int * u_int ** 2)
    c_int = np.sqrt(gamma * p_int / rho_int)

    mach_int = np.abs(u_int) / c_int

    if mach_int >= 1.0:
        rho_b, u_b, p_b = rho_int, u_int, p_int
    else:
        J_plus = u_int + 2.0 * c_int / (gamma - 1.0)

        s_int = p_int / (rho_int ** gamma)

        p_b = p_inf
        rho_b = (p_b / s_int) ** (1.0 / gamma)
        c_b = np.sqrt(gamma * p_b / rho_b)
        u_b = J_plus - 2.0 * c_b / (gamma - 1.0)

    e_b = p_b / (gamma - 1.0) + 0.5 * rho_b * u_b ** 2
    for i in range(N - ng, N):
        U[0, i] = rho_b * A[i]
        U[1, i] = rho_b * u_b * A[i]
        U[2, i] = e_b * A[i]

    return U


@njit(fastmath=True, cache=True)
def boundary_inlet_reflective(U, A, gamma, R, p0, t0, p_back, ng):
    i0 = ng

    for i in range(ng):
        i_mirror = i0 + (ng - 1 - i)

        U[0, i] =  U[0, i_mirror] * A[i] / A[i_mirror]  
        U[1, i] = -U[1, i_mirror] * A[i] / A[i_mirror]
        U[2, i] =  U[2, i_mirror] * A[i] / A[i_mirror]

    return U


@njit
def boundary_outlet_reflective(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        ghost_idx = -ng + i
        interior_idx = -ng - 1 - i

        U[0, ghost_idx] =  U[0, interior_idx] * A[ghost_idx] / A[interior_idx]
        U[1, ghost_idx] = -U[1, interior_idx] * A[ghost_idx] / A[interior_idx]
        U[2, ghost_idx] =  U[2, interior_idx] * A[ghost_idx] / A[interior_idx]

    return U


@njit(fastmath=True, cache=True)
def boundary_inlet_transmissive(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        U[0, i] = U[0, ng] * A[i] / A[ng]
        U[1, i] = U[1, ng] * A[i] / A[ng]
        U[2, i] = U[2, ng] * A[i] / A[ng]
    return U


@njit(fastmath=True, cache=True)
def boundary_outlet_transmissive(U, A, gamma, R, p0, t0, p_back, ng):
    for i in range(ng):
        ghost_idx = -1 - i
        interior_idx = -ng - 1

        U[0, -1 - i] = U[0, -ng - 1] * A[ghost_idx] / A[interior_idx]
        U[1, -1 - i] = U[1, -ng - 1] * A[ghost_idx] / A[interior_idx]
        U[2, -1 - i] = U[2, -ng - 1] * A[ghost_idx] / A[interior_idx]
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


