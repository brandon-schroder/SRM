import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def boundary_inlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng):
    """
    Characteristic-based subsonic inlet BC.

    For subsonic inlet (M < 1):
    - 1 characteristic enters from upstream (exterior)
    - 3 characteristics exit to downstream (interior)

    We specify:
    - Total pressure p0
    - Total temperature t0
    - Assume incoming characteristic carries entropy from interior

    This allows the flow to adjust naturally.
    """
    i0 = ng  # First interior cell

    # Extract interior state
    rho_int = U[0, i0] / A[i0]
    u_int = U[1, i0] / U[0, i0]
    p_int = (gamma - 1.0) * (U[2, i0] / A[i0] - 0.5 * rho_int * u_int * u_int)
    p_int = max(p_int, 1e-8)
    c_int = np.sqrt(gamma * p_int / rho_int)

    # Riemann invariant (entering from interior)
    J_minus = u_int - 2.0 * c_int / (gamma - 1.0)

    # Assume isentropic relation from total conditions
    # For simplicity, we can specify target Mach or let it evolve
    # Here we use a simple approach: assume the flow accelerates gradually

    # Method 1: Fix total enthalpy and use Riemann invariant
    # H_total = gamma/(gamma-1) * p/rho + 0.5*u^2 = constant
    H_total = gamma / (gamma - 1.0) * R * t0

    # Solve for ghost cell state using:
    # J_minus = u - 2*c/(gamma-1) and H_total
    # This is complex, so we use a simpler approach:

    # Method 2 (simpler, stable):
    # Extrapolate entropy, fix total conditions
    s_int = p_int / (rho_int ** gamma)  # Entropy measure

    # Assume low Mach at inlet (initial guess)
    M_guess = 0.3

    # Isentropic relations
    T_static = t0 / (1.0 + 0.5 * (gamma - 1.0) * M_guess ** 2)
    p_static = p0 * (T_static / t0) ** (gamma / (gamma - 1.0))
    rho_static = p_static / (R * T_static)
    c_static = np.sqrt(gamma * p_static / rho_static)
    u_static = M_guess * c_static

    # Compute total energy per unit volume
    e_vol = p_static / (gamma - 1.0) + 0.5 * rho_static * u_static ** 2

    # Fill ghost cells with linear variation toward inlet condition
    for i in range(ng):
        # Blend between inlet and interior state
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
    """
    Characteristic-based outlet BC.

    For subsonic outlet (M < 1):
    - 3 characteristics enter from upstream (interior)
    - 1 characteristic exits to downstream (exterior)

    We specify:
    - Back pressure p_back (the outgoing characteristic)
    - Extrapolate the 3 incoming characteristics
    """
    N = U.shape[1]
    im1 = N - ng - 1  # Last interior cell
    im2 = N - ng - 2  # Second to last interior cell

    # Extract state from last two interior cells
    rho_m1 = U[0, im1] / A[im1]
    u_m1 = U[1, im1] / U[0, im1]
    p_m1 = (gamma - 1.0) * (U[2, im1] / A[im1] - 0.5 * rho_m1 * u_m1 * u_m1)
    p_m1 = max(p_m1, 1e-8)
    c_m1 = np.sqrt(gamma * p_m1 / rho_m1)
    M_m1 = abs(u_m1) / max(c_m1, 1e-8)

    if M_m1 < 1.0:
        # Subsonic: use characteristic BC

        # Extrapolate Riemann invariants from interior
        rho_m2 = U[0, im2] / A[im2]
        u_m2 = U[1, im2] / U[0, im2]
        p_m2 = (gamma - 1.0) * (U[2, im2] / A[im2] - 0.5 * rho_m2 * u_m2 * u_m2)
        p_m2 = max(p_m2, 1e-8)
        c_m2 = np.sqrt(gamma * p_m2 / rho_m2)

        # Extrapolate outgoing Riemann invariant
        # J_plus = u + 2*c/(gamma-1)
        J_plus_m1 = u_m1 + 2.0 * c_m1 / (gamma - 1.0)
        J_plus_m2 = u_m2 + 2.0 * c_m2 / (gamma - 1.0)
        J_plus_extrap = 2.0 * J_plus_m1 - J_plus_m2

        # Extrapolate entropy
        s_m1 = p_m1 / (rho_m1 ** gamma)
        s_m2 = p_m2 / (rho_m2 ** gamma)
        s_extrap = 2.0 * s_m1 - s_m2
        s_extrap = max(s_extrap, 1e-10)

        # Given p_back and J_plus, s → solve for u, rho, c
        # From s = p/rho^gamma and p = rho*c^2/gamma:
        # c^2 = gamma * s * rho^(gamma-1)

        # From J_plus = u + 2*c/(gamma-1) and other relations
        # This requires iteration. Simplified approach:

        # Use back pressure directly
        p_g = p_back

        # From entropy: rho = (p/s)^(1/gamma)
        rho_g = (p_g / s_extrap) ** (1.0 / gamma)
        rho_g = max(rho_g, 1e-10)

        c_g = np.sqrt(gamma * p_g / rho_g)

        # From Riemann invariant
        u_g = J_plus_extrap - 2.0 * c_g / (gamma - 1.0)

        # Compute energy
        e_g = p_g / (gamma - 1.0) + 0.5 * rho_g * u_g ** 2

        # Fill ghost cells
        for i in range(N - ng, N):
            U[0, i] = rho_g * A[i]
            U[1, i] = rho_g * u_g * A[i]
            U[2, i] = e_g * A[i]

    else:
        # Supersonic: all characteristics leave domain, extrapolate everything
        for i in range(N - ng, N):
            U[0, i] = U[0, im1] * A[i] / A[im1]
            U[1, i] = U[1, im1] * A[i] / A[im1]
            U[2, i] = U[2, im1] * A[i] / A[im1]

    return U


@njit(fastmath=True, cache=True)
def boundary_inlet_reflective(U, A, gamma, R, p0, t0, p_back, ng):
    """
    Reflective wall boundary condition at inlet.

    For a solid wall:
    - Reflect the flow state across the boundary
    - Reverse the velocity to enforce zero normal velocity at the wall
    """
    i0 = ng  # First interior cell

    # Fill ghost cells by reflecting interior cells
    for i in range(ng):
        i_mirror = i0 + (ng - 1 - i)  # Mirror: i=0→ng+(ng-1), i=1→ng+(ng-2), etc.

        # Copy conserved variables but reverse momentum
        U[0, i] = U[0, i_mirror] * A[i] / A[i_mirror]  # Mass
        U[1, i] = -U[1, i_mirror] * A[i] / A[i_mirror]  # Momentum (reversed)
        U[2, i] = U[2, i_mirror] * A[i] / A[i_mirror]  # Energy

    return U



# ==============================================================

@njit(fastmath=True, cache=True)
def apply_boundary_jit(U, A, gamma, R, p0, t0, p_inf, ng):
    """Apply characteristic-based boundary conditions."""

    U = boundary_inlet_reflective(U, A, gamma, R, p0, t0, p_inf, ng)
    # U = boundary_inlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng)
    U = boundary_outlet_characteristic(U, A, gamma, R, p0, t0, p_inf, ng)

    return U


