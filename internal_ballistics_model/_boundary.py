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
    Robust Characteristic-based outlet BC.

    Handles Subsonic/Supersonic switching safely using 0th-order
    extrapolation for invariants to prevent overshoots.
    """
    N = U.shape[1]
    im1 = N - ng - 1  # Last interior cell index

    # 1. Recover Primitive Variables at Last Interior Cell
    rho_int = U[0, im1] / A[im1]

    # Safety check for vacuum
    if rho_int < 1e-9:
        rho_int = 1e-9

    u_int = U[1, im1] / U[0, im1]

    # Calculate Pressure: p = (gamma-1)*(E - 0.5*rho*u^2)
    p_int = (gamma - 1.0) * (U[2, im1] / A[im1] - 0.5 * rho_int * u_int ** 2)
    p_int = max(p_int, 1e-9)  # Clamp pressure

    c_int = np.sqrt(gamma * p_int / rho_int)
    M_int = abs(u_int) / c_int

    # 2. Determine Boundary State (Ghost Cell State)
    rho_g, u_g, p_g = rho_int, u_int, p_int  # Default to 0th-order extrapolation

    # SUBSONIC OUTFLOW LOGIC (M < 1.0)
    # We must enforce p_back, but calculate rho and u consistent with outgoing characteristics.
    if M_int < 1.0:
        # A. Extrapolate Invariants (Zeroth Order for Stability)
        # J+ (outgoing) comes from interior
        J_plus = u_int + 2.0 * c_int / (gamma - 1.0)

        # Entropy (outgoing) comes from interior
        s_int = p_int / (rho_int ** gamma)

        # B. Enforce Back Pressure
        # If the flow is subsonic, the exit pressure is controlled by back pressure
        p_g = p_back

        # C. Solve for remaining variables using p_back and invariants
        # rho_g determined by Entropy: s = p / rho^gamma  -> rho = (p/s)^(1/gamma)
        rho_g = (p_g / s_int) ** (1.0 / gamma)
        rho_g = max(rho_g, 1e-9)

        c_g = np.sqrt(gamma * p_g / rho_g)

        # u_g determined by Riemann Invariant: J+ = u + 2c/(gamma-1)
        u_g = J_plus - 2.0 * c_g / (gamma - 1.0)

        # D. Inflow Check (Optional but recommended)
        # If back pressure is extremely high, it might force u_g < 0 (reverse flow).
        # For a standard nozzle, we typically clamp to 0 or allow it if your solver handles inflow.
        u_g = max(u_g, 0.0)

    # SUPERSONIC OUTFLOW LOGIC (M >= 1.0)
    # All characteristics travel downstream.
    # The boundary state is simply the interior state extrapolated.
    # We already set rho_g, u_g, p_g = rho_int, u_int, p_int above.
    else:
        pass

        # 3. Fill Ghost Cells
    # We apply the calculated boundary state (rho_g, u_g, p_g) to the ghost cells
    e_g = p_g / (gamma - 1.0) + 0.5 * rho_g * u_g ** 2

    for i in range(N - ng, N):
        # Scale by area ratio to conserve mass/momentum/energy FLUX properly if area changes
        # However, for ghost cells acting as boundary values, using the primitive values
        # combined with the *ghost cell Area* is usually the correct approach.

        U[0, i] = rho_g * A[i]
        U[1, i] = rho_g * u_g * A[i]
        U[2, i] = e_g * A[i]

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


