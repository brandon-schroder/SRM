import numpy as np
from numba import njit, prange

from core.wenos import weno5_left as weno_left
from core.wenos import weno5_right as weno_right

# =============================================================================
# NUMBA-COMPILED WENO GODUNOV UPWINDING
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def weno_godunov(phi, dx, r_coords, ng=3):
    """Compute WENO Godunov upwinding on physical domain"""

    nr_full, ntheta_full, nz_full = phi.shape
    nr = nr_full - 2 * ng
    ntheta = ntheta_full - 1
    nz = nz_full - 2 * ng

    dr_half = dx[0] * 0.5
    dt_half = dx[1] * 0.5
    dz_half = dx[2] * 0.5

    D_r = np.empty((nr, ntheta, nz))
    D_theta = np.empty((nr, ntheta, nz))
    D_z = np.empty((nr, ntheta, nz))

    for i in prange(nr):
        for j in prange(ntheta):
            for k in prange(nz):
                ii = i + ng
                kk = k + ng

                # Periodic indexing for theta
                jm3 = (j - 3) % ntheta
                jm2 = (j - 2) % ntheta
                jm1 = (j - 1) % ntheta
                jp1 = (j + 1) % ntheta
                jp2 = (j + 2) % ntheta
                jp3 = (j + 3) % ntheta

                # WENO reconstruction (r)
                f_m_r = weno_left(
                    phi[ii - 3, j, kk], phi[ii - 2, j, kk], phi[ii - 1, j, kk],
                    phi[ii, j, kk], phi[ii + 1, j, kk], phi[ii + 2, j, kk])
                f_p_r = weno_right(
                    phi[ii - 2, j, kk], phi[ii - 1, j, kk], phi[ii, j, kk],
                    phi[ii + 1, j, kk], phi[ii + 2, j, kk], phi[ii + 3, j, kk])

                # WENO reconstruction (theta)
                f_m_t = weno_left(
                    phi[ii, jm3, kk], phi[ii, jm2, kk], phi[ii, jm1, kk],
                    phi[ii, j, kk], phi[ii, jp1, kk], phi[ii, jp2, kk])
                f_p_t = weno_right(
                    phi[ii, jm2, kk], phi[ii, jm1, kk], phi[ii, j, kk],
                    phi[ii, jp1, kk], phi[ii, jp2, kk], phi[ii, jp3, kk])

                # WENO reconstruction (z)
                f_m_z = weno_left(
                    phi[ii, j, kk - 3], phi[ii, j, kk - 2], phi[ii, j, kk - 1],
                    phi[ii, j, kk], phi[ii, j, kk + 1], phi[ii, j, kk + 2])
                f_p_z = weno_right(
                    phi[ii, j, kk - 2], phi[ii, j, kk - 1], phi[ii, j, kk],
                    phi[ii, j, kk + 1], phi[ii, j, kk + 2], phi[ii, j, kk + 3])

                # Godunov upwinding (r)
                D_m_r = (phi[ii, j, kk] - f_m_r) / dr_half
                D_p_r = (f_p_r - phi[ii, j, kk]) / dr_half

                # Godunov upwinding (theta) (scaled by r for cylindrical)
                D_m_t = (phi[ii, j, kk] - f_m_t) / (dt_half * r_coords[ii, j, kk])
                D_p_t = (f_p_t - phi[ii, j, kk]) / (dt_half * r_coords[ii, j, kk])

                # Godunov upwinding (z)
                D_m_z = (phi[ii, j, kk] - f_m_z) / dz_half
                D_p_z = (f_p_z - phi[ii, j, kk]) / dz_half

                D_r[i, j, k] = max(D_m_r, 0.0) ** 2 + min(D_p_r, 0.0) ** 2
                D_theta[i, j, k] = max(D_m_t, 0.0) ** 2 + min(D_p_t, 0.0) ** 2
                D_z[i, j, k] = max(D_m_z, 0.0) ** 2 + min(D_p_z, 0.0) ** 2

    return np.sqrt(D_r + D_theta + D_z) # -|∇φ|_Godunov


# =============================================================================
# ADAPTIVE TIMESTEP COMPUTATION
# =============================================================================

@njit(fastmath=True, cache=True)
def adaptive_timestep(grad_mag, dx, r_coords, ng, CFL, t_end, br, t=0.0):

    """
    Physically accurate and fast adaptive timestep for cylindrical level-set
    φ_t = -br * |∇φ| with directionally split CFL in (r, θ, z).

    Optimisation: instead of looping over every cell, we only need the
    maximum |∇φ| in each radial shell i — the θ-resolution restriction is
    the same for the entire shell.

    """
    # grad_mag = grid.grad_mag
    # dx = grid.dx
    # r_coords = grid.polar_coords[0]
    # ng = grid.ng

    # CFL = simulation_inputs.CFL
    # t_end = simulation_inputs.t_end
    # br = simulation_inputs.br

    nr, ntheta, nz = grad_mag.shape
    dr, dtheta, dz = dx

    # Physical r-coordinates for each radial index (shape: nr,)
    r_physical = r_coords[ng:-ng, 0, ng]        # assuming r independent of θ,z

    # Pre-compute maximum |∇φ| in each radial layer (nr values)
    max_grad_per_r = np.zeros(nr)
    for i in range(nr):
        max_val = 0.0
        for j in range(ntheta):
            for k in range(nz):
                if grad_mag[i, j, k] > max_val:
                    max_val = grad_mag[i, j, k]
        max_grad_per_r[i] = max_val

    dt_min = 1e20  # very large initial value

    for i in range(nr):
        r_i = r_physical[i]
        max_grad_i = max_grad_per_r[i]

        if max_grad_i < 1e-12:
            continue                                    # nothing moving here

        wave_speed = br * max_grad_i                    # local max propagation speed

        # Effective grid spacing in θ-direction at this radius
        h_theta = r_i * dtheta
        if h_theta < 1e-12:                             # protect axis (if present)
            h_theta = dr * 0.5                          # treat as Cartesian near r=0

        # Directional CFL limits
        dt_r     = dr     / wave_speed
        dt_theta = h_theta / wave_speed
        dt_z     = dz     / wave_speed

        dt_local = min(dt_r, dt_theta, dt_z)

        if dt_local < dt_min:
            dt_min = dt_local

    if dt_min > 1e19:                     # no significant gradient anywhere
        dt = 0.1
    else:
        dt = CFL * dt_min

    # Global safety caps
    dt = max(dt, 1e-8)
    dt = min(dt, 0.5)

    # Do not overshoot final time
    if t_end is not None:
        dt = min(dt, t_end - t)

    return dt


