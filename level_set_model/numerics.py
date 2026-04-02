import numpy as np
from numba import njit, prange

from schemes.spatial_reconstruction import weno5_left as left_biased, weno5_right as right_biased


@njit(fastmath=True, cache=True, parallel=True)
def weno_godunov(phi, dx, r_coords, ng=3):
    nr_full, ntheta_full, nz_full = phi.shape
    nr = nr_full - 2 * ng
    ntheta = ntheta_full - 1
    nz = nz_full - 2 * ng

    dr_half = dx[0] * 0.5
    dt_half = dx[1] * 0.5
    dz_half = dx[2] * 0.5

    D_r = np.asfortranarray(np.empty((nr, ntheta, nz), dtype=phi.dtype))
    D_theta = np.asfortranarray(np.empty((nr, ntheta, nz), dtype=phi.dtype))
    D_z = np.asfortranarray(np.empty((nr, ntheta, nz), dtype=phi.dtype))

    for k in prange(nz):
        for j in prange(ntheta):
            for i in prange(nr):

                ii = i + ng
                kk = k + ng

                jm3 = (j - 3) % ntheta
                jm2 = (j - 2) % ntheta
                jm1 = (j - 1) % ntheta
                jp1 = (j + 1) % ntheta
                jp2 = (j + 2) % ntheta
                jp3 = (j + 3) % ntheta

                f_m_r = left_biased(
                    phi[ii - 3, j, kk], phi[ii - 2, j, kk], phi[ii - 1, j, kk],
                    phi[ii, j, kk], phi[ii + 1, j, kk], phi[ii + 2, j, kk])
                f_p_r = right_biased(
                    phi[ii - 2, j, kk], phi[ii - 1, j, kk], phi[ii, j, kk],
                    phi[ii + 1, j, kk], phi[ii + 2, j, kk], phi[ii + 3, j, kk])

                f_m_t = left_biased(
                    phi[ii, jm3, kk], phi[ii, jm2, kk], phi[ii, jm1, kk],
                    phi[ii, j, kk], phi[ii, jp1, kk], phi[ii, jp2, kk])
                f_p_t = right_biased(
                    phi[ii, jm2, kk], phi[ii, jm1, kk], phi[ii, j, kk],
                    phi[ii, jp1, kk], phi[ii, jp2, kk], phi[ii, jp3, kk])

                f_m_z = left_biased(
                    phi[ii, j, kk - 3], phi[ii, j, kk - 2], phi[ii, j, kk - 1],
                    phi[ii, j, kk], phi[ii, j, kk + 1], phi[ii, j, kk + 2])
                f_p_z = right_biased(
                    phi[ii, j, kk - 2], phi[ii, j, kk - 1], phi[ii, j, kk],
                    phi[ii, j, kk + 1], phi[ii, j, kk + 2], phi[ii, j, kk + 3])

                D_m_r = (phi[ii, j, kk] - f_m_r) / dr_half
                D_p_r = (f_p_r - phi[ii, j, kk]) / dr_half

                D_m_t = (phi[ii, j, kk] - f_m_t) / (dt_half * r_coords[ii, j, kk])
                D_p_t = (f_p_t - phi[ii, j, kk]) / (dt_half * r_coords[ii, j, kk])

                D_m_z = (phi[ii, j, kk] - f_m_z) / dz_half
                D_p_z = (f_p_z - phi[ii, j, kk]) / dz_half

                D_r[i, j, k] = max(D_m_r, 0.0) ** 2 + min(D_p_r, 0.0) ** 2
                D_theta[i, j, k] = max(D_m_t, 0.0) ** 2 + min(D_p_t, 0.0) ** 2
                D_z[i, j, k] = max(D_m_z, 0.0) ** 2 + min(D_p_z, 0.0) ** 2

    return np.sqrt(D_r + D_theta + D_z)


@njit(fastmath=True, cache=True)
def adaptive_timestep(dx, r_coords, ng, CFL, t_end, br, t=0.0):
    dr, dtheta, dz = dx

    v_max = br.max()
    if v_max < 1e-16:
        v_max = 1e-16

    r_min = r_coords[ng, 0, ng]

    h_theta_min = (r_min * dtheta)
    if h_theta_min < 1e-16:
        h_theta_min = dr * 0.5

    dt_stable = CFL / (v_max * ((1.0 / dr) + (1.0 / h_theta_min) + (1.0 / dz)))

    dt_stable = min(max(dt_stable, 1e-16), 0.5)

    t_remaining = t_end - t

    if t_remaining <= 1e-16 or t_remaining < 0:
        return 0.0

    if t_remaining <= dt_stable:
        dt = t_remaining
    elif t_remaining < 2.0 * dt_stable:
        dt = t_remaining / 2.0
    else:
        dt = dt_stable

    return dt


