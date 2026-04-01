import numpy as np
from numba import njit


# ===================================================================
# HLLC (Harten-Lax-van Leer-Contact)
# ===================================================================
@njit(fastmath=True, cache=True)
def hllc_flux(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    e_L = p_L / ((gamma - 1) * rho_L)
    E_L = e_L + 0.5 * u_L ** 2
    H_L = E_L + p_L / rho_L
    c_L = np.sqrt(gamma * p_L / rho_L)

    e_R = p_R / ((gamma - 1) * rho_R)
    E_R = e_R + 0.5 * u_R ** 2
    H_R = E_R + p_R / rho_R
    c_R = np.sqrt(gamma * p_R / rho_R)

    c_bar = 0.5 * (c_L + c_R)
    rho_bar = 0.5 * (rho_L + rho_R)

    p_star = 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) * rho_bar * c_bar
    p_star = max(0.0, p_star)

    if p_star <= p_L:
        q_L = 1.0
    else:
        q_L = np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / p_L - 1.0))

    if p_star <= p_R:
        q_R = 1.0
    else:
        q_R = np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / p_R - 1.0))

    S_L = u_L - c_L * q_L
    S_R = u_R + c_R * q_R

    num = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / (den + 1e-16)

    if 0 <= S_L:
        f0 = rho_L * u_L
        f1 = rho_L * u_L ** 2 + p_L
        f2 = rho_L * u_L * H_L

    elif S_L <= 0 <= S_star:
        factor = rho_L * (S_L - u_L) / (S_L - S_star)
        U_star_0 = factor
        U_star_1 = factor * S_star
        U_star_2 = factor * (E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))

        f0 = rho_L * u_L + S_L * (U_star_0 - rho_L)
        f1 = (rho_L * u_L ** 2 + p_L) + S_L * (U_star_1 - rho_L * u_L)
        f2 = (rho_L * u_L * H_L) + S_L * (U_star_2 - rho_L * E_L)

    elif S_star <= 0 <= S_R:
        factor = rho_R * (S_R - u_R) / (S_R - S_star)
        U_star_0 = factor
        U_star_1 = factor * S_star
        U_star_2 = factor * (E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))

        f0 = rho_R * u_R + S_R * (U_star_0 - rho_R)
        f1 = (rho_R * u_R ** 2 + p_R) + S_R * (U_star_1 - rho_R * u_R)
        f2 = (rho_R * u_R * H_R) + S_R * (U_star_2 - rho_R * E_R)

    else:
        f0 = rho_R * u_R
        f1 = rho_R * u_R ** 2 + p_R
        f2 = rho_R * u_R * H_R

    return f0, f1, f2

# ===================================================================
# ROE (With Harten's Entropy Fix)
# ===================================================================
@njit(fastmath=True, cache=True)
def roe_flux(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    e_L = p_L / ((gamma - 1.0) * rho_L)
    E_L = e_L + 0.5 * u_L ** 2
    H_L = E_L + p_L / rho_L
    f0_L = rho_L * u_L
    f1_L = rho_L * u_L ** 2 + p_L
    f2_L = rho_L * u_L * H_L

    e_R = p_R / ((gamma - 1.0) * rho_R)
    E_R = e_R + 0.5 * u_R ** 2
    H_R = E_R + p_R / rho_R
    f0_R = rho_R * u_R
    f1_R = rho_R * u_R ** 2 + p_R
    f2_R = rho_R * u_R * H_R

    R = np.sqrt(rho_R / rho_L)
    rho_avg = R * rho_L
    u_avg = (u_L + R * u_R) / (1.0 + R)
    H_avg = (H_L + R * H_R) / (1.0 + R)
    c_avg = np.sqrt((gamma - 1.0) * (H_avg - 0.5 * u_avg ** 2))

    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    a1 = 0.5 / c_avg ** 2 * (dp - rho_avg * c_avg * du)
    a2 = drho - dp / c_avg ** 2
    a3 = 0.5 / c_avg ** 2 * (dp + rho_avg * c_avg * du)

    lambda1 = u_avg - c_avg
    lambda2 = u_avg
    lambda3 = u_avg + c_avg

    epsilon = 0.1 * c_avg
    L1 = abs(lambda1) if abs(lambda1) >= epsilon else (lambda1**2 + epsilon**2) / (2.0 * epsilon)
    L2 = abs(lambda2) if abs(lambda2) >= epsilon else (lambda2**2 + epsilon**2) / (2.0 * epsilon)
    L3 = abs(lambda3) if abs(lambda3) >= epsilon else (lambda3**2 + epsilon**2) / (2.0 * epsilon)

    D0 = L1 * a1 + L2 * a2 + L3 * a3
    D1 = L1 * a1 * (u_avg - c_avg) + L2 * a2 * u_avg + L3 * a3 * (u_avg + c_avg)
    D2 = L1 * a1 * (H_avg - u_avg * c_avg) + L2 * a2 * 0.5 * u_avg ** 2 + L3 * a3 * (H_avg + u_avg * c_avg)

    f0 = 0.5 * (f0_L + f0_R - D0)
    f1 = 0.5 * (f1_L + f1_R - D1)
    f2 = 0.5 * (f2_L + f2_R - D2)

    return f0, f1, f2

# ===================================================================
# RUSANOV (Local Lax-Friedrichs)
# ===================================================================
@njit(fastmath=True, cache=True)
def rusanov_flux(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    e_L = p_L / ((gamma - 1.0) * rho_L)
    E_L = e_L + 0.5 * u_L ** 2
    H_L = E_L + p_L / rho_L
    c_L = np.sqrt(gamma * p_L / rho_L)

    f0_L = rho_L * u_L
    f1_L = rho_L * u_L ** 2 + p_L
    f2_L = rho_L * u_L * H_L

    e_R = p_R / ((gamma - 1.0) * rho_R)
    E_R = e_R + 0.5 * u_R ** 2
    H_R = E_R + p_R / rho_R
    c_R = np.sqrt(gamma * p_R / rho_R)

    f0_R = rho_R * u_R
    f1_R = rho_R * u_R ** 2 + p_R
    f2_R = rho_R * u_R * H_R

    S_max = max(abs(u_L) + c_L, abs(u_R) + c_R)

    f0 = 0.5 * (f0_L + f0_R) - 0.5 * S_max * (rho_R - rho_L)
    f1 = 0.5 * (f1_L + f1_R) - 0.5 * S_max * (rho_R * u_R - rho_L * u_L)
    f2 = 0.5 * (f2_L + f2_R) - 0.5 * S_max * (rho_R * E_R - rho_L * E_L)

    return f0, f1, f2