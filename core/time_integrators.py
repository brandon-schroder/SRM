import numpy as np

# ===================================================================
# Total Variation Diminishing Runge-Kutta
# ===================================================================


def ssp_rk_1_1(u: np.ndarray, dt: float, L) -> np.ndarray:
    """1st-order, 1-stage SSP Runge-Kutta."""
    Lu = L(u)
    u_next = u + dt * Lu

    return u_next


def ssp_rk_2_2(u: np.ndarray, dt: float, L) -> np.ndarray:
    """2nd-order, 2-stage SSP Runge-Kutta."""
    # Stage 1
    Lu = L(u)
    u1 = u + dt * Lu

    # Stage 2
    Lu1 = L(u1)
    u_next = 0.5 * (u + u1 + dt * Lu1)

    return u_next


def ssp_rk_3_3(u: np.ndarray, dt: float, L) -> np.ndarray:
    """3rd-order, 3-stage SSP Runge-Kutta."""
    # Stage 1
    Lu = L(u)
    u1 = u + dt * Lu

    # Stage 2
    Lu1 = L(u1)
    u2 = (3.0 / 4.0) * u + (1.0 / 4.0) * u1 + (1.0 / 4.0) * dt * Lu1

    # Stage 3
    Lu2 = L(u2)
    u_next = (1.0 / 3.0) * u + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * Lu2

    return u_next


def ssp_rk_3_3_low_storage(u: np.ndarray, dt: float, L) -> np.ndarray:
    """Low-storage 3-stage, 3rd-order SSP Runge–Kutta (explicit) with SSP coefficient ~2."""
    u0 = u.copy()

    # Stage 1
    u = u0 + dt * L(u0)

    # Stage 2
    u = (3.0 / 4.0) * u0 + (1.0 / 4.0) * (u + dt * L(u))

    # Stage 3
    u = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u + dt * L(u))

    return u


def ssp_rk_4_4(u: np.ndarray, dt: float, L) -> np.ndarray:
    """4th-order, 4-stage SSP Runge-Kutta."""
    # Stage 1
    Lu = L(u)
    u1 = u + (1.0 / 2.0) * dt * Lu

    # Stage 2
    Lu1 = L(u1)
    u2 = (1.0 / 2.0) * u - (1.0 / 4.0) * dt * Lu + (1.0 / 2.0) * u1 + (1.0 / 2.0) * dt * Lu1

    # Stage 3
    Lu2 = L(u2)
    u3 = (1.0 / 9.0) * u - (1.0 / 9.0) * dt * Lu + (2.0 / 9.0) * u1 - (1.0 / 3.0) * dt * Lu1 + (2.0 / 3.0) * u2 + dt * Lu2

    # Stage 4
    Lu3 = L(u3)
    u_next = (1.0 / 3.0) * u1 + (1.0 / 6.0) * dt * Lu1 + (1.0 / 3.0) * u2 + (1.0 / 3.0) * u3 + (1.0 / 6.0) * dt * Lu3

    return u_next


def ssp_rk_5_3(un, dt, F):
    """
    SSP-RK(5,3) with exact fractions
    """
    # Stage 1
    u1 = un + (2 / 5) * dt * F(un)

    # Stage 2
    u2 = (1 / 2) * un + (1 / 2) * u1 + (1 / 5) * dt * F(u1)

    # Stage 3
    u3 = (3 / 5) * un + (2 / 5) * u2 + (1 / 4) * dt * F(u2)

    # Stage 4
    u4 = (1 / 5) * un + (4 / 5) * u3 + (1 / 2) * dt * F(u3)

    # Stage 5
    un1 = (1 / 2) * u2 + (1 / 10) * u3 + (2 / 5) * u4 + (3 / 50) * dt * F(u3) + (1 / 5) * dt * F(u4)

    return un1