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


def classic_rk4(u: np.ndarray, dt: float, L) -> np.ndarray:
    """4th-order, 4-stage Classical Runge-Kutta."""
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

    Lu = F(un)
    u1 = un + (2.0 / 5.0) * dt * Lu

    Lu1 = F(u1)
    u2 = 0.5 * un + 0.5 * u1 + (1.0 / 5.0) * dt * Lu1

    Lu2 = F(u2)
    u3 = 0.6 * un + 0.4 * u2 + (1.0 / 4.0) * dt * Lu2

    Lu3 = F(u3)
    u4 = 0.2 * un + 0.8 * u3 + 0.5 * dt * Lu3

    Lu4 = F(u4)
    u_next = 0.5 * u2 + 0.1 * u3 + 0.4 * u4 + (3.0 / 50.0) * dt * Lu3 + 0.2 * dt * Lu4

    return u_next