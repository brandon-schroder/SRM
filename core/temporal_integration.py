import numpy as np

# ===================================================================
# Total Variation Diminishing Runge-Kutta
# ===================================================================

class SSPRK11:
    """1st-order, 1-stage SSP Runge-Kutta (Forward Euler)."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.Lu = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        self.Lu[:] = rhs_func(u)
        u += dt * self.Lu
        return u


class SSPRK22:
    """2nd-order, 2-stage SSP Runge-Kutta."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.u1 = np.zeros(shape, dtype=dtype)
        self.Lu = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        # Stage 1
        self.Lu[:] = rhs_func(u)
        self.u1[:] = u + dt * self.Lu

        # Stage 2
        self.Lu[:] = rhs_func(self.u1)
        u[:] = 0.5 * (u + self.u1 + dt * self.Lu)

        return u


class SSPRK33:
    """3rd-order, 3-stage SSP Runge-Kutta."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.u1 = np.zeros(shape, dtype=dtype)
        self.u2 = np.zeros(shape, dtype=dtype)
        self.Lu = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        # Stage 1
        self.Lu[:] = rhs_func(u)
        self.u1[:] = u + dt * self.Lu

        # Stage 2
        self.Lu[:] = rhs_func(self.u1)
        self.u2[:] = (3.0 / 4.0) * u + (1.0 / 4.0) * self.u1 + (1.0 / 4.0) * dt * self.Lu

        # Stage 3
        self.Lu[:] = rhs_func(self.u2)
        u[:] = (1.0 / 3.0) * u + (2.0 / 3.0) * self.u2 + (2.0 / 3.0) * dt * self.Lu

        return u


class SSPRK33LowStorage:
    """Low-storage 3-stage, 3rd-order SSP Runge–Kutta (explicit)."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.u0 = np.zeros(shape, dtype=dtype)
        self.Lu = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        # Save initial state
        self.u0[:] = u

        # Stage 1
        self.Lu[:] = rhs_func(self.u0)
        u[:] = self.u0 + dt * self.Lu

        # Stage 2
        self.Lu[:] = rhs_func(u)
        u[:] = (3.0 / 4.0) * self.u0 + (1.0 / 4.0) * (u + dt * self.Lu)

        # Stage 3
        self.Lu[:] = rhs_func(u)
        u[:] = (1.0 / 3.0) * self.u0 + (2.0 / 3.0) * (u + dt * self.Lu)

        return u


class ClassicRK4:
    """4th-order, 4-stage Classical Runge-Kutta."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.u1 = np.zeros(shape, dtype=dtype)
        self.u2 = np.zeros(shape, dtype=dtype)
        self.u3 = np.zeros(shape, dtype=dtype)
        self.Lu = np.zeros(shape, dtype=dtype)
        self.Lu1 = np.zeros(shape, dtype=dtype)
        self.L_temp = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        # Stage 1
        self.Lu[:] = rhs_func(u)
        self.u1[:] = u + (1.0 / 2.0) * dt * self.Lu

        # Stage 2
        self.Lu1[:] = rhs_func(self.u1)
        self.u2[:] = (1.0 / 2.0) * u - (1.0 / 4.0) * dt * self.Lu + (1.0 / 2.0) * self.u1 + (1.0 / 2.0) * dt * self.Lu1

        # Stage 3
        self.L_temp[:] = rhs_func(self.u2)
        self.u3[:] = (1.0 / 9.0) * u - (1.0 / 9.0) * dt * self.Lu + (2.0 / 9.0) * self.u1 - (
                    1.0 / 3.0) * dt * self.Lu1 + (2.0 / 3.0) * self.u2 + dt * self.L_temp

        # Stage 4
        self.L_temp[:] = rhs_func(self.u3)
        u[:] = (1.0 / 3.0) * self.u1 + (1.0 / 6.0) * dt * self.Lu1 + (1.0 / 3.0) * self.u2 + (1.0 / 3.0) * self.u3 + (
                    1.0 / 6.0) * dt * self.L_temp

        return u


class SSPRK53:
    """3rd-order, 5-stage SSP Runge-Kutta."""

    def __init__(self, shape: tuple, dtype=np.float64):
        self.un = np.zeros(shape, dtype=dtype)
        self.u1 = np.zeros(shape, dtype=dtype)
        self.u2 = np.zeros(shape, dtype=dtype)
        self.u3 = np.zeros(shape, dtype=dtype)
        self.u4 = np.zeros(shape, dtype=dtype)

        self.L_temp = np.zeros(shape, dtype=dtype)
        self.Lu3 = np.zeros(shape, dtype=dtype)

    def step(self, u: np.ndarray, dt: float, rhs_func) -> np.ndarray:
        # Save initial state
        self.un[:] = u

        # Stage 1
        self.L_temp[:] = rhs_func(self.un)
        self.u1[:] = self.un + (2.0 / 5.0) * dt * self.L_temp

        # Stage 2
        self.L_temp[:] = rhs_func(self.u1)
        self.u2[:] = 0.5 * self.un + 0.5 * self.u1 + (1.0 / 5.0) * dt * self.L_temp

        # Stage 3
        self.L_temp[:] = rhs_func(self.u2)
        self.u3[:] = 0.6 * self.un + 0.4 * self.u2 + (1.0 / 4.0) * dt * self.L_temp

        # Stage 4 (Need to persist Lu3 for the final stage)
        self.Lu3[:] = rhs_func(self.u3)
        self.u4[:] = 0.2 * self.un + 0.8 * self.u3 + 0.5 * dt * self.Lu3

        # Stage 5
        self.L_temp[:] = rhs_func(self.u4)
        u[:] = 0.5 * self.u2 + 0.1 * self.u3 + 0.4 * self.u4 + (3.0 / 50.0) * dt * self.Lu3 + 0.2 * dt * self.L_temp

        return u