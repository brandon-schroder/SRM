import numpy as np

class PerformanceAnalyzer:
    def __init__(self, cfg):
        """
        Analyzes the solver state to compute engineering performance metrics.
        Decouples physics calculations from data I/O.
        """
        self.cfg = cfg
        # Default to standard atmosphere if not specified in config
        self.p_inf = getattr(cfg, "p_inf", 101325.0)
        self.g0 = 9.80665

    def compute_metrics(self, state, grid):
        """
        Computes instantaneous performance metrics from the current state.

        Args:
            state: The current solver state object (with .p, .u, .rho, .A, .c).
            grid: The solver grid object (for ghost cell indices).

        Returns:
            dict: A dictionary containing both scalar 'monitors' and field 'data'.
        """
        # Exclude ghost cells for boundary values
        idx_head = grid.ng
        idx_exit = -1 - grid.ng

        # Extract primitive variables at key locations
        p_exit = state.p[idx_exit]
        u_exit = state.u[idx_exit]
        rho_exit = state.rho[idx_exit]
        area_exit = state.A[idx_exit]
        p_head = state.p[idx_head]

        # 1. Mass Flow Rate
        m_dot = rho_exit * u_exit * area_exit

        # 2. Thrust (Vacuum + Pressure term)
        thrust = m_dot * u_exit + (p_exit - self.p_inf) * area_exit

        # 3. Specific Impulse (Isp)
        isp = thrust / (m_dot * self.g0) if m_dot > 1e-9 else 0.0

        # 4. Mach Number Field
        mach = state.u / (state.c + 1e-16)

        return {
            "scalars": {
                "p_head": p_head,
                "thrust": thrust,
                "isp": isp,
                "mass_flow": m_dot,
                "p_exit": p_exit
            },
            "fields": {
                "mach": mach
            }
        }