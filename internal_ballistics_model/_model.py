import numpy as np
import pandas as pd
from typing import Callable, Tuple

# Assuming Grid1D is in grid.py
from _grid import Grid1D
from _structure import SimulationConfig, FlowState

# Import JIT functions
from _burn_rate import *
from internal_ballistics_solver import *
from core.time_integrators import ssp_rk_3_3


class IBSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # 1. Build Grid
        self.grid = Grid1D(config)

        # 2. Allocate State
        self.state = FlowState(n_cells=self.grid.dims[0])

        # Initialize defaults to safe values (avoid divide by zero in empty state)
        self.state.A[:] = 1.0e-6
        self.state.P[:] = 1.0e-2

        # Select burn rate laws
        base = StRobertsLaw(config)
        self.burn_model = MukundaPaulErosive(config, base_model=base)


    def set_geometry(self, x_imp: np.ndarray, A_imp: np.ndarray, P_imp: np.ndarray):
        """
        Interpolates external geometry onto the grid.
        CRITICAL: Clamps Area to prevent division by zero in boundary conditions.
        """
        # Interpolate onto the solver's x coordinates (including ghosts)
        self.state.A = np.interp(self.grid.x_coords, x_imp, A_imp)
        self.state.P = np.interp(self.grid.x_coords, x_imp, P_imp)

        # Calculate gradients (Central Difference)
        self.state.dAdz[1:-1] = (self.state.A[2:] - self.state.A[:-2]) / (2 * self.grid.dx)
        # Boundaries (Neumann)
        self.state.dAdz[0] = self.state.dAdz[1]
        self.state.dAdz[-1] = self.state.dAdz[-2]

    def initialize(self):
        """Sets initial conditions."""
        # Verification
        if np.min(self.state.A) <= 0:
            raise ValueError("Geometry Error: Area contains zeros. set_geometry() clamp failed.")

        self.state.rho[:] = self.cfg.p_inf / (self.cfg.R * self.cfg.t_initial)
        self.state.p[:] = self.cfg.p_inf
        self.state.u[:] = self.cfg.u_initial
        self.state.br = 0.0

        self.state.U = primitive_to_conserved(
            self.state.rho, self.state.u, self.state.p,
            self.state.A, self.cfg.gamma
        )

        # Explicitly update primitives once to ensure consistency
        self.state.c[:] = np.sqrt(self.cfg.gamma * self.state.p / self.state.rho)

    def _compute_rhs(self, U_interior: np.ndarray) -> np.ndarray:
        """
        The Spatial Operator L(U).
        Calculates dU/dt = -dF/dx + S
        """
        # Map interior to full state
        U_full = self.state.U
        U_full[:, self.grid.interior] = U_interior

        # Apply Boundary Conditions
        # Note: We pass self.state.A which is now guaranteed > 1e-12
        U_full = apply_boundary_jit(
            U_full,
            self.state.A,
            self.cfg.gamma,
            self.cfg.R,
            self.cfg.p0_inlet,
            self.cfg.t0_inlet,
            self.cfg.p_inf,
            self.cfg.ng
        )

        # Update Primitives
        self.state.rho, self.state.u, self.state.p, self.state.c = \
            compute_primitives_jit(U_full, self.state.A, self.cfg.gamma)

        # Adaptive dissipation
        alpha = np.max(np.abs(self.state.u[self.grid.interior]) + self.state.c[self.grid.interior])
        # Fallback if alpha is NaN (rare)
        if not np.isfinite(alpha): alpha = 1000.0

        # Fluxes
        F_hat = compute_numerical_flux_jit(
            U_full, self.state.A, self.state.rho, self.state.u,
            self.state.p, alpha, self.cfg.ng
        )

        # Source Terms
        S = source_jit(
            self.cfg.rho_p, self.cfg.Tf, self.state.br,
            self.cfg.R, self.cfg.gamma, self.state.p[self.grid.interior],
            self.state.P[self.grid.interior], self.state.dAdz[self.grid.interior]
        )

        dFdx = (F_hat[:, 1:] - F_hat[:, :-1]) / self.grid.dx
        return S - dFdx

    def step(self) -> Tuple[float, float]:
        # Adaptive Timestep
        # Use simple max wave speed
        u_max = np.max(np.abs(self.state.u)) + np.max(self.state.c)
        dt = self.cfg.CFL * self.grid.dx / (u_max + 1e-16)

        # Update Burn Rate
        self.state.br = self.burn_model.compute_burn_rate(self.state, self.grid)

        # Time Integration (SSP-RK3)
        U_int = self.state.U[:, self.grid.interior]
        U_new = ssp_rk_3_3(U_int, dt, self._compute_rhs)
        self.state.U[:, self.grid.interior] = U_new

        self.state.t += dt
        return dt, self.state.t

    def get_dataframe(self) -> pd.DataFrame:
        sl = self.grid.interior
        return pd.DataFrame({
            "x": self.grid.x_interior,
            "rho": self.state.rho[sl],
            "u": self.state.u[sl],
            "p": self.state.p[sl],
            "Mach": self.state.u[sl] / (self.state.c[sl] + 1e-16),
            "Area": self.state.A[sl]
        })