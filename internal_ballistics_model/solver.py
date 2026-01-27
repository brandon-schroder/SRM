import pandas as pd
import numpy as np
from typing import Tuple

# Imports
from .grid import *
from .boundary import *
from .burn_rate import *
from .numerics import *
from .config import *
from .postprocess import *

from core.logger import *
from core.time_integrators import ssp_rk_3_3


class IBSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.grid = Grid1D(config)
        self.state = FlowState(n_cells=self.grid.dims[2], dtype=self.cfg.dtype)
        self.dt = 0.0

        # --- Internalized Recorder ---
        self.recorder = SimulationRecorder(
            solver=self,
            state_map={
                "pressure": {"attr": "p", "unit": "Pa"},
                "velocity": {"attr": "u", "unit": "m/s"},
                "density": {"attr": "rho", "unit": "kg/m^3"},
                "area": {"attr": "A", "unit": "m^2"}
            },
            metrics_def=METRICS,
            geometry_callback=save_1d_geometry,
            summary_callback=compute_summary_stats
        )

    def set_geometry(self, z: np.ndarray, A: np.ndarray, P: np.ndarray, P_wetted: np.ndarray,
                     A_propellant: np.ndarray, A_casing: np.ndarray):
        """
        Interpolates external geometry onto the grid.
        """
        ng = self.grid.ng
        target_dtype = self.cfg.dtype

        # Use cart_coords[2] (Z-axis) for interpolation
        z_grid = self.grid.cart_coords[2]

        self.state.A = np.interp(z_grid, z, A).astype(target_dtype)
        self.state.P = np.interp(z_grid, z, P).astype(target_dtype)
        self.state.P_wetted = np.interp(z_grid, z, P_wetted).astype(target_dtype)

        self.state.A_propellant = np.interp(z_grid, z, A_propellant).astype(target_dtype)
        self.state.A_casing = np.interp(z_grid, z, A_casing).astype(target_dtype)

        # Assign values to ghost cells
        geom_arrays = [self.state.A, self.state.P, self.state.P_wetted, self.state.A_propellant, self.state.A_casing]
        for arr in geom_arrays:
            arr[:ng] = arr[ng]
            arr[-ng:] = arr[-ng - 1]

        # Calculate gradients using grid.dx[2] (Axial spacing)
        dz = self.grid.dx[2]
        self.state.dAdz[1:-1] = (self.state.A[2:] - self.state.A[:-2]) / (2 * dz)
        self.state.dAdz[0] = self.state.dAdz[1]
        self.state.dAdz[-1] = self.state.dAdz[-2]

    def initialize(self):
        """Sets initial conditions."""
        self.state.rho[:] = self.cfg.p_inf / (self.cfg.R * self.cfg.t_initial)
        self.state.p[:] = self.cfg.p_inf
        self.state.u[:] = self.cfg.u_initial
        self.state.br[:] = self.state.br + self.cfg.br_initial

        self.state.U[:] = primitive_to_conserved(
            self.state.rho, self.state.u, self.state.p, self.state.A, self.cfg.gamma
        )
        self.state.c[:] = np.sqrt(self.cfg.gamma * self.state.p / self.state.rho)

        # Log initial state
        self.recorder.save()

    def _compute_rhs(self, U_interior: np.ndarray) -> np.ndarray:
        U_full = self.state.U
        U_full[:, self.grid.interior] = U_interior

        U_full = apply_boundary_jit(
            U_full, self.state.A, self.cfg.gamma, self.cfg.R,
            self.cfg.p0_inlet, self.cfg.t0_inlet, self.cfg.p_inf, self.cfg.ng
        )

        self.state.rho, self.state.u, self.state.p, self.state.c = \
            compute_primitives_jit(U_full, self.state.A, self.cfg.gamma)

        self.state.br, self.state.eta = burn_rate(self.cfg, self.state, model="MP")

        alpha = np.abs(self.state.u) + self.state.c
        alpha = np.nan_to_num(alpha, nan=1000.0)

        F_hat = compute_numerical_flux_jit(
            U_full, self.state.A, self.state.rho, self.state.u,
            self.state.p, alpha, self.cfg.ng
        )

        S = source_jit(
            self.cfg.rho_p, self.cfg.Tf, self.state.br[self.grid.interior],
            self.cfg.R, self.cfg.gamma, self.state.p[self.grid.interior],
            self.state.P[self.grid.interior], self.state.dAdz[self.grid.interior]
        )

        # Compute flux gradient using grid.dx[2]
        dFdz = (F_hat[:, 1:] - F_hat[:, :-1]) / self.grid.dx[2]
        return S - dFdz

    def step(self) -> Tuple[float, float]:
        # Pass grid.dx[2] (scalar) to adaptive_timestep
        self.dt = adaptive_timestep(
            self.cfg.CFL, self.state.U, self.state.A, self.cfg.gamma,
            self.grid.dx[2], self.grid.ng,
            self.state.t, self.cfg.t_end)

        U_int = self.state.U[:, self.grid.interior]
        U_new = ssp_rk_3_3(U_int, self.dt, self._compute_rhs)
        self.state.U[:, self.grid.interior] = U_new

        self.state.t += self.dt

        # Log new state
        self.recorder.save()

        return self.dt, self.state.t

    def get_derived_quantities(self):
        data = compute_metrics(self.state, self.grid, self.cfg)
        data["scalars"]["time"] = self.state.t
        data["scalars"]["dt"] = self.dt
        return data

    def finalize(self):
        """Must be called at end of simulation."""
        self.recorder.finalize()

    def get_dataframe(self) -> pd.DataFrame:
        sl = self.grid.interior
        return pd.DataFrame({
            "z": self.grid.cart_coords[2][sl],  # Return Z as primary axis
            "rho": self.state.rho[sl],
            "u": self.state.u[sl],
            "p": self.state.p[sl],
            "Mach": self.state.u[sl] / (self.state.c[sl] + 1e-16),
            "Area": self.state.A[sl]
        })