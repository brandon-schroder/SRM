import pandas as pd
import numpy as np
from typing import Tuple

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
        # Using index 2 to match your Z-axial axial consistency
        self.state = FlowState(n_cells=self.grid.dims[2], dtype=self.cfg.dtype)
        self.dt = 0.0

        self.residuals = {"res_rho": 0.0, "res_mom": 0.0, "res_E": 0.0}

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
        ng = self.grid.ng
        target_dtype = self.cfg.dtype
        z_grid = self.grid.cart_coords[2]

        self.state.A = np.interp(z_grid, z, A).astype(target_dtype)
        self.state.P = np.interp(z_grid, z, P).astype(target_dtype)
        self.state.P_wetted = np.interp(z_grid, z, P_wetted).astype(target_dtype)
        self.state.A_propellant = np.interp(z_grid, z, A_propellant).astype(target_dtype)
        self.state.A_casing = np.interp(z_grid, z, A_casing).astype(target_dtype)

        geom_arrays = [self.state.A, self.state.P, self.state.P_wetted, self.state.A_propellant, self.state.A_casing]
        for arr in geom_arrays:
            arr[:ng] = arr[ng]
            arr[-ng:] = arr[-ng - 1]

        # NOTE: dAdz is no longer pre-calculated here to ensure well-balanced source terms.

    def initialize(self):
        self.state.rho[:] = self.cfg.p_inf / (self.cfg.R * self.cfg.t_initial)
        self.state.p[:] = self.cfg.p_inf
        self.state.u[:] = self.cfg.u_initial
        self.state.br[:] = self.cfg.br_initial  # simplified initialization

        self.state.U[:] = primitives_to_conserved(self.state.rho, self.state.u, self.state.p, self.state.A,
                                                  self.cfg.gamma)
        self.state.c[:] = np.sqrt(self.cfg.gamma * self.state.p / self.state.rho)
        self.recorder.save()

    def _compute_rhs(self, U_interior: np.ndarray) -> np.ndarray:
        U_full = self.state.U
        U_full[:, self.grid.interior] = U_interior

        A_interfaces = 0.5 * (self.state.A[self.grid.ng - 1: -self.grid.ng] +
                              self.state.A[self.grid.ng: -self.grid.ng + 1])

        U_full = apply_boundary_jit(
            U_full, self.state.A, self.cfg.gamma, self.cfg.R,
            self.cfg.p0_inlet, self.cfg.t0_inlet, self.cfg.p_inf, self.cfg.ng
        )

        self.state.rho, self.state.u, self.state.p, self.state.c = \
            conserved_to_primitives(U_full, self.state.A, self.cfg.gamma)

        # Ensure burn rate calculation handles low pressure safely
        self.state.br, self.state.eta = burn_rate(self.cfg, self.state, model="none")

        F_hat = compute_numerical_flux(U_full, A_interfaces, self.state.rho, self.state.u, self.state.p, self.state.c,
                                       self.cfg.gamma, self.cfg.ng)

        S = source(self.cfg.rho_p, self.cfg.Tf, self.state.br[self.grid.interior], self.cfg.R, self.cfg.gamma,
                   self.state.p[self.grid.interior], self.state.P[self.grid.interior], A_interfaces, self.grid.dx[2])

        dFdz = (F_hat[:, 1:] - F_hat[:, :-1]) / self.grid.dx[2]
        return S - dFdz

    def step(self) -> Tuple[float, float]:
        self.dt = adaptive_timestep(
            self.cfg.CFL, self.state.U, self.state.A, self.cfg.gamma,
            self.grid.dx[2], self.grid.ng, self.state.t, self.cfg.t_end)

        U_int_old = self.state.U[:, self.grid.interior].copy()
        U_int_new = ssp_rk_3_3(U_int_old, self.dt, self._compute_rhs)

        if self.dt > 1e-16:
            rate_of_change = (U_int_new - U_int_old) / self.dt
            self.residuals["res_rho"] = np.sqrt(np.mean(rate_of_change[0] ** 2))
            self.residuals["res_mom"] = np.sqrt(np.mean(rate_of_change[1] ** 2))
            self.residuals["res_E"] = np.sqrt(np.mean(rate_of_change[2] ** 2))

        self.state.U[:, self.grid.interior] = U_int_new
        self.state.t += self.dt
        self.recorder.save()

        return self.dt, self.state.t

    def get_derived_quantities(self):
        data = compute_metrics(self.state, self.grid, self.cfg)
        data["scalars"]["time"] = self.state.t
        data["scalars"]["dt"] = self.dt
        data["residuals"] = self.residuals
        return data

    def finalize(self):
        self.recorder.finalize()

    def get_dataframe(self) -> pd.DataFrame:
        sl = self.grid.interior
        return pd.DataFrame({
            "z": self.grid.cart_coords[2][sl],
            "rho": self.state.rho[sl],
            "u": self.state.u[sl],
            "p": self.state.p[sl],
            "Mach": self.state.u[sl] / (self.state.c[sl] + 1e-16),
            "Area": self.state.A[sl]
        })