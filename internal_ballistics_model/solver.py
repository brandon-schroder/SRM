import pandas as pd

from .grid import *
from .boundary import *
from .burn_rate import *
from .numerics import *
from .config import *
from .postprocess import *

from utils.hdf5_logger import *
from schemes.temporal_integration import SSPRK33LowStorage as rk_step


class IBSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.grid = Grid1D(config)
        self.state = FlowState(n_cells=self.grid.dims[2], dtype=self.cfg.dtype)
        self.step_count = 0
        self.dt = 0.0

        interior_shape = self.state.U[:, self.grid.interior].shape
        self.integrator = rk_step(shape=interior_shape, dtype=self.cfg.dtype)

        self.A_interfaces = np.zeros(interior_shape[1] + 1, dtype=self.cfg.dtype)
        self.F_hat = np.zeros((3, interior_shape[1] + 1), dtype=self.cfg.dtype)
        self.S = np.zeros((3, interior_shape[1]), dtype=self.cfg.dtype)
        self.rhs_out = np.zeros((3, interior_shape[1]), dtype=self.cfg.dtype)

        self.inlet_bc_flag = BCType[self.cfg.inlet_bc_type.upper()].value
        self.outlet_bc_flag = BCType[self.cfg.outlet_bc_type.upper()].value

        self.burn_model_flag = BurnModel[self.cfg.burn_model.upper()].value

        self.hdf5_recorder = None
        if getattr(self.cfg, "log_interval", 0) and self.cfg.log_interval > 0:
            self.hdf5_recorder = HDF5Recorder(
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

    def set_geometry(self, z, A, P, P_wetted, A_propellant, A_casing):
        ng = self.grid.ng
        target_dtype = self.cfg.dtype
        z_grid = self.grid.cart_coords[2]

        self.state.A[:] = np.interp(z_grid, z, A).astype(target_dtype)
        self.state.P[:] = np.interp(z_grid, z, P).astype(target_dtype)
        self.state.P_wetted[:] = np.interp(z_grid, z, P_wetted).astype(target_dtype)
        self.state.A_propellant[:] = np.interp(z_grid, z, A_propellant).astype(target_dtype)
        self.state.A_casing[:] = np.interp(z_grid, z, A_casing).astype(target_dtype)

        geom_arrays = [self.state.A, self.state.P, self.state.P_wetted, self.state.A_propellant, self.state.A_casing]

        for arr in geom_arrays:
            arr[:ng] = arr[ng]
            arr[-ng:] = arr[-ng - 1]

    def initialize(self):
        self.state.rho[:] = self.cfg.p_inf / (self.cfg.R * self.cfg.t_initial)
        self.state.p[:] = self.cfg.p_inf
        self.state.u[:] = self.cfg.u_initial
        self.state.br[:] = self.cfg.br_initial

        self.state.U = primitives_to_conserved(
            self.state.rho, self.state.u, self.state.p,
            self.state.A, self.cfg.gamma, self.state.U)

        self.state.c[:] = np.sqrt(self.cfg.gamma * self.state.p / self.state.rho)

        if self.hdf5_recorder:
            self.hdf5_recorder.save()

    def _compute_rhs(self, U_interior: np.ndarray) -> np.ndarray:
        rhs_out = rhs_numerics(
            U_interior, self.state.U, self.state.A, self.cfg.gamma, self.cfg.R,
            self.cfg.p0_inlet, self.cfg.t0_inlet, self.cfg.p_inf, self.grid.ng,
            self.inlet_bc_flag, self.outlet_bc_flag, self.cfg.rho_p, self.cfg.Tf,
            self.state.br, self.state.P, self.grid.dx[2],
            self.state.rho, self.state.u, self.state.p, self.state.c,
            self.A_interfaces, self.F_hat, self.S, self.rhs_out
        )

        return rhs_out

    def step(self, save: bool = True) -> Tuple[float, float]:
        self.dt = adaptive_timestep(
            self.cfg.CFL, self.state.u, self.state.c,
            self.grid.dx[2], self.grid.ng, self.state.t, self.cfg.t_end)

        if self.step_count % self.cfg.burn_rate_update_interval == 0:
            self.state.br = burn_rate_model(self.cfg, self.state, model_flag=self.burn_model_flag)

        self.integrator.step(self.state.U[:, self.grid.interior], self.dt, self._compute_rhs)

        if save:
            if self.hdf5_recorder and (self.step_count % self.cfg.log_interval == 0 or self.state.t >= self.cfg.t_end):
                self.hdf5_recorder.save()

        self.state.t += self.dt
        self.step_count += 1

        return self.dt, self.state.t

    def get_derived_quantities(self):
        data = compute_metrics(self.state, self.grid, self.cfg)
        data["scalars"]["time"] = self.state.t
        data["scalars"]["dt"] = self.dt
        return data

    def finalize(self):
        if self.hdf5_recorder:  
            self.hdf5_recorder.finalize()

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