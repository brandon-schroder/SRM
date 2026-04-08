from .grid import *
from .boundary import *
from .burn_rate import *
from .numerics import *
from .config import *

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

    def step(self):
        self.dt = adaptive_timestep(
            self.cfg.CFL, self.state.u, self.state.c,
            self.grid.dx[2], self.grid.ng, self.state.t, self.cfg.t_end)

        if self.step_count % self.cfg.burn_rate_update_interval == 0:
            self.state.br = burn_rate_model(self.cfg, self.state, model_flag=self.burn_model_flag)

        self.integrator.step(self.state.U[:, self.grid.interior], self.dt, self._compute_rhs)

        self.state.t += self.dt
        self.step_count += 1

        return self.dt, self.state.t
