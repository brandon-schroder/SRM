from .grid import *
from .config import *
from .numerics import *
from .geometry import *
from .boundary import *

from schemes.temporal_integration import SSPRK33LowStorage as rk_step


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        self.grid = Grid3D(config)
        self.state = State(dims=self.grid.dims, dtype=self.cfg.dtype)

        self.bc_flag = BCType[self.cfg.bc_type.upper()].value
        self.step_count = 0
        self.dt = 0.0

        interior_shape = self.state.phi[self.grid.interior].shape
        self.integrator = rk_step(shape=interior_shape, dtype=self.cfg.dtype)


    def initialize(self):
        filename_prop = self.cfg.file_prop
        filename_case = self.cfg.file_case

        # Fast initialization via mesh cleaning and decimation
        prop = pv.read(filename_prop).scale(self.cfg.file_scale)#.clean().decimate(0.9)
        case = pv.read(filename_case).scale(self.cfg.file_scale)#.clean().decimate(0.9)

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(case)
        self.grid.pv_grid.point_data["casing"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(prop)
        self.grid.pv_grid.point_data["propellant"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.state.phi = np.array(self.grid.pv_grid["propellant"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)
        self.state.casing = np.array(self.grid.pv_grid["casing"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)

        self.grid.pv_grid.point_data.remove("implicit_distance")

        self._get_geometry()

        self.state.br = self.state.br + self.cfg.br_initial
        self.state.t = 0.0

    def _compute_rhs(self, phi_interior: np.ndarray) -> np.ndarray:

        self.state.phi[self.grid.interior] = phi_interior
        self.state.phi = apply_boundary_conditions(self.state.phi, self.grid.ng, self.bc_flag)

        self.state.grad_mag = weno_godunov(self.state.phi, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)

        return -self.state.br[self.grid.interior] * self.state.grad_mag

    def _get_geometry(self):
        z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas, propellant_areas = \
            compute_geometric_distributions(self.grid, self.state)

        self.state.x = z_coords
        self.state.P_propellant = perimeters
        self.state.P_wetted = hydraulic_perimeters
        self.state.A_flow = flow_areas
        self.state.A_casing = casing_areas
        self.state.A_propellant = propellant_areas

    def set_burn_rate(self, x: np.ndarray, br: np.ndarray):
        z_ls = self.grid.polar_coords[2]
        self.state.br[:] = np.interp(z_ls, x, br)
        return self.state.br

    def step(self):
        dt = adaptive_timestep(
            self.grid.dx, self.grid.polar_coords[0], self.grid.ng,
            self.cfg.CFL, self.cfg.t_end, self.state.br, self.state.t)

        self.integrator.step(self.state.phi[self.grid.interior], dt, self._compute_rhs)

        self._get_geometry()

        self.state.t += dt
        self.step_count += 1

        return dt, self.state.t
