from typing import Tuple
import pandas as pd

# Import Solver Functions
from .grid import *
from .config import *
from .numerics import *
from .geometry import *
from .boundary import *

from core.time_integrators import ssp_rk_3_3_low_storage as rk_step


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # 1. Build Grid
        self.grid = Grid3D(config)

        # 2. Allocate State
        self.state = State(dims=self.grid.dims, dtype=self.cfg.dtype)

        # 3. Initialize Solver
        self.initialize()


    def initialize(self):
        """
        Loads initial geometry (SDFs) and calculates initial properties.
        """
        filename_prop = self.cfg.file_prop
        filename_case = self.cfg.file_case

        prop = pv.read(filename_prop).scale(self.cfg.file_scale)
        case = pv.read(filename_case).scale(self.cfg.file_scale)
        prop = prop.clip_surface(case, invert=True)

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(case)
        self.grid.pv_grid.point_data["casing"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(prop)
        self.grid.pv_grid.point_data["propellant"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.state.phi = np.array(self.grid.pv_grid["propellant"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)
        self.state.casing = np.array(self.grid.pv_grid["casing"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)

        self._get_geometry()

        self.state.grad_mag = weno_godunov(self.state.phi, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)
        self.state.br = self.state.br + self.cfg.br_initial
        self.state.t = 0.0


    def _compute_rhs(self, phi_interior: np.ndarray) -> np.ndarray:

        # Map interior to full state
        phi_full = self.state.phi
        phi_full[self.grid.interior] = phi_interior

        phi_full = apply_boundary_conditions(phi_full, self.grid.ng)

        self.state.grad_mag = weno_godunov(phi_full, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)

        return -self.state.br[self.grid.interior] * self.state.grad_mag

    def _get_geometry(self):

        z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas, propellant_areas = compute_geometric_distributions(self.grid, self.state)

        self.state.x = z_coords
        self.state.A_flow = flow_areas
        self.state.P_propellant = perimeters
        self.state.P_wetted = hydraulic_perimeters
        self.state.A_casing = casing_areas


    def set_burn_rate(self, x: np.ndarray, br: np.ndarray):

        # Get the Z-coordinates of the 3D grid (Axial direction)
        z_ls = self.grid.polar_coords[2]

        # Interpolate the 1D burn rate onto the 3D grid
        self.state.br = np.interp(z_ls, x, br)

        return self.state.br


    def step(self) -> Tuple[float, float]:
        """
        Performs one time step:
        """
        # 1. Adaptive Timestep

        dt = adaptive_timestep(
            self.state.grad_mag, self.grid.dx, self.grid.polar_coords[0], self.grid.ng,
            self.cfg.CFL, self.cfg.t_end, self.state.br, self.state.t)

        # 2. Advance Level Set
        # level_set_step handles the RK3 integration and calls get_geometry internally
        phi_int = self.state.phi[self.grid.interior]
        phi_new = rk_step(phi_int, dt, self._compute_rhs)
        self.state.phi[self.grid.interior] = phi_new

        self._get_geometry()

        self.grid.pv_grid["propellant"] = self.state.phi.flatten(order='F')

        self.state.t += dt
        return dt, self.state.t

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the current geometric distributions along the rocket axis.
        """
        return pd.DataFrame({
            "x": self.state.x,
            "Area_Prop": self.state.A_flow,
            "Area_Case": self.state.A_casing,
            "Perimeter": self.state.P_propellant
        })