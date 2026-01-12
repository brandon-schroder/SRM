from typing import Tuple
import pandas as pd

# Import Solver Functions
from .grid import *
from .config import *
from .numerics import *
from .geometry import *
from .boundary import *

from core.time_integrators import ssp_rk_3_3 as rk_step


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # 1. Build Grid
        self.grid = Grid3D(config)

        # 2. Allocate State
        self.state = State(dims=self.grid.dims)

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

        self.state.phi = np.array(self.grid.pv_grid["propellant"].reshape(self.grid.dims, order='F'))
        self.state.casing = np.array(self.grid.pv_grid["casing"].reshape(self.grid.dims, order='F'))

        self._get_geometry("propellant")
        self._get_geometry("casing")

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

    def _get_geometry(self, sdf="propellant"):

        phi = self.state.phi
        phi_cas = self.state.casing
        cart_coords = self.grid.cart_coords

        if sdf == "propellant":
            z_distances, areas, perimeters, wetted_perimeters = calculate_axial_distributions(phi, phi_cas, cart_coords)

            self.state.x = z_distances
            self.state.A_propellant = areas*self.cfg.n_periodics
            self.state.P_propellant = perimeters*self.cfg.n_periodics
            self.state.P_wetted = wetted_perimeters * self.cfg.n_periodics
        elif sdf == "casing":
            z_distances, areas, _, _ = calculate_axial_distributions(phi_cas, phi_cas, cart_coords)

            self.state.x = z_distances
            self.state.A_casing = areas*self.cfg.n_periodics

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

        self._get_geometry(sdf="propellant")

        self.grid.pv_grid["propellant"] = self.state.phi.flatten(order='F')

        self.state.t += dt
        return dt, self.state.t

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the current geometric distributions along the rocket axis.
        """
        return pd.DataFrame({
            "x": self.state.x,
            "Area_Prop": self.state.A_propellant,
            "Area_Case": self.state.A_casing,
            "Perimeter": self.state.P_propellant
        })