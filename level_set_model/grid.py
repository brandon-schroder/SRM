import numpy as np
import pyvista as pv

from .config import SimulationConfig

class Grid3D:
    def __init__(self, config: SimulationConfig):
        """
        Generates the 3D Periodic Computational Grid based on the configuration.
        """
        self.ng = config.ng
        self.size = config.size
        self.n_periodics = config.n_periodics
        self.bounds = config.bounds
        self.bounds[2] = 0
        self.bounds[3] = self.bounds[2] + 2.0 * np.pi / self.n_periodics

        # Cell widths
        r_min, r_max, theta_min, theta_max, z_min, z_max = self.bounds
        n_r, n_theta, n_z = self.size
        dr = abs(r_min - r_max) / n_r
        dz = abs(z_min - z_max) / n_z
        dtheta = abs(theta_min - theta_max) / n_theta
        dims_full = [n_r + 2 * self.ng, n_theta + 1, n_z + 2 * self.ng]

        # Full grid with ghost cells
        r_full = np.linspace(r_min + 0.5 * dr - self.ng * dr, r_max - 0.5 * dr + self.ng * dr, n_r + 2 * self.ng)
        z_full = np.linspace(z_min + 0.5 * dz - self.ng * dz, z_max - 0.5 * dz + self.ng * dz, n_z + 2 * self.ng)
        theta = np.linspace(theta_min + 0.5 * dtheta, theta_max + 0.5 * dtheta, n_theta + 1)

        # Meshgrid
        R_full, THETA_full, Z_full = np.meshgrid(r_full, theta, z_full, indexing='ij')
        X_full = R_full * np.cos(THETA_full)
        Y_full = R_full * np.sin(THETA_full)

        # Pyvista Grid
        grid_full = pv.StructuredGrid(X_full, Y_full, Z_full)

        self.pv_grid=grid_full
        self.dx=[dr, dtheta, dz]
        self.dims=dims_full
        self.cart_coords=np.array([X_full, Y_full, Z_full])
        self.polar_coords=np.array([R_full, THETA_full, Z_full])
        self.interior = np.s_[self.ng:-self.ng, :-1, self.ng:-self.ng]






