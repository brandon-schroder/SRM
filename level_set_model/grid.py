import numpy as np
import pyvista as pv
import h5py

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
        r_full = np.linspace(r_min + 0.5 * dr - self.ng * dr, r_max - 0.5 * dr + self.ng * dr, n_r + 2 * self.ng, dtype=config.dtype)
        z_full = np.linspace(z_min + 0.5 * dz - self.ng * dz, z_max - 0.5 * dz + self.ng * dz, n_z + 2 * self.ng, dtype=config.dtype)
        theta = np.linspace(theta_min + 0.5 * dtheta, theta_max + 0.5 * dtheta, n_theta + 1, dtype=config.dtype)

        # Meshgrid
        R_full, THETA_full, Z_full = np.meshgrid(r_full, theta, z_full, indexing='ij')
        X_full = R_full * np.cos(THETA_full)
        Y_full = R_full * np.sin(THETA_full)

        # Pyvista Grid
        grid_full = pv.StructuredGrid(X_full, Y_full, Z_full)

        self.pv_grid=grid_full
        self.dx=[dr, dtheta, dz]
        self.dims=dims_full
        self.cart_coords=np.array([X_full, Y_full, Z_full], dtype=config.dtype)
        self.polar_coords=np.array([R_full, THETA_full, Z_full], dtype=config.dtype)
        self.interior = np.s_[self.ng:-self.ng, :-1, self.ng:-self.ng]


def save_3d_geometry(filename, solver):
    """
    Callback function to save 3D geometry to HDF5.
    Saves the 1D axes (r, theta, z) to define the structured grid.
    """
    with h5py.File(filename, "a") as f:
        if "geometry" in f:
            return

        g_geo = f.create_group("geometry")
        grid = solver.grid

        # Extract 1D axes from the 3D polar coordinates to save space
        # R: varies along dim 0
        r_axis = grid.polar_coords[0][:, 0, 0]
        # Theta: varies along dim 1
        th_axis = grid.polar_coords[1][0, :, 0]
        # Z: varies along dim 2
        z_axis = grid.polar_coords[2][0, 0, :]

        dset_r = g_geo.create_dataset("r", data=r_axis)
        dset_r.attrs["units"] = "m"

        dset_th = g_geo.create_dataset("theta", data=th_axis)
        dset_th.attrs["units"] = "rad"

        dset_z = g_geo.create_dataset("z", data=z_axis)
        dset_z.attrs["units"] = "m"

        g_geo.attrs["dims"] = grid.dims






