import numpy as np
import h5py

from .config import SimulationConfig

class Grid1D:
    def __init__(self, config: SimulationConfig):
        self.ng = config.ng
        self.n_cells = config.n_cells

        length = abs(config.bounds[1] - config.bounds[0])
        dz = length / config.n_cells

        self.dx = [0.0, 0.0, dz]

        start = config.bounds[0] + 0.5 * dz - self.ng * dz
        end = config.bounds[1] - 0.5 * dz + self.ng * dz

        self.dims = [0, 0, config.n_cells + 2 * self.ng]

        z_array = np.linspace(start, end, self.dims[2], dtype=config.dtype)
        zeros = np.zeros_like(z_array)

        self.cart_coords = np.array([zeros, zeros, z_array], dtype=config.dtype)
        self.polar_coords = np.array([zeros, zeros, z_array], dtype=config.dtype)

        self.interior = slice(self.ng, -self.ng)


def save_1d_geometry(filename, solver):
    with h5py.File(filename, "a") as f:
        if "geometry" in f:
            return

        g_geo = f.create_group("geometry")

        grid = solver.grid

        dset_z = g_geo.create_dataset("z", data=grid.cart_coords[2])
        dset_z.attrs["units"] = "m"

        dset_x = g_geo.create_dataset("x", data=grid.cart_coords[0])
        dset_x.attrs["units"] = "m"
        dset_y = g_geo.create_dataset("y", data=grid.cart_coords[1])
        dset_y.attrs["units"] = "m"