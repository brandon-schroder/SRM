import numpy as np
import h5py

from .config import SimulationConfig

class Grid1D:
    def __init__(self, config: SimulationConfig):
        """
        Generates the 1D Computational Grid based on the configuration.
        """
        self.ng = config.ng
        self.n_cells = config.n_cells

        # 1. Calculate spacing
        length = abs(config.bounds[1] - config.bounds[0])
        self.dx = length / config.n_cells

        # 2. Generate Coordinate Arrays (including ghost cells)
        # We start at x_min - (ng * dx) and end at x_max + (ng * dx)
        # The 0.5 offset puts the node at the cell center.
        start = config.bounds[0] + 0.5 * self.dx - self.ng * self.dx
        end = config.bounds[1] - 0.5 * self.dx + self.ng * self.dx

        self.dims = (config.n_cells + 2 * self.ng,)
        self.x_coords = np.linspace(start, end, self.dims[0], dtype=config.dtype)

        # 3. Define the "Interior" slice for easy access
        # This allows solver.u[grid.interior] to skip ghost cells
        self.interior = slice(self.ng, -self.ng)


def save_1d_geometry(filename, solver):
    """
    Callback function to save 1D geometry to HDF5.
    """
    with h5py.File(filename, "a") as f:
        if "geometry" in f:
            return

        g_geo = f.create_group("geometry")

        # Access the grid from the solver
        grid = solver.grid

        # Save X coordinates
        dset_x = g_geo.create_dataset("x", data=grid.x_coords)
        dset_x.attrs["units"] = "m"

        # Save Zeros (For future 3D visualization compatibility)
        zeros = np.zeros_like(grid.x_coords)
        dset_z = g_geo.create_dataset("zeros", data=zeros)
        dset_z.attrs["units"] = "m"
