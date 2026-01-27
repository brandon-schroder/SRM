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

        # 1. Calculate spacing (Scalar calculation first)
        length = abs(config.bounds[1] - config.bounds[0])
        dz = length / config.n_cells

        # Format: [dx, dy, dz] or [dr, dtheta, dz]
        self.dx = [0.0, 0.0, dz]

        # 2. Generate Coordinate Arrays (including ghost cells)
        # We start at z_min - (ng * dz) and end at z_max + (ng * dz)
        # The 0.5 offset puts the node at the cell center.
        start = config.bounds[0] + 0.5 * dz - self.ng * dz
        end = config.bounds[1] - 0.5 * dz + self.ng * dz

        # DIMS: Keep as a list of 1 integer so solver slicing works
        self.dims = [0, 0, config.n_cells + 2 * self.ng]

        # Create the axial array
        z_array = np.linspace(start, end, self.dims[2], dtype=config.dtype)
        zeros = np.zeros_like(z_array)

        # 3. Define Standard Coordinate Systems (Shape: 3 x N)
        # Z is axial (index 2) to match 3D convention
        self.cart_coords = np.array([zeros, zeros, z_array], dtype=config.dtype)
        self.polar_coords = np.array([zeros, zeros, z_array], dtype=config.dtype)

        # 4. Define the "Interior" slice for easy access
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

        # Save Z coordinates (Axial)
        dset_z = g_geo.create_dataset("z", data=grid.cart_coords[2])
        dset_z.attrs["units"] = "m"

        # Save X/Y (Zeros)
        dset_x = g_geo.create_dataset("x", data=grid.cart_coords[0])
        dset_x.attrs["units"] = "m"
        dset_y = g_geo.create_dataset("y", data=grid.cart_coords[1])
        dset_y.attrs["units"] = "m"