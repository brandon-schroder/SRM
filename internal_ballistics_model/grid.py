import numpy as np

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