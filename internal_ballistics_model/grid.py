import numpy as np

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
        self.x_coords = np.linspace(start, end, self.dims[0])

        # 3. Define the "Interior" slice for easy access
        # This allows solver.u[grid.interior] to skip ghost cells
        self.interior = slice(self.ng, -self.ng)

    @property
    def x_interior(self):
        """Helper to get only the physical coordinates (no ghosts)."""
        return self.x_coords[self.interior]