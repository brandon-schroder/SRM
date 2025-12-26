import numpy as np
import pandas as pd
from typing import Callable, Tuple

# Assuming Grid1D is in grid.py
from _grid import Grid3D
from _structure import SimulationConfig, State

# Import JIT functions
from level_set_solver import *
from core.time_integrators import ssp_rk_3_3 as rk


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # 1. Build Grid
        self.grid = Grid3D(config)

        # 2. Allocate State
        self.state = State(self.grid.dims)