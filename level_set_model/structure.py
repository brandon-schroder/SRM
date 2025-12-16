from dataclasses import dataclass, field
import pyvista as pv
import numpy as np
from pathlib import Path
from typing import Any



@dataclass
class LS_Solver:
    params: 'LS_Solver.Parameters' = None
    grid: 'LS_Solver.Grid' = None
    states: 'LS_Solver.States' = None

    @dataclass
    class Parameters:
        n_periodics: int
        size: tuple[int, int, int]
        bounds: list[float]
        file_prop: str | Path
        file_case: str | Path
        ng: int = 3

        CFL: float = 0.8
        t_end: float = 1.0
        t_start: float = 0.0

        br_initial: float = 1.0E-6


    @dataclass
    class Grid:

        pv_grid: pv.StructuredGrid
        dx: tuple[float, float, float]
        dims: tuple[int, int, int]
        cart_coords: tuple[np.ndarray, np.ndarray, np.ndarray]
        polar_coords: tuple[np.ndarray, np.ndarray, np.ndarray]
        ng: int = 3

        # Solver helpers (auto-built)
        interior: Any = field(default=None, init=False, repr=False)
        L: Any = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            self.interior = np.s_[self.ng:-self.ng, :-1, self.ng:-self.ng]
            from level_set_solver import make_L_variable_br
            self.L = make_L_variable_br(np.ones(self.dims))


    @dataclass
    class States:
        t: float = 0.0

        phi: np.ndarray = field(init=False)
        casing: np.ndarray = field(init=False)
        grad_mag: np.ndarray = field(init=False)

        br: float = 1.0E-6

        A_propellant: np.ndarray = field(init=False)
        A_casing: np.ndarray = field(init=False)
        P_propellant: np.ndarray = field(init=False)
        x: np.ndarray = field(init=False)


