from dataclasses import dataclass, field
import numpy as np
from typing import Any, Tuple


@dataclass
class IB_Solver:
    params: 'IB_Solver.Parameters' = None
    grid: 'IB_Solver.Grid' = None
    states: 'IB_Solver.States' = None

    @dataclass
    class Parameters:
        size: int
        bounds: Tuple[float]
        ng: int = 3

        CFL: float = 0.6
        t_end: float = 1.0
        t_start: float = 0.0

        u_initial: float = 1.0
        t_initial: float = 288.15

        p_inf: float = 100.0E3
        t0_inlet: float = 288.15
        p0_inlet: float = 150.0e3
        R: float = 287.0
        gamma: float = 1.4

        rho_p: float = 1600
        Tf: float = 2888


    @dataclass
    class Grid:
        dx: Tuple[float]
        dims: Tuple[int]
        cart_coords: Tuple[np.ndarray]
        ng: int = 3

        # Solver helpers (auto-built)
        interior: Any = field(default=None, init=False, repr=False)
        L: Any = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            self.interior = np.s_[self.ng:-self.ng]
            from internal_ballistics_solver import make_L_variable
            self.L = make_L_variable(np.zeros((3, self.dims[0])))


    @dataclass
    class States:
        t: float = 0.0
        gamma: float = 1.4
        R: float = 287.0

        U: np.ndarray = field(init=False)

        rho: np.ndarray = field(init=False)
        p: np.ndarray = field(init=False)
        u: np.ndarray = field(init=False)
        e_T: np.ndarray = field(init=False)
        c: np.ndarray = field(init=False)

        A: np.ndarray = field(init=False)
        dAdz: np.ndarray = field(init=False)
        P: np.ndarray = field(init=False)

        rho_p: float = field(init=False)
        Tf: float = field(init=False)
        br: float = 1.0E-6

