import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SimulationConfig:
    n_cells: int
    bounds: Tuple[float, float]
    ng: int = 3
    CFL: float = 0.6
    t_end: float = 1.0
    t_start: float = 0.0

    u_initial: float = 1.0
    t_initial: float = 288.15
    p_inf: float = 100.0E3
    t0_inlet: float = 288.15
    p0_inlet: float = 100.0e3
    br_initial: float = 1E-6
    inlet_bc_type: str = "reflective"
    outlet_bc_type: str = "characteristic"

    R: float = 287.0
    gamma: float = 1.4

    rho_p: float = 1600.0
    Tf: float = 2888.0

    a_coef: float = 0.000035
    n_exp: float = 0.36
    burn_model: str = "none"
    burn_rate_update_interval: int = 1

    output_filename: str = "internal_ballistics.h5"
    log_interval: int = 10
    dtype: np.dtype = np.float64


@dataclass
class FlowState:
    n_cells: int
    t: float = 0.0

    dtype: np.dtype = np.float64

    U: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)
    c: np.ndarray = field(init=False)

    A: np.ndarray = field(init=False)
    dAdz: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    P_wetted: np.ndarray = field(init=False)
    A_casing: np.ndarray = field(init=False)
    A_propellant: np.ndarray = field(init=False)

    br: np.ndarray = field(init=False)

    def __post_init__(self):
        shape = (self.n_cells,)
        self.U = np.zeros((3, self.n_cells), dtype=self.dtype)
        self.rho = np.zeros(shape, dtype=self.dtype)
        self.u = np.zeros(shape, dtype=self.dtype)
        self.p = np.zeros(shape, dtype=self.dtype)
        self.c = np.zeros(shape, dtype=self.dtype)

        self.A = np.zeros(shape, dtype=self.dtype)
        self.dAdz = np.zeros(shape, dtype=self.dtype)
        self.P = np.zeros(shape, dtype=self.dtype)
        self.P_wetted = np.zeros(shape, dtype=self.dtype)
        self.A_casing = np.zeros(shape, dtype=self.dtype)
        self.A_propellant = np.zeros(shape, dtype=self.dtype)

        self.br = np.zeros(shape, dtype=self.dtype)
        self.eta = np.zeros(shape, dtype=self.dtype)