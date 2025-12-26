import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SimulationConfig:
    """Static configuration parameters."""
    n_cells: int
    bounds: Tuple[float, float]
    ng: int = 3
    CFL: float = 0.6
    t_end: float = 1.0
    t_start: float = 0.0

    # Initial & Boundary Conditions
    u_initial: float = 1.0
    t_initial: float = 288.15
    p_inf: float = 100.0E3
    t0_inlet: float = 288.15
    p0_inlet: float = 150.0e3
    br_initial: float = 1E-6

    # Properties
    R: float = 287.0
    gamma: float = 1.4

    # --- Propellant Physics ---
    rho_p: float = 1600.0
    Tf: float = 2888.0

    # Base Burn Law (r = a * P^n)
    a_coef: float = 0.000035
    n_exp: float = 0.36

    # Erosive Constants (Mukunda-Paul)
    erosive_mu: float = 8.85E-5
    erosive_K1: float = 0.023
    erosive_K2: float = 2.3714
    erosive_m: float = -0.125
    erosive_gth: float = 35.0


@dataclass
class FlowState:
    """
    Holds the evolving solution variables.
    """
    n_cells: int
    t: float = 0.0
    br: float = 0.0
    eta: float = 1.0

    # Arrays
    U: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)
    c: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)
    dAdz: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)

    def __post_init__(self):
        shape = (self.n_cells,)
        self.U = np.zeros((3, self.n_cells))
        self.rho = np.zeros(shape)
        self.u = np.zeros(shape)
        self.p = np.zeros(shape)
        self.c = np.zeros(shape)
        self.A = np.zeros(shape)
        self.dAdz = np.zeros(shape)
        self.P = np.zeros(shape)