import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Any
import pyvista as pv


@dataclass
class SimulationConfig:
    """
    Static configuration parameters.
    """

    n_periodics: int
    size: tuple[int, int, int]
    bounds: list[float]
    file_prop: str | Path
    file_case: str | Path
    ng: int = 3

    CFL: float = 0.8
    t_end: float = 1.0
    t_start: float = 0.0

    # Initial & Boundary Conditions
    br_initial: float = 1.0E-6


@dataclass
class State:
    """
    Holds the evolving solution variables.
    """
    dims: tuple[int, int, int]
    t: float = 0.0
    br: float = 0.0

    # Arrays
    phi: np.ndarray = field(init=False)
    casing: np.ndarray = field(init=False)
    grad_mag: np.ndarray = field(init=False)
    A_propellant: np.ndarray = field(init=False)
    A_casing: np.ndarray = field(init=False)
    P_propellant: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        phi = np.zeros(self.dims)
        casing = np.zeros(self.dims)
        grad_mag = np.zeros(self.dims)
        A_propellant = np.zeros(self.dims[2])
        A_casing = np.zeros(self.dims[2])
        P_propellant = np.zeros(self.dims[2])
        x = np.zeros(self.dims[2])
