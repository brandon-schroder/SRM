import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


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
    file_scale: float = 1.0 # Meters as default
    ng: int = 3

    CFL: float = 0.8
    t_end: float = 1.0
    t_start: float = 0.0

    # Initial & Boundary Conditions
    br_initial: float = 1.0E-6

    output_filename: str = "output.h5"
    vtk_dir: str = "vtk_output"


@dataclass
class State:
    """
    Holds the evolving solution variables.
    """
    dims: tuple[int, int, int]
    t: float = 0.0

    # Arrays
    phi: np.ndarray = field(init=False)
    casing: np.ndarray = field(init=False)
    grad_mag: np.ndarray = field(init=False)
    A_propellant: np.ndarray = field(init=False)
    A_casing: np.ndarray = field(init=False)
    P_propellant: np.ndarray = field(init=False)
    P_wetted: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    br: np.ndarray = field(init=False)

    def __post_init__(self):

        self.phi = np.zeros(self.dims)
        self.casing = np.zeros(self.dims)
        self.grad_mag = np.zeros(self.dims)
        self.A_propellant = np.zeros(self.dims[2])
        self.A_casing = np.zeros(self.dims[2])
        self.P_propellant = np.zeros(self.dims[2])
        self.P_wetted = np.zeros(self.dims[2])
        self.x = np.zeros(self.dims[2])
        self.br = np.zeros(self.dims)
