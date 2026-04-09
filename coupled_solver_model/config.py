from dataclasses import dataclass

from internal_ballistics_model import SimulationConfig as IBConfig
from level_set_model import SimulationConfig as LSConfig

@dataclass
class CoupledConfig:
    ib_config: IBConfig
    ls_config: LSConfig

    coupling_scheme: str = 'explicit'
    max_iter: int = 5
    rel_tol: float = 1e-3

    t_end: float = 1.0

    output_dir: str = "results"
    save_csv: bool = True
    save_vtk_3d: bool = False
    save_vtp_1d: bool = True
    save_frequency: int = 1