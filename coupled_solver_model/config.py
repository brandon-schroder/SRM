from dataclasses import dataclass


from internal_ballistics_model import SimulationConfig as IBConfig
from level_set_model import SimulationConfig as LSConfig

@dataclass
class CoupledConfig:
    """Configuration for the coupled simulation."""
    ib_config: IBConfig
    ls_config: LSConfig
    t_end: float = 1.0