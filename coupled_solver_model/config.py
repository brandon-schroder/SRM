from dataclasses import dataclass
from pathlib import Path
import yaml

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

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CoupledConfig":

        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        glob = data.get('global', {})
        ib_data = data.get('internal_ballistics', {})
        ls_data = data.get('level_set', {})

        master_bounds = tuple(glob.get('master_bounds'))
        cfl = glob.get('cfl', 0.95)

        ib_data['bounds'] = (master_bounds[4], master_bounds[5])
        ib_data['CFL'] = cfl

        ls_data['bounds'] = master_bounds
        ls_data['CFL'] = cfl

        if 'file_prop' in ls_data:
            ls_data['file_prop'] = Path(ls_data['file_prop'])
        if 'file_case' in ls_data:
            ls_data['file_case'] = Path(ls_data['file_case'])

        ib_config = IBConfig(**ib_data)
        ls_config = LSConfig(**ls_data)

        return cls(
            ib_config=ib_config,
            ls_config=ls_config,
            t_end=glob.get('t_end', 1.0),
            coupling_scheme=glob.get('coupling_scheme', 'explicit')
        )