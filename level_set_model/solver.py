import pandas as pd
from typing import Tuple

from .grid import *
from .config import *
from .numerics import *
from .geometry import *
from .boundary import *
from .postprocess import *

from schemes.temporal_integration import SSPRK33LowStorage as rk_step
from utils.hdf5_logger import HDF5Recorder
from utils.vtk_logger import VTKRecorder


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        self.grid = Grid3D(config)
        self.state = State(dims=self.grid.dims, dtype=self.cfg.dtype)

        self.bc_flag = BCType[self.cfg.bc_type.upper()].value
        self.step_count = 0
        self.dt = 0.0

        interior_shape = self.state.phi[self.grid.interior].shape
        self.integrator = rk_step(shape=interior_shape, dtype=self.cfg.dtype)

        # 1D HDF5 Data Recorder
        self.hdf5_recorder = None
        if getattr(self.cfg, "log_interval", 0) and self.cfg.log_interval > 0:
            self.hdf5_recorder = HDF5Recorder(
                solver=self,
                state_map={
                    "x": {"attr": "x", "unit": "m"},
                    "A_flow": {"attr": "A_flow", "unit": "m^2"},
                    "A_casing": {"attr": "A_casing", "unit": "m^2"},
                    "A_propellant": {"attr": "A_propellant", "unit": "m^2"},
                    "P_propellant": {"attr": "P_propellant", "unit": "m"},
                    "P_wetted": {"attr": "P_wetted", "unit": "m"},
                },
                metrics_def=METRICS,
                geometry_callback=save_3d_geometry,
                summary_callback=compute_summary_stats
            )

        # 3D VTK Background Recorder
        self.vtk_recorder = None
        if getattr(self.cfg, "vtk_interval", 0) and self.cfg.vtk_interval > 0:
            self.vtk_recorder = VTKRecorder(
                solver=self,
                state_map={
                    "propellant": "phi",
                    "casing": "casing"
                },
                data_callback=prepare_vtk_data
            )

    def initialize(self):
        filename_prop = self.cfg.file_prop
        filename_case = self.cfg.file_case

        # Fast initialization via mesh cleaning and decimation
        prop = pv.read(filename_prop).scale(self.cfg.file_scale).clean().decimate(0.8)
        case = pv.read(filename_case).scale(self.cfg.file_scale).clean().decimate(0.8)

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(case)
        self.grid.pv_grid.point_data["casing"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(prop)
        self.grid.pv_grid.point_data["propellant"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.state.phi = np.array(self.grid.pv_grid["propellant"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)
        self.state.casing = np.array(self.grid.pv_grid["casing"].reshape(self.grid.dims, order='F'), dtype=self.cfg.dtype)

        self._get_geometry()

        self.state.br = self.state.br + self.cfg.br_initial
        self.state.t = 0.0

        # Run initial save for both loggers
        if self.hdf5_recorder:
            self.hdf5_recorder.save()
        if self.vtk_recorder:
            self.vtk_recorder.save()

    def _compute_rhs(self, phi_interior: np.ndarray) -> np.ndarray:

        self.state.phi[self.grid.interior] = phi_interior
        self.state.phi = apply_boundary_conditions(self.state.phi, self.grid.ng, self.bc_flag)

        self.state.grad_mag = weno_godunov(self.state.phi, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)

        return -self.state.br[self.grid.interior] * self.state.grad_mag

    def _get_geometry(self):
        z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas, propellant_areas = \
            compute_geometric_distributions(self.grid, self.state)

        self.state.x = z_coords
        self.state.P_propellant = perimeters
        self.state.P_wetted = hydraulic_perimeters
        self.state.A_flow = flow_areas
        self.state.A_casing = casing_areas
        self.state.A_propellant = propellant_areas

    def set_burn_rate(self, x: np.ndarray, br: np.ndarray):
        z_ls = self.grid.polar_coords[2]
        self.state.br[:] = np.interp(z_ls, x, br)
        return self.state.br

    def step(self) -> Tuple[float, float]:
        dt = adaptive_timestep(
            self.grid.dx, self.grid.polar_coords[0], self.grid.ng,
            self.cfg.CFL, self.cfg.t_end, self.state.br, self.state.t)

        self.integrator.step(self.state.phi[self.grid.interior], dt, self._compute_rhs)

        self._get_geometry()

        self.state.t += dt
        self.step_count += 1

        if self.hdf5_recorder and (self.step_count % self.cfg.log_interval == 0 or self.state.t >= self.cfg.t_end):
            self.hdf5_recorder.save()

        if self.vtk_recorder and (self.step_count % self.cfg.vtk_interval == 0 or self.state.t >= self.cfg.t_end):
            self.vtk_recorder.save()

        return dt, self.state.t

    def get_derived_quantities(self):
        data = compute_metrics(self.state, self.grid, self.cfg)
        data["scalars"]["time"] = self.state.t
        return data

    def finalize(self):
        if self.hdf5_recorder:
            self.hdf5_recorder.finalize()
        if self.vtk_recorder:
            self.vtk_recorder.finalize()

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "x": self.state.x,
            "Area_Flow": self.state.A_flow,
            "Area_Case": self.state.A_casing,
            "Perimeter": self.state.P_propellant
        })