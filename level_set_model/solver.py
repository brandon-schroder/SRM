import os
import json
import pandas as pd
import numpy as np
from typing import Tuple

from .grid import *
from .config import *
from .numerics import *
from .geometry import *
from .boundary import *
from .postprocess import *

from core.time_integrators import ssp_rk_3_3_low_storage as rk_step
from core.logger import SimulationRecorder


class LSSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # 1. Build Grid
        self.grid = Grid3D(config)

        # 2. Allocate State
        self.state = State(dims=self.grid.dims, dtype=self.cfg.dtype)

        # 3. Initialize Solver State Variables
        self.step_count = 0
        self.dt = 0.0
        self.vtk_times = []

        # 4. Initialize Recorder
        self.recorder = SimulationRecorder(
            solver=self,
            state_map={
                # 1D Geometric Distributions
                "x": {"attr": "x", "unit": "m"},
                "A_flow": {"attr": "A_flow", "unit": "m^2"},
                "A_casing": {"attr": "A_casing", "unit": "m^2"},
                "A_propellant": {"attr": "A_propellant", "unit": "m^2"},
                "P_propellant": {"attr": "P_propellant", "unit": "m"},
                "P_wetted": {"attr": "P_wetted", "unit": "m"},
            },
            metrics_def=METRICS,
            geometry_callback=save_3d_geometry,  # Callback from .grid
            summary_callback=compute_summary_stats
        )

        # 5. VTK Output Setup
        self.vtk_dir = getattr(self.cfg, "vtk_dir", "vtk_output")
        if not os.path.exists(self.vtk_dir):
            os.makedirs(self.vtk_dir)

        # 6. Initialize Solver
        self.initialize()

    def initialize(self):
        filename_prop = self.cfg.file_prop
        filename_case = self.cfg.file_case

        prop = pv.read(filename_prop).scale(self.cfg.file_scale)
        case = pv.read(filename_case).scale(self.cfg.file_scale)
        prop = prop.clip_surface(case, invert=True)

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(case)
        self.grid.pv_grid.point_data["casing"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.grid.pv_grid = self.grid.pv_grid.compute_implicit_distance(prop)
        self.grid.pv_grid.point_data["propellant"] = self.grid.pv_grid.point_data["implicit_distance"]

        self.state.phi = np.array(self.grid.pv_grid["propellant"].reshape(self.grid.dims, order='F'),
                                  dtype=self.cfg.dtype)
        self.state.casing = np.array(self.grid.pv_grid["casing"].reshape(self.grid.dims, order='F'),
                                     dtype=self.cfg.dtype)

        self._get_geometry()

        self.state.grad_mag = weno_godunov(self.state.phi, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)
        self.state.br = self.state.br + self.cfg.br_initial
        self.state.t = 0.0

        # Save initial state
        self.recorder.save()
        self._save_vtk()

    def _compute_rhs(self, phi_interior: np.ndarray) -> np.ndarray:
        phi_full = self.state.phi
        phi_full[self.grid.interior] = phi_interior
        phi_full = apply_boundary_conditions(phi_full, self.grid.ng)

        self.state.grad_mag = weno_godunov(phi_full, self.grid.dx, self.grid.polar_coords[0], self.grid.ng)

        return -self.state.br[self.grid.interior] * self.state.grad_mag

    def _get_geometry(self):
        z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas, propellant_areas = compute_geometric_distributions(
            self.grid, self.state)

        self.state.x = z_coords
        self.state.P_propellant = perimeters
        self.state.P_wetted = hydraulic_perimeters
        self.state.A_flow = flow_areas
        self.state.A_casing = casing_areas
        self.state.A_propellant = propellant_areas

    def set_burn_rate(self, x: np.ndarray, br: np.ndarray):
        z_ls = self.grid.polar_coords[2]
        self.state.br = np.interp(z_ls, x, br)
        return self.state.br

    def get_derived_quantities(self):
        """
        Called by SimulationRecorder to fetch derived metrics for logging.
        """
        data = compute_metrics(self.state, self.grid, self.cfg)
        data["scalars"]["time"] = self.state.t
        return data

    def step(self) -> Tuple[float, float]:
        dt = adaptive_timestep(
            self.state.grad_mag, self.grid.dx, self.grid.polar_coords[0], self.grid.ng,
            self.cfg.CFL, self.cfg.t_end, self.state.br, self.state.t)

        phi_int = self.state.phi[self.grid.interior]
        phi_new = rk_step(phi_int, dt, self._compute_rhs)
        self.state.phi[self.grid.interior] = phi_new

        self._get_geometry()

        self.state.t += dt
        self.step_count += 1

        self.recorder.save()
        self._save_vtk()

        return dt, self.state.t

    def _save_vtk(self):
        """
        Handles 3D visualization output separately from HDF5 logging.
        """
        int_vtk = getattr(self.cfg, 'log_interval_vtk', 100)

        if self.step_count % int_vtk == 0:
            vtk_name = f"step_{self.step_count:05d}.vtk"
            vtk_path = os.path.join(self.vtk_dir, vtk_name)

            self.vtk_times.append(self.state.t)

            self.grid.pv_grid["propellant"] = self.state.phi.flatten(order='F')
            if hasattr(self.state, 'br'):
                self.grid.pv_grid["burn_rate"] = self.state.br.flatten(order='F')

            self.grid.pv_grid.save(vtk_path)

    def finalize(self):
        self.recorder.finalize()

        # Generate VTK Series File
        series_path = os.path.join(self.vtk_dir, "results.vtk.series")
        interval = getattr(self.cfg, 'log_interval_vtk', 100)

        series_data = {
            "file-series-version": "1.0",
            "files": [
                {"name": f"step_{i * interval:05d}.vtk", "time": t}
                for i, t in enumerate(self.vtk_times)
            ]
        }

        try:
            with open(series_path, 'w') as f:
                json.dump(series_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not create .vtk.series file: {e}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "x": self.state.x,
            "Area_Flow": self.state.A_flow,
            "Area_Case": self.state.A_casing,
            "Perimeter": self.state.P_propellant
        })