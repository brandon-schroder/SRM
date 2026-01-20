import os
import json
import numpy as np
from core.logger import HDF5Logger


class LSRecorder(HDF5Logger):
    def __init__(self, solver, buffer_size=100_000):
        """
        Level Set Recorder.
        Logs 1D geometric distributions to HDF5 and 3D Level Set grid to VTK.
        """
        self.solver = solver
        self.cfg = solver.cfg

        # 1. Define Variables to Track in HDF5
        scalar_names = ["time", "dt"]
        field_names =  ["x", "A_propellant", "A_casing", "P_propellant", "P_wetted"]

        # 2. Determine Field Shape
        nz = solver.grid.dims[2]
        field_shape = (nz - 1,) # geometry.py works on intermediate grid points

        # 3. Initialize Base Logger (HDF5)
        filename = getattr(self.cfg, "output_filename", "level_set.h5")
        super().__init__(filename, scalar_names, field_names, field_shape, buffer_size)

        # 4. Save Configuration Metadata
        self.save_config(self.cfg)

        # 5. Setup VTK Output and Time Tracking
        self.vtk_dir = getattr(self.cfg, "vtk_dir", "vtk_output")
        if not os.path.exists(self.vtk_dir):
            os.makedirs(self.vtk_dir)

        self.step_count = 0
        self.recorded_times = []  # <--- New: Track physical time for each step

    def save(self):
        """
        Calculates current performance metrics and pushes data to buffers.
        Saves the 3D grid state to VTK using PyVista.
        """
        s = self.solver.state

        # Track the time for this step
        self.recorded_times.append(float(s.t))

        # --- HDF5 Calculations & Logging ---

        self.log_scalar("time", s.t)
        self.log_scalar("dt", getattr(self.solver, 'dt', 0.0))

        self.log_field("x", s.x)
        self.log_field("A_propellant", s.A_propellant)
        self.log_field("A_casing", s.A_casing)
        self.log_field("P_propellant", s.P_propellant)
        self.log_field("P_wetted", s.P_wetted)

        self.check_buffer()

        # --- VTK Logging ---
        vtk_name = f"step_{self.step_count:05d}.vtk"
        vtk_path = os.path.join(self.vtk_dir, vtk_name)

        # Update and save the 3D grid
        self.solver.grid.pv_grid["propellant"] = s.phi.flatten(order='F')
        if hasattr(s, 'br'):
            self.solver.grid.pv_grid["burn_rate"] = s.br.flatten(order='F')

        self.solver.grid.pv_grid.save(vtk_path)

        self.step_count += 1

    def finalize(self):
        """
        Finalizes the HDF5 file and creates a .vtk.series file for ParaView.
        """
        # 1. Flush HDF5 buffers
        super().finalize()

        # 2. Create ParaView Time Series Metadata
        # This allows ParaView to display the actual simulation time instead of step numbers
        series_path = os.path.join(self.vtk_dir, "results.vtk.series")
        series_data = {
            "file-series-version": "1.0",
            "files": [
                {"name": f"step_{i:05d}.vtk", "time": t}
                for i, t in enumerate(self.recorded_times)
            ]
        }

        try:
            with open(series_path, 'w') as f:
                json.dump(series_data, f, indent=4)
            print(f"Created VTK time series manifest: {series_path}")
        except Exception as e:
            print(f"Warning: Could not create .vtk.series file: {e}")