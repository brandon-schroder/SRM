import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor


class BackgroundVTKLogger:
    """Low-level threaded I/O engine for VTK files."""

    def __init__(self, output_dir, max_workers=3):
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.vtk_times = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            for f in glob.glob(os.path.join(self.output_dir, "*.vts")):
                try:
                    os.remove(f)
                except OSError:
                    pass

            for f in glob.glob(os.path.join(self.output_dir, "*.series")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    @staticmethod
    def _write_worker(grid, path, point_data):
        grid.clear_data()
        for name, array in point_data.items():
            grid.point_data[name] = array
        grid.save(path)

    def write(self, step_count, t, grid, point_data_dict):
        vtk_name = f"step_{step_count:05d}.vts"
        vtk_path = os.path.join(self.output_dir, vtk_name)
        self.vtk_times.append((vtk_name, t))

        # Memory isolation
        safe_data = {
            name: arr.flatten(order='F').copy()
            for name, arr in point_data_dict.items()
        }
        grid_isolated = grid.copy(deep=True)

        future = self.executor.submit(
            self._write_worker, grid_isolated, vtk_path, safe_data
        )
        self.futures.append(future)
        self.futures = [f for f in self.futures if not f.done()]

    def finalize(self):
        print("Waiting for background VTK writers to finish...")
        self.executor.shutdown(wait=True)
        print("All VTK files saved successfully.")

        series_path = os.path.join(self.output_dir, "results.vts.series")
        series_data = {
            "file-series-version": "1.0",
            "files": [{"name": name, "time": t} for name, t in self.vtk_times]
        }
        try:
            with open(series_path, 'w') as f:
                json.dump(series_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not create .vts.series file: {e}")


class VTKRecorder(BackgroundVTKLogger):
    """High-level hdf5_recorder that maps solver state to VTK arrays."""

    def __init__(self, solver, state_map=None, data_callback=None):
        output_dir = getattr(solver.cfg, "vtk_dir", "vtk_output")
        super().__init__(output_dir=output_dir)

        self.solver = solver
        self.state_map = state_map or {}
        self.data_callback = data_callback

    def save(self):
        """Pulls mapped attributes from the solver and writes to disk."""
        data_to_save = {}

        # 1. Automatically pull mapped arrays from the solver state
        for vtk_name, attr_name in self.state_map.items():
            data_to_save[vtk_name] = getattr(self.solver.state, attr_name)

        # 2. Append any derived arrays (like bounded propellant or 1D mapped pressure)
        if self.data_callback:
            data_to_save.update(self.data_callback(self.solver.state))

        # 3. Dispatch to the background thread
        self.write(
            step_count=self.solver.step_count,
            t=self.solver.state.t,
            grid=self.solver.grid.pv_grid,
            point_data_dict=data_to_save
        )