import h5py
import numpy as np
import subprocess
from collections import defaultdict


class HDF5Logger:
    def __init__(self, filename, config=None, units=None, buffer_size=1000, dtype=np.float32):
        self.filename = filename
        self.buffer = defaultdict(list)
        self.buffer_size = buffer_size
        self.units = units or {}
        self.default_dtype = dtype

        self.file = h5py.File(self.filename, "w", libver='latest')
        self.file.create_group("timeseries")
        self.file.create_group("fields")

        if config:
            self._save_config(self.file, config)

        self.file.swmr_mode = True

    def log(self, name, value):
        self.buffer[name].append(value)
        if len(self.buffer[name]) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer: return

        for name, values in self.buffer.items():
            if not values: continue

            data = np.array(values, dtype=self.default_dtype)
            group_name = "timeseries" if data.ndim == 1 else "fields"
            group = self.file[group_name]

            if name not in group:
                base_shape = data.shape[1:]
                max_shape = (None,) + base_shape

                chunk_shape = (self.buffer_size,) + base_shape if base_shape else (self.buffer_size,)

                dset = group.create_dataset(
                    name,
                    shape=(0,) + base_shape,
                    maxshape=max_shape,
                    dtype=self.default_dtype,
                    compression="lzf",
                    chunks=chunk_shape
                )
                if name in self.units:
                    dset.attrs["units"] = self.units[name]

            dset = group[name]
            n_curr = dset.shape[0]
            dset.resize(n_curr + len(values), axis=0)
            dset[n_curr:] = data

        self.buffer.clear()

        self.file.flush()

    def _save_config(self, f, config_obj):
        try:
            h = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
            f.attrs['git_commit'] = h.strip().decode('utf-8')
        except:
            f.attrs['git_commit'] = "unknown"

        for k, v in vars(config_obj).items():
            if isinstance(v, (int, float, str, bool, list)):
                f.attrs[k] = str(v) if isinstance(v, list) else v

    def finalize(self):
        self.flush()
        if hasattr(self, 'file') and self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


class SimulationRecorder(HDF5Logger):
    def __init__(self, solver, state_map, metrics_def=None, geometry_callback=None, summary_callback=None):
        self.solver = solver
        self.state_map = state_map
        self.metrics_def = metrics_def or {}
        self.summary_callback = summary_callback

        units = {k: v.get("unit", "") for k, v in self.state_map.items()}
        for cat in self.metrics_def.values():
            for k, v in cat.items(): units[k] = v.get("unit", "")
        units.update({"time": "s", "dt": "s"})

        super().__init__(
            filename=getattr(solver.cfg, 'output_filename', 'output.h5'),
            config=getattr(solver, 'cfg', None),
            units=units,
            dtype=getattr(solver.cfg, 'dtype', np.float32)
        )

        if geometry_callback:
            geometry_callback(self.filename, solver)

    def save(self):
        self.log("time", self.solver.state.t)
        self.log("dt", self.solver.dt)

        for name, meta in self.state_map.items():
            val = getattr(self.solver.state, meta["attr"], None)
            if val is not None:
                self.log(name, val)

        if hasattr(self.solver, "get_derived_quantities"):
            derived = self.solver.get_derived_quantities()
            for cat in derived.values():
                for name, val in cat.items():
                    if name not in ["time", "dt"]:
                        self.log(name, val)

    def finalize(self):
        super().finalize()
        if self.summary_callback:
            try:
                stats = self.summary_callback(self.filename)
                with h5py.File(self.filename, "a") as f:
                    for k, v in stats.items(): f.attrs[k] = v
            except Exception as e:
                print(f"Warning: Failed to write summary stats: {e}")