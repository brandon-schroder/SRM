import h5py
import numpy as np
import subprocess
import os
from collections import defaultdict


class HDF5Logger:
    def __init__(self, filename, config=None, units=None, buffer_size=100, dtype=np.float32):
        self.filename = filename
        self.buffer = defaultdict(list)
        self.buffer_size = buffer_size
        self.units = units or {}
        self.default_dtype = dtype  # [Opt] Force float32 to save 50% disk space

        # [Opt] libver='latest' enables performance features like SWMR
        with h5py.File(self.filename, "w", libver='latest') as f:
            f.create_group("timeseries")
            f.create_group("fields")
            if config:
                self._save_config(f, config)

    def log(self, name, value):
        self.buffer[name].append(value)
        # Check size of the list, not the name string
        if len(self.buffer[name]) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer: return

        # [Safety] Use append mode
        with h5py.File(self.filename, "a", libver='latest') as f:
            for name, values in self.buffer.items():
                if not values: continue

                # Convert and Cast
                data = np.array(values, dtype=self.default_dtype)

                # Auto-Classify
                group_name = "timeseries" if data.ndim == 1 else "fields"
                group = f[group_name]

                # Lazy Create
                if name not in group:
                    base_shape = data.shape[1:]
                    max_shape = (None,) + base_shape

                    dset = group.create_dataset(
                        name, shape=(0,) + base_shape, maxshape=max_shape,
                        dtype=self.default_dtype, compression="gzip", chunks=True
                    )
                    if name in self.units:
                        dset.attrs["units"] = self.units[name]

                # Append
                dset = group[name]
                n_curr = dset.shape[0]
                dset.resize(n_curr + len(values), axis=0)
                dset[n_curr:] = data

        self.buffer.clear()

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

    # --- [Safety] Context Manager Support ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flushes data even if the simulation crashes
        self.finalize()


class SimulationRecorder(HDF5Logger):
    def __init__(self, solver, state_map, metrics_def=None, geometry_callback=None, summary_callback=None):
        self.solver = solver
        self.state_map = state_map
        self.metrics_def = metrics_def or {}
        self.summary_callback = summary_callback

        # Collect units
        units = {k: v.get("unit", "") for k, v in self.state_map.items()}
        for cat in self.metrics_def.values():
            for k, v in cat.items(): units[k] = v.get("unit", "")
        units.update({"time": "s", "dt": "s"})

        super().__init__(
            filename=getattr(solver.cfg, 'output_filename', 'output.h5'),
            config=getattr(solver, 'cfg', None),
            units=units,
            dtype=getattr(solver.cfg, 'dtype', np.float32)  # Inherit solver precision
        )

        if geometry_callback:
            geometry_callback(self.filename, solver)

    def save(self):
        # 1. Log Time & DT
        self.log("time", self.solver.state.t)
        self.log("dt", self.solver.dt)

        # 2. Log State Fields
        for name, meta in self.state_map.items():
            val = getattr(self.solver.state, meta["attr"], None)
            if val is not None:
                self.log(name, val)

        # 3. Log Metrics
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