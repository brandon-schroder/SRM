import h5py
import numpy as np


class HDF5Logger:
    def __init__(self, filename, scalar_names, field_names, field_shape, buffer_size=100, units=None):
        self.filename = filename
        self.scalar_names = scalar_names
        self.field_names = field_names
        self.field_shape = field_shape
        self.buffer_size = buffer_size
        self.units = units or {}

        # In-memory buffers
        self.scalar_buffer = {name: [] for name in scalar_names}
        self.field_buffer = {name: [] for name in field_names}
        self.buffer_count = 0

        # Initialize file (truncate if exists)
        with h5py.File(self.filename, "w") as f:
            g_scalars = f.create_group("timeseries")
            g_fields = f.create_group("fields")

            for name in scalar_names:
                unit = self.units.get(name, "")
                self._create_dataset_with_units(g_scalars, name, (0,), (None,), unit)

            max_shape = (None,) + field_shape
            chunk_shape = (1,) + field_shape
            for name in field_names:
                unit = self.units.get(name, "")
                self._create_dataset_with_units(g_fields, name, (0,) + field_shape, max_shape, unit, chunks=chunk_shape)

    def _create_dataset_with_units(self, group, name, shape, maxshape, unit, chunks=None):
        dset = group.create_dataset(name, shape=shape, maxshape=maxshape, dtype='float32', chunks=chunks)
        if unit:
            dset.attrs["units"] = unit
        return dset

    def save_config(self, config_obj):
        with h5py.File(self.filename, "a") as f:
            for key, val in vars(config_obj).items():
                if isinstance(val, (int, float, str, bool)):
                    f.attrs[key] = val
                elif isinstance(val, list):
                    f.attrs[key] = str(val)

    def save_summary(self, summary_dict):
        with h5py.File(self.filename, "a") as f:
            for key, val in summary_dict.items():
                f.attrs[key] = val

    def log_scalar(self, name, value):
        if name in self.scalar_buffer:
            self.scalar_buffer[name].append(value)

    def log_field(self, name, value):
        if name in self.field_buffer:
            self.field_buffer[name].append(value)

    def check_buffer(self):
        self.buffer_count += 1
        if self.buffer_count >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer_count == 0:
            return

        with h5py.File(self.filename, "a") as f:
            g_scalars = f["timeseries"]
            for name in self.scalar_names:
                data = self.scalar_buffer[name]
                if not data: continue
                dset = g_scalars[name]
                n_current = dset.shape[0]
                dset.resize(n_current + len(data), axis=0)
                dset[n_current:] = data
                self.scalar_buffer[name] = []

            g_fields = f["fields"]
            for name in self.field_names:
                data = self.field_buffer[name]
                if not data: continue
                dset = g_fields[name]
                n_current = dset.shape[0]
                dset.resize(n_current + len(data), axis=0)
                dset[n_current:] = np.array(data)
                self.field_buffer[name] = []

        self.buffer_count = 0

    def finalize(self):
        self.flush()


class SimulationRecorder(HDF5Logger):
    """
    A generic recorder that configures itself via dictionaries and callbacks.
    Compatible with any solver that exposes a 'state' object.
    """

    def __init__(self, solver, state_map, metrics_def=None,
                 geometry_callback=None, summary_callback=None,
                 buffer_size=100_000):

        self.solver = solver
        self.cfg = getattr(solver, 'cfg', None)
        self.step_count = 0

        # Configuration
        self.state_map = state_map
        self.metrics_def = metrics_def or {"scalars": {}, "fields": {}}
        self.geometry_callback = geometry_callback
        self.summary_callback = summary_callback

        # 1. Gather Names
        pp_scalars = list(self.metrics_def.get("scalars", {}).keys())
        pp_fields = list(self.metrics_def.get("fields", {}).keys())
        state_vars = list(self.state_map.keys())

        all_scalars = ["time", "dt"] + pp_scalars
        all_fields = state_vars + pp_fields

        # 2. Gather Units
        units = {"time": "s", "dt": "s"}

        # A. Units from Metrics
        for cat in self.metrics_def.values():
            for name, meta in cat.items():
                units[name] = meta.get("unit", "")

        # B. Units from State Map
        for name, meta in self.state_map.items():
            units[name] = meta.get("unit", "")

        # 3. Determine Field Shape
        if hasattr(solver, "grid"):
            n_cells = getattr(solver.grid, "n_cells", 0)
            ng = getattr(solver.grid, "ng", 0)
            if n_cells > 0:
                field_shape = (n_cells + 2 * ng,)
            else:
                field_shape = (1,)
        else:
            field_shape = (1,)

        super().__init__(
            filename=self.cfg.output_filename if self.cfg else "output.h5",
            scalar_names=all_scalars,
            field_names=all_fields,
            field_shape=field_shape,
            buffer_size=buffer_size,
            units=units
        )

        # 4. Save Static Geometry & Config
        if self.geometry_callback:
            self.geometry_callback(self.filename, self.solver)

        if self.cfg:
            self.save_config(self.cfg)

    def save(self):
        # Interval check
        interval = getattr(self.cfg, 'log_interval', 1)
        if self.step_count % interval != 0:
            self.step_count += 1
            return

        # 1. Get Derived Data
        derived = {}
        if hasattr(self.solver, "get_derived_quantities"):
            derived = self.solver.get_derived_quantities()

        scalars = derived.get("scalars", {})
        derived_fields = derived.get("fields", {})

        # 2. Log Scalars
        # Explicitly log time/dt if they exist
        if hasattr(self.solver.state, "t"):
            self.log_scalar("time", self.solver.state.t)
        if hasattr(self.solver, "dt"):
            self.log_scalar("dt", self.solver.dt)

        for name, val in scalars.items():
            # [CRITICAL FIX] Prevent double logging of time/dt
            if name in ["time", "dt"]:
                continue
            self.log_scalar(name, val)

        # 3. Log Raw State Fields
        for name, meta in self.state_map.items():
            attr_name = meta["attr"]
            val = getattr(self.solver.state, attr_name, None)
            if val is not None:
                self.log_field(name, val)

        # 4. Log Derived Fields
        for name, val in derived_fields.items():
            self.log_field(name, val)

        self.check_buffer()
        self.step_count += 1

    def finalize(self):
        super().finalize()

        # Summary Stats
        if self.summary_callback:
            summary_data = self.summary_callback(self.filename)
            self.save_summary(summary_data)