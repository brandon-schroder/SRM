import h5py
import numpy as np
import os


class HDF5Logger:
    def __init__(self, filename, scalar_names, field_names, field_shape, buffer_size=100, units=None):
        """
        Base HDF5 Logger.
        Handles buffering and writing of time-series data to disk.

        Args:
            units (dict, optional): Dictionary mapping variable names to unit strings.
                                    e.g., {"pressure": "Pa", "time": "s"}
        """
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
            # Create Groups
            g_scalars = f.create_group("timeseries")
            g_fields = f.create_group("fields")

            # Initialize Resizable Datasets for Scalars
            for name in scalar_names:
                unit = self.units.get(name, "")
                self._create_dataset_with_units(g_scalars, name, (0,), (None,), unit)

            # Initialize Resizable Datasets for Fields
            # Shape is (Time, Space...)
            max_shape = (None,) + field_shape
            chunk_shape = (1,) + field_shape
            for name in field_names:
                unit = self.units.get(name, "")
                self._create_dataset_with_units(g_fields, name, (0,) + field_shape, max_shape, unit, chunks=chunk_shape)

    def _create_dataset_with_units(self, group, name, shape, maxshape, unit, chunks=None):
        """Helper to create a dataset and attach the unit attribute."""
        dset = group.create_dataset(name, shape=shape, maxshape=maxshape, dtype='float32', chunks=chunks)
        if unit:
            dset.attrs["units"] = unit
        return dset

    def save_config(self, config_obj):
        """
        Saves configuration attributes to the root of the HDF5 file.
        """
        with h5py.File(self.filename, "a") as f:
            for key, val in vars(config_obj).items():
                # HDF5 attributes support limited types (int, float, string)
                if isinstance(val, (int, float, str, bool)):
                    f.attrs[key] = val
                elif isinstance(val, list):
                    # Convert list to string representation if needed
                    f.attrs[key] = str(val)

    def save_summary(self, summary_dict):
        """
        Saves summary statistics (e.g., total impulse) to attributes.
        """
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
            # Flush Scalars
            g_scalars = f["timeseries"]
            for name in self.scalar_names:
                data = self.scalar_buffer[name]
                if not data: continue

                dset = g_scalars[name]
                n_current = dset.shape[0]
                n_new = len(data)

                dset.resize(n_current + n_new, axis=0)
                dset[n_current:] = data

                self.scalar_buffer[name] = []  # Clear buffer

            # Flush Fields
            g_fields = f["fields"]
            for name in self.field_names:
                data = self.field_buffer[name]
                if not data: continue

                dset = g_fields[name]
                n_current = dset.shape[0]
                n_new = len(data)

                # Resize only the time axis (axis 0)
                dset.resize(n_current + n_new, axis=0)
                dset[n_current:] = np.array(data)

                self.field_buffer[name] = []  # Clear buffer

        self.buffer_count = 0

    def finalize(self):
        self.flush()