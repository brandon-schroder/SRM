import h5py
import numpy as np
import os


class HDF5Logger:
    def __init__(self, filename, scalar_names, field_names, field_shape, buffer_size=5000, overwrite=True):
        """
        Generic HDF5 Logger.

        Args:
            filename (str): Path to the output .h5 file.
            scalar_names (list): List of strings for 0D values (e.g., 'time', 'thrust').
            field_names (list): List of strings for spatial arrays (e.g., 'pressure').
            field_shape (tuple or int): The shape of a single field snapshot (e.g., (n_cells,)).
            buffer_size (int): Number of steps to store in RAM before flushing to disk.
            overwrite (bool): If True, deletes existing file at filename.
        """
        self.filename = filename
        self.buffer_size = buffer_size
        self.scalar_names = scalar_names
        self.field_names = field_names

        # Ensure field_shape is a tuple
        if isinstance(field_shape, int):
            self.field_shape = (field_shape,)
        else:
            self.field_shape = tuple(field_shape)

        # Initialize Buffer
        self.buffer = {name: [] for name in scalar_names + field_names}

        # 1. Setup File
        if overwrite and os.path.exists(filename):
            os.remove(filename)

        with h5py.File(self.filename, "w") as f:
            # Create Groups
            f.create_group("config")
            f.create_group("summary")
            g_ts = f.create_group("timeseries")
            g_fields = f.create_group("fields")

            # --- Create Scalar Datasets (1D) ---
            for name in scalar_names:
                # Chunking is crucial for performance when resizing
                g_ts.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1000,))

            # --- Create Field Datasets (N+1 Dimensions) ---
            # Shape = (Time, Space...)
            max_shape = (None,) + self.field_shape
            # Heuristic chunking: 100 time steps x full spatial domain
            chunk_shape = (100,) + self.field_shape

            for name in field_names:
                g_fields.create_dataset(name, shape=(0,) + self.field_shape,
                                        maxshape=max_shape, chunks=chunk_shape)

    def save_config(self, config_obj):
        """Saves a configuration object or dictionary to the /config group."""
        with h5py.File(self.filename, "a") as f:
            g_conf = f["config"]

            # Handle object vs dict
            conf_dict = config_obj.__dict__ if hasattr(config_obj, '__dict__') else config_obj

            for k, v in conf_dict.items():
                # HDF5 attributes support limited types
                if isinstance(v, (int, float, str, bool)):
                    g_conf.attrs[k] = v
                elif v is None:
                    g_conf.attrs[k] = "None"

    def log_scalar(self, name, value):
        """Append a scalar value to the buffer."""
        if name in self.buffer:
            self.buffer[name].append(value)
        else:
            raise KeyError(f"Scalar '{name}' was not defined in init.")

    def log_field(self, name, array):
        """Append a spatial field to the buffer (automatically copies)."""
        if name in self.buffer:
            self.buffer[name].append(np.array(array, copy=True))
        else:
            raise KeyError(f"Field '{name}' was not defined in init.")

    def check_buffer(self):
        """Checks if buffer is full and flushes if necessary."""
        # Check length of the first scalar (usually 'time') or first field
        check_key = self.scalar_names[0] if self.scalar_names else self.field_names[0]

        if len(self.buffer[check_key]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Writes the buffer to disk and clears it."""
        # Determine number of new records based on the first key
        first_key = self.scalar_names[0] if self.scalar_names else self.field_names[0]
        n_new = len(self.buffer[first_key])

        if n_new == 0: return

        with h5py.File(self.filename, "a") as f:
            # 1. Write Scalars
            for name in self.scalar_names:
                dset = f[f"timeseries/{name}"]
                dset.resize((dset.shape[0] + n_new,))
                dset[-n_new:] = self.buffer[name]
                self.buffer[name] = []  # Clear buffer

            # 2. Write Fields
            for name in self.field_names:
                dset = f[f"fields/{name}"]
                # Resize the time axis (axis 0)
                current_len = dset.shape[0]
                dset.resize((current_len + n_new,) + self.field_shape)

                # Stack list of arrays into a (N_new, Space...) array
                dset[current_len:, ...] = np.stack(self.buffer[name])
                self.buffer[name] = []  # Clear buffer

    def save_summary(self, summary_dict):
        """Saves calculated summary stats to /summary attributes."""
        with h5py.File(self.filename, "a") as f:
            g_sum = f["summary"]
            for k, v in summary_dict.items():
                g_sum.attrs[k] = v

    def finalize(self):
        """Flushes remaining data."""
        self.flush()
