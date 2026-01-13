import h5py
import numpy as np
import os


class IBRecorder:
    def __init__(self, filename, solver, buffer_size=5000):
        self.filename = filename
        self.solver = solver
        self.buffer_size = buffer_size

        # Initialize Buffer
        self.buffer = {
            # Scalars
            "time": [], "p_head": [], "thrust": [], "isp": [], "mass_flow": [],
            # Fields (will store a list of arrays)
            "pressure": [], "velocity": [], "density": [], "mach": [], "area": []
        }

        # 1. Setup File
        if os.path.exists(filename): os.remove(filename)

        with h5py.File(self.filename, "w") as f:
            # --- A. Save Config (Metadata) ---
            g_conf = f.create_group("config")
            # Handle config being an object or a dict
            conf_dict = self.solver.cfg.__dict__ if hasattr(self.solver.cfg, '__dict__') else self.solver.cfg
            for k, v in conf_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    g_conf.attrs[k] = v

            # --- B. Create Datasets ---
            n_cells = self.solver.grid.n_cells + 2 * self.solver.grid.ng

            # 1D Time Series (Scalars)
            g_ts = f.create_group("timeseries")
            for name in ["time", "p_head", "thrust", "isp", "mass_flow"]:
                # Chunking is crucial for performance when resizing
                g_ts.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1000,))

            # 2D Fields (Spatial Distributions)
            g_fields = f.create_group("fields")
            for name in ["pressure", "velocity", "density", "mach", "area"]:
                # Chunk along time axis
                g_fields.create_dataset(name, shape=(0, n_cells), maxshape=(None, n_cells), chunks=(100, n_cells))

    def save(self):
        """Calculates values and adds them to the memory buffer."""
        s = self.solver.state

        # --- 1. Calculate Derived Quantities ---
        idx_exit = -1 - self.solver.grid.ng
        p_exit = s.p[idx_exit]
        u_exit = s.u[idx_exit]
        rho_exit = s.rho[idx_exit]
        A_exit = s.A[idx_exit]

        mdot = rho_exit * u_exit * A_exit
        thrust = mdot * u_exit + (p_exit - 101325.0) * A_exit
        isp = thrust / (mdot * 9.81) if mdot > 1e-9 else 0.0
        p_head = np.max(s.p)

        # --- 2. Append to Buffer (Fast RAM operation) ---
        self.buffer["time"].append(s.t)
        self.buffer["p_head"].append(p_head)
        self.buffer["thrust"].append(thrust)
        self.buffer["isp"].append(isp)
        self.buffer["mass_flow"].append(mdot)

        self.buffer["pressure"].append(s.p.copy())  # .copy() is essential!
        self.buffer["velocity"].append(s.u.copy())
        self.buffer["density"].append(s.rho.copy())
        self.buffer["area"].append(s.A.copy())
        self.buffer["mach"].append((s.u / (s.c + 1e-16)).copy())

        # --- 3. Flush if buffer is full ---
        if len(self.buffer["time"]) >= self.buffer_size:
            self._flush()

    def _flush(self):
        """Writes the buffer to disk."""
        n_new = len(self.buffer["time"])
        if n_new == 0: return

        with h5py.File(self.filename, "a") as f:
            # Helper for scalars
            for name in ["time", "p_head", "thrust", "isp", "mass_flow"]:
                dset = f[f"timeseries/{name}"]
                dset.resize((dset.shape[0] + n_new,))
                dset[-n_new:] = self.buffer[name]
                self.buffer[name] = []  # Clear buffer

            # Helper for fields
            for name in ["pressure", "velocity", "density", "mach", "area"]:
                dset = f[f"fields/{name}"]
                dset.resize((dset.shape[0] + n_new, dset.shape[1]))
                # Stack list of arrays into a 2D array
                dset[-n_new:, :] = np.stack(self.buffer[name])
                self.buffer[name] = []  # Clear buffer

    def finalize(self):
        """Call this once at the end to flush remaining data and calculate summaries."""
        self._flush()  # Write whatever is left in the buffer

        with h5py.File(self.filename, "r+") as f:
            t = f["timeseries/time"][:]
            F = f["timeseries/thrust"][:]
            p = f["timeseries/p_head"][:]

            if len(t) > 1:
                total_impulse = np.trapezoid(F, t)
                max_pressure = np.max(p)
            else:
                total_impulse = 0.0
                max_pressure = 0.0

            # Save Summary
            g_sum = f.create_group("summary")
            g_sum.attrs["total_impulse"] = total_impulse
            g_sum.attrs["max_pressure"] = max_pressure

            print(f"Recorded: Itot={total_impulse:.2f} Ns | P_max={max_pressure / 1e6:.2f} MPa")