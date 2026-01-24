import numpy as np
import h5py
from core.logger import HDF5Logger


class IBRecorder(HDF5Logger):
    def __init__(self, solver, buffer_size=100_000):
        """
        Internal Ballistics Recorder.
        Calculates rocket performance metrics and logs them to HDF5.
        """
        self.solver = solver
        self.cfg = solver.cfg

        # 1. Define Variables to Track
        scalar_names = ["time", "dt", "p_head", "thrust", "isp", "mass_flow"]
        field_names = ["pressure", "velocity", "density", "mach", "area"]

        # 2. Determine Field Shape
        n_cells_total = solver.grid.n_cells + 2 * solver.grid.ng
        field_shape = (n_cells_total,)

        # 3. Initialize Base Logger
        # We now trust the config to have this field
        filename = self.cfg.output_filename

        super().__init__(filename, scalar_names, field_names, field_shape, buffer_size)

        # 4. Save Configuration Metadata
        self.save_config(self.cfg)

        # 5. Initialize Step Counter
        self.step_count = 0

    def save(self):
        """
        Calculates current performance metrics and pushes data to the logger buffer.
        Respects log_interval configuration.
        """
        # Check if we should log this step
        interval = getattr(self.cfg, 'log_interval', 1)
        if self.step_count % interval != 0:
            self.step_count += 1
            return

        s = self.solver.state
        grid = self.solver.grid

        # --- Physics Calculations ---

        # Use indices to skip ghost cells for boundary evaluation
        idx_head = grid.ng
        idx_exit = -1 - grid.ng

        p_exit = s.p[idx_exit]
        u_exit = s.u[idx_exit]
        rho_exit = s.rho[idx_exit]
        area_exit = s.A[idx_exit]
        p_head = s.p[idx_head]

        # 1. Mass Flow Rate
        m_dot = rho_exit * u_exit * area_exit

        # 2. Thrust (F = m_dot * u_e + (p_e - p_inf) * A_e)
        p_inf = getattr(self.cfg, "p_inf", 101325.0)
        thrust = m_dot * u_exit + (p_exit - p_inf) * area_exit

        # 3. Isp
        g0 = 9.80665
        isp = thrust / (m_dot * g0) if m_dot > 1e-9 else 0.0

        # 4. Mach Number (Derived Field)
        mach = s.u / (s.c + 1e-16)

        # --- Log Data ---
        self.log_scalar("time", s.t)
        self.log_scalar("dt", getattr(self.solver, 'dt', 0.0))
        self.log_scalar("p_head", p_head)
        self.log_scalar("thrust", thrust)
        self.log_scalar("isp", isp)
        self.log_scalar("mass_flow", m_dot)

        # Fields
        self.log_field("pressure", s.p)
        self.log_field("velocity", s.u)
        self.log_field("density", s.rho)
        self.log_field("mach", mach)
        self.log_field("area", s.A)

        # Handle buffering/flushing
        self.check_buffer()

        # Increment step count
        self.step_count += 1

    def finalize(self):
        super().finalize()

        with h5py.File(self.filename, "r") as f:
            # Check if datasets exist and have data before reading
            if "timeseries/time" in f and f["timeseries/time"].shape[0] > 1:
                t = f["timeseries/time"][:]
                F = f["timeseries/thrust"][:]
                p_head = f["timeseries/p_head"][:]

                total_impulse = np.trapezoid(F, x=t)
                max_p_head = np.max(p_head)
            else:
                total_impulse = 0.0
                max_p_head = 0.0

        summary = {
            "total_impulse": total_impulse,
            "max_p_head": max_p_head
        }
        self.save_summary(summary)