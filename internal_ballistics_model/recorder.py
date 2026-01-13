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
        filename = getattr(self.cfg, "output_filename", "internal_ballistics.h5")

        super().__init__(filename, scalar_names, field_names, field_shape, buffer_size)

        # 4. Save Configuration Metadata
        self.save_config(self.cfg)

    def save(self):
        """
        Calculates current performance metrics and pushes data to the logger buffer.
        """
        s = self.solver.state
        grid = self.solver.grid

        # --- Physics Calculations ---

        # Use indices to skip ghost cells for boundary evaluation
        idx_head = grid.ng
        idx_exit = -1 - grid.ng

        # [FIX] Changed s.area to s.A to match FlowState in solver.py
        p_exit = s.p[idx_exit]
        u_exit = s.u[idx_exit]
        rho_exit = s.rho[idx_exit]
        area_exit = s.A[idx_exit]  # Corrected from .area
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
        # Using s.c (speed of sound) which is already computed in the solver
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
        self.log_field("area", s.A)  # Corrected from s.area

        # Handle buffering/flushing
        self.check_buffer()

    def finalize(self):
        super().finalize()

        with h5py.File(self.filename, "r") as f:
            t = f["timeseries/time"][:]
            F = f["timeseries/thrust"][:]
            p_head = f["timeseries/p_head"][:]

            if len(t) > 1:
                # Support both NumPy 2.0 and 1.x
                if hasattr(np, 'trapezoid'):
                    total_impulse = np.trapezoid(F, x=t)
                else:
                    total_impulse = np.trapz(F, x=t)
                max_p_head = np.max(p_head)
            else:
                total_impulse = 0.0
                max_p_head = 0.0

        summary = {
            "total_impulse": total_impulse,
            "max_p_head": max_p_head
        }
        self.save_summary(summary)