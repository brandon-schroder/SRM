import numpy as np
import h5py

# Single Source of Truth for Metric Definitions
METRICS = {
    "scalars": {
        "p_head": {"unit": "Pa"},
        "p_exit": {"unit": "Pa"},
        "thrust": {"unit": "N"},
        "isp": {"unit": "s"},
        "mass_flow": {"unit": "kg/s"},
    },
    "fields": {
        "mach": {"unit": ""}
    },
    "residuals": {
        "res_rho": {"unit": "kg/m^3/s"},
        "res_mom": {"unit": "Pa/m"},
        "res_E":   {"unit": "Pa/s"}
    }
}


def compute_metrics(state, grid, cfg):
    """
    Computes instantaneous performance metrics during the simulation.
    """
    # 1. Setup Constants
    p_inf = getattr(cfg, "p_inf", 101325.0)
    g0 = 9.80665

    # 2. Identify Indices (Boundary logic)
    idx_head = grid.ng
    idx_exit = -1 - grid.ng

    # 3. Extract Primitive Variables
    p_head = state.p[idx_head]
    p_exit = state.p[idx_exit]
    u_exit = state.u[idx_exit]
    rho_exit = state.rho[idx_exit]
    area_exit = state.A[idx_exit]

    # 4. Compute Mechanics
    m_dot = rho_exit * u_exit * area_exit
    thrust = m_dot * u_exit + (p_exit - p_inf) * area_exit
    isp = thrust / (m_dot * g0) if m_dot > 1e-9 else 0.0

    # Field Calculation
    mach = state.u / (state.c + 1e-16)

    # 5. Return Data
    return {
        "scalars": {
            "p_head": p_head,
            "p_exit": p_exit,
            "thrust": thrust,
            "isp": isp,
            "mass_flow": m_dot,
        },
        "fields": {
            "mach": mach
        }
    }


def compute_summary_stats(filename: str) -> dict:
    """
    Opens the finalized HDF5 file to compute global summary metrics.
    """
    stats = {
        "total_impulse": 0.0,
        "max_p_head": 0.0,
        "num_steps": 0
    }

    try:
        with h5py.File(filename, "r") as f:
            if "timeseries/time" in f:
                t_dset = f["timeseries/time"]

                if t_dset.shape[0] > 1:
                    t = t_dset[:]
                    stats["num_steps"] = len(t)

                    # Calculate Total Impulse
                    if "timeseries/thrust" in f:
                        F = f["timeseries/thrust"][:]
                        stats["total_impulse"] = np.trapezoid(F, x=t)

                    # Calculate Peak Pressure
                    if "timeseries/p_head" in f:
                        p_head = f["timeseries/p_head"][:]
                        stats["max_p_head"] = np.max(p_head)
    except (OSError, KeyError):
        pass

    return stats