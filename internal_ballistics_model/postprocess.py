import numpy as np
import h5py

METRICS = {
    "scalars": {
        "p_head": {"unit": "Pa"},
        "p_exit": {"unit": "Pa"},
        "thrust": {"unit": "N"},
        "isp": {"unit": "s"},
        "mass_flow": {"unit": "kg/s"},

        "res_rho": {"unit": "kg/m^3/s"},
        "res_mom": {"unit": "Pa/m"},
        "res_E":   {"unit": "Pa/s"}
    },
    "fields": {
        "mach": {"unit": ""}
    },
}


def compute_metrics(state, grid, cfg):
    p_inf = getattr(cfg, "p_inf", 101325.0)
    g0 = 9.80665

    idx_exit = -1 - grid.ng

    p_head = np.max(state.p)
    p_exit = state.p[idx_exit]
    u_exit = state.u[idx_exit]
    rho_exit = state.rho[idx_exit]
    area_exit = state.A[idx_exit]

    m_dot = rho_exit * u_exit * area_exit
    thrust = m_dot * u_exit + (p_exit - p_inf) * area_exit
    isp = thrust / (m_dot * g0) if m_dot > 1e-9 else 0.0

    mach = state.u / (state.c + 1e-16)

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

                    if "timeseries/thrust" in f:
                        F = f["timeseries/thrust"][:]
                        stats["total_impulse"] = np.trapezoid(F, x=t)

                    if "timeseries/p_head" in f:
                        p_head = f["timeseries/p_head"][:]
                        stats["max_p_head"] = np.max(p_head)
    except (OSError, KeyError):
        pass

    return stats