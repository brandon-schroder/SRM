import numpy as np
import h5py

METRICS = {
    "scalars": {
        "time": {"unit": "s"},
        "max_burn_rate": {"unit": "m/s"},
        "propellant_volume": {"unit": "m^3"},
    },
    "fields": {
        # Derived fields can be added here
    }
}


def compute_metrics(state, grid, cfg):
    metrics = {
        "scalars": {},
        "fields": {}
    }

    if hasattr(state, 'br') and state.br.size > 0:
        metrics["scalars"]["max_burn_rate"] = np.max(state.br)
    else:
        metrics["scalars"]["max_burn_rate"] = 0.0

    if hasattr(state, 'phi'):
        is_propellant = state.phi < 0
        dr, dtheta, dz = grid.dx
        dV = grid.polar_coords[0] * dr * dtheta * dz
        metrics["scalars"]["propellant_volume"] = np.sum(dV[is_propellant])

    return metrics


def compute_summary_stats(filename: str) -> dict:
    stats = {
        "final_time": 0.0,
        "steps_recorded": 0
    }

    try:
        with h5py.File(filename, "r") as f:
            if "timeseries/time" in f:
                t = f["timeseries/time"][:]
                stats["final_time"] = t[-1] if len(t) > 0 else 0.0
                stats["steps_recorded"] = len(t)
    except (OSError, KeyError):
        pass

    return stats