import numpy as np
import h5py

METRICS = {
    "scalars": {
        "time": {"unit": "s"},
        "max_burn_rate": {"unit": "m/s"},
        "propellant_volume": {"unit": "m^3"},
        # Stability / Residual Metrics
        "res_eikonal_avg": {"unit": "-"},  # Mean deviation from |grad(phi)|=1
        "res_eikonal_max": {"unit": "-"},  # Max deviation (detects local kinks)
    },
    "fields": {
        # Derived fields can be added here
    }
}


def compute_metrics(state, grid, cfg):
    """
    Computes instantaneous performance metrics during the simulation.
    """
    metrics = {
        "scalars": {},
        "fields": {}
    }

    # 1. Physical Metrics
    if hasattr(state, 'br') and state.br.size > 0:
        metrics["scalars"]["max_burn_rate"] = np.max(state.br)
    else:
        metrics["scalars"]["max_burn_rate"] = 0.0

    # Propellant Volume Integration: sum( r * dr * dtheta * dz ) where phi < 0
    if hasattr(state, 'phi'):
        is_propellant = state.phi < 0
        dr, dtheta, dz = grid.dx
        dV = grid.polar_coords[0] * dr * dtheta * dz
        metrics["scalars"]["propellant_volume"] = np.sum(dV[is_propellant])

    # 2. Stability / Residual Metrics (Eikonal Constraint)
    # We measure how much |grad(phi)| deviates from 1.0
    if hasattr(state, 'grad_mag'):
        # Only evaluate on the interior to avoid Ghost Cell artifacts
        gm_int = state.grad_mag[grid.interior]

        # Calculate residual: | |grad(phi)| - 1 |
        eikonal_resid = np.abs(gm_int - 1.0)

        metrics["scalars"]["res_eikonal_avg"] = np.mean(eikonal_resid)
        metrics["scalars"]["res_eikonal_max"] = np.max(eikonal_resid)
    else:
        metrics["scalars"]["res_eikonal_avg"] = 0.0
        metrics["scalars"]["res_eikonal_max"] = 0.0

    return metrics


def compute_summary_stats(filename: str) -> dict:
    """
    Computes global summary metrics from the final HDF5 file.
    """
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