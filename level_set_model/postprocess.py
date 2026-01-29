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


def save_3d_geometry(filename, solver):
    """
    Callback function to save 3D geometry to HDF5.
    Saves the 1D axes (r, theta, z) to define the structured grid.
    """
    with h5py.File(filename, "a") as f:
        if "geometry" in f:
            return

        g_geo = f.create_group("geometry")
        grid = solver.grid

        # Extract 1D axes from the 3D polar coordinates to save space
        # R: varies along dim 0
        r_axis = grid.polar_coords[0][:, 0, 0]
        # Theta: varies along dim 1
        th_axis = grid.polar_coords[1][0, :, 0]
        # Z: varies along dim 2
        z_axis = grid.polar_coords[2][0, 0, :]

        dset_r = g_geo.create_dataset("r", data=r_axis)
        dset_r.attrs["units"] = "m"

        dset_th = g_geo.create_dataset("theta", data=th_axis)
        dset_th.attrs["units"] = "rad"

        dset_z = g_geo.create_dataset("z", data=z_axis)
        dset_z.attrs["units"] = "m"

        g_geo.attrs["dims"] = grid.dims


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