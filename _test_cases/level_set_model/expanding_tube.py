import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import sys
from pathlib import Path

# --- Path Management ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from level_set_model import SimulationConfig, LSSolver


def create_dummy_stls():
    """Create temporary STLs just to satisfy the LSSolver initializer."""
    pv.Cylinder(radius=0.035, height=0.1).save("dummy_prop.stl")
    pv.Cylinder(radius=0.036, height=0.1).save("dummy_case.stl")


def main():
    create_dummy_stls()

    # Test Parameters
    br_initial = 0.01  # Burn rate: 10 mm/s
    R0 = 0.015  # Initial bore radius: 15 mm
    R_case = 0.034  # Casing radius: 34 mm

    # Grid Bounds [r_min, r_max, th_min, th_max, z_min, z_max]
    bounds = (0.01, 0.035, None, None, 0.0, 0.1)

    config = SimulationConfig(
        n_periodics=11,  # Defines a sector (360/11 degrees)
        size=(50, 4, 20),  # Low Y-resolution needed for a purely radial test
        bounds=bounds,
        file_scale=1.0,
        file_prop=Path("dummy_prop.stl"),
        file_case=Path("dummy_case.stl"),
        ng=3,  # Ghost cells for WENO-5
        CFL=0.5,
        t_end=1.0,  # Run for 1 second (10mm of burning)
        br_initial=br_initial,
    )

    config.dtype = np.float64  # Use float64 for strict verification

    # 1. Initialize Solver (Loads dummy STLs)
    solver = LSSolver(config)

    # 2. Override with Exact Analytical Cylindrical SDF
    # ---------------------------------------------------------
    # In polar coords, the distance to a cylinder surface is simply r - R
    r_coords = solver.grid.polar_coords[0]

    # phi < 0 is the flow port (void), phi > 0 is the propellant
    solver.state.phi[:] = r_coords - R0
    solver.state.casing[:] = r_coords - R_case

    # Force the geometry engine to recalculate areas based on our exact SDF
    solver._get_geometry()

    # Tracking arrays
    time_hist = []
    r_num_hist = []
    r_exact_hist = []

    print("Running 1D Expanding Cylinder Advection Test...")

    # 3. Time Integration Loop
    # ---------------------------------------------------------
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            # The solver outputs the FULL 360-degree flow area.
            # We calculate the numerical radius from the area at the mid-point Z-slice.
            mid_z = config.size[2] // 2
            A_flow = solver.state.A_flow[mid_z]
            r_num = np.sqrt(A_flow / np.pi)

            # Calculate exact analytical radius
            r_exact = R0 + br_initial * current_time

            time_hist.append(current_time)
            r_num_hist.append(r_num)
            r_exact_hist.append(r_exact)

            if solver.step_count % 10 == 0:
                print(f"t={current_time:.3f}s "
                      f"| R_num={r_num * 1000:.3f} mm "
                      f"| R_exact={r_exact * 1000:.3f} mm "
                      f"| R_error={(r_num-r_exact) * 1000:.3f} mm")


    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Cleanup dummy files
    if os.path.exists("dummy_prop.stl"): os.remove("dummy_prop.stl")
    if os.path.exists("dummy_case.stl"): os.remove("dummy_case.stl")

    # 4. Plotting Results
    # ---------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Radius Comparison Plot
    ax[0].plot(time_hist, [r * 1000 for r in r_exact_hist], 'k-', lw=2, label="Analytical (Exact)")
    ax[0].plot(time_hist, [r * 1000 for r in r_num_hist], 'r--', lw=2, label="Numerical (LSSolver)")
    ax[0].set_ylabel("Bore Radius [mm]")
    ax[0].set_title("Level Set Advection: Expanding Core")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Error Plot
    error_mm = np.abs(np.array(r_num_hist) - np.array(r_exact_hist)) * 1000
    ax[1].plot(time_hist, error_mm, 'm-', lw=1.5)
    ax[1].set_ylabel("Absolute Error [mm]")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_title("Radius Tracking Error")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()