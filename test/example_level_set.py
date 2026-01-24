import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from level_set_model import SimulationConfig, LSSolver
from level_set_model.recorder import LSRecorder  # Import the new recorder


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    prop_file = Path("geometry/07R-SRM-Propellant.STL")
    case_file = Path("geometry/07R-SRM-Casing.STL")

    # Domain bounds: [r_min, r_max, theta_min, theta_max, z_min, z_max]
    bounds = [10.0 * 1e-3, 35.0 * 1e-3, None, None, 0.0 * 1e-3, 100.0 * 1e-3]
    br_initial = 10e-3

    x_ib = np.linspace(bounds[4], bounds[5], 100)
    br_ib = np.ones_like(x_ib) * br_initial

    config = SimulationConfig(
        n_periodics=11,
        size=(50, 40, 100),
        bounds=bounds,
        file_scale=1.0e-3,
        file_prop=prop_file,
        file_case=case_file,
        ng=3,
        CFL=0.8,
        t_end=0.1,
        br_initial=br_initial,
        # Added recorder-specific configuration attributes
        output_filename="level_set_results.h5",
        vtk_dir="vtk_output"
    )

    config.dtype = np.float32

    print(f"--- Initializing Level Set Simulation: {config.size} cells ---")

    # ---------------------------------------------------------
    # 2. Solver & Recorder Setup
    # ---------------------------------------------------------
    solver = LSSolver(config)

    print(f"Grid Memory: {solver.state.phi.nbytes / 1e6:.2f} MB")

    # Initialize the recorder with the solver instance
    recorder = LSRecorder(solver)

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    # Save initial state (t=0)
    recorder.save()

    import time

    t_start = time.time()

    while solver.state.t < config.t_end:
        # Perform one step
        dt, current_time = solver.step()

        # Update burn rate from the internal ballistics model
        solver.state.br = solver.set_burn_rate(x_ib, br_ib)

        # Record data to HDF5 (1D/0D) and VTK (3D)
        # This replaces the manual 'history' lists used previously
        recorder.save()

        # Monitor progress
        avg_phi = np.mean(solver.state.phi)
        print(f"t={current_time:.5f}s | dt={dt:.2e} | Avg Phi={avg_phi:.2e}")

    t_end = time.time()

    print(f"Simulation took {t_end - t_start:.2f} seconds.")

    # ---------------------------------------------------------
    # 4. Finalization & Post-Processing
    # ---------------------------------------------------------
    # Ensure all buffered HDF5 data is written to disk
    recorder.finalize()

    print("\nSimulation Complete. Data saved to HDF5 and VTK.")

    # ---------------------------------------------------------
    # 5. Post-Processing from HDF5
    # ---------------------------------------------------------
    h5_path = "level_set_results.h5"

    with h5py.File(h5_path, "r") as f:
        # 1. Load Time Series (0D) from the 'timeseries' group
        t_hist = f["timeseries/time"][:]

        # 2. Load Axial Distributions (1D) from the 'fields' group
        # Note: HDF5Logger stores these as (Time, Space)
        # We take the last time step [-1] for the spatial plot
        x_dist = f["fields/x"][-1, :]
        a_prop_dist = f["fields/A_flow"][-1, :]
        a_case_dist = f["fields/A_casing"][-1, :]

        # 3. Load Midpoint History
        # Calculate index for the middle of the grain along the Z-axis
        mid_idx = a_prop_dist.shape[0] // 2
        a_mid_hist = f["fields/A_flow"][:, mid_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Geometric Distributions along Axis (Final State)
    ax[0].plot(x_dist, a_prop_dist, 'b-', label='Propellant Area', linewidth=2)
    ax[0].plot(x_dist, a_case_dist, 'k--', label='Casing Area')
    ax[0].set_ylabel('Flow Area [m^2]')
    ax[0].set_xlabel('Axial Position [m]')
    ax[0].set_title('Final Geometry Distributions (from HDF5)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Burn History at Midpoint
    ax[1].plot(t_hist, a_mid_hist, 'r-')
    ax[1].set_ylabel('Propellant Area [m^2] (Mid-Grain)')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_title('Port Area Progression over Time (from HDF5)')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()