import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import time

# Internal Solver Import
from level_set_model import SimulationConfig, LSSolver


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    prop_file = Path("geometry/07R-SRM-Propellant.STL")
    case_file = Path("geometry/07R-SRM-Casing.STL")

    # Domain bounds: [r_min, r_max, theta_min, theta_max, z_min, z_max]
    # Adjust z bounds to match your geometry scaling
    bounds = [10.0 * 1e-3, 35.0 * 1e-3, None, None, 0.0 * 1e-3, 100.0 * 1e-3]
    br_initial = 10e-3

    # Dummy Burn Rate Distribution (Uniform)
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
        CFL=0.5,
        t_end=1.0,
        br_initial=br_initial,
        output_filename="level_set_results.h5",
        vtk_dir="vtk_output",
        log_interval_vtk=50
    )

    config.dtype = np.float32

    print(f"--- Initializing Level Set Simulation: {config.size} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------
    solver = LSSolver(config)
    print(f"Grid Memory: {solver.state.phi.nbytes / 1e6:.2f} MB")

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")
    t_start = time.time()

    while solver.state.t < config.t_end:
        # Step returns dt and new time
        dt, current_time = solver.step()

        # Update burn rate (if coupled, this would come from internal ballistics)
        solver.state.br = solver.set_burn_rate(x_ib, br_ib)

        metrics = solver.get_derived_quantities()

        res_avg = metrics["scalars"].get("res_eikonal_avg", 0.0)
        res_max = metrics["scalars"].get("res_eikonal_max", 0.0)

        if solver.step_count % 1 == 0:
            print(f"Step {solver.step_count}: t={current_time:.5f}s | dt={dt:.2e} | "
                  f"Eikonal Avg={res_avg:.2e} | Max={res_max:.2e}")

    t_end = time.time()
    print(f"Simulation took {t_end - t_start:.2f} seconds.")

    # ---------------------------------------------------------
    # 4. Finalization
    # ---------------------------------------------------------
    solver.finalize()
    print("\nSimulation Complete. Data saved to HDF5 and VTK.")

    # ---------------------------------------------------------
    # 5. Post-Processing & Plotting
    # ---------------------------------------------------------
    h5_path = "level_set_results.h5"

    with h5py.File(h5_path, "r") as f:
        # A. Load Time Vector
        t_hist = f["timeseries/time"][:]

        # B. Load Spatial Distributions (Final Time Step)
        # Shape: (Time, Space) -> Take last row [-1, :]
        x_dist = f["fields/x"][-1, :]
        a_prop_dist = f["fields/A_flow"][-1, :]
        a_case_dist = f["fields/A_casing"][-1, :]

        # C. Load Midpoint Area History
        mid_idx = a_prop_dist.shape[0] // 2
        a_mid_hist = f["fields/A_flow"][:, mid_idx]

        # D. Load Stability Metrics (Eikonal Error)
        # These keys must match METRICS defined in postprocess.py
        if "timeseries/res_eikonal_avg" in f:
            res_avg = f["timeseries/res_eikonal_avg"][:]
            res_max = f["timeseries/res_eikonal_max"][:]
        else:
            res_avg = np.zeros_like(t_hist)
            res_max = np.zeros_like(t_hist)
            print("Warning: Stability metrics not found in HDF5.")

    # --- Plotting ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # Plot 1: Geometric Distributions (Final State)
    ax[0].plot(x_dist, a_prop_dist, 'b-', label='Flow Area', linewidth=2)
    ax[0].plot(x_dist, a_case_dist, 'k--', label='Casing Area')
    ax[0].fill_between(x_dist, a_prop_dist, a_case_dist, color='gray', alpha=0.2, label='Propellant Web')
    ax[0].set_ylabel('Area [m^2]')
    ax[0].set_xlabel('Axial Position [m]')
    ax[0].set_title('Final Geometric Distribution')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Midpoint Burn History
    ax[1].plot(t_hist, a_mid_hist, 'r-', linewidth=2)
    ax[1].set_ylabel('Mid-Grain Flow Area [m^2]')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_title('Burn Progression at Midpoint')
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Stability / Eikonal Residuals
    # Ideal Level Set fields have |grad(phi)| = 1.0. Deviations indicate numerical error.
    ax[2].semilogy(t_hist, res_avg, 'g-', label='Mean Error', linewidth=1.5)
    ax[2].semilogy(t_hist, res_max, 'm--', label='Max Error', linewidth=1.5)
    ax[2].set_ylabel('Eikonal Residual || |∇φ|-1 ||')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_title('Level Set Field Stability (Eikonal Error)')
    ax[2].legend()
    ax[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()