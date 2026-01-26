import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os

# 1. Import the Recorder and Solver
from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.recorder import IBRecorder


def main():
    precision = np.float32

    # Load geometry data
    df = pd.read_csv('nozzle_area_dist.csv')

    # Force the arrays to match your simulation precision
    x_geom = (df['x'] * 1e-3).values
    A_geom = (df['a'] * 1e-6).values
    P_geom = (np.ones_like(x_geom) * 1.0)

    for i in range(len(x_geom)):
        # No propellant in the nozzle section
        if x_geom[i] > 0.1:
            P_geom[i] = 0.0

    P_wetted = P_geom
    A_casing = A_geom
    A_propellant = A_geom

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    config = SimulationConfig(
        n_cells=200,
        ng=3,
        bounds=(x_geom.min(), x_geom.max()),
        CFL=0.8,
        t_end=0.005,
        p0_inlet=105.0e3,
        p_inf=100.0e3,
        rho_p=1600.0,
        a_coef=0.000035,
        n_exp=0.36,
        br_initial=0,
    )

    # Set the output filename (Required by the new IBRecorder)
    config.output_filename = "output_ib.h5"
    config.log_interval = 1  # Log every step
    config.dtype = precision

    print(f"--- Initializing Simulation: {config.n_cells} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------
    solver = IBSolver(config)
    solver.set_geometry(x_geom, A_geom, P_geom, P_wetted, A_propellant, A_casing)
    solver.initialize()

    print(f"Grid Memory: {solver.state.U.nbytes / 1e3:.3f} kB")

    # ---------------------------------------------------------
    # 3. Recorder Setup
    # ---------------------------------------------------------
    # Instantiates the new IBRecorder.
    # This will also write the static /geometry/x data to the file immediately.
    recorder = IBRecorder(solver)

    # ---------------------------------------------------------
    # 4. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    while solver.state.t < config.t_end:
        dt, current_time = solver.step()

        # --- Save Data ---
        # The recorder now delegates math to solver.get_derived_quantities()
        # and logs units automatically.
        recorder.save()

        # Monitor progress (Basic console output)
        step_count = int(solver.state.t / dt)
        if step_count % 1000 == 0:
            max_p = np.max(solver.state.p) / 1e6
            print(f"t={current_time:.5f}s | dt={dt:.2e} | P_max={max_p:.2f} MPa")

    # ---------------------------------------------------------
    # 5. Finalize
    # ---------------------------------------------------------
    # This flushes the buffer, calculates summary stats (Total Impulse),
    # AND generates the XDMF file for ParaView.
    recorder.finalize()
    print(f"XDMF file generated: {config.output_filename.replace('.h5', '.xmf')}")

    # ---------------------------------------------------------
    # 6. Post-Processing & Verification
    # ---------------------------------------------------------
    print(f"\nSimulation Complete (t = {solver.state.t:.4f} s)")
    print(f"Fetching data from {config.output_filename}...")

    # Open the HDF5 file
    with h5py.File(config.output_filename, "r") as f:
        # 1. Load Spatial Data (Fields)
        # We access the last timestep using index -1
        p_final = f["fields/pressure"][-1, :]
        u_final = f["fields/velocity"][-1, :]
        A_final = f["fields/area"][-1, :]

        # [NEW] Load Mach directly (Pre-calculated by PerformanceAnalyzer)
        # This verifies Phase 1 architecture changes
        mach_final = f["fields/mach"][-1, :]

        # [NEW] Load Geometry from file
        # This verifies Phase 2 architecture changes
        if "geometry/x" in f:
            x_grid = f["geometry/x"][:]
        else:
            # Fallback if something went wrong
            x_grid = solver.grid.x_coords

        # 2. Load Time-Series Data (Scalars)
        t_hist = f["timeseries/time"][:]
        p_head_hist = f["timeseries/p_head"][:]
        thrust_hist = f["timeseries/thrust"][:]

        # 3. Retrieve Summary Metrics
        # [UPDATED] In the new logger, save_summary() writes to root attributes (f.attrs),
        # not a "summary" group.
        total_impulse = f.attrs.get("total_impulse", 0.0)
        max_p_head = f.attrs.get("max_p_head", 0.0)

        # [NEW] Verify Units Attribute (Phase 2)
        p_units = f["fields/pressure"].attrs.get("units", "Unknown")

        print(f"Total Impulse: {total_impulse:.2f} Ns")
        print(f"Max Head Pressure: {max_p_head / 1e6:.2f} MPa")
        print(f"Pressure Units detected in file: {p_units}")

    # --- Plotting ---

    # Figure 1: Spatial Distribution (Final State)
    fig1, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pressure
    ax[0].plot(x_grid, p_final / 1e5, 'r-', linewidth=2)
    ax[0].set_ylabel(f'Pressure [bar]')  # We know it's bar because we div by 1e5
    ax[0].set_title(f'Internal Ballistics State at t={solver.state.t:.4f}s')
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Velocity
    ax[1].plot(x_grid, u_final, 'b-', label='Velocity')
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Geometry & Mach
    ax[2].fill_between(x_grid, 0, A_final, color='gray', alpha=0.5, label='Nozzle Area')
    ax[2].set_ylabel('Area [m^2]')
    ax[2].set_xlabel('Position [m]')
    ax[2].grid(True, alpha=0.3)

    ax2_twin = ax[2].twinx()
    # Plotting the Mach number retrieved directly from HDF5
    ax2_twin.plot(x_grid, mach_final, 'k--', label='Mach (Logged)')
    ax2_twin.set_ylabel('Mach Number')

    plt.tight_layout()

    # Figure 2: Time History
    fig2, ax_hist = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # History 1: Pressure
    ax_hist[0].plot(t_hist, p_head_hist / 1e6, 'r-')
    ax_hist[0].set_ylabel('Head Pressure [MPa]')
    ax_hist[0].set_title('Simulation History')
    ax_hist[0].grid(True, alpha=0.3)

    # History 2: Thrust
    ax_hist[1].plot(t_hist, thrust_hist / 1000, 'k-')
    ax_hist[1].set_ylabel('Thrust [kN]')
    ax_hist[1].set_xlabel('Time [s]')
    ax_hist[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()