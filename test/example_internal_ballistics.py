import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os

# 1. Import Config and Solver
from internal_ballistics_model import SimulationConfig, IBSolver


def main():
    precision = np.float32

    # Load geometry data
    # Ensure the CSV exists or handle the error
    if os.path.exists('nozzle_area_dist.csv'):
        df = pd.read_csv('nozzle_area_dist.csv')
        # Force the arrays to match your simulation precision
        x_geom = (df['x'] * 1e-3).values.astype(precision)
        A_geom = (df['a'] * 1e-6).values.astype(precision)
    else:
        print("Warning: CSV not found. Using synthetic geometry for demo.")
        x_geom = np.linspace(0, 0.5, 200, dtype=precision)
        A_geom = np.ones_like(x_geom) * 0.001

    P_geom = (np.ones_like(x_geom) * 1.0).astype(precision)

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

    # Set the output filename
    # [CRITICAL] This must be set BEFORE initializing the solver,
    # because the solver initializes the internal recorder using this config.
    config.output_filename = "output_ib.h5"
    config.log_interval = 1  # Log every step
    config.dtype = precision

    print(f"--- Initializing Simulation: {config.n_cells} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------
    # The Recorder is now automatically instantiated inside IBSolver.
    # It will also write the static /geometry/x data immediately upon initialization.
    solver = IBSolver(config)
    solver.set_geometry(x_geom, A_geom, P_geom, P_wetted, A_propellant, A_casing)
    solver.initialize()

    print(f"Grid Memory: {solver.state.U.nbytes / 1e3:.3f} kB")

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    while solver.state.t < config.t_end:
        # solver.step() automatically calls self.recorder.save()
        dt, current_time = solver.step()

        # Monitor progress (Basic console output)
        # Using a small epsilon to avoid division by zero on first step
        step_count = int(solver.state.t / (dt + 1e-16))
        if step_count % 50 == 0:
            max_p = np.max(solver.state.p) / 1e6
            print(f"t={current_time:.5f}s | dt={dt:.2e} | P_max={max_p:.2f} MPa")

    # ---------------------------------------------------------
    # 4. Finalize
    # ---------------------------------------------------------
    # This flushes the buffer, calculates summary stats (Total Impulse),
    # and generates the XDMF file for ParaView.
    solver.finalize()
    print(f"XDMF file generated: {config.output_filename.replace('.h5', '.xdmf')}")

    # ---------------------------------------------------------
    # 5. Post-Processing & Verification
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

        # Load Mach directly (Calculated via postprocess.METRICS)
        mach_final = f["fields/mach"][-1, :]

        # Load Geometry from file (Verifies geometry callback worked)
        if "geometry/x" in f:
            x_grid = f["geometry/x"][:]
        else:
            x_grid = solver.grid.x_coords

        # 2. Load Time-Series Data (Scalars)
        t_hist = f["timeseries/time"][:]
        p_head_hist = f["timeseries/p_head"][:]
        thrust_hist = f["timeseries/thrust"][:]

        # 3. Retrieve Summary Metrics
        # These are now stored as root attributes by SimulationRecorder
        total_impulse = f.attrs.get("total_impulse", 0.0)
        max_p_head = f.attrs.get("max_p_head", 0.0)

        # Verify Units (Feature from Phase 2)
        p_units = f["fields/pressure"].attrs.get("units", "Unknown")

        print(f"Total Impulse: {total_impulse:.2f} Ns")
        print(f"Max Head Pressure: {max_p_head / 1e6:.2f} MPa")
        print(f"Pressure Units detected in file: {p_units}")

    # --- Plotting ---
    fig1, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pressure
    ax[0].plot(x_grid, p_final / 1e5, 'r-', linewidth=2)
    ax[0].set_ylabel(f'Pressure [bar]')
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