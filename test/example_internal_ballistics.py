import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
import sys

# 1. Import Config and Solver
# Adjust import path as necessary based on your folder structure
from internal_ballistics_model import SimulationConfig, IBSolver


def main():
    precision = np.float32

    # ---------------------------------------------------------
    # 0. Load / Generate Geometry
    # ---------------------------------------------------------
    if os.path.exists('nozzle_area_dist.csv'):
        df = pd.read_csv('nozzle_area_dist.csv')
        x_geom = (df['x'] * 1e-3).values.astype(precision)
        A_geom = (df['a'] * 1e-6).values.astype(precision)
    else:
        print("Warning: CSV not found. Using synthetic geometry for demo.")
        x_geom = np.linspace(0, 0.5, 200, dtype=precision)
        A_geom = np.ones_like(x_geom) * 0.001

    # Define propellant grain logic
    P_geom = (np.ones_like(x_geom) * 1.0).astype(precision)
    for i in range(len(x_geom)):
        # No propellant in the nozzle section (x > 0.1)
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

    # [CRITICAL] Set output settings before solver init
    config.output_filename = "output_ib.h5"
    config.log_interval = 1
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
    # 3. Main Loop (With Crash Safety)
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    # [UPDATE] Use the context manager.
    # If the simulation crashes inside this block, data is automatically saved.
    with solver.recorder:
        try:
            while solver.state.t < config.t_end:
                dt, current_time = solver.step()

                # Monitor progress
                step_count = int(solver.state.t / (dt + 1e-16))
                if step_count % 1000 == 0:
                    max_p = np.max(solver.state.p) / 1e6
                    print(f"t={current_time:.5f}s | dt={dt:.2e} | P_max={max_p:.2f} MPa")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user. Saving data...")
        except Exception as e:
            print(f"\n[ERROR] Simulation crashed: {e}")
            # The 'with' block ensures recorder.finalize() is called here automatically
            raise

    # ---------------------------------------------------------
    # 4. Post-Processing & Verification
    # ---------------------------------------------------------
    print(f"\nSimulation Complete (t = {solver.state.t:.4f} s)")

    # Check if file exists (it should, thanks to the recorder)
    if not os.path.exists(config.output_filename):
        print("Error: Output file not found.")
        return

    print(f"Fetching data from {config.output_filename}...")

    with h5py.File(config.output_filename, "r") as f:
        # 1. Load Spatial Data (Fields)
        # We access the last timestep using index -1
        # [Note] The new logger names groups dynamically based on metrics
        p_final = f["fields/pressure"][-1, :]
        u_final = f["fields/velocity"][-1, :]

        # Load Mach directly if available (Logged via metrics)
        if "fields/mach" in f:
            mach_final = f["fields/mach"][-1, :]
        else:
            mach_final = np.zeros_like(p_final)

        # 2. Load Time-Series Data (Scalars)
        t_hist = f["timeseries/time"][:]
        p_head_hist = f["timeseries/p_head"][:]
        thrust_hist = f["timeseries/thrust"][:]

        # 3. Retrieve Summary Metrics
        total_impulse = f.attrs.get("total_impulse", 0.0)
        max_p_head = f.attrs.get("max_p_head", 0.0)

        print(f"Total Impulse: {total_impulse:.2f} Ns")
        print(f"Max Head Pressure: {max_p_head / 1e6:.2f} MPa")

    # --- Plotting ---
    fig1, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pressure
    ax[0].plot(solver.grid.x_coords, p_final / 1e5, 'r-', linewidth=2)
    ax[0].set_ylabel(f'Pressure [bar]')
    ax[0].set_title(f'Internal Ballistics State at t={solver.state.t:.4f}s')
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Velocity
    ax[1].plot(solver.grid.x_coords, u_final, 'b-', label='Velocity')
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Mach
    ax[2].plot(solver.grid.x_coords, mach_final, 'k--', label='Mach')
    ax[2].set_ylabel('Mach Number')
    ax[2].set_xlabel('Position [m]')
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Figure 2: Time History
    fig2, ax_hist = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_hist[0].plot(t_hist, p_head_hist / 1e6, 'r-')
    ax_hist[0].set_ylabel('Head Pressure [MPa]')
    ax_hist[0].set_title('Simulation History')
    ax_hist[0].grid(True, alpha=0.3)

    ax_hist[1].plot(t_hist, thrust_hist / 1000, 'k-')
    ax_hist[1].set_ylabel('Thrust [kN]')
    ax_hist[1].set_xlabel('Time [s]')
    ax_hist[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()