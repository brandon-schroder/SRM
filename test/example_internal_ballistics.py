import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
import sys

# Adjust import path as necessary based on your folder structure
from internal_ballistics_model import SimulationConfig, IBSolver


def main():
    precision = np.float32

    # ---------------------------------------------------------
    # 0. Load / Generate Geometry
    # ---------------------------------------------------------
    if os.path.exists('nozzle_area_dist.csv'):
        df = pd.read_csv('nozzle_area_dist.csv')
        # Map file 'x' to simulation 'z'
        z_geom = (df['x'] * 1e-3).values.astype(precision)
        A_geom = (df['a'] * 1e-6).values.astype(precision)
    else:
        print("Warning: CSV not found. Using synthetic geometry for demo.")
        z_geom = np.linspace(0, 0.5, 200, dtype=precision)
        A_geom = np.ones_like(z_geom) * 0.001

    # Define propellant grain logic
    P_geom = (np.ones_like(z_geom) * 1.0).astype(precision)
    for i in range(len(z_geom)):
        # No propellant in the nozzle section (z > 0.1)
        if z_geom[i] > 0.1:
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
        bounds=(z_geom.min(), z_geom.max()),
        CFL=0.9,
        t_end=0.1,
        p0_inlet=100.0e3,
        p_inf=100.0e3,
        rho_p=1600.0,
        u_initial=0.0,
        # a_coef=0.000035,
        a_coef=0.0,
        n_exp=0.36,
        br_initial=0,
    )

    # Output settings
    config.output_filename = "output_ib.h5"
    config.log_interval = 1
    config.dtype = precision

    print(f"--- Initializing Simulation: {config.n_cells} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------
    solver = IBSolver(config)
    solver.set_geometry(z_geom, A_geom, P_geom, P_wetted, A_propellant, A_casing)
    solver.initialize()

    print(f"Grid Memory: {solver.state.U.nbytes / 1e3:.3f} kB")

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    with solver.recorder:
        try:
            while solver.state.t < config.t_end:
                dt, current_time = solver.step()

                step_count = int(solver.state.t / (dt + 1e-16))
                if step_count % 1000 == 0:
                    max_p = np.max(solver.state.p) / 1e6
                    # Monitor Residuals during run
                    res_rho = solver.residuals.get("res_rho", 0.0)
                    print(f"t={current_time:.5f}s | dt={dt:.2e} | P_max={max_p:.2f} MPa | Res(rho)={res_rho:.2e}")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user. Saving data...")
        except Exception as e:
            print(f"\n[ERROR] Simulation crashed: {e}")
            raise

    # ---------------------------------------------------------
    # 4. Post-Processing
    # ---------------------------------------------------------
    print(f"\nSimulation Complete (t = {solver.state.t:.4f} s)")

    if not os.path.exists(config.output_filename):
        print("Error: Output file not found.")
        return

    print(f"Fetching data from {config.output_filename}...")

    with h5py.File(config.output_filename, "r") as f:
        # 1. Spatial Fields (Final Step)
        p_final = f["fields/pressure"][-1, :]
        u_final = f["fields/velocity"][-1, :]
        if "fields/mach" in f:
            mach_final = f["fields/mach"][-1, :]
        else:
            mach_final = np.zeros_like(p_final)

        # 2. Time-Series Scalars
        t_hist = f["timeseries/time"][:]
        p_head_hist = f["timeseries/p_head"][:]
        thrust_hist = f["timeseries/thrust"][:]

        # 3. Residuals (Check if they exist)
        res_rho_hist = None
        if "residuals/res_rho" in f:
            res_rho_hist = f["residuals/res_rho"][:]
            res_mom_hist = f["residuals/res_mom"][:]
            res_E_hist = f["residuals/res_E"][:]

        # 4. Summary Metrics
        total_impulse = f.attrs.get("total_impulse", 0.0)
        max_p_head = f.attrs.get("max_p_head", 0.0)

        print(f"Total Impulse: {total_impulse:.2f} Ns")
        print(f"Max Head Pressure: {max_p_head / 1e6:.2f} MPa")

        # ---------------------------------------------------------
        # 4. Post-Processing & Verification
        # ---------------------------------------------------------
        print(f"\nSimulation Complete (t = {solver.state.t:.4f} s)")

        if not os.path.exists(config.output_filename):
            print("Error: Output file not found.")
            return

        print(f"Fetching data from {config.output_filename}...")

        with h5py.File(config.output_filename, "r") as f:
            # 1. Load Spatial Data (Fields)
            p_final = f["fields/pressure"][-1, :]
            u_final = f["fields/velocity"][-1, :]

            if "fields/mach" in f:
                mach_final = f["fields/mach"][-1, :]
            else:
                mach_final = np.zeros_like(p_final)

            # 2. Load Time-Series Data (Scalars)
            t_hist = f["timeseries/time"][:]
            p_head_hist = f["timeseries/p_head"][:]
            thrust_hist = f["timeseries/thrust"][:]

            # 3. Load Residuals (Found in 'timeseries' because they are scalars)
            # We use .get() or check existence to be safe
            res_rho_hist = None
            if "timeseries/res_rho" in f:
                print("Found residuals in 'timeseries' group.")
                res_rho_hist = f["timeseries/res_rho"][:]
                res_mom_hist = f["timeseries/res_mom"][:]
                res_E_hist = f["timeseries/res_E"][:]
            else:
                print("Warning: Residuals not found in HDF5 file.")

            # 4. Retrieve Summary Metrics
            total_impulse = f.attrs.get("total_impulse", 0.0)
            max_p_head = f.attrs.get("max_p_head", 0.0)

            print(f"Total Impulse: {total_impulse:.2f} Ns")
            print(f"Max Head Pressure: {max_p_head / 1e6:.2f} MPa")

        # --- Plotting ---
        fig1, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Use cart_coords[2] for plotting Z-axis
        z_coords = solver.grid.cart_coords[2]

        # Plot 1: Pressure
        ax[0].plot(z_coords, p_final / 1e5, 'r-', linewidth=2)
        ax[0].set_ylabel(f'Pressure [bar]')
        ax[0].set_title(f'Internal Ballistics State at t={solver.state.t:.4f}s')
        ax[0].grid(True, alpha=0.3)

        # Plot 2: Velocity
        ax[1].plot(z_coords, u_final, 'b-', label='Velocity')
        ax[1].set_ylabel('Velocity [m/s]')
        ax[1].grid(True, alpha=0.3)

        # Plot 3: Mach
        ax[2].plot(z_coords, mach_final, 'k--', label='Mach')
        ax[2].set_ylabel('Mach Number')
        ax[2].set_xlabel('Position Z [m]')
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

        # Figure 3: Residuals
        if res_rho_hist is not None:
            fig3, ax_res = plt.subplots(figsize=(10, 5))

            # Use semi-log plot for residuals
            ax_res.semilogy(t_hist, res_rho_hist, label=r'Res($\rho$)', alpha=0.8)
            ax_res.semilogy(t_hist, res_mom_hist, label=r'Res($\rho u$)', alpha=0.8)
            ax_res.semilogy(t_hist, res_E_hist, label=r'Res($E$)', alpha=0.8)

            ax_res.set_title('Solver Residuals (RMS of dU/dt)')
            ax_res.set_ylabel('Residual Magnitude')
            ax_res.set_xlabel('Time [s]')
            ax_res.grid(True, which="both", alpha=0.3)
            ax_res.legend()
            plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()