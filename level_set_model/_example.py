import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import our new modules
from _structure import SimulationConfig
from _model import LSSolver


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # Define paths to your geometry files (Change these to your actual file names)
    # These should be PyVista-readable files (vtk, vtm, stl, etc.)
    prop_file = Path("07R-SRM-Propellant.STL")
    case_file = Path("07R-SRM-Casing.STL")

    # Domain bounds: [r_min, r_max, theta_min, theta_max, z_min, z_max]
    # Note: theta bounds (indices 2, 3) are automatically handled by
    # n_periodics in the grid generation, but we provide placeholders.
    bounds = [10.0, 35.0, None, None, 0.0, 100.0]

    config = SimulationConfig(
        n_periodics=12,  # Number of symmetric segments
        size=(50, 40, 100),  # Resolution: (nr, ntheta, nz)
        bounds=bounds,  # Physical dimensions

        file_prop=prop_file,  # Propellant SDF input
        file_case=case_file,  # Casing SDF input

        ng=3,  # Ghost cells
        CFL=0.8,  # Stability factor
        t_end=0.5,  # Simulation duration
        br_initial=10.0  # Initial burn rate (mm/s)
    )

    print(f"--- Initializing Level Set Simulation: {config.size} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------
    # The solver will automatically load the mesh files specified in config
    solver = LSSolver(config)


    # ---------------------------------------------------------
    # 4. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    history_t = []
    history_A_mid = []  # Track area at the middle of the grain

    # Index for the middle of the domain to monitor
    mid_idx = config.size[2] // 2

    while solver.state.t < config.t_end:
        # Perform one step
        dt, current_time = solver.step()

        # Monitor progress every 10 steps
        step_count = int(solver.state.t / dt)
        if step_count % 1 == 0:
            # We can access solver state variables cleanly
            # For example, checking the average value of the level set field
            avg_phi = np.mean(solver.state.phi)
            print(f"t={current_time:.5f}s | dt={dt:.2e} | Avg Phi={avg_phi:.2e}")

        # Record history
        history_t.append(current_time)
        history_A_mid.append(solver.state.A_propellant[mid_idx])

    # ---------------------------------------------------------
    # 5. Post-Processing
    # ---------------------------------------------------------
    print("\nSimulation Complete.")

    # Get the final state as a DataFrame (Geometry along Z-axis)
    df = solver.get_dataframe()
    print(df.head())

    # Plot Results
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Geometric Distributions along Axis
    ax[0].plot(df['x'], df['Area_Prop'], 'b-', label='Propellant Area', linewidth=2)
    ax[0].plot(df['x'], df['Area_Case'], 'k--', label='Casing Area')
    ax[0].set_ylabel('Flow Area [m^2]')
    ax[0].set_xlabel('Axial Position [m]')
    ax[0].set_title(f'Geometry Distributions at t={solver.state.t:.4f}s')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Burn History at Midpoint
    ax[1].plot(history_t, history_A_mid, 'r-o', markersize=3)
    ax[1].set_ylabel('Propellant Area [m^2] (Mid-Grain)')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_title('Port Area Progression over Time')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()