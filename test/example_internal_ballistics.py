import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import our new modules
from internal_ballistics_model import SimulationConfig, IBSolver

def main():

    df = pd.read_csv('nozzle_area_dist.csv')
    x_geom = df['x'] * 1e-3
    A_geom = df['a'] * 1e-6
    P_geom = np.ones_like(x_geom) * 1E-0

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # Note: IDE autocompletion works here!
    config = SimulationConfig(
        # Grid parameters
        n_cells=200,  # Spatial resolution
        ng=3,  # Ghost cells
        bounds=(x_geom.min(), x_geom.max()),  # Domain length (meters)
        CFL=0.5,  # Stability factor
        t_end=0.005,  # Simulation duration

        # Initial Conditions
        p0_inlet=3.5e6,  # 3.5 MPa Chamber Pressure
        p_inf=100.0e3,  # 1 atm Ambient Pressure

        rho_p=1600.0,  # Propellant Density
        a_coef=0.000035,
        n_exp=0.36,
        br_initial=1e-3,

    )

    print(f"--- Initializing Simulation: {config.n_cells} cells ---")

    # ---------------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------------

    solver = IBSolver(config)

    solver.set_geometry(x_geom, A_geom, P_geom)

    solver.initialize()

    # ---------------------------------------------------------
    # 5. Main Loop
    # ---------------------------------------------------------
    print(f"Starting Time Integration (Target: {config.t_end}s)...")

    # Lists to store history for plotting later (optional)
    history_p_head = []
    history_time = []

    while solver.state.t < config.t_end:
        dt, current_time = solver.step()

        # Monitor progress every 50 steps
        step_count = int(solver.state.t / dt)
        if step_count % 1000 == 0:
            # We can access solver state variables cleanly
            max_p = np.max(solver.state.p) / 1e5  # Convert Pa to Bar
            max_u = np.max(solver.state.u)
            print(f"t={current_time:.5f}s | dt={dt:.2e} | P_max={max_p:.2f} bar | U_max={max_u:.1f} m/s")

        # Record head-end pressure
        history_p_head.append(solver.state.p[solver.grid.interior][0])
        history_time.append(current_time)

    # ---------------------------------------------------------
    # 6. Post-Processing
    # ---------------------------------------------------------
    print(f"\nSimulation Complete (t = {solver.state.t} s)")

    # Get the final state as a DataFrame
    df = solver.get_dataframe()
    print(df.head())

    # Plot Results
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pressure
    ax[0].plot(df['x'], df['p'] / 1e5, 'r-', linewidth=2)
    ax[0].set_ylabel('Pressure [bar]')
    ax[0].set_title(f'Internal Ballistics State at t={solver.state.t:.4f}s')
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Velocity & Mach
    ax[1].plot(df['x'], df['u'], 'b-', label='Velocity')
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Geometry Check
    ax[2].fill_between(df['x'], 0, df['Area'], color='gray', alpha=0.5, label='Nozzle Area')
    ax[2].set_ylabel('Area [m^2]')
    ax[2].set_xlabel('Position [m]')
    ax[2].grid(True, alpha=0.3)

    # Calculate Mach (u/c)
    mach = df['u'] / np.sqrt(config.gamma * df['p'] / df['rho'])
    ax2_twin = ax[2].twinx()
    ax2_twin.plot(df['x'], mach, 'k--', label='Mach')
    ax2_twin.set_ylabel('Mach Number')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()