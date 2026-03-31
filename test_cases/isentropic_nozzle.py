import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the parent directory is in the path to import internal_ballistics_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved


def main():
    # Use float64 for better convergence stability in steady-state tests
    precision = np.float64

    # 1. Geometry Setup (Converging-Diverging Nozzle)
    # ---------------------------------------------------------
    n_cells = 800
    z_geom = np.linspace(0.0, 1.0, n_cells, dtype=precision)

    # A parabolic nozzle: throat at z=0.5 (Area=1.0). Inlet/Outlet Area=1.5
    A_geom = 1.0 + 2.0 * (z_geom - 0.5) ** 2
    P_geom = np.zeros_like(z_geom)  # Perimeter (not critical for isentropic test)

    # 2. Configuration Setup
    # ---------------------------------------------------------
    config = SimulationConfig(
        n_cells=n_cells,
        bounds=(0.0, 1.0),
        CFL=0.9,  # Lower CFL for initial transient stability
        t_end=0.2,  # Longer time to ensure steady state is reached
        gamma=1.4,
        R=287.05,  # Gas constant for Air
        br_initial=0.0,  # No mass addition
        a_coef=0.0,  # No combustion
        p0_inlet=500.0e3,  # 500 kPa Stagnation
        t0_inlet=288.15,  # 15 degC Stagnation
        p_inf=100.0e3,  # 100 kPa Ambient
        inlet_bc_type="characteristic",
        outlet_bc_type="characteristic"  # Switched from transmissive for startup stability
    )

    solver = IBSolver(config)
    # Area_propellant and Area_flow are the same for this test
    solver.set_geometry(z_geom, A_geom, P_geom, P_geom, A_geom, A_geom)
    solver.initialize()

    # 3. Initial Conditions (Start close to Stagnation)
    # ---------------------------------------------------------
    # Calculate stagnation density from p0 and t0
    rho0 = config.p0_inlet / (config.R * config.t0_inlet)

    # Initialize the whole domain with stagnation properties
    # This prevents a massive shock at the inlet on step 1
    solver.state.rho[:] = rho0
    solver.state.p[:] = config.p0_inlet
    solver.state.u[:] = 0.01  # Small initial velocity to define flow direction

    # CRITICAL: Synchronize the conserved variable vector U with these primitives
    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p, solver.state.A, config.gamma
    )

    print(f"Running Isentropic Nozzle to t={config.t_end}s...")

    # 4. Main Loop
    # ---------------------------------------------------------
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            # Monitor progress every 500 steps
            if solver.step_count % 500 == 0:
                # Track Mach number at the throat (middle of the domain)
                mid = n_cells // 2
                c_throat = np.sqrt(config.gamma * solver.state.p[mid] / solver.state.rho[mid])
                mach_throat = np.abs(solver.state.u[mid]) / c_throat
                print(f"t={current_time:.4f}s | Step={solver.step_count} | Throat Mach={mach_throat:.3f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    # 5. Post-Processing & Plotting
    # ---------------------------------------------------------
    interior = solver.grid.interior
    z = solver.grid.cart_coords[2][interior]
    p = solver.state.p[interior]
    rho = solver.state.rho[interior]
    u = solver.state.u[interior]

    # Calculate Mach Number: M = u / sqrt(gamma * P / rho)
    mach = np.abs(u) / np.sqrt(config.gamma * p / rho)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Mach Number Plot
    axs[0].plot(z, mach, 'b-', lw=2, label='Simulation')
    axs[0].set_ylabel('Mach Number')
    axs[0].set_title('Converging-Diverging Nozzle: Steady State Results')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].axvline(0.5, color='k', linestyle=':', label='Throat (z=0.5)')
    axs[0].legend()

    # Pressure Plot
    axs[1].plot(z, p / 1e3, 'r-', lw=2)
    axs[1].set_ylabel('Pressure (kPa)')
    axs[1].set_xlabel('Position (z)')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].axvline(0.5, color='k', linestyle=':')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()