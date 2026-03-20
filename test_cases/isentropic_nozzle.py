import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved


def main():
    precision = np.float32

    # 1. Geometry Setup (Converging-Diverging Nozzle)
    # ---------------------------------------------------------
    z_geom = np.linspace(0.0, 1.0, 200, dtype=precision)
    # A parabolic nozzle: throat at z=0.5 with Area=1.0. Inlet/Outlet Area=1.5
    A_geom = 1.0 + 2.0 * (z_geom - 0.5) ** 2
    P_geom = np.zeros_like(z_geom)

    # 2. Configuration Setup
    # ---------------------------------------------------------
    config = SimulationConfig(
        n_cells=200,
        bounds=(0.0, 1.0),
        CFL=0.9,
        t_end=0.2,  # Run long enough to reach steady state
        gamma=1.4,
        br_initial=0.0,
        a_coef=0.0,
        p0_inlet=100.0e3,
        inlet_bc_type="characteristic",  # Use your rocket motor inlet (stagnation)
        outlet_bc_type="transmissive"  # Supersonic outflow doesn't need a back-pressure BC
    )

    # NOTE: Ensure your config.p0_inlet is something standard, e.g., 100000.0 Pa
    # and your config.t0_inlet is set.

    solver = IBSolver(config)
    solver.set_geometry(z_geom, A_geom, P_geom, P_geom, A_geom, A_geom)
    solver.initialize()

    # 3. Initial Conditions (Stagnation everywhere to start)
    # ---------------------------------------------------------
    interior = solver.grid.interior

    # Initialize the whole domain with stagnation pressure/density
    solver.state.rho[:] = 1.0  # Assuming non-dimensional for now, or use actual stagnation density
    solver.state.p[:] = 1.0*100E3
    solver.state.u[:] = 0.1

    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p, solver.state.A, config.gamma
    )

    print("Running Isentropic Nozzle to steady state...")

    # 4. Main Loop
    # ---------------------------------------------------------
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            # Optional: You can track the max change in density here to monitor convergence

            step_count = int(solver.state.t / (dt + 1e-16))
            if step_count % 1000 == 0:
                print(f"t={current_time:.4f}s | dt={dt:.2e}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    # 5. Plotting
    # ---------------------------------------------------------
    z = solver.grid.cart_coords[2][interior]
    p = solver.state.p[interior]
    M = np.abs(solver.state.u[interior]) / np.sqrt(config.gamma * p / solver.state.rho[interior])

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(z, M, 'b-', lw=2)
    axs[0].set_ylabel('Mach Number')
    axs[0].set_title('Steady State Nozzle Flow')
    axs[0].grid(True)

    # Mark the throat
    axs[0].axvline(0.5, color='k', linestyle='--', alpha=0.5)

    axs[1].plot(z, p, 'r-', lw=2)
    axs[1].set_ylabel('Pressure')
    axs[1].set_xlabel('Position (z)')
    axs[1].grid(True)
    axs[1].axvline(0.5, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()