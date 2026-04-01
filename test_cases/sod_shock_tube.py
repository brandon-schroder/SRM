import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Adjust import path so it can find the model if running from the test_cases folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved

try:
    import sodshock
except ImportError:
    print("[ERROR] The 'sodshock' package is not installed.")
    print("Please install it by running: pip install sodshock")
    sys.exit(1)


def main():
    precision = np.float64

    # 1. Geometry Setup (Constant Area Tube)
    # ---------------------------------------------------------
    z_geom = np.linspace(0.0, 1.0, 200, dtype=precision)
    A_geom = np.ones_like(z_geom) * 1.0  # A(z) = 1.0
    P_geom = np.zeros_like(z_geom)  # No burning perimeter

    A_casing = A_geom
    A_propellant = A_geom
    P_wetted = P_geom

    # 2. Configuration Setup
    # ---------------------------------------------------------
    config = SimulationConfig(
        n_cells=200,
        bounds=(0.0, 1.0),
        CFL=0.6,  # Safe CFL for strong shocks
        t_end=0.2,  # Standard Sod end time
        gamma=1.4,
        br_initial=0.0,  # Disable source terms
        a_coef=0.0,
        inlet_bc_type="transmissive",  # Transmissive boundary
        outlet_bc_type="transmissive"  # Transmissive boundary
    )

    config.output_filename = "output_sod.h5"
    config.dtype = precision

    # 3. Solver Setup & Initialization
    # ---------------------------------------------------------
    solver = IBSolver(config)
    solver.set_geometry(z_geom, A_geom, P_geom, P_wetted, A_propellant, A_casing)

    # Run the default uniform initialization
    solver.initialize()

    # 4. Apply Sod's Shock Tube Initial Conditions
    # ---------------------------------------------------------
    # Get the interior slice and z-coordinates
    interior = solver.grid.interior
    z_coords = solver.grid.cart_coords[2][interior]

    # Create a mask for the left side of the diaphragm (z < 0.5)
    left_mask = z_coords < 0.5
    right_mask = ~left_mask

    # Left State (High pressure/density)
    solver.state.rho[interior][left_mask] = 1.0
    solver.state.p[interior][left_mask] = 1.0
    solver.state.u[interior][left_mask] = 0.0

    # Right State (Low pressure/density)
    solver.state.rho[interior][right_mask] = 0.125
    solver.state.p[interior][right_mask] = 0.1
    solver.state.u[interior][right_mask] = 0.0

    # Apply the exact same states to the ghost cells to mimic transmissive boundaries
    # (Though waves won't reach the boundaries by t=0.2 anyway)
    solver.state.rho[:solver.grid.ng] = 1.0
    solver.state.p[:solver.grid.ng] = 1.0
    solver.state.u[:solver.grid.ng] = 0.0

    solver.state.rho[-solver.grid.ng:] = 0.125
    solver.state.p[-solver.grid.ng:] = 0.1
    solver.state.u[-solver.grid.ng:] = 0.0

    # Recalculate the conserved variables (U) based on the new primitives
    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p, solver.state.A, config.gamma, solver.state.U
    )

    print("Running Sod Shock Tube...")

    # 5. Main Loop
    # ---------------------------------------------------------
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            step_count = int(solver.state.t / (dt + 1e-16))
            if step_count % 50 == 0:
                print(f"t={current_time:.4f}s | dt={dt:.2e}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    except Exception as e:
        print(f"\n[ERROR] Simulation crashed: {e}")
        raise

    # 6. Analytical Solution & Plotting
    # ---------------------------------------------------------
    print("Computing exact analytical solution...")

    # sodshock takes states in the order: (pressure, density, velocity)
    left_state = (1.0, 1.0, 0.0)
    right_state = (0.1, 0.125, 0.0)

    # Calculate exact solution
    positions, regions, values = sodshock.solve(
        left_state=left_state,
        right_state=right_state,
        geometry=(0.0, 1.0, 0.5),
        t=config.t_end,
        gamma=config.gamma,
        npts=500  # High resolution for the analytical line
    )

    z_exact = values['x']
    rho_exact = values['rho']
    u_exact = values['u']
    p_exact = values['p']

    # Extract numerical results
    z_num = solver.grid.cart_coords[2][interior]
    rho_num = solver.state.rho[interior]
    p_num = solver.state.p[interior]
    u_num = solver.state.u[interior]

    # Plotting Comparison
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(z_exact, rho_exact, 'k-', linewidth=1.5, label='Exact Analytical')
    axs[0].plot(z_num, rho_num, 'ro', markersize=2, label='Numerical (IBSolver)')
    axs[0].set_ylabel('Density')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[0].set_title(f"Sod Shock Tube Validation at t={solver.state.t:.3f}s")

    axs[1].plot(z_exact, u_exact, 'k-', linewidth=1.5)
    axs[1].plot(z_num, u_num, 'go', markersize=2)
    axs[1].set_ylabel('Velocity')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(z_exact, p_exact, 'k-', linewidth=1.5)
    axs[2].plot(z_num, p_num, 'bo', markersize=2)
    axs[2].set_ylabel('Pressure')
    axs[2].set_xlabel('Position (z)')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()