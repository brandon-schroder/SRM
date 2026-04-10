import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path Management ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


import sodshock
from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved


def setup_sod_ic(solver, config):
    """Sets the initial conditions for the Sod Shock Tube."""
    interior = solver.grid.interior
    z_coords = solver.grid.cart_coords[2][interior]
    ng = solver.grid.ng

    # Define regions
    left = z_coords < 0.5
    right = ~left

    # Set Primitives
    solver.state.rho[interior] = np.where(left, 1.0, 0.125)
    solver.state.p[interior] = np.where(left, 1.0, 0.1)
    solver.state.u[interior] = 0.0

    # Sync Ghost Cells (Transmissive)
    solver.state.rho[:ng], solver.state.p[:ng] = 1.0, 1.0
    solver.state.rho[-ng:], solver.state.p[-ng:] = 0.125, 0.1

    # Update Conserved variables
    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p,
        solver.state.A, config.gamma, solver.state.U
    )


def run_simulation(solver, config):
    """Main execution loop with formatted logging."""
    print(f"Starting Sod Shock Tube: t_end={config.t_end}s, CFL={config.CFL}")
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            # Log every ~50 steps based on t/dt
            if int(current_time / dt) % 50 == 0:
                print(f" > t: {current_time:6.4f}s | dt: {dt:.2e}")
    except Exception as e:
        print(f"\n[CRITICAL] Simulation failed: {e}")
        raise
    print("Simulation complete.\n")


def plot_comparison(solver, config):
    """Generates a high-quality comparison plot with analytical data."""
    # Analytical Solution
    _, _, val = sodshock.solve(
        left_state=(1.0, 1.0, 0.0),
        right_state=(0.1, 0.125, 0.0),
        geometry=(0.0, 1.0, 0.5), t=config.t_end,
        gamma=config.gamma, npts=500
    )

    # Numerical Data
    idx = solver.grid.interior
    z_num = solver.grid.cart_coords[2][idx]

    plots = [
        ('Density', val['rho'], solver.state.rho[idx], 'r'),
        ('Velocity', val['u'], solver.state.u[idx], 'g'),
        ('Pressure', val['p'], solver.state.p[idx], 'b')
    ]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axs[0].set_title(f"Sod Shock Tube Validation (t={solver.state.t:.3f}s)")

    for i, (label, exact, num, color) in enumerate(plots):
        axs[i].plot(val['x'], exact, 'k-', lw=1.5, label='Exact' if i == 0 else None)
        axs[i].plot(z_num, num, f'{color}o', ms=2, alpha=0.6, label='Numerical' if i == 0 else None)
        axs[i].set_ylabel(label)
        axs[i].grid(True, alpha=0.3)

    axs[0].legend()
    axs[-1].set_xlabel('Position (z)')
    plt.tight_layout()
    plt.show()


def main():
    # 1. Config
    config = SimulationConfig(
        n_cells=1000,
        bounds=(0.0, 1.0),
        CFL=0.6,
        t_end=0.2,
        gamma=1.4,
        br_initial=0.0, 
        a_coef=0.0,
        inlet_bc_type="transmissive",
        outlet_bc_type="transmissive",
        dtype=np.float32
    )

    # 2. Setup Solver & Geometry
    z_g = np.linspace(0, 1, 200)
    solver = IBSolver(config)
    solver.set_geometry(z_g, np.ones_like(z_g), np.zeros_like(z_g),
                        np.zeros_like(z_g), np.ones_like(z_g), np.ones_like(z_g))
    solver.initialize()

    # 3. Run
    setup_sod_ic(solver, config)
    run_simulation(solver, config)
    plot_comparison(solver, config)


if __name__ == "__main__":
    main()