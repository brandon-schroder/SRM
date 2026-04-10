import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- Path Management ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved


def get_geometry(n_cells, L_inlet=0.05, L_conv=0.15, L_div=0.2,
                 D_inlet=0.1, D_throat=0.04, D_exit=0.08,
                 precision=np.float64):

    L_total = L_inlet + L_conv + L_div
    z_geom = np.linspace(0, L_total, n_cells)

    z1 = L_inlet
    z2 = L_inlet + L_conv

    diam = np.zeros_like(z_geom)

    for i, z in enumerate(z_geom):
        if z <= z1:
            diam[i] = D_inlet
        elif z <= z2:
            frac = (z - z1) / L_conv
            diam[i] = D_throat + 0.5 * (D_inlet - D_throat) * (1 + np.cos(np.pi * frac))
        else:
            frac = (z - z2) / L_div
            diam[i] = D_throat + 0.5 * (D_exit - D_throat) * (1 - np.cos(np.pi * frac))

    area = (np.pi / 4.0) * (diam ** 2)
    perim = np.pi * diam

    throat_idx = np.argmin(area)
    z_throat = z_geom[throat_idx]

    return (z_geom.astype(precision),
            area.astype(precision),
            perim.astype(precision),
            precision(z_throat),
            int(throat_idx))


def calculate_mach(p, rho, u, gamma):
    sound_speed = np.sqrt(gamma * p / rho)
    return np.abs(u) / sound_speed


def initialize_solver(config, z, A, P):
    solver = IBSolver(config)
    solver.set_geometry(z, A, P, P, A, A)
    solver.initialize()

    rho0 = config.p0_inlet / (config.R * config.t0_inlet)
    solver.state.rho[:] = rho0
    solver.state.p[:] = config.p0_inlet
    solver.state.u[:] = 0.0

    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p,
        solver.state.A, config.gamma, solver.state.U
    )
    return solver


def run_simulation(solver, config, throat_idx):
    print(f"Running Isentropic Nozzle to t={config.t_end}s...")
    try:
        while solver.state.t < config.t_end:
            solver.step()

            if solver.step_count % 1000 == 0:
                # Use the calculated throat index for monitoring
                m_throat = calculate_mach(
                    solver.state.p[throat_idx], solver.state.rho[throat_idx],
                    solver.state.u[throat_idx], config.gamma
                )
                print(f"t={solver.state.t:.4f}s | Step={solver.step_count} | Throat Mach={m_throat:.3f}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    return solver


def plot_results(solver, config, z_throat):
    idx = solver.grid.interior
    z = solver.grid.cart_coords[2][idx]
    p = solver.state.p[idx]
    rho = solver.state.rho[idx]
    u = solver.state.u[idx]
    area = solver.state.A[idx]

    mach = calculate_mach(p, rho, u, config.gamma)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Mach Number Plot + Area Overlay ---
    ax1.plot(z, mach, 'b-', lw=2.5, label='Mach Number', zorder=5)
    ax1.set_ylabel('Mach Number', color='b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Nozzle Performance: Mach & Pressure vs. Geometry', fontsize=14)

    ax1_area = ax1.twinx()
    ax1_area.plot(z, area, 'k--', alpha=0.3, label='Nozzle Area')
    ax1_area.fill_between(z, area, color='gray', alpha=0.1, label='Nozzle Profile')
    ax1_area.set_ylabel('Cross-Sectional Area ($m^2$)', color='k', alpha=0.6)

    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axvline(z_throat, color='k', linestyle=':', label=f'Throat (z={z_throat:.3f})')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_area.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- Pressure Plot ---
    ax2.plot(z, p / 1e3, 'r-', lw=2)
    ax2.set_ylabel('Static Pressure (kPa)', color='r', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position (z)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axvline(z_throat, color='k', linestyle=':')

    plt.tight_layout()
    plt.show()


def main():
    n_cells = 800
    z, A, P, z_throat, throat_idx = get_geometry(n_cells)

    config = SimulationConfig(
        n_cells=n_cells,
        bounds=(z.min(), z.max()),
        CFL=0.95,
        t_end=0.05,
        gamma=1.4,
        R=287.05,
        br_initial=0.0,
        a_coef=0.0,
        p0_inlet=500.0e3,
        t0_inlet=288.15,
        p_inf=100.0e3,
        inlet_bc_type="characteristic",
        outlet_bc_type="characteristic"
    )

    solver = initialize_solver(config, z, A, P)
    solver = run_simulation(solver, config, throat_idx)
    plot_results(solver, config, z_throat)


if __name__ == "__main__":
    main()