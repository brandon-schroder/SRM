import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- SRM Project Imports ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from level_set_model import SimulationConfig, LSSolver

# --- Constants ---
RC, RF, N_LOBES = 1.2, 4.8, 9
N_CONTOURS = 40


def setup_solver():
    """Configures and initializes the Level Set Solver."""
    config = SimulationConfig(
        n_periodics=N_LOBES,
        size=(300, 300, 1),
        bounds=(1.0, 7.0, None, None, 0.0, 1.0),
        CFL=0.8,
        t_end=2.9,
        br_initial=1.0
    )

    solver = LSSolver(config)

    # Initialize phi with the Osher-Sethian Star geometry
    r, theta, _ = solver.grid.polar_coords
    phi_init = (r - ((RC + RF) / 2.0 + (RF - RC) / 2.0 * np.sin(N_LOBES * theta)))
    solver.state.phi[:] = phi_init

    return solver, config


def plot_full_star(ax, r_s, th_s, phi_s, color, n_sects):
    """Rotates and plots the fundamental sector to show the complete star."""
    sector_width = 2.0 * np.pi / n_sects
    for i in range(n_sects):
        rotation = i * sector_width
        x = r_s * np.cos(th_s + rotation)
        y = r_s * np.sin(th_s + rotation)
        # Plot only the zero-level set (the interface)
        ax.contour(x, y, phi_s, levels=[0], colors=[color], linewidths=1.2, alpha=0.8)


def plot_mesh_domain(ax, solver):
    """Plots the boundaries of the computational mesh (fundamental sector)."""
    # Radial and Angular bounds from Grid3D
    r_min, r_max = solver.grid.bounds[0], solver.grid.bounds[1]
    theta_step = 2.0 * np.pi / solver.grid.n_periodics
    theta_arc = np.linspace(0, theta_step, 100)

    # Arcs
    ax.plot(r_min * np.cos(theta_arc), r_min * np.sin(theta_arc), 'k-', lw=1.5, zorder=5)
    ax.plot(r_max * np.cos(theta_arc), r_max * np.sin(theta_arc), 'k-', lw=1.5, zorder=5)

    # Radial Symmetry Lines
    for th in [0, theta_step]:
        ax.plot([r_min * np.cos(th), r_max * np.cos(th)],
                [r_min * np.sin(th), r_max * np.sin(th)], 'k-', lw=1.5, zorder=5,
                label="Mesh Domain" if th == 0 else "")


def main():
    # 1. Initialization
    solver, config = setup_solver()
    r, theta, _ = solver.grid.polar_coords

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_times = np.linspace(0, config.t_end, N_CONTOURS)
    colors = plt.cm.plasma(np.linspace(0, 1, N_CONTOURS))
    plot_idx = 0

    # 2. Slicing logic defined by Grid3D.interior
    ng = solver.grid.ng
    r_slice = slice(ng, -ng)
    th_slice = slice(None, -1)
    z_idx = ng

    print(f"Starting Star Regression (N_Lobes={N_LOBES})...")

    # 3. Simulation Loop
    while solver.state.t < config.t_end:
        if plot_idx < N_CONTOURS and solver.state.t >= plot_times[plot_idx]:
            # Extract interior physical data
            phi_p = solver.state.phi[r_slice, th_slice, z_idx]
            r_p = r[r_slice, th_slice, z_idx]
            th_p = theta[r_slice, th_slice, z_idx]

            plot_full_star(ax, r_p, th_p, phi_p, colors[plot_idx], N_LOBES)
            plot_idx += 1

        solver.step()

    # 4. Final Formatting
    plot_mesh_domain(ax, solver)

    # Colorbar for temporal evolution
    norm = plt.Normalize(vmin=0, vmax=config.t_end)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Time [s]', rotation=270, labelpad=15)

    ax.set(aspect='equal', xlabel='x [m]', ylabel='y [m]',
           title=f"Star Front Regression: {N_LOBES} Lobes")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()