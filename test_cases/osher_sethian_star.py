import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from level_set_model import SimulationConfig, LSSolver

RC, RF, N_LOBES = 1.2, 4.8, 9
N_CONTOURS = 40

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

config = SimulationConfig(
    n_periodics=N_LOBES,
    size=(50, 50, 1),
    bounds=(1.0, 7.0, None, None, 0.0, 1.0),
    CFL=0.05,
    t_end=2.9,
    br_initial=1
)

solver = LSSolver(config)

r, theta, _ = solver.grid.polar_coords
phi_init = (r - ((RC + RF) / 2.0 + (RF - RC) / 2.0 * np.sin(N_LOBES * theta)))
solver.state.phi[:] = phi_init

fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.plasma(np.linspace(0, 1, N_CONTOURS))
plot_times = np.linspace(0, config.t_end, N_CONTOURS)
plot_idx = 0


ng = solver.cfg.ng
z_idx = ng

def plot_full_star(ax, r_s, th_s, phi_s, color, label, n_sects):
    for i in range(n_sects):
        rotation = i * (2.0 * np.pi / n_sects)
        x = r_s * np.cos(th_s + rotation)
        y = r_s * np.sin(th_s + rotation)

        ax.contour(x, y, phi_s, levels=[0], colors=[color], linewidths=1.5)

    ax.plot([], [], color=color, label=label)


print(f"Starting Star Regression (N={N_LOBES})...")

while solver.state.t < config.t_end:
    # Capture contour based on the pre-defined time array
    if plot_idx < N_CONTOURS and solver.state.t >= plot_times[plot_idx]:
        # Slice data
        phi_slice = solver.state.phi[ng:-ng, :, z_idx]
        r_slice = r[ng:-ng, :, z_idx]
        theta_slice = theta[ng:-ng, :, z_idx]

        plot_full_star(
            ax, r_slice, theta_slice, phi_slice,
            colors[plot_idx], f"t = {solver.state.t:.2f}s", N_LOBES
        )
        plot_idx += 1

    solver.step()


ax.set(aspect='equal', xlabel='x [m]', ylabel='y [m]',
       title=f"Osher-Setian Star Front Regression (N={N_LOBES})")
ax.legend(loc='upper right', fontsize='small', ncol=2)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()