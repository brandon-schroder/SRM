import pandas as pd
from dataclasses import dataclass

from internal_ballistics_solver import *
from internal_ballistics_grid import *
from structure import *

@dataclass
class GeomData: # emulates areas/perimeters extracted from lsm model
    z_distance: np.ndarray
    prop_area: np.ndarray
    case_area: np.ndarray
    prop_perim: np.ndarray

@dataclass
class LS_Solver:
    states: 'LS_Solver.States' = None

    @dataclass
    class States:
        A_propellant: np.ndarray
        A_casing: np.ndarray
        P_propellant: np.ndarray
        x: np.ndarray
        br: float = 1.0E-6




df = pd.read_csv('nozzle_area_dist.csv')
x_geom = df['x'] * 1e-3
A_geom = df['a'] * 1e-3

ls_solver = LS_Solver()
ls_solver.states = LS_Solver.States(
    A_propellant = A_geom,
    A_casing = np.zeros_like(x_geom),
    P_propellant = np.zeros_like(x_geom),
    x = x_geom,
    br = 1.0E-6,)


ib_solver = IB_Solver()
ib_solver.params = IB_Solver.Parameters(
    size=500,
    bounds=[x_geom.min(), x_geom.max()],
    ng=3,

    CFL=0.9,
    t_end=0.1,
    t_start=0.0,

    u_initial=1.0,
    t_initial=288.15,

    p_inf=100.0E3,
    t0_inlet=288.15,
    p0_inlet=150.0e3,
    R=287.0,
    gamma=1.4,
)


ib_solver = generate_grid(ib_solver)

ib_solver = get_geom(ib_solver, ls_solver)

ib_solver = initialise(ib_solver)

t=0
time_step = -1
run = True
interior = ib_solver.grid.interior

while run:
    dt = adaptive_timestep(ib_solver)

    ib_solver = ib_step(ib_solver, dt)

    if time_step % 1000 == 0:
        print(f"Step {time_step:3d}: dt={dt:.6f}, t={t:.6f}")

    if t >= ib_solver.params.t_end:
        print(f"Simulation finished at t = {t:.4f}")
        break

    t += dt
    time_step += 1


ng = ib_solver.grid.ng
U = ib_solver.states.U
rho = U[0, ng:-ng] / ib_solver.states.A[ng:-ng]
u = U[1, ng:-ng] / U[0, ng:-ng]
rho_eT = U[2, ng:-ng] / ib_solver.states.A[ng:-ng]
p = (ib_solver.states.gamma - 1.0) * (rho_eT - 0.5 * rho * u ** 2)

results = {"x": ib_solver.grid.cart_coords[0][ng:-ng], "rho": rho, "u": u, "p": p, "A": ib_solver.states.A[ng:-ng]}
res_df = pd.DataFrame(results)


# res_df.to_csv('res.csv', index=False) # Save df for plotting later
# results = pd.read_csv('res.csv')
# res_df = pd.DataFrame(results)

x = res_df["x"]
rho = res_df["rho"]
u = res_df["u"]
p = res_df["p"]
A = res_df["A"]

print(f"Results saved to temp_res.csv")
print(f"Final max velocity: {np.max(u):.2f} m/s")
c = np.sqrt(ib_solver.states.gamma * p / rho)
print(f"Final max Mach: {np.max(u / c):.3f}")

import matplotlib.pyplot as plt

t = p / (rho * ib_solver.states.R)
M = u / np.sqrt(ib_solver.states.gamma * ib_solver.states.R * t)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(x, p / 1e3, 'b-', linewidth=2, label='Static Pressure')
ax.legend(loc='upper left')
ax = axes[1]
ax.plot(x, u, 'g-', linewidth=2, label='Velocity')
ax.legend(loc='upper left')
ax = axes[2]
ax.plot(x, rho, 'k-', linewidth=2, label='Density')
ax.legend(loc='upper left')

ax1 = axes[3]  # left y-axis (primary)
ax2 = ax1.twinx()  # right y-axis (secondary)

# Plot Mach number on the left y-axis
ax1.plot(x, M, 'r-', linewidth=2, label='Mach Number')
ax1.set_ylabel('Mach Number', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# Plot Area on the right y-axis
ax2.fill_between(x, 0, A, alpha=0.3, color='gray', label='Nozzle Area')
ax2.set_ylabel('Nozzle Area', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

plt.show()