from level_set_grid import *
from level_set_solver import *

from structure import *

import time

# =============================================================================
# MAIN SIMULATION (Now using the correct L function)
# =============================================================================
ls_solver = LS_Solver()
ls_solver.params = LS_Solver.Parameters(
    n_periodics = 11,
    size = [40, 40, 40],
    bounds=[10.0, 35.0, None, None, 0.0, 100.0],
    file_prop="C:\\Users\\brandon.schroder\\PycharmProjects\\SRM-LSM\\07R-SRM-Propellant.STL",
    file_case="C:\\Users\\brandon.schroder\\PycharmProjects\\SRM-LSM\\07R-SRM-Casing.STL",

    CFL = 0.8,
    t_end = 1.0,
    t_start = 0.0,

    br_initial = 10,

)

ls_solver = build_grid(ls_solver)

print("\nStarting simulation...")
time_step = -1
st = time.time()
t = 0.0
dt = 1E-5
while True:
    # 2. Compute timestep using the self-contained function

    ls_solver = level_set_step(ls_solver, dt)
    t += dt
    time_step += 1
    print(f"Step {time_step + 1:3d}: dt={dt:.6f}, t={t:.6f}")
    if t >= ls_solver.params.t_end:
        print(f"Simulation finished at t = {t:.4f}")
        break
    dt = adaptive_timestep_wrapper(ls_solver)
et = time.time()
elapsed = et-st

print(f"Elapsed time = {elapsed:.5f}s for {time_step} Timesteps ({elapsed / time_step:.5f}s per step)")

import matplotlib.pyplot as plt

inputs = ls_solver.params
states = ls_solver.states

plt.figure()
plt.plot(states.x, states.A_propellant * inputs.n_periodics, label="propellant area")
plt.plot(states.x, states.A_casing * inputs.n_periodics, label="casing area")
plt.plot(states.x, states.P_propellant * inputs.n_periodics, label="propellant perimeter")
plt.legend()
plt.show()