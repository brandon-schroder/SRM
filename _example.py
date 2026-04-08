from pathlib import Path
import pyvista as pv
import pandas as pd
import time

import coupled_solver_model
import internal_ballistics_model
import level_set_model

bounds=[14.0*1e-3, 67.0*1e-3, None, None, 4.83*1e-3, 1899*1e-3] # 1035.5
prop_file = Path("test\\geometry\\NAWC6-SRM-Propellant.STL")
case_file = Path("test\\geometry\\NAWC6-SRM-Casing.STL")

ib_config = internal_ballistics_model.config.SimulationConfig(
    # Grid parameters
    n_cells=500,  # Spatial resolution
    ng=3,  # Ghost cells
    bounds=(bounds[4], bounds[5]),  # Domain length (meters)
    CFL=0.95,  # Stability factor

    # Initial Conditions
    p_inf=100.0e3,  # 1 atm Ambient Pressure

    rho_p=1800.0,  # Propellant Density
    a_coef=0.00678/(6.9E6**0.360),
    n_exp=0.36,
    br_initial=10.0e-3,

    Tf = 2713+273.15,

    R = 325.62,
    gamma = (1083.0**2)/(325.62*(2713+273.15)),

    log_interval = 500,

    burn_model = "mp",
    burn_rate_update_interval = 1,
)

ls_config = level_set_model.config.SimulationConfig(
    n_periodics=6,  # Number of symmetric segments
    size=(50, 40, 500),  # Resolution: (nr, ntheta, nz)
    bounds=bounds,  # Physical dimensions
    file_scale=1.0e-3,

    file_prop=prop_file,  # Propellant SDF input
    file_case=case_file,  # Casing SDF input

    ng=3,  # Ghost cells
    CFL=0.95,  # Stability factor
    br_initial=1.0e-3,  # Initial burn rate
    log_interval=1,
    vtk_interval=0
)

coupled_conf = coupled_solver_model.config.CoupledConfig(
    ib_config=ib_config,
    ls_config=ls_config,
    t_end=1.0,
    coupling_scheme='explicit',
)

solver = coupled_solver_model.solver.CoupledSolver(coupled_conf)

initial_surface = solver.ls.grid.pv_grid.contour(scalars="propellant", isosurfaces=[0.0])

print(f"Starting Simulation (Target: {coupled_conf.t_end}s)")

times = []
pressures = []
burn_rates = []

history = []

start_time = time.time()
while solver.t < coupled_conf.t_end:
    dt_ls, t_current = solver.step()

    p_head = solver.ib.state.p.max()

    # Create small DF for this step
    df_step = pd.DataFrame({
        "t": solver.ls.state.t,
        "x": solver.ls.state.x,
        "A": solver.ls.state.A_flow,
        "P": solver.ls.state.P_propellant
    })
    history.append(df_step)

    print(f"Time: {t_current:.4f} s | dt_ls: {dt_ls:.2e} | P_head: {p_head / 1e6:.2f} MPa | Sub-cycles: {solver.sub_steps} |")

    if t_current >= coupled_conf.t_end or p_head < ib_config.p_inf:
        break

end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} s")

final_surface = solver.ls.grid.pv_grid.contour(scalars="propellant", isosurfaces=[0.0])


b=final_surface.bounds
n=ls_config.size[2]/2
initial_surface = initial_surface.slice_along_axis(axis='z', n=20, bounds=[b[0], b[1], b[2], b[3], 800E-3, b[5]])
final_surface = final_surface.slice_along_axis(axis='z', n=20, bounds=[b[0], b[1], b[2], b[3], 800E-3, b[5]])

plotter = pv.Plotter()
plotter.add_mesh(initial_surface, color="red", opacity=0.8)
plotter.add_mesh(final_surface, color="blue", opacity=0.8)
plotter.show()

