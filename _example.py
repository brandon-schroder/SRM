from pathlib import Path
import pyvista as pv

from coupled_solver import *
import internal_ballistics_model
import level_set_model

bounds=[10.0*1e-3, 35.0*1e-3, None, None, 0.0*1e-3, 1035.0*1e-3]
prop_file = Path("test\\07R-SRM-Propellant.STL")
case_file = Path("test\\07R-SRM-Casing.STL")

ib_config = internal_ballistics_model._structure.SimulationConfig(
    # Grid parameters
    n_cells=200,  # Spatial resolution
    ng=3,  # Ghost cells
    bounds=(bounds[4], bounds[5]),  # Domain length (meters)
    CFL=0.9,  # Stability factor
    t_end=0.005,  # Simulation duration

    # Initial Conditions
    p0_inlet=3.5e6,  # 3.5 MPa Chamber Pressure
    p_inf=100.0e3,  # 1 atm Ambient Pressure

    rho_p=1600.0,  # Propellant Density
    a_coef=0.000035,
    n_exp=0.36,
    br_initial=10.0e-3,

)

ls_config = level_set_model._structure.SimulationConfig(
    n_periodics=11,  # Number of symmetric segments
    size=(50, 40, 300),  # Resolution: (nr, ntheta, nz)
    bounds=bounds,  # Physical dimensions
    file_scale=1.0e-3,

    file_prop=prop_file,  # Propellant SDF input
    file_case=case_file,  # Casing SDF input

    ng=3,  # Ghost cells
    CFL=0.9,  # Stability factor
    t_end=0.1,  # Simulation duration
    br_initial=10.0e-3  # Initial burn rate
)

coupled_conf = CoupledConfig(
    ib_config=ib_config,
    ls_config=ls_config,
    t_end=0.2)

solver = CoupledSolver(coupled_conf)

initial_surface = solver.ls.grid.pv_grid.contour(scalars="propellant", isosurfaces=[0.0])

print(f"Starting Simulation (Target: {coupled_conf.t_end}s)")

times = []
pressures = []
burn_rates = []

while solver.t < coupled_conf.t_end:
    dt_ls, t_current = solver.step()

    p_head = solver.ib.state.p.max()
    br_curr = solver.ib.state.br

    if dt_ls <= 1E-10:
        break

    print(f"Time: {t_current:.4f} s | dt_ls: {dt_ls:.2e} | P_head: {p_head / 1e6:.2f} MPa | BR: {br_curr * 1000:.2f} mm/s")

solver.ls.grid.pv_grid["propellant"] = solver.ls.state.phi.flatten(order='F')
final_surface = solver.ls.grid.pv_grid.contour(scalars="propellant", isosurfaces=[0.0])

z=100*1E-3

initial_surface = initial_surface.slice(normal='z', origin=(0, 0, z))
final_surface = final_surface.slice(normal='z', origin=(0, 0, z))

plotter = pv.Plotter()
plotter.add_mesh(initial_surface, color="red", opacity=0.8)
plotter.add_mesh(final_surface, color="blue", opacity=0.8)
plotter.show()