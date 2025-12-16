import numpy as np

def get_geom(ib_solver, ls_solver):

    grid = ib_solver.grid
    states = ls_solver.states

    # Interpolate area and perimeter
    ib_solver.states.A = np.interp(grid.cart_coords[0], states.x, states.A_propellant)
    ib_solver.states.P = np.interp(grid.cart_coords[0], states.x, states.A_propellant)
    ib_solver.states.dAdx = np.gradient(ib_solver.states.A, grid.cart_coords[0])

    return ib_solver