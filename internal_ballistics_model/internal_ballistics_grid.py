import numpy as np

from structure import *


def generate_grid(ib_solver):

    params = ib_solver.params

    z_min, z_max = params.bounds
    n_z = params.size
    ng = params.ng

    dz = abs(z_min - z_max) / n_z

    z_full = np.linspace(z_min + 0.5 * dz - ng * dz, z_max - 0.5 * dz + ng * dz, n_z + 2 * ng)
    dims_full = [n_z + 2 * ng]

    ib_solver.grid = IB_Solver.Grid(
        dx = [dz],
        dims = dims_full,
        cart_coords = np.array([z_full]),
        ng = ng,
    )

    ib_solver.states = IB_Solver.States()

    return ib_solver