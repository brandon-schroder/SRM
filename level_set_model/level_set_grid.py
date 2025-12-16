from structure import *
from level_set_geometry import *

# =============================================================================
# GRID GENERATION
# =============================================================================

def generate_grid(ls_solver):
    """Generate cylindrical grid with ghost cells"""

    params = ls_solver.params

    params.bounds[2] = 0
    params.bounds[3] = params.bounds[2] + 2.0 * np.pi / params.n_periodics

    ng = params.ng

    # Cell widths
    r_min, r_max, theta_min, theta_max, z_min, z_max = params.bounds
    n_r, n_theta, n_z = params.size
    dr = abs(r_min - r_max) / n_r
    dz = abs(z_min - z_max) / n_z
    dtheta = abs(theta_min - theta_max) / n_theta

    # Full grid with ghost cells
    r_full = np.linspace(r_min + 0.5 * dr - ng * dr, r_max - 0.5 * dr + ng * dr, n_r + 2 * ng)
    z_full = np.linspace(z_min + 0.5 * dz - ng * dz, z_max - 0.5 * dz + ng * dz, n_z + 2 * ng)
    theta = np.linspace(theta_min + 0.5 * dtheta, theta_max + 0.5 * dtheta, n_theta + 1)

    # Meshgrid
    R_full, THETA_full, Z_full = np.meshgrid(r_full, theta, z_full, indexing='ij')
    X_full = R_full * np.cos(THETA_full)
    Y_full = R_full * np.sin(THETA_full)

    grid_full = pv.StructuredGrid(X_full, Y_full, Z_full)
    dims_full = [n_r + 2 * ng, n_theta + 1, n_z + 2 * ng]

    ls_solver.grid = LS_Solver.Grid(
        pv_grid=grid_full,
        dx=[dr, dtheta, dz],
        dims=dims_full,
        cart_coords=np.array([X_full, Y_full, Z_full]),
        polar_coords=np.array([R_full, THETA_full, Z_full]),
        ng=ng
    )

    return ls_solver


# =============================================================================
# INITIALISE SIGNED DISTANCES
# =============================================================================

def initialise_sdf(ls_solver):
    """Initialize signed distance function from STL geometry"""
    params = ls_solver.params
    grid = ls_solver.grid
    ls_solver.states = LS_Solver.States()

    filename_prop = params.file_prop
    filename_case = params.file_case

    prop = pv.read(filename_prop)
    case = pv.read(filename_case)
    prop = prop.clip_surface(case, invert=True)

    ls_solver.grid.pv_grid = grid.pv_grid.compute_implicit_distance(case)
    ls_solver.grid.pv_grid.point_data["casing"] = grid.pv_grid.point_data["implicit_distance"]

    ls_solver.grid.pv_grid = grid.pv_grid.compute_implicit_distance(prop)
    ls_solver.grid.pv_grid.point_data["propellant"] = grid.pv_grid.point_data["implicit_distance"]



    ls_solver.states.phi = np.array(grid.pv_grid["propellant"].reshape(grid.dims, order='F'))
    ls_solver.states.casing = np.array(grid.pv_grid["casing"].reshape(grid.dims, order='F'))

    ls_solver.states.grad_mag = np.ones_like(ls_solver.states.phi)

    ls_solver.states.br = ls_solver.params.br_initial

    return ls_solver


def build_grid(ls_solver):

    ls_solver = generate_grid(ls_solver)
    ls_solver = initialise_sdf(ls_solver)

    ls_solver = get_geometry(ls_solver, sdf="casing")

    return ls_solver







