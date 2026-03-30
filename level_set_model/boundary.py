from numba import njit
from enum import IntEnum


@njit(cache=True)
def apply_extrapolative_bc(phi, ng):
    """
    Applies second-order linear extrapolation to fill ghost cells.
    Used for r and z boundaries to allow the level set to exit the domain smoothly.
    """
    phi[:ng, :, :]     = 2 * phi[ng:ng+1, :, :]     - phi[ng+1:2*ng+1, :, :][::-1, :, :]
    phi[-ng:, :, :]    = 2 * phi[-ng-1:-ng-2:-1, :, :] - phi[-ng-2:-2*ng-2:-1, :, :][::-1, :, :]

    # z boundaries
    phi[:, :, :ng]     = 2 * phi[:, :, ng:ng+1]     - phi[:, :, ng+1:2*ng+1][:, :, ::-1]
    phi[:, :, -ng:]    = 2 * phi[:, :, -ng-1:-ng-2:-1] - phi[:, :, -ng-2:-2*ng-2:-1][:, :, ::-1]

    return phi


@njit(cache=True)
def apply_periodic_bc(phi):
    """
    Applies periodic boundary conditions for the angular (theta) dimension.
    Assumes the angular axis is axis 1.
    """
    # Simply wrap the last interior cell to the first ghost and vice versa
    # This matches the logic from your original implementation
    phi[:, -1, :] = phi[:, 0, :]
    return phi

class BCType(IntEnum):
    """
    Boundary condition type flags for the Level Set solver.
    Matches the IntEnum pattern used in internal_ballistics_model.
    """
    DEFAULT = 0

@njit(cache=True)
def apply_boundary_conditions(phi, ng, bc_type):
    """
    Main entry point for boundary enforcement.
    Matches the apply_boundary_jit pattern in internal_ballistics_model.
    """
    # 1. Handle Radial Boundaries
    if bc_type == BCType.DEFAULT:

        phi = apply_extrapolative_bc(phi, ng)
        phi = apply_periodic_bc(phi)

    return phi