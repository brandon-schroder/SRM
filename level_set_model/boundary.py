from numba import njit
from enum import IntEnum


@njit(cache=True)
def apply_extrapolative_bc(phi, ng):
    phi[:ng, :, :]     = 2 * phi[ng:ng+1, :, :]     - phi[ng+1:2*ng+1, :, :][::-1, :, :]
    phi[-ng:, :, :]    = 2 * phi[-ng-1:-ng-2:-1, :, :] - phi[-ng-2:-2*ng-2:-1, :, :][::-1, :, :]

    # z boundaries
    phi[:, :, :ng]     = 2 * phi[:, :, ng:ng+1]     - phi[:, :, ng+1:2*ng+1][:, :, ::-1]
    phi[:, :, -ng:]    = 2 * phi[:, :, -ng-1:-ng-2:-1] - phi[:, :, -ng-2:-2*ng-2:-1][:, :, ::-1]

    return phi


@njit(cache=True)
def apply_periodic_bc(phi):
    phi[:, -1, :] = phi[:, 0, :]
    return phi

class BCType(IntEnum):
    DEFAULT = 0

@njit(cache=True)
def apply_boundary_conditions(phi, ng, bc_type):
    if bc_type == BCType.DEFAULT:

        phi = apply_extrapolative_bc(phi, ng)
        phi = apply_periodic_bc(phi)

    return phi