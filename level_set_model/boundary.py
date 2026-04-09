from numba import njit
from enum import IntEnum


@njit(cache=True)
def apply_extrapolative_bc(phi, ng):
    for i in range(ng):
        phi[i, :, :] = phi[ng, :, :]
        phi[-i-1, :, :] = phi[-ng-1, :, :]

    for k in range(ng):
        phi[:, :, k] = phi[:, :, ng]
        phi[:, :, -k-1] = phi[:, :, -ng-1]

    return phi


@njit(cache=True)
def apply_periodic_bc(phi, ng):
    phi[:, -1, :] = phi[:, 0, :]

    return phi

class BCType(IntEnum):
    DEFAULT = 0

@njit(cache=True)
def apply_boundary_conditions(phi, ng, bc_type):
    if bc_type == BCType.DEFAULT:

        phi = apply_periodic_bc(phi, ng)
        phi = apply_extrapolative_bc(phi, ng)

    return phi