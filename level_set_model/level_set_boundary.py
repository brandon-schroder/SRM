from numba import njit, prange
import numpy as np


# =============================================================================
# NUMBA-COMPILED BOUNDARY CONDITIONS
# =============================================================================

@njit(fastmath=True, cache=True)
def apply_boundary_conditions(phi: np.ndarray, ng: int = 3) -> np.ndarray:
    # r boundaries
    phi[:ng, :, :]     = 2 * phi[ng:ng+1, :, :]     - phi[ng+1:2*ng+1, :, :][::-1, :, :]
    phi[-ng:, :, :]    = 2 * phi[-ng-1:-ng-2:-1, :, :] - phi[-ng-2:-2*ng-2:-1, :, :][::-1, :, :]

    # z boundaries
    phi[:, :, :ng]     = 2 * phi[:, :, ng:ng+1]     - phi[:, :, ng+1:2*ng+1][:, :, ::-1]
    phi[:, :, -ng:]    = 2 * phi[:, :, -ng-1:-ng-2:-1] - phi[:, :, -ng-2:-2*ng-2:-1][:, :, ::-1]

    phi[:, -1, :] = phi[:, 0, :]
    return phi