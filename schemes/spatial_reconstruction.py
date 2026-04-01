from numba import njit
import numpy as np

# Global constant
EPS = 1E-12

# ===================================================================
# GODUNOV: 1st-order
# ===================================================================

@njit(fastmath=True, cache=True)
def upwind1_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """
    1st-order upwind reconstruction for the left state at interface i+1/2.
    Ignores the extended stencil and simply returns the value at cell i.
    """
    return f_i

@njit(fastmath=True, cache=True)
def upwind1_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """
    1st-order upwind reconstruction for the right state at interface i+1/2.
    Ignores the extended stencil and simply returns the value at cell i+1.
    """
    return f_ip1

# ===================================================================
# MUSCL: 2nd-order
# ===================================================================

@njit(fastmath=True, cache=True)
def muscl_minmod_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    du_plus = f_ip1 - f_i
    du_minus = f_i - f_im1

    # Minmod function
    if du_plus * du_minus > 0.0:
        slope = du_plus if abs(du_plus) < abs(du_minus) else du_minus
    else:
        slope = 0.0

    return f_i + 0.5 * slope

@njit(fastmath=True, cache=True)
def muscl_minmod_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    du_plus = f_ip2 - f_ip1
    du_minus = f_ip1 - f_i

    # Minmod function
    if du_plus * du_minus > 0.0:
        slope = du_plus if abs(du_plus) < abs(du_minus) else du_minus
    else:
        slope = 0.0

    return f_ip1 - 0.5 * slope


@njit(fastmath=True, cache=True)
def muscl_vanleer_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    du_plus = f_ip1 - f_i
    du_minus = f_i - f_im1

    # Van Leer harmonic mean limiter
    if du_plus * du_minus > 0.0:
        slope = (2.0 * du_plus * du_minus) / (du_plus + du_minus)
    else:
        slope = 0.0

    return f_i + 0.5 * slope


@njit(fastmath=True, cache=True)
def muscl_vanleer_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    du_plus = f_ip2 - f_ip1
    du_minus = f_ip1 - f_i

    # Van Leer harmonic mean limiter
    if du_plus * du_minus > 0.0:
        slope = (2.0 * du_plus * du_minus) / (du_plus + du_minus)
    else:
        slope = 0.0

    return f_ip1 - 0.5 * slope

# ===================================================================
# WENO: 3rd-order & 5th-order
# ===================================================================

@njit(fastmath=True, cache=True)
def weno3_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO3 left-biased reconstruction at interface i+1/2."""
    # Smoothness indicators
    beta0 = (f_i   - f_im1) ** 2
    beta1 = (f_ip1 - f_i)   ** 2

    # Nonlinear weights
    a0 = (1.0 / 3.0) / (eps + beta0) ** 2
    a1 = (2.0 / 3.0) / (eps + beta1) ** 2
    w0, w1 = a0 / (a0 + a1), a1 / (a0 + a1)

    # Candidate polynomials
    p0 = -0.5 * f_im1 + 1.5 * f_i
    p1 =  0.5 * f_i   + 0.5 * f_ip1

    return w0 * p0 + w1 * p1


@njit(fastmath=True, cache=True)
def weno3_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO3 right-biased reconstruction at interface i+1/2."""
    # Smoothness indicators
    beta0 = (f_ip2 - f_ip1) ** 2
    beta1 = (f_ip1 - f_i)   ** 2

    # Nonlinear weights
    a0 = (1.0 / 3.0) / (eps + beta0) ** 2
    a1 = (2.0 / 3.0) / (eps + beta1) ** 2
    w0, w1 = a0 / (a0 + a1), a1 / (a0 + a1)

    # Candidate polynomials
    p0 = -0.5 * f_ip2 + 1.5 * f_ip1
    p1 =  0.5 * f_i   + 0.5 * f_ip1

    return w0 * p0 + w1 * p1


@njit(fastmath=True, cache=True)
def weno5_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO5 left-biased reconstruction at interface i+1/2."""
    # Candidate polynomials
    p0 = (1.0/3.0)*f_im2 - (7.0/6.0)*f_im1 + (11.0/6.0)*f_i
    p1 = (-1.0/6.0)*f_im1 + (5.0/6.0)*f_i   + (1.0/3.0)*f_ip1
    p2 = (1.0/3.0)*f_i   + (5.0/6.0)*f_ip1 - (1.0/6.0)*f_ip2

    # Smoothness indicators (Jiang & Shu)
    beta0 = (13.0/12.0)*(f_im2 - 2.0*f_im1 + f_i)**2     + 0.25*(f_im2 - 4.0*f_im1 + 3.0*f_i)**2
    beta1 = (13.0/12.0)*(f_im1 - 2.0*f_i   + f_ip1)**2   + 0.25*(f_im1 - f_ip1)**2
    beta2 = (13.0/12.0)*(f_i   - 2.0*f_ip1 + f_ip2)**2   + 0.25*(3.0*f_i - 4.0*f_ip1 + f_ip2)**2

    # Linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # Nonlinear weights
    a0 = d0 / (eps + beta0)**2
    a1 = d1 / (eps + beta1)**2
    a2 = d2 / (eps + beta2)**2
    w0, w1, w2 = a0 / (a0 + a1 + a2), a1 / (a0 + a1 + a2), a2 / (a0 + a1 + a2)

    return w0 * p0 + w1 * p1 + w2 * p2


@njit(fastmath=True, cache=True)
def weno5_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO5 right-biased reconstruction at interface i+1/2."""
    # Candidate polynomials
    p0 = (1.0/3.0)*f_ip3 - (7.0/6.0)*f_ip2 + (11.0/6.0)*f_ip1
    p1 = (-1.0/6.0)*f_ip2 + (5.0/6.0)*f_ip1 + (1.0/3.0)*f_i
    p2 = (1.0/3.0)*f_ip1 + (5.0/6.0)*f_i   - (1.0/6.0)*f_im1

    # Smoothness indicators
    beta0 = (13.0/12.0)*(f_ip3 - 2.0*f_ip2 + f_ip1)**2 + 0.25*(f_ip3 - 4.0*f_ip2 + 3.0*f_ip1)**2
    beta1 = (13.0/12.0)*(f_ip2 - 2.0*f_ip1 + f_i)**2   + 0.25*(f_ip2 - f_i)**2
    beta2 = (13.0/12.0)*(f_ip1 - 2.0*f_i   + f_im1)**2 + 0.25*(3.0*f_ip1 - 4.0*f_i + f_im1)**2

    # Linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # Nonlinear weights
    a0 = d0 / (eps + beta0)**2
    a1 = d1 / (eps + beta1)**2
    a2 = d2 / (eps + beta2)**2
    w0, w1, w2 = a0 / (a0 + a1 + a2), a1 / (a0 + a1 + a2), a2 / (a0 + a1 + a2)

    return w0 * p0 + w1 * p1 + w2 * p2


@njit(fastmath=True, cache=True)
def wenoz5_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO-Z5 left-biased reconstruction at interface i+1/2."""
    # Candidate polynomials
    p0 = (1.0 / 3.0) * f_im2 - (7.0 / 6.0) * f_im1 + (11.0 / 6.0) * f_i
    p1 = (-1.0 / 6.0) * f_im1 + (5.0 / 6.0) * f_i + (1.0 / 3.0) * f_ip1
    p2 = (1.0 / 3.0) * f_i + (5.0 / 6.0) * f_ip1 - (1.0 / 6.0) * f_ip2

    # Smoothness indicators (Jiang & Shu)
    beta0 = (13.0 / 12.0) * (f_im2 - 2.0 * f_im1 + f_i) ** 2 + 0.25 * (f_im2 - 4.0 * f_im1 + 3.0 * f_i) ** 2
    beta1 = (13.0 / 12.0) * (f_im1 - 2.0 * f_i + f_ip1) ** 2 + 0.25 * (f_im1 - f_ip1) ** 2
    beta2 = (13.0 / 12.0) * (f_i - 2.0 * f_ip1 + f_ip2) ** 2 + 0.25 * (3.0 * f_i - 4.0 * f_ip1 + f_ip2) ** 2

    # WENO-Z Global smoothness indicator
    tau5 = np.abs(beta0 - beta2)

    # Linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # Nonlinear weights (WENO-Z formulation)
    a0 = d0 * (1.0 + (tau5 / (beta0 + eps)) ** 2)
    a1 = d1 * (1.0 + (tau5 / (beta1 + eps)) ** 2)
    a2 = d2 * (1.0 + (tau5 / (beta2 + eps)) ** 2)

    w0, w1, w2 = a0 / (a0 + a1 + a2), a1 / (a0 + a1 + a2), a2 / (a0 + a1 + a2)

    return w0 * p0 + w1 * p1 + w2 * p2


@njit(fastmath=True, cache=True)
def wenoz5_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS):
    """WENO-Z5 right-biased reconstruction at interface i+1/2."""
    # Candidate polynomials
    p0 = (1.0 / 3.0) * f_ip3 - (7.0 / 6.0) * f_ip2 + (11.0 / 6.0) * f_ip1
    p1 = (-1.0 / 6.0) * f_ip2 + (5.0 / 6.0) * f_ip1 + (1.0 / 3.0) * f_i
    p2 = (1.0 / 3.0) * f_ip1 + (5.0 / 6.0) * f_i - (1.0 / 6.0) * f_im1

    # Smoothness indicators
    beta0 = (13.0 / 12.0) * (f_ip3 - 2.0 * f_ip2 + f_ip1) ** 2 + 0.25 * (f_ip3 - 4.0 * f_ip2 + 3.0 * f_ip1) ** 2
    beta1 = (13.0 / 12.0) * (f_ip2 - 2.0 * f_ip1 + f_i) ** 2 + 0.25 * (f_ip2 - f_i) ** 2
    beta2 = (13.0 / 12.0) * (f_ip1 - 2.0 * f_i + f_im1) ** 2 + 0.25 * (3.0 * f_ip1 - 4.0 * f_i + f_im1) ** 2

    # WENO-Z Global smoothness indicator
    tau5 = np.abs(beta0 - beta2)

    # Linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # Nonlinear weights (WENO-Z formulation)
    a0 = d0 * (1.0 + (tau5 / (beta0 + eps)) ** 2)
    a1 = d1 * (1.0 + (tau5 / (beta1 + eps)) ** 2)
    a2 = d2 * (1.0 + (tau5 / (beta2 + eps)) ** 2)

    w0, w1, w2 = a0 / (a0 + a1 + a2), a1 / (a0 + a1 + a2), a2 / (a0 + a1 + a2)

    return w0 * p0 + w1 * p1 + w2 * p2