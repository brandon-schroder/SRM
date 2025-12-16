from numba import njit

# Global constant (shared across all WENO calls)
EPS_WENO = 1e-6

# ===================================================================
# WENO3: 3rd-order
# ===================================================================
@njit(fastmath=True, cache=True)
def weno3_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS_WENO):
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
def weno3_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS_WENO):
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


# ===================================================================
# WENO5: 5th-order
# ===================================================================
@njit(fastmath=True, cache=True)
def weno5_left(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS_WENO):
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
def weno5_right(f_im2, f_im1, f_i, f_ip1, f_ip2, f_ip3, eps=EPS_WENO):
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