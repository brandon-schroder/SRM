import numpy as np
from abc import ABC, abstractmethod
from _structure import SimulationConfig, FlowState
from _grid import Grid1D


class BurnRateModel(ABC):
    """Abstract base class for all burn rate logic."""

    @abstractmethod
    def compute_burn_rate(self, state: FlowState, grid: Grid1D) -> float:
        pass


class StRobertsLaw(BurnRateModel):
    """
    Standard Law: r = a * P^n
    """

    def __init__(self, config: SimulationConfig):
        self.a = config.a_coef
        self.n = config.n_exp

    def compute_burn_rate(self, state: FlowState, grid: Grid1D) -> float:
        # Use max pressure in the INTERIOR domain (ignore ghost cells)
        p_interior = state.p[grid.interior]
        p_ref = np.max(p_interior)

        return self.a * (p_ref ** self.n)


class MukundaPaulErosive(BurnRateModel):
    """
    Applies the Mukunda-Paul erosive augmentation factor (eta) on top of a base burn model.
    """

    def __init__(self, config: SimulationConfig, base_model: BurnRateModel):
        self.cfg = config
        self.base_model = base_model  # We 'wrap' the standard law

    def compute_burn_rate(self, state: FlowState, grid: Grid1D) -> float:
        # 1. Calculate Base Rate (r0)
        br_base = self.base_model.compute_burn_rate(state, grid)

        # 2. Extract Interior Arrays (Exclude ghost cells for averages)
        A_int = state.A[grid.interior]
        P_int = state.P[grid.interior]
        rho_int = state.rho[grid.interior]
        u_int = state.u[grid.interior]

        # 3. Calculate Hydraulic Diameter (Mean approximation)
        # Dh = 4 * Area / Perimeter
        Dh = 4 * np.mean(A_int) / np.mean(P_int)

        # 4. Calculate Reynolds Number Base (Re0)
        # Re0 = rho_mean * r_base * Dh / mu
        rho_mean = np.mean(rho_int)
        Re0 = (rho_mean * br_base * Dh) / self.cfg.erosive_mu

        # 5. Calculate Mass Flux parameter (G) at the Exit (Port)
        # We take the last physical cell [-1]
        A_port = A_int[-1]
        mf_port = rho_int[-1] * u_int[-1] * A_port

        G = mf_port / A_port

        # 6. Calculate scaling factors
        # g0 = G / (rho_p * r_base)
        g0 = G / (self.cfg.rho_p * br_base)

        # g = g0 * K2 * Re0^m
        g = g0 * self.cfg.erosive_K2 * (Re0 ** self.cfg.erosive_m)

        # 7. Heaviside & Eta Calculation
        # H = 1 if g > gth else 0
        if g > self.cfg.erosive_gth:
            term = (g ** 0.8) - (self.cfg.erosive_gth ** 0.8)
            eta = 1 + self.cfg.erosive_K1 * term
        else:
            eta = 1.0

        # Return total rate
        return eta * br_base