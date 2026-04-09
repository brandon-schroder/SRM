from .config import CoupledConfig

from internal_ballistics_model import IBSolver
from level_set_model import LSSolver

from coupled_solver_model.output import *

class CoupledSolver:
    def __init__(self, config: CoupledConfig):
        self.cfg = config
        self.ib = IBSolver(config.ib_config)
        self.ls = LSSolver(config.ls_config)
        self.sub_steps = 0
        self.t = 0.0

        self.coupling_scheme = config.coupling_scheme
        self.max_iter = config.max_iter
        self.tolerance = config.rel_tol

        self.recorder = CoupledRecorder(self.cfg)

        self.initialize()


    def initialize(self):
        self.ls.initialize()
        self._sync_geometry()
        self.ib.initialize()

        self.recorder.record_step(self)


    def _sync_geometry(self):

        self.ls.state.A_flow = np.maximum(self.ls.state.A_flow, 1e-12)

        self.ib.set_geometry(
            self.ls.state.z,
            self.ls.state.A_flow,
            self.ls.state.P_propellant,
            self.ls.state.P_wetted,
            self.ls.state.A_propellant,
            self.ls.state.A_casing
        )

    def step(self):
        """Dispatches to the preferred coupling scheme."""
        if self.cfg.coupling_scheme.lower() == 'implicit':
            dt, t = self.implicit_step()
        else:
            dt, t = self.explicit_step()

        self.recorder.record_step(self)

        return dt, t

    def explicit_step(self):
        self.ls.state.br = self.ls.set_burn_rate(self.ib.grid.cart_coords[2], self.ib.state.br)

        dt_ls, t_ls_next = self.ls.step()

        t_target = t_ls_next
        self.ib.cfg.t_end = t_target

        self.sub_steps = 0
        while self.ib.state.t < t_target:
            dt_ib, t_ib = self.ib.step()

            self.sub_steps+=1
            if dt_ib <= 1E-10:
                raise RuntimeError(f"IB Solver stalled at t={t_ib} (dt < 1e-10).")

        self._sync_geometry()

        self.t = t_target
        return dt_ls, self.t

    def implicit_step(self):
        phi_old = self.ls.state.phi.copy()
        U_old = self.ib.state.U.copy()
        t_start = self.t
        br_prev = self.ib.state.br.copy()

        for i in range(self.max_iter):
            is_last_iter = (i == self.max_iter - 1)

            # Reset states for new iteration
            self.ls.state.phi[:] = phi_old
            self.ls.state.t = t_start
            self.ib.state.U[:] = U_old
            self.ib.state.t = t_start

            # 1. LS Step
            self.ls.state.br = self.ls.set_burn_rate(self.ib.grid.cart_coords[2], br_prev)
            dt_ls, t_target = self.ls.step()

            # 2. Sync and IB Sub-stepping
            self._sync_geometry()
            self.ib.cfg.t_end = t_target

            iter_sub_steps = 0  # Track steps for THIS iteration
            while self.ib.state.t < t_target:
                dt_ib, t_ib = self.ib.step()
                iter_sub_steps += 1
                if dt_ib <= 1E-10: break

            # 3. Convergence Check
            error = np.linalg.norm(self.ib.state.br - br_prev) / (np.linalg.norm(br_prev) + 1e-10)
            br_prev = self.ib.state.br.copy()

            if error < self.tolerance or is_last_iter:
                self.sub_steps = iter_sub_steps
                break

        self.t = t_target
        return dt_ls, self.t

