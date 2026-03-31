from .config import CoupledConfig

from internal_ballistics_model import IBSolver
from level_set_model import LSSolver

class CoupledSolver:
    def __init__(self, config: CoupledConfig):
        self.cfg = config
        self.sub_steps = 0

        self.ib = IBSolver(config.ib_config)
        self.ls = LSSolver(config.ls_config)

        self.t = 0.0

        self.initialize()


    def initialize(self):
        self.ls.initialize()

        self._sync_geometry()

        self.ib.initialize()


    def _sync_geometry(self):
        self.ib.set_geometry(
            self.ls.state.x,
            self.ls.state.A_flow,
            self.ls.state.P_propellant,
            self.ls.state.P_wetted,
            self.ls.state.A_propellant,
            self.ls.state.A_casing
        )

    def step(self):
        self.ls.state.br = self.ls.set_burn_rate(self.ib.grid.cart_coords[2], self.ib.state.br)

        dt_ls, t_ls_next = self.ls.step()

        t_target = t_ls_next
        self.ib.cfg.t_end = t_target

        self.sub_steps = 0
        while self.ib.state.t < t_target:
            dt_ib, t_ib = self.ib.step()
            self.sub_steps+=1

            if dt_ib <= 1E-10:
                break

        self._sync_geometry()

        self.t = t_target
        return dt_ls, self.t

    def get_dataframe(self):
        df_ib = self.ib.get_dataframe()

        return df_ib