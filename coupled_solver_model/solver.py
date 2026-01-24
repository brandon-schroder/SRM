
from .config import CoupledConfig

from internal_ballistics_model import IBSolver
from level_set_model import LSSolver

class CoupledSolver:
    def __init__(self, config: CoupledConfig):

        self.cfg = config

        # Instantiate the individual solvers
        self.ib = IBSolver(config.ib_config)
        self.ls = LSSolver(config.ls_config)

        self.t = 0.0

        self.initialize()


    def initialize(self):
        """Initializes both solvers and performs the first geometry sync."""
        # 1. Initialize Level Set (loads geometry, computes SDFs)
        # LSSolver automatically initializes in __init__, but we ensure state is ready

        self.ls.initialize()

        # 2. Sync Geometry (LS -> IB) before initializing flow
        self._sync_geometry()

        # 3. Initialize Flow (uses the geometry we just set)
        self.ib.initialize()


    def _sync_geometry(self):
        """
        Extracts A(z) and P(z) from the Level Set solver and updates the Flow solver.
        """
        # Get geometric distributions from LS
        x = self.ls.state.x
        A = self.ls.state.A_flow
        P = self.ls.state.P_propellant
        P_wetted = self.ls.state.P_wetted

        # Update IB Geometry
        self.ib.set_geometry(x, A, P, P_wetted)

    def step(self):
        """
        Advances the coupled simulation by one 'Level Set' timestep.
        The IB solver sub-cycles to catch up.
        """

        # --- 1. Advance Level Set (Master Time Step) ---
        # The LS solver determines the stable time step for geometry evolution
        # based on the current burn rate and grid resolution.

        # Update LS burn rate with the latest from IB
        self.ls.state.br = self.ls.set_burn_rate(self.ib.grid.x_coords, self.ib.state.br)

        # Take one step with Level Set solver
        dt_ls, t_ls_next = self.ls.step()

        # --- 2. Advance Internal Ballistics (Sub-cycling) ---
        # The fluid timescales are much faster than erosion.
        # We run the fluid solver until it reaches the new LS time.

        t_target = t_ls_next
        self.ib.cfg.t_end = t_target  # Ensure IB solver stops exactly at sync point

        while self.ib.state.t < t_target:
            dt_ib, t_ib = self.ib.step()

            # Break if solver finished or stalled
            if dt_ib <= 1E-10:
                break

        # --- 3. Sync Geometry for Next Step ---
        # Now that geometry has evolved to t_ls_next, update the flow path
        self._sync_geometry()

        self.t = t_target
        return dt_ls, self.t

    def get_dataframe(self):
        """Returns a combined dataframe of the current state."""
        df_ib = self.ib.get_dataframe()
        # You could merge this with LS data if needed, but IB data
        # now contains the interpolated Area used in the physics.
        return df_ib