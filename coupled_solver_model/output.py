import os
import pandas as pd
import numpy as np
import pyvista as pv

class CSVHandler:

    def __init__(self, output_dir):
        self.timeseries_path = os.path.join(output_dir, "timeseries.csv")
        self.summary_path = os.path.join(output_dir, "summary.csv")
        self.history = []

    @staticmethod
    def _compute_metrics(solver):
        cfg = solver.ib.cfg
        state = solver.ib.state

        mass_flow_exit = state.rho[-1] * state.u[-1] * state.A[-1]
        thrust = (mass_flow_exit * state.u[-1]) + (state.p[-1] - cfg.p_inf) * state.A[-1]
        specific_impulse = thrust * mass_flow_exit / 9.81
        max_pressure = np.max(state.p)
        burn_area = np.sum(state.P * solver.ib.grid.dx[2])
        propellant_mass = np.sum(state.A_propellant * solver.ib.grid.dx[2] * cfg.rho_p)

        metrics = {
            "time": solver.t,
            "thrust": thrust,
            "specific_impulse": specific_impulse,
            "max_p": max_pressure,
            "ave_br": np.mean(state.br),
            "burn_area": burn_area,
            "prop_mass": propellant_mass,
            "mass_flow": state.rho[-1] * state.u[-1] * state.A[-1]
        }
        return metrics

    def record_step(self, solver):
        """Computes metrics, buffers them, and appends to timeseries CSV."""
        metrics = self._compute_metrics(solver)
        self.history.append(metrics)

        df_step = pd.DataFrame([metrics])
        header = not os.path.exists(self.timeseries_path)
        df_step.to_csv(self.timeseries_path, mode='a', index=False, header=header)

    def finalize(self):
        """Calculates final run statistics using buffered history."""
        if not self.history:
            return

        df: pd.DataFrame = pd.DataFrame(self.history)

        summary_data = {
            "total_impulse": [np.trapezoid(df['thrust'], x=df['time'])],
            "peak_pressure": [df['max_p'].max()],
            "avg_burn_rate": [df['ave_br'].mean()],
            "total_burn_time": [df['time'].iloc[-1] - df['time'].iloc[0]]
        }
        pd.DataFrame(summary_data).to_csv(self.summary_path, index=False)


class VTKHandler3D:
    """Handles 3D Level Set output as .vtk files."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_step(self, step, grid_pv, phi):
        """Saves the 3D grid with the current phi (level set) values."""
        plot_grid = grid_pv.copy()
        plot_grid.point_data["phi"] = phi.flatten(order="F")

        file_path = os.path.join(self.output_dir, f"level_set_{step:04d}.vtk")
        plot_grid.save(file_path)


class VTPHandler1D:
    """Handles 1D Flow data output as .vtp files."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_step(self, step, z_coords, state):
        """Saves 1D state data onto a 3D line representation."""
        points = np.column_stack((np.zeros_like(z_coords), np.zeros_like(z_coords), z_coords))
        poly = pv.PolyData(points)

        poly.point_data["pressure"] = state.p
        poly.point_data["velocity"] = state.u
        poly.point_data["density"] = state.rho
        poly.point_data["area"] = state.A
        poly.point_data["perimeter"] = state.P
        poly.point_data["area_p"] = state.A_propellant
        poly.point_data["perimeter_w"] = state.P_wetted
        poly.point_data["burn_rate"] = state.br

        file_path = os.path.join(self.output_dir, f"flow_1d_{step:04d}.vtp")
        poly.save(file_path)


class CoupledRecorder:
    def __init__(self, config):

        self.cfg = config
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.csv = None
        if self.cfg.save_csv:
            self.csv = CSVHandler(self.cfg.output_dir)

        self.vtk = None
        if self.cfg.save_vtk_3d:
            self.vtk = VTKHandler3D(os.path.join(self.cfg.output_dir, "vtk_3d"))

        self.vtp = None
        if self.cfg.save_vtp_1d:
            self.vtp = VTPHandler1D(os.path.join(self.cfg.output_dir, "vtp_1d"))

        self.step_count = 0

    def record_step(self, solver):
        if self.csv:
            self.csv.record_step(solver)

        if self.step_count % self.cfg.save_frequency == 0:
            if self.vtk:
                self.vtk.save_step(self.step_count, solver.ls.grid.pv_grid, solver.ls.phi)
            if self.vtp:
                self.vtp.save_step(self.step_count, solver.ib.grid.cart_coords[2], solver.ib.state)

        self.step_count += 1

    def finalize(self):
        if self.csv:
            self.csv.finalize()