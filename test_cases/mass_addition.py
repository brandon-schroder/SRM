import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from internal_ballistics_model import SimulationConfig, IBSolver
from internal_ballistics_model.numerics import primitives_to_conserved


def main():
    precision = np.float64

    # 1. Geometry Setup (Closed Constant-Volume Tube)
    # ---------------------------------------------------------
    z_geom = np.linspace(0.0, 1.0, 50, dtype=precision)  # Fewer cells needed for uniform flow
    A_geom = np.ones_like(z_geom) * 1.0  # Area = 1.0 m^2
    P_geom = np.ones_like(z_geom) * 1.0  # Burning Perimeter = 1.0 m

    A_casing = A_geom
    A_propellant = A_geom
    P_wetted = P_geom

    # 2. Configuration Setup
    # ---------------------------------------------------------
    config = SimulationConfig(
        n_cells=50,
        bounds=(0.0, 1.0),
        CFL=0.4,
        t_end=0.05,  # Short burn time is enough to verify rates
        gamma=1.4,
        R=287.0,
        br_initial=0.01,  # Hardcode burn rate to 1 cm/s
        a_coef=0.0,  # Disable pressure-dependent burning
        rho_p=1600.0,  # Propellant density
        Tf=2888.0,  # Flame temperature
        inlet_bc_type="reflective",   # Solid wall
        outlet_bc_type="reflective",  # Solid wall
        burn_model="override",

    )

    solver = IBSolver(config)
    solver.set_geometry(z_geom, A_geom, P_geom, P_wetted, A_propellant, A_casing)
    solver.initialize()

    # 3. Initial Conditions (Stagnant ambient air)
    # ---------------------------------------------------------
    interior = solver.grid.interior

    p_init = 100000.0
    T_init = 288.15
    rho_init = p_init / (config.R * T_init)

    solver.state.rho[:] = rho_init
    solver.state.p[:] = p_init
    solver.state.u[:] = 0.0  # Perfectly stagnant

    # Override burn rate array just to be absolutely certain it's uniform and constant
    solver.state.br[:] = config.br_initial

    solver.state.U[:] = primitives_to_conserved(
        solver.state.rho, solver.state.u, solver.state.p, solver.state.A, config.gamma, solver.state.U
    )

    print("Running Closed-Volume Mass Addition Test...")

    # Data trackers for the theoretical comparison
    time_history = []
    rho_history = []
    p_history = []

    time_history.append(0.0)
    rho_history.append(rho_init)
    p_history.append(p_init)

    # 4. Main Loop
    # ---------------------------------------------------------
    try:
        while solver.state.t < config.t_end:
            dt, current_time = solver.step()

            # Re-enforce constant burn rate in case the solver tries to update it
            solver.state.br[:] = config.br_initial

            # Record state at the exact center of the tube
            mid_idx = config.n_cells // 2 + solver.grid.ng
            time_history.append(current_time)
            rho_history.append(solver.state.rho[mid_idx])
            p_history.append(solver.state.p[mid_idx])

            step_count = int(solver.state.t / (dt + 1e-16))
            if step_count % 50 == 0:
                print(f"t={current_time:.4f}s | dt={dt:.2e} | p={p_history[-1] / 1000:.2f} kPa")

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    # 5. Theoretical Calculation
    # ---------------------------------------------------------
    t_array = np.array(time_history)

    # Mass conservation: d(rho*A)/dt = rho_p * P * r_b
    # Therefore: rho(t) = rho_init + (rho_p * P * r_b / A) * t
    mass_injection_rate = config.rho_p * 1.0 * config.br_initial / 1.0  # kg/m^3/s
    rho_theoretical = rho_init + mass_injection_rate * t_array

    # Energy conservation for a closed volume injected with stagnant gas at Tf
    # Specific enthalpy of injected gas: h_f = Cp * Tf = (gamma * R / (gamma - 1)) * Tf
    Cp = (config.gamma * config.R) / (config.gamma - 1.0)
    h_f = Cp * config.Tf

    # Energy equation: d(E)/dt = m_dot * h_f
    # Total Energy E = p/(gamma-1) (since u=0).
    # d(p)/dt = (gamma - 1) * mass_injection_rate * h_f
    dp_dt = (config.gamma - 1.0) * mass_injection_rate * h_f
    p_theoretical = p_init + dp_dt * t_array

    # 6. Plotting
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Density Plot
    axs[0].plot(t_array, rho_theoretical, 'k-', lw=2, label="Analytical (Conservation Laws)")
    axs[0].plot(time_history, rho_history, 'r--', lw=2, label="Numerical Solver")
    axs[0].set_ylabel('Density (kg/m³)')
    axs[0].set_title('Closed-Volume Bomb Calorimeter Test')
    axs[0].grid(True)
    axs[0].legend()

    # Pressure Plot
    axs[1].plot(t_array, p_theoretical, 'k-', lw=2)
    axs[1].plot(time_history, p_history, 'b--', lw=2)
    axs[1].set_ylabel('Pressure (Pa)')
    axs[1].set_xlabel('Time (s)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Check velocity to ensure it remains essentially zero (no spurious momentum)
    max_u = np.max(np.abs(solver.state.u[interior]))
    print(f"\nMaximum velocity generated in closed tube: {max_u:.4e} m/s")
    if max_u < 1e-3:
        print("SUCCESS: Spurious momentum generation is negligible.")


if __name__ == "__main__":
    main()