import time
from pathlib import Path
import coupled_solver_model

# 1. Robustly locate the YAML file relative to this script's directory
script_dir = Path(__file__).parent
yaml_file = script_dir / "NAWC6_INPUTS.yaml"

# 2. Load configuration from YAML
coupled_conf = coupled_solver_model.config.CoupledConfig.from_yaml(yaml_file)

# 3. Initialize solver
solver = coupled_solver_model.solver.CoupledSolver(coupled_conf)

print(f"Starting NAWC6 Simulation (Target: {coupled_conf.t_end}s)")

start_time = time.time()

# 4. Main execution loop
while solver.t < coupled_conf.t_end:
    dt_ls, t_current = solver.step()

    p_head = solver.ib.state.p.max()
    print(f"Time: {t_current:.4f} s | dt_ls: {dt_ls:.2e} | P_head: {p_head / 1e6:.2f} MPa | Sub-cycles: {solver.sub_steps} |")

    if t_current >= coupled_conf.t_end or p_head < coupled_conf.ib_config.p_inf:
        break

end_time = time.time()

solver.recorder.finalize()
print(f"Simulation time: {end_time - start_time:.2f} s")