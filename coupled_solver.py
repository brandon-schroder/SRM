from level_set_model.structure import *
from level_set_model.level_set_grid import *
from internal_ballistics_model.structure import *




ls_solver = LS_Solver()
ls_solver.params = LS_Solver.Parameters(
    n_periodics = 11,
    size = [40, 40, 40],
    bounds=[10.0, 35.0, None, None, 0.0, 100.0],
    file_prop="C:\\Users\\brandon.schroder\\PycharmProjects\\SRM-LSM\\07R-SRM-Propellant.STL",
    file_case="C:\\Users\\brandon.schroder\\PycharmProjects\\SRM-LSM\\07R-SRM-Casing.STL",

    CFL = 0.8,
    t_end = 1.0,
    t_start = 0.0,

    br_initial = 10,

)

ls_solver = build_grid(ls_solver)
