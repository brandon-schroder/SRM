import pyvista as pv

propellant = pv.read("../NAWC6/geometry/NAWC6-SRM-Propellant.STL")
casing = pv.read("../NAWC6/geometry/NAWC6-SRM-Casing.STL")

plotter = pv.Plotter()
plotter.add_mesh(propellant, color="red", opacity=0.8)
plotter.add_mesh(casing, color="blue", opacity=0.8)
plotter.show()