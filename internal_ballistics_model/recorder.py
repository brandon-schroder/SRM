import numpy as np
import h5py
import os
from core.logger import HDF5Logger


class IBRecorder(HDF5Logger):
    def __init__(self, solver, buffer_size=100_000):
        """
        Internal Ballistics Recorder.
        """
        self.solver = solver
        self.cfg = solver.cfg

        # 1. Define Variables
        scalar_names = [
            "time", "dt", "p_head", "p_exit",
            "thrust", "isp", "mass_flow"
        ]

        field_names = [
            "pressure", "velocity", "density", "mach", "area"
        ]

        # 2. Define Units
        units = {
            "time": "s", "dt": "s",
            "p_head": "Pa", "p_exit": "Pa", "pressure": "Pa",
            "thrust": "N",
            "isp": "s",
            "mass_flow": "kg/s",
            "velocity": "m/s", "mach": "",
            "density": "kg/m^3",
            "area": "m^2"
        }

        # 3. Determine Field Shape
        n_cells_total = solver.grid.n_cells + 2 * solver.grid.ng
        field_shape = (n_cells_total,)

        # 4. Initialize Base Logger with Units
        filename = self.cfg.output_filename
        super().__init__(filename, scalar_names, field_names, field_shape, buffer_size, units)

        # 5. Save Static Geometry to HDF5
        with h5py.File(self.filename, "a") as f:
            if "geometry" not in f:
                g_geo = f.create_group("geometry")

                # Save full x coords (including ghosts)
                dset_x = g_geo.create_dataset("x", data=self.solver.grid.x_coords)
                dset_x.attrs["units"] = "m"

                # [FIX] Save a Zeros array for Y and Z coordinates
                # This prevents us from having to write "0.0 0.0..." as text in the XDMF
                zeros = np.zeros_like(self.solver.grid.x_coords)
                dset_z = g_geo.create_dataset("zeros", data=zeros)
                dset_z.attrs["units"] = "m"

        self.save_config(self.cfg)
        self.step_count = 0

    def save(self):
        """Retrieves data from solver and logs it."""
        interval = getattr(self.cfg, 'log_interval', 1)
        if self.step_count % interval != 0:
            self.step_count += 1
            return

        # Retrieve Data
        s = self.solver.state
        data = self.solver.get_derived_quantities()
        scalars = data["scalars"]
        fields = data["fields"]

        # Log Scalars
        for name in self.scalar_names:
            if name in scalars:
                self.log_scalar(name, scalars[name])

        # Log Fields
        self.log_field("pressure", s.p)
        self.log_field("velocity", s.u)
        self.log_field("density", s.rho)
        self.log_field("area", s.A)
        if "mach" in fields:
            self.log_field("mach", fields["mach"])

        self.check_buffer()
        self.step_count += 1

    def finalize(self):
        super().finalize()

        # 1. Calc Summary Stats
        with h5py.File(self.filename, "r") as f:
            if "timeseries/time" in f and f["timeseries/time"].shape[0] > 1:
                t = f["timeseries/time"][:]
                F = f["timeseries/thrust"][:]
                p_head = f["timeseries/p_head"][:]

                total_impulse = np.trapezoid(F, x=t)
                max_p_head = np.max(p_head)
                num_steps = len(t)
            else:
                total_impulse = 0.0
                max_p_head = 0.0
                num_steps = 0

        summary = {"total_impulse": total_impulse, "max_p_head": max_p_head}
        self.save_summary(summary)

        # 2. Generate XDMF
        if num_steps > 0:
            self.write_xdmf(num_steps)

    def write_xdmf(self, num_steps):
        """
        Generates an optimized XDMF file (.xmf) using XPath references
        to avoid duplicating grid geometry for every time step.
        """
        xmf_filename = self.filename.replace(".h5", ".xmf")
        h5_filename = os.path.basename(self.filename)

        # Dimensions
        n_points = self.field_shape[0]

        # Get actual time values for the animation
        with h5py.File(self.filename, 'r') as f:
            times = f['timeseries/time'][:num_steps]

        with open(xmf_filename, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            f.write('<Xdmf Version="3.0">\n')
            f.write(' <Domain>\n')

            # ==========================================================
            # 1. DEFINE THE STATIC MESH ONCE (The "Template")
            # ==========================================================
            f.write(f'  <Grid Name="Mesh_Definition" GridType="Uniform">\n')

            # Topology (Connectivity)
            # For a 1D line, we need to list points 0, 1, ... N-1
            f.write(f'   <Topology TopologyType="Polyline" NodesPerElement="{n_points}">\n')
            f.write(f'    <DataItem Dimensions="1 {n_points}" Format="XML">\n')
            f.write(f'      {" ".join(map(str, range(n_points)))}\n')
            f.write('    </DataItem>\n')
            f.write('   </Topology>\n')

            # Geometry (Coordinates)
            f.write('   <Geometry GeometryType="X_Y_Z">\n')

            # X (From HDF5)
            f.write(f'    <DataItem Name="X" Dimensions="{n_points}" NumberType="Float" Precision="4" Format="HDF">\n')
            f.write(f'     {h5_filename}:/geometry/x\n')
            f.write('    </DataItem>\n')

            # Y (From HDF5 - Zeros)
            f.write(f'    <DataItem Name="Y" Dimensions="{n_points}" NumberType="Float" Precision="4" Format="HDF">\n')
            f.write(f'     {h5_filename}:/geometry/zeros\n')
            f.write('    </DataItem>\n')

            # Z (From HDF5 - Zeros)
            f.write(f'    <DataItem Name="Z" Dimensions="{n_points}" NumberType="Float" Precision="4" Format="HDF">\n')
            f.write(f'     {h5_filename}:/geometry/zeros\n')
            f.write('    </DataItem>\n')

            f.write('   </Geometry>\n')
            f.write('  </Grid>\n')

            # ==========================================================
            # 2. DEFINE THE TIME SERIES (Referencing the Mesh)
            # ==========================================================
            f.write('  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')

            for i in range(num_steps):
                f.write(f'   <Grid Name="Step_{i}" GridType="Uniform">\n')
                f.write(f'    <Time Value="{times[i]}" />\n')

                # --- REFERENCE THE STATIC MESH ---
                # Instead of re-writing Topology/Geometry, point to the block above
                f.write(f'    <Topology Reference="/Xdmf/Domain/Grid[@Name=\'Mesh_Definition\']/Topology"/>\n')
                f.write(f'    <Geometry Reference="/Xdmf/Domain/Grid[@Name=\'Mesh_Definition\']/Geometry"/>\n')

                # --- ATTRIBUTES (These change every step) ---
                for field in self.field_names:
                    f.write(f'    <Attribute Name="{field}" AttributeType="Scalar" Center="Node">\n')
                    # Match dimensions to the 1xN slice produced by the hyperslab
                    f.write(f'     <DataItem ItemType="HyperSlab" Dimensions="1 {n_points}" Type="HyperSlab">\n')

                    # The Slicing Instructions (Start, Stride, Count)
                    f.write(f'       <DataItem Dimensions="3 2" Format="XML">\n')
                    f.write(f'         {i} 0\n')  # Start: Row i (Time), Col 0 (Space)
                    f.write(f'         1 1\n')  # Stride
                    f.write(f'         1 {n_points}\n')  # Count: 1 Row, N Columns
                    f.write('       </DataItem>\n')

                    # The Source HDF5
                    # Ensure dimensions match the actual dataset shape in HDF5
                    f.write(f'       <DataItem Dimensions="{num_steps} {n_points}" Format="HDF">\n')
                    f.write(f'         {h5_filename}:/fields/{field}\n')
                    f.write('       </DataItem>\n')

                    f.write('     </DataItem>\n')
                    f.write('    </Attribute>\n')

                f.write('   </Grid>\n')

            f.write('  </Grid>\n')
            f.write(' </Domain>\n')
            f.write('</Xdmf>\n')