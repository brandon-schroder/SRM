import numpy as np
from numba import njit, prange

# ==============================================================================
# PART 1: CONSTANTS & LOOKUP TABLES
# ==============================================================================

# Kuhn Triangulation Offsets (6 tets per cube)
# (Tet Index, Vertex Index, Dimension)
TET_OFFSETS = np.array([
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],  # Tet 0
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],  # Tet 1
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],  # Tet 2
    [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],  # Tet 3
    [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],  # Tet 4
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],  # Tet 5
], dtype=np.int8)

EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int8)
NODE_TO_EDGES = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]], dtype=np.int8)

# ==============================================================================
# SUB-GRID INTERFACE LOOKUP TABLES (Indexed by 4-bit mask: 0 to 15)
# Bit values: Node 0=1, Node 1=2, Node 2=4, Node 3=8
# ==============================================================================

# ---------------------------------------------------------
# CORNER CUTS (1 Node Fluid or 3 Nodes Fluid)
# ---------------------------------------------------------
# Maps mask to the single isolated node (Inside for 1-node, Outside for 3-node)
LUT_1_NODE    = np.array([-1,  0,  1, -1,  2, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1], dtype=np.int8)
LUT_3_NODE    = np.array([-1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1,  2, -1,  1,  0, -1], dtype=np.int8)
LUT_SURF_NODE = np.array([-1,  0,  1, -1,  2, -1, -1,  3,  3, -1, -1,  2, -1,  1,  0, -1], dtype=np.int8)

# ---------------------------------------------------------
# PRISM CUTS (Exactly 2 Nodes Fluid / 2 Nodes Solid)
# ---------------------------------------------------------
LUT_PRISM_NODES = np.array([
    [-1, -1], #  0: 0000 (0 nodes)
    [-1, -1], #  1: 0001 (1 node)
    [-1, -1], #  2: 0010 (1 node)
    [ 0,  1], #  3: 0011 (Nodes 0,1)
    [-1, -1], #  4: 0100 (1 node)
    [ 0,  2], #  5: 0101 (Nodes 0,2)
    [ 1,  2], #  6: 0110 (Nodes 1,2)
    [-1, -1], #  7: 0111 (3 nodes)
    [-1, -1], #  8: 1000 (1 node)
    [ 0,  3], #  9: 1001 (Nodes 0,3)
    [ 1,  3], # 10: 1010 (Nodes 1,3)
    [-1, -1], # 11: 1011 (3 nodes)
    [ 2,  3], # 12: 1100 (Nodes 2,3)
    [-1, -1], # 13: 1101 (3 nodes)
    [-1, -1], # 14: 1110 (3 nodes)
    [-1, -1], # 15: 1111 (4 nodes)
], dtype=np.int8)

LUT_PRISM_EDGES = np.array([
    [-1, -1, -1, -1], #  0: 0000
    [-1, -1, -1, -1], #  1: 0001
    [-1, -1, -1, -1], #  2: 0010
    [ 1,  2,  4,  3], #  3: 0011 (Nodes 0,1)
    [-1, -1, -1, -1], #  4: 0100
    [ 0,  2,  5,  3], #  5: 0101 (Nodes 0,2)
    [ 0,  1,  5,  4], #  6: 0110 (Nodes 1,2)
    [-1, -1, -1, -1], #  7: 0111
    [-1, -1, -1, -1], #  8: 1000
    [ 0,  4,  5,  1], #  9: 1001 (Nodes 0,3)
    [ 0,  3,  5,  2], # 10: 1010 (Nodes 1,3)
    [-1, -1, -1, -1], # 11: 1011
    [ 1,  3,  4,  2], # 12: 1100 (Nodes 2,3)
    [-1, -1, -1, -1], # 13: 1101
    [-1, -1, -1, -1], # 14: 1110
    [-1, -1, -1, -1], # 15: 1111
], dtype=np.int8)

LUT_PRISM_ORDER = np.array([
    [ 0,  1,  2,  3], #  0
    [ 0,  1,  2,  3], #  1
    [ 0,  1,  2,  3], #  2
    [ 0,  1,  2,  3], #  3
    [ 0,  1,  2,  3], #  4
    [ 0,  1,  2,  3], #  5
    [ 3,  0,  1,  2], #  6: Winding correction
    [ 0,  1,  2,  3], #  7
    [ 0,  1,  2,  3], #  8
    [ 3,  0,  1,  2], #  9: Winding correction
    [ 0,  1,  2,  3], # 10
    [ 0,  1,  2,  3], # 11
    [ 0,  1,  2,  3], # 12
    [ 0,  1,  2,  3], # 13
    [ 0,  1,  2,  3], # 14
    [ 0,  1,  2,  3], # 15
], dtype=np.int8)


# ==============================================================================
# PART 2: MATH HELPERS (Inlined, FastMath Enabled)
# ==============================================================================

@njit(inline='always', fastmath=True)
def _vol_tet(p0, p1, p2, p3):
    """Volume of a tetrahedron defined by 4 points."""
    cx = (p1[1] - p3[1]) * (p2[2] - p3[2]) - (p1[2] - p3[2]) * (p2[1] - p3[1])
    cy = (p1[2] - p3[2]) * (p2[0] - p3[0]) - (p1[0] - p3[0]) * (p2[2] - p3[2])
    cz = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    return abs((p0[0] - p3[0]) * cx + (p0[1] - p3[1]) * cy + (p0[2] - p3[2]) * cz) / 6.0

@njit(inline='always', fastmath=True)
def _area_tri_proj(p1, p2, p3):
    """Area projected onto lateral walls (perpendicular to Z)."""
    cx = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    cy = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    return 0.5 * np.sqrt(cx * cx + cy * cy)

@njit(inline='always', fastmath=True)
def _get_intersections(coords, phi, pts_out, ts_out):
    """
    Computes intersection points and t-values for 6 edges.
    Writes result into pre-allocated `pts_out` and `ts_out` to avoid heap allocation.
    """
    for i in range(6):
        m, n = EDGES[i]
        pm, pn = phi[m], phi[n]
        denom = pm - pn
        t = 0.0
        if denom != 0:
            t = pm / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        ts_out[i] = t

        # Unroll coordinate calculation for speed
        p_m, p_n = coords[m], coords[n]
        pts_out[i, 0] = p_m[0] + t * (p_n[0] - p_m[0])
        pts_out[i, 1] = p_m[1] + t * (p_n[1] - p_m[1])
        pts_out[i, 2] = p_m[2] + t * (p_n[2] - p_m[2])


# ==============================================================================
# PART 3: MICRO-KERNELS (Processing One Tet, Zero Allocation)
# ==============================================================================

@njit(inline='always', fastmath=True)
def calc_vol_tet(nodes, phi, pts_ws, ts_ws):
    """Calculates volume of region where phi > 0 inside a tet."""
    mask = 0
    if phi[0] > 0: mask |= 1
    if phi[1] > 0: mask |= 2
    if phi[2] > 0: mask |= 4
    if phi[3] > 0: mask |= 8

    if mask == 0: return 0.0

    full_vol = _vol_tet(nodes[0], nodes[1], nodes[2], nodes[3])
    if mask == 15: return full_vol

    # Pass pre-allocated workspaces instead of creating local arrays
    _get_intersections(nodes, phi, pts_ws, ts_ws)

    vol = 0.0
    if LUT_1_NODE[mask] != -1:  # 1 Node Inside
        node_idx = LUT_1_NODE[mask]
        e = NODE_TO_EDGES[node_idx]
        vol = _vol_tet(nodes[node_idx], pts_ws[e[0]], pts_ws[e[1]], pts_ws[e[2]])
    elif LUT_3_NODE[mask] != -1:  # 3 Nodes Inside
        out_node = LUT_3_NODE[mask]
        e = NODE_TO_EDGES[out_node]
        vol = full_vol - _vol_tet(nodes[out_node], pts_ws[e[0]], pts_ws[e[1]], pts_ws[e[2]])
    else:  # Prism
        edge_idxs = LUT_PRISM_EDGES[mask]
        if edge_idxs[0] != -1:
            node_idxs = LUT_PRISM_NODES[mask]
            perm = LUT_PRISM_ORDER[mask]

            q0 = pts_ws[edge_idxs[perm[0]]]
            q1 = pts_ws[edge_idxs[perm[1]]]
            q2 = pts_ws[edge_idxs[perm[2]]]
            q3 = pts_ws[edge_idxs[perm[3]]]

            nA, nB = nodes[node_idxs[0]], nodes[node_idxs[1]]
            vol = _vol_tet(nA, nB, q3, q2) + _vol_tet(nA, q2, q3, q1) + _vol_tet(nA, q1, q2, q0)
    return vol


@njit(inline='always', fastmath=True)
def calc_surf_tet(nodes, phi_target, phi_filter, pts_ws, ts_ws):
    """Calculates surface area of phi_target=0 interface, clipped by phi_filter > 0."""
    mask = 0
    if phi_target[0] > 0: mask |= 1
    if phi_target[1] > 0: mask |= 2
    if phi_target[2] > 0: mask |= 4
    if phi_target[3] > 0: mask |= 8

    if mask == 0 or mask == 15: return 0.0

    _get_intersections(nodes, phi_target, pts_ws, ts_ws)

    area = 0.0
    if LUT_SURF_NODE[mask] != -1:  # Triangle
        node_idx = LUT_SURF_NODE[mask]
        edges = NODE_TO_EDGES[node_idx]

        valid = 0
        for k in range(3):
            e_idx = edges[k]
            m, n = EDGES[e_idx]
            val = phi_filter[m] + ts_ws[e_idx] * (phi_filter[n] - phi_filter[m])
            if val > 0: valid += 1

        if valid == 3:
            area = _area_tri_proj(pts_ws[edges[0]], pts_ws[edges[1]], pts_ws[edges[2]])

    else:  # Quad
        edge_idxs = LUT_PRISM_EDGES[mask]
        if edge_idxs[0] != -1:
            valid = 0
            for k in range(4):
                e_idx = edge_idxs[k]
                m, n = EDGES[e_idx]
                val = phi_filter[m] + ts_ws[e_idx] * (phi_filter[n] - phi_filter[m])
                if val > 0: valid += 1

            if valid == 4:
                q0, q1, q2, q3 = pts_ws[edge_idxs[0]], pts_ws[edge_idxs[1]], pts_ws[edge_idxs[2]], pts_ws[edge_idxs[3]]
                area = _area_tri_proj(q0, q1, q2) + _area_tri_proj(q0, q2, q3)

    return area


# ==============================================================================
# PART 4: THE FUSED KERNEL (ZERO-COPY & PRE-ALLOCATED)
# ==============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def compute_geometry_3d_fused(nx, ny, nz, Xg, Yg, Zg, phi_p, phi_c):
    """
    Computes geometry for the entire 3D domain in a single parallel call.
    Parallelizes over the Z-axis (k) to eliminate thread pool overhead.
    """
    v_c_arr = np.empty(nz)
    v_g_arr = np.empty(nz)
    s_p_arr = np.empty(nz)
    s_c_arr = np.empty(nz)

    for k in prange(nz):
        # Local accumulators for this specific Z-slice
        slice_v_c = 0.0
        slice_v_g = 0.0
        slice_s_p = 0.0
        slice_s_c = 0.0

        # Pre-allocate Thread-Local Workspaces using np.empty
        # This completely eliminates runtime malloc/memset overhead in the inner loops
        loc_nodes = np.empty((4, 3), dtype=np.float64)
        loc_p = np.empty(4, dtype=np.float64)
        loc_c = np.empty(4, dtype=np.float64)
        loc_eff = np.empty(4, dtype=np.float64)
        pts_ws = np.empty((6, 3), dtype=np.float64)
        ts_ws = np.empty(6, dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                for t in range(6):
                    for v in range(4):
                        ii = i + TET_OFFSETS[t, v, 0]
                        jj = j + TET_OFFSETS[t, v, 1]
                        kk = k + TET_OFFSETS[t, v, 2]

                        loc_nodes[v, 0] = Xg[ii, jj, kk]
                        loc_nodes[v, 1] = Yg[ii, jj, kk]
                        loc_nodes[v, 2] = Zg[ii, jj, kk]

                        loc_p[v] = phi_p[ii, jj, kk]
                        loc_c[v] = phi_c[ii, jj, kk]

                        val_p = loc_p[v]
                        val_c = loc_c[v]
                        loc_eff[v] = val_p if val_p < val_c else val_c

                    # Pass workspaces down by reference
                    slice_v_c += calc_vol_tet(loc_nodes, loc_c, pts_ws, ts_ws)
                    slice_v_g += calc_vol_tet(loc_nodes, loc_eff, pts_ws, ts_ws)
                    slice_s_p += calc_surf_tet(loc_nodes, loc_p, loc_c, pts_ws, ts_ws)
                    slice_s_c += calc_surf_tet(loc_nodes, loc_c, loc_p, pts_ws, ts_ws)

        v_c_arr[k] = slice_v_c
        v_g_arr[k] = slice_v_g
        s_p_arr[k] = slice_s_p
        s_c_arr[k] = slice_s_c

    return v_c_arr, v_g_arr, s_p_arr, s_c_arr


# ==============================================================================
# PART 5: MAIN DRIVER
# ==============================================================================

def compute_geometric_distributions(grid, state):
    """
    Computes axial geometric distributions for the motor.

    Logic:
    - state.phi < 0 is fluid. We invert this to phi_prop > 0 for the volume kernel.
    - The kernels calculate values for a single SECTOR (defined by the grid).
    - We then scale by n_periodics and add the central core to get full 360 deg values.
    """
    ng = grid.ng

    # Invert phi so that the fluid/void zone is > 0 for the calc_vol_tet logic
    # Slice off the ghost cells so the 3D volume calculation strictly covers the physical domain
    phi_prop = np.ascontiguousarray(-state.phi[ng:-ng, :, ng:-ng])
    phi_case = np.ascontiguousarray(-state.casing[ng:-ng, :, ng:-ng])

    Xg = np.ascontiguousarray(grid.cart_coords[0, ng:-ng, :, ng:-ng])
    Yg = np.ascontiguousarray(grid.cart_coords[1, ng:-ng, :, ng:-ng])
    Zg = np.ascontiguousarray(grid.cart_coords[2, ng:-ng, :, ng:-ng])

    shape = Xg.shape
    nx_dual = shape[0] - 1
    ny_dual = shape[1] - 1
    nz_dual = shape[2] - 1

    r_min = grid.polar_coords[0, ng, 0, ng]
    core_area_base = np.pi * (r_min ** 2)
    n_periodics = grid.n_periodics

    # Execute fused computation block
    v_c_sector, v_void_sector, s_p_sector, s_c_sector = compute_geometry_3d_fused(
        nx_dual, ny_dual, nz_dual,
        Xg, Yg, Zg,
        phi_prop, phi_case
    )

    # Vectorized post-processing to convert volumes/areas to 1D axial distributions
    z_coords = Zg[0, 0, :-1]
    z_next = Zg[0, 0, 1:]
    dz = z_next - z_coords
    dz = np.where(dz == 0, 1.0, dz)  # Guard against division by zero

    # Convert 3D slice metrics to average 2D cross-sectional metrics
    casing_area_sector = v_c_sector / dz
    void_area_sector = v_void_sector / dz
    perimeter_sector = s_p_sector / dz
    casing_exposed_sector = s_c_sector / dz

    # The solid propellant area in this sector is the total casing area minus the void
    propellant_area_sector = casing_area_sector - void_area_sector

    # --- DYNAMIC CORE CATEGORIZATION ---
    # Determine the state of the core at the innermost radial cell
    # Since ghost cells are stripped, the inner physical boundary is exactly at index 0
    inner_idx = 0

    # Extract 1D axial profiles at the inner boundary for all azimuthal cells
    phi_prop_inner = 0.5 * (phi_prop[inner_idx, :, :-1] + phi_prop[inner_idx, :, 1:])
    phi_case_inner = 0.5 * (phi_case[inner_idx, :, :-1] + phi_case[inner_idx, :, 1:])

    # Compute the fractional state across the azimuthal ring (axis=0)
    fraction_chamber = np.mean(phi_case_inner > 0, axis=0, dtype=grid.cart_coords.dtype)
    fraction_void = np.mean(phi_prop_inner > 0, axis=0, dtype=grid.cart_coords.dtype)

    # Explicitly calculate the propellant fraction (where the level set is <= 0)
    fraction_propellant = np.mean(phi_prop_inner <= 0, axis=0, dtype=grid.cart_coords.dtype)

    # Partition the core area continuously based on the azimuthal fractions
    core_flow_areas = core_area_base * fraction_chamber * fraction_void
    core_propellant_areas = core_area_base * fraction_chamber * fraction_propellant
    core_chamber_areas = core_area_base * fraction_chamber
    # -----------------------------------

    # Final 360-degree Scaling
    # 1. Total Perimeter of the burning surface
    perimeters = perimeter_sector * n_periodics

    # 2. Hydraulic Perimeter (includes exposed casing for friction/heat transfer)
    hydraulic_perimeters = (perimeter_sector + casing_exposed_sector) * n_periodics

    # 3. Total Flow Area (Sum of all sector voids + the central hollow core)
    flow_areas = (void_area_sector * n_periodics) + core_flow_areas

    # 4. Total Casing Internal Area (The maximum possible flow area)
    casing_areas_final = (casing_area_sector * n_periodics) + core_chamber_areas

    # 5. Total Solid Propellant Area
    propellant_areas = (propellant_area_sector * n_periodics) + core_propellant_areas

    return z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas_final, propellant_areas