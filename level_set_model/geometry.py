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

# Prism Decomposition Tables
LUT_PRISM_EDGES = np.full((16, 4), -1, dtype=np.int8)
LUT_PRISM_EDGES[3]  = [1, 2, 4, 3]
LUT_PRISM_EDGES[5]  = [0, 2, 5, 3]
LUT_PRISM_EDGES[6]  = [0, 1, 5, 4]
LUT_PRISM_EDGES[9]  = [0, 4, 5, 1]
LUT_PRISM_EDGES[10] = [0, 3, 5, 2]
LUT_PRISM_EDGES[12] = [1, 3, 4, 2]

LUT_PRISM_NODES = np.full((16, 2), -1, dtype=np.int8)
LUT_PRISM_NODES[3]  = [0, 1]
LUT_PRISM_NODES[5]  = [0, 2]
LUT_PRISM_NODES[6]  = [1, 2]
LUT_PRISM_NODES[9]  = [0, 3]
LUT_PRISM_NODES[10] = [1, 3]
LUT_PRISM_NODES[12] = [2, 3]

LUT_PRISM_ORDER = np.zeros((16, 4), dtype=np.int8)
for i in range(16): LUT_PRISM_ORDER[i] = [0, 1, 2, 3]
LUT_PRISM_ORDER[6] = [3, 0, 1, 2]
LUT_PRISM_ORDER[9] = [3, 0, 1, 2]


# ==============================================================================
# PART 2: MATH HELPERS (Inlined)
# ==============================================================================

@njit(inline='always')
def _vol_tet(p0, p1, p2, p3):
    """Volume of a tetrahedron defined by 4 points."""
    cx = (p1[1] - p3[1]) * (p2[2] - p3[2]) - (p1[2] - p3[2]) * (p2[1] - p3[1])
    cy = (p1[2] - p3[2]) * (p2[0] - p3[0]) - (p1[0] - p3[0]) * (p2[2] - p3[2])
    cz = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    return abs((p0[0] - p3[0]) * cx + (p0[1] - p3[1]) * cy + (p0[2] - p3[2]) * cz) / 6.0


@njit(inline='always')
def _area_tri_proj(p1, p2, p3):
    """Area projected onto lateral walls (perpendicular to Z)."""
    cx = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    cy = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    return 0.5 * np.sqrt(cx * cx + cy * cy)


@njit(inline='always')
def _get_intersections(coords, phi, pts_out):
    """
    Computes intersection points and t-values for 6 edges.
    Writes result into `pts_out` to avoid allocation.
    Returns array of t-values.
    """
    ts = np.zeros(6, dtype=np.float64)
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
        ts[i] = t

        # Unroll coordinate calculation for speed
        p_m, p_n = coords[m], coords[n]
        pts_out[i, 0] = p_m[0] + t * (p_n[0] - p_m[0])
        pts_out[i, 1] = p_m[1] + t * (p_n[1] - p_m[1])
        pts_out[i, 2] = p_m[2] + t * (p_n[2] - p_m[2])
    return ts


# ==============================================================================
# PART 3: MICRO-KERNELS (Processing One Tet)
# ==============================================================================

@njit(inline='always')
def calc_vol_tet(nodes, phi):
    """Calculates volume of region where phi > 0 inside a tet."""
    mask = 0
    if phi[0] > 0: mask |= 1
    if phi[1] > 0: mask |= 2
    if phi[2] > 0: mask |= 4
    if phi[3] > 0: mask |= 8

    if mask == 0: return 0.0

    full_vol = _vol_tet(nodes[0], nodes[1], nodes[2], nodes[3])
    if mask == 15: return full_vol

    # Intersection Points Buffer (Stack allocated)
    pts = np.zeros((6, 3), dtype=np.float64)
    _get_intersections(nodes, phi, pts)

    vol = 0.0
    if mask in [1, 2, 4, 8]:  # 1 Node Inside
        node_idx = {1: 0, 2: 1, 4: 2, 8: 3}[mask]
        e = NODE_TO_EDGES[node_idx]
        vol = _vol_tet(nodes[node_idx], pts[e[0]], pts[e[1]], pts[e[2]])
    elif mask in [14, 13, 11, 7]:  # 3 Nodes Inside
        out_node = {14: 0, 13: 1, 11: 2, 7: 3}[mask]
        e = NODE_TO_EDGES[out_node]
        vol = full_vol - _vol_tet(nodes[out_node], pts[e[0]], pts[e[1]], pts[e[2]])
    else:  # Prism
        edge_idxs = LUT_PRISM_EDGES[mask]
        if edge_idxs[0] != -1:
            node_idxs = LUT_PRISM_NODES[mask]
            perm = LUT_PRISM_ORDER[mask]

            # Gather prism points manually to avoid array slicing overhead
            q0 = pts[edge_idxs[perm[0]]]
            q1 = pts[edge_idxs[perm[1]]]
            q2 = pts[edge_idxs[perm[2]]]
            q3 = pts[edge_idxs[perm[3]]]

            nA, nB = nodes[node_idxs[0]], nodes[node_idxs[1]]
            vol = _vol_tet(nA, nB, q3, q2) + _vol_tet(nA, q2, q3, q1) + _vol_tet(nA, q1, q2, q0)
    return vol


@njit(inline='always')
def calc_surf_tet(nodes, phi_target, phi_filter):
    """Calculates surface area of phi_target=0 interface, clipped by phi_filter > 0."""
    mask = 0
    if phi_target[0] > 0: mask |= 1
    if phi_target[1] > 0: mask |= 2
    if phi_target[2] > 0: mask |= 4
    if phi_target[3] > 0: mask |= 8

    if mask == 0 or mask == 15: return 0.0

    pts = np.zeros((6, 3), dtype=np.float64)
    ts = _get_intersections(nodes, phi_target, pts)

    area = 0.0

    if mask in [1, 2, 4, 8, 7, 11, 13, 14]:  # Triangle
        node_idx = {1: 0, 14: 0, 2: 1, 13: 1, 4: 2, 11: 2, 8: 3, 7: 3}[mask]
        edges = NODE_TO_EDGES[node_idx]

        # Check center of triangle (simplified check) or vertices
        valid = 0
        for k in range(3):
            e_idx = edges[k]
            m, n = EDGES[e_idx]
            val = phi_filter[m] + ts[e_idx] * (phi_filter[n] - phi_filter[m])
            if val > 0: valid += 1

        if valid == 3:
            area = _area_tri_proj(pts[edges[0]], pts[edges[1]], pts[edges[2]])

    else:  # Quad
        edge_idxs = LUT_PRISM_EDGES[mask]
        if edge_idxs[0] != -1:
            valid = 0
            for k in range(4):
                e_idx = edge_idxs[k]
                m, n = EDGES[e_idx]
                val = phi_filter[m] + ts[e_idx] * (phi_filter[n] - phi_filter[m])
                if val > 0: valid += 1

            if valid == 4:
                q0, q1, q2, q3 = pts[edge_idxs[0]], pts[edge_idxs[1]], pts[edge_idxs[2]], pts[edge_idxs[3]]
                area = _area_tri_proj(q0, q1, q2) + _area_tri_proj(q0, q2, q3)

    return area


# ==============================================================================
# PART 4: THE FUSED KERNEL (ZERO-COPY)
# ==============================================================================

# 1. New 3D Fused Kernel (Replaces compute_slice_fused)
@njit(parallel=True, cache=True)
def compute_geometry_3d_fused(nx, ny, nz, Xg, Yg, Zg, phi_p, phi_c):
    """
    Computes geometry for the entire 3D domain in a single parallel call.
    Parallelizes over the Z-axis (k) to eliminate thread pool overhead.
    """
    # Pre-allocate output arrays for the Z-slices
    v_c_arr = np.empty(nz)
    v_g_arr = np.empty(nz)
    s_p_arr = np.empty(nz)
    s_c_arr = np.empty(nz)

    # Parallelize the OUTER loop
    for k in prange(nz):
        # Local accumulators for this specific Z-slice
        slice_v_c = 0.0
        slice_v_g = 0.0
        slice_s_p = 0.0
        slice_s_c = 0.0

        # Standard inner loops
        for i in range(nx):
            for j in range(ny):
                loc_nodes = np.zeros((4, 3), dtype=np.float64)
                loc_p = np.zeros(4, dtype=np.float64)
                loc_c = np.zeros(4, dtype=np.float64)
                loc_eff = np.zeros(4, dtype=np.float64)

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

                    slice_v_c += calc_vol_tet(loc_nodes, loc_c)
                    slice_v_g += calc_vol_tet(loc_nodes, loc_eff)
                    slice_s_p += calc_surf_tet(loc_nodes, loc_p, loc_c)
                    slice_s_c += calc_surf_tet(loc_nodes, loc_c, loc_p)

        # Store results for this slice
        v_c_arr[k] = slice_v_c
        v_g_arr[k] = slice_v_g
        s_p_arr[k] = slice_s_p
        s_c_arr[k] = slice_s_c

    return v_c_arr, v_g_arr, s_p_arr, s_c_arr

# ==============================================================================
# PART 5: MAIN DRIVER
# ==============================================================================

def compute_geometric_distributions(grid, state):
    phi_prop = np.ascontiguousarray(-state.phi)
    phi_case = np.ascontiguousarray(-state.casing)

    Xg = np.ascontiguousarray(grid.cart_coords[0])
    Yg = np.ascontiguousarray(grid.cart_coords[1])
    Zg = np.ascontiguousarray(grid.cart_coords[2])

    shape = grid.cart_coords.shape
    nx_dual = shape[1] - 1
    ny_dual = shape[2] - 1
    nz_dual = shape[3] - 1

    r_min = grid.polar_coords[0, :].min()
    core_area_base = np.pi * (r_min ** 2)
    n_periodics = grid.n_periodics

    # Single parallel call handles the entire grid
    v_c_arr, v_g_arr, s_p_arr, s_c_arr = compute_geometry_3d_fused(
        nx_dual, ny_dual, nz_dual,
        Xg, Yg, Zg,
        phi_prop, phi_case
    )

    # Vectorized post-processing
    z_coords = Zg[0, 0, :-1]  # Get all z-coords at once
    z_next = Zg[0, 0, 1:]
    dz = z_next - z_coords
    dz = np.where(dz == 0, 1.0, dz)  # Protect against division by zero

    casing_areas = v_c_arr / dz
    grain_areas = v_g_arr / dz
    perimeters_raw = s_p_arr / dz
    casing_exposed = s_c_arr / dz

    flow_port_areas = casing_areas - grain_areas

    # Core correction logic using numpy vectorization
    core_areas = np.full(nz_dual, core_area_base)
    core_areas[casing_areas < 1e-9] = 0.0

    perimeters = perimeters_raw * n_periodics
    hydraulic_perimeters = (perimeters_raw + casing_exposed) * n_periodics
    casing_areas_final = casing_areas * n_periodics + core_areas
    flow_areas = grain_areas * n_periodics + core_areas
    propellant_areas = flow_port_areas * n_periodics

    return z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas_final, propellant_areas
