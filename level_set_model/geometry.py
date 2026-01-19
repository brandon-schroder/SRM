import numpy as np
from numba import njit, prange


# ==============================================================================
# PART 1: SIMPLEX GRID LOGIC
# ==============================================================================

class SimplexGrid:
    def __init__(self, grid):
        self.grid = grid
        shape = grid.cart_coords.shape
        self.nx_dual = shape[1] - 1
        self.ny_dual = shape[2] - 1
        self.nz_dual = shape[3] - 1

        # Kuhn Triangulation Offsets (6 tets per cube)
        self.tet_offsets = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],  # Tet 0
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],  # Tet 1
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],  # Tet 2
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],  # Tet 3
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],  # Tet 4
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],  # Tet 5
        ], dtype=np.int8)


# ==============================================================================
# PART 2: INTEGRATOR KERNELS
# ==============================================================================

EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int8)
NODE_TO_EDGES = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]], dtype=np.int8)

LUT_PRISM_EDGES = np.full((16, 4), -1, dtype=np.int8)
LUT_PRISM_EDGES[3] =  [1, 2, 4, 3]
LUT_PRISM_EDGES[5] =  [0, 2, 5, 3]
LUT_PRISM_EDGES[6] =  [0, 1, 5, 4]
LUT_PRISM_EDGES[9] =  [0, 4, 5, 1]
LUT_PRISM_EDGES[10] = [0, 3, 5, 2]
LUT_PRISM_EDGES[12] = [1, 3, 4, 2]

LUT_PRISM_NODES = np.full((16, 2), -1, dtype=np.int8)
LUT_PRISM_NODES[3] =  [0, 1]
LUT_PRISM_NODES[5] =  [0, 2]
LUT_PRISM_NODES[6] =  [1, 2]
LUT_PRISM_NODES[9] =  [0, 3]
LUT_PRISM_NODES[10] = [1, 3]
LUT_PRISM_NODES[12] = [2, 3]

LUT_PRISM_ORDER = np.zeros((16, 4), dtype=np.int8)
for i in range(16): LUT_PRISM_ORDER[i] = [0, 1, 2, 3]
LUT_PRISM_ORDER[6] = [3, 0, 1, 2]
LUT_PRISM_ORDER[9] = [3, 0, 1, 2]


@njit(cache=True, fastmath=True)
def _vol_tet(a, b, c, d):
    cx = (b[1] - d[1]) * (c[2] - d[2]) - (b[2] - d[2]) * (c[1] - d[1])
    cy = (b[2] - d[2]) * (c[0] - d[0]) - (b[0] - d[0]) * (c[2] - d[2])
    cz = (b[0] - d[0]) * (c[1] - d[1]) - (b[1] - d[1]) * (c[0] - d[0])
    return abs((a[0] - d[0]) * cx + (a[1] - d[1]) * cy + (a[2] - d[2]) * cz) / 6.0


@njit(cache=True, fastmath=True)
def _area_tri_proj(p1, p2, p3):
    cx = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    cy = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    return 0.5 * np.sqrt(cx * cx + cy * cy)


@njit(cache=True)
def _get_intersections(coords, phi):
    pts = np.zeros((6, 3), dtype=np.float64)
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
        for k in range(3):
            pts[i, k] = coords[m, k] + t * (coords[n, k] - coords[m, k])
    return pts, ts


@njit(parallel=True, cache=True)
def integrate_volume_kernel(nodes, phi_vals):
    n_tets = nodes.shape[0]
    total_volume = 0.0
    for i in prange(n_tets):
        mask = 0
        if phi_vals[i, 0] > 0: mask |= 1
        if phi_vals[i, 1] > 0: mask |= 2
        if phi_vals[i, 2] > 0: mask |= 4
        if phi_vals[i, 3] > 0: mask |= 8

        if mask == 0: continue
        full_vol = _vol_tet(nodes[i, 0], nodes[i, 1], nodes[i, 2], nodes[i, 3])
        if mask == 15:
            total_volume += full_vol
            continue

        pts, _ = _get_intersections(nodes[i], phi_vals[i])

        if mask in [1, 2, 4, 8]:
            node_idx = {1: 0, 2: 1, 4: 2, 8: 3}[mask]
            e = NODE_TO_EDGES[node_idx]
            total_volume += _vol_tet(nodes[i, node_idx], pts[e[0]], pts[e[1]], pts[e[2]])
        elif mask in [14, 13, 11, 7]:
            out_node = {14: 0, 13: 1, 11: 2, 7: 3}[mask]
            e = NODE_TO_EDGES[out_node]
            total_volume += (full_vol - _vol_tet(nodes[i, out_node], pts[e[0]], pts[e[1]], pts[e[2]]))
        else:
            edge_idxs = LUT_PRISM_EDGES[mask]
            if edge_idxs[0] != -1:
                node_idxs = LUT_PRISM_NODES[mask]
                perm = LUT_PRISM_ORDER[mask]
                p_ring = np.empty((4, 3), dtype=np.float64)
                for k in range(4): p_ring[k] = pts[edge_idxs[k]]
                nA, nB = nodes[i, node_idxs[0]], nodes[i, node_idxs[1]]
                q0, q1, q2, q3 = p_ring[perm[0]], p_ring[perm[1]], p_ring[perm[2]], p_ring[perm[3]]
                total_volume += (_vol_tet(nA, nB, q3, q2) + _vol_tet(nA, q2, q3, q1) + _vol_tet(nA, q1, q2, q0))
    return total_volume


@njit(parallel=True, cache=True)
def integrate_surface_kernel(nodes, phi_target, phi_filter):
    n_tets = nodes.shape[0]
    total_area = 0.0
    for i in prange(n_tets):
        mask = 0
        if phi_target[i, 0] > 0: mask |= 1
        if phi_target[i, 1] > 0: mask |= 2
        if phi_target[i, 2] > 0: mask |= 4
        if phi_target[i, 3] > 0: mask |= 8

        if mask == 0 or mask == 15: continue
        pts, ts = _get_intersections(nodes[i], phi_target[i])

        if mask in [1, 2, 4, 8, 7, 11, 13, 14]:
            node_idx = {1: 0, 14: 0, 2: 1, 13: 1, 4: 2, 11: 2, 8: 3, 7: 3}[mask]
            edges = NODE_TO_EDGES[node_idx]
            valid_cnt = 0
            for k in range(3):
                e_idx = edges[k]
                m, n = EDGES[e_idx]
                if (phi_filter[i, m] + ts[e_idx] * (phi_filter[i, n] - phi_filter[i, m])) > 0:
                    valid_cnt += 1
            if valid_cnt == 3:
                total_area += _area_tri_proj(pts[edges[0]], pts[edges[1]], pts[edges[2]])
        else:
            edge_idxs = LUT_PRISM_EDGES[mask]
            if edge_idxs[0] != -1:
                valid_cnt = 0
                p_quad = np.empty((4, 3), dtype=np.float64)
                for k in range(4):
                    e_idx = edge_idxs[k]
                    m, n = EDGES[e_idx]
                    if (phi_filter[i, m] + ts[e_idx] * (phi_filter[i, n] - phi_filter[i, m])) > 0:
                        valid_cnt += 1
                if valid_cnt == 4:
                    p_quad = np.empty((4, 3), dtype=np.float64)
                    for k in range(4): p_quad[k] = pts[edge_idxs[k]]
                    total_area += (_area_tri_proj(p_quad[0], p_quad[1], p_quad[2]) +
                                   _area_tri_proj(p_quad[0], p_quad[2], p_quad[3]))
    return total_area


class SimplexIntegrator:
    def __init__(self, n_periodics=1, r_min=0.0):
        self.n_periodics = n_periodics
        self.r_min = r_min

    def compute_slice_metrics(self, nodes, phi_grain, phi_casing, phi_eff_buffer):
        # 1. FIXED dz CALCULATION
        # Use Vertex 3 (top) - Vertex 0 (bottom) of the first tetrahedron to get true layer thickness
        dz = nodes[0, 3, 2] - nodes[0, 0, 2]
        if dz == 0: dz = 1.0

        casing_area = integrate_volume_kernel(nodes, phi_casing) / dz

        # Intersection (Grain Material)
        np.minimum(phi_grain, phi_casing, out=phi_eff_buffer)

        # 'grain_area_local' contains the Volume of the Propellant
        grain_area_local = integrate_volume_kernel(nodes, phi_eff_buffer) / dz

        perimeter = integrate_surface_kernel(nodes, phi_grain, phi_casing) / dz
        casing_exposed = integrate_surface_kernel(nodes, phi_casing, phi_grain) / dz

        core_area = np.pi * (self.r_min ** 2)
        if casing_area < 1E-9: core_area = 0.0

        flow_port_area = casing_area - grain_area_local  # The actual empty space

        perimeter_ = perimeter * self.n_periodics
        hydraulic_perimeter_ = (perimeter + casing_exposed) * self.n_periodics
        propellant_area_ = flow_port_area * self.n_periodics
        casing_area_ = casing_area * self.n_periodics + core_area
        flow_area_ = grain_area_local * self.n_periodics + core_area

        return {
            "perimeter": perimeter_,
            "hydraulic_perimeter": hydraulic_perimeter_,
            "propellant_area": propellant_area_,  # Actually Port Area
            "casing_area": casing_area_,
            "flow_area": flow_area_  # Actually Propellant Area
        }


# ==============================================================================
# PART 3: OPTIMIZED GATHER & MAIN LOOP
# ==============================================================================

@njit(parallel=True, cache=True)
def fill_slice_buffers(k, nx, ny, tet_offsets, Xg, Yg, Zg, phi_p, phi_c,
                       out_coords, out_phi_p, out_phi_c):
    for i in prange(nx):
        for j in range(ny):
            base_idx = (i * ny + j) * 6
            for t in range(6):
                buf_idx = base_idx + t
                for v in range(4):
                    ii = i + tet_offsets[t, v, 0]
                    jj = j + tet_offsets[t, v, 1]
                    kk = k + tet_offsets[t, v, 2]

                    out_coords[buf_idx, v, 0] = Xg[ii, jj, kk]
                    out_coords[buf_idx, v, 1] = Yg[ii, jj, kk]
                    out_coords[buf_idx, v, 2] = Zg[ii, jj, kk]

                    out_phi_p[buf_idx, v] = phi_p[ii, jj, kk]
                    out_phi_c[buf_idx, v] = phi_c[ii, jj, kk]


def compute_geometric_distributions(grid, state):

    phi_prop = np.ascontiguousarray(-state.phi)
    phi_case = np.ascontiguousarray(-state.casing)

    Xg = np.ascontiguousarray(grid.cart_coords[0])
    Yg = np.ascontiguousarray(grid.cart_coords[1])
    Zg = np.ascontiguousarray(grid.cart_coords[2])

    simplex = SimplexGrid(grid)
    integrator = SimplexIntegrator(n_periodics=grid.n_periodics, r_min=grid.polar_coords[0, :].min())

    nx, ny, nz = simplex.nx_dual, simplex.ny_dual, simplex.nz_dual

    perimeters = np.zeros(nz)
    hydraulic_perimeters = np.zeros(nz)
    flow_areas = np.zeros(nz)
    casing_areas = np.zeros(nz)
    propellant_areas = np.zeros(nz)
    z_coords = np.zeros(nz)

    n_tets_slice = nx * ny * 6
    slab_coords = np.zeros((n_tets_slice, 4, 3), dtype=np.float64)
    slab_phi_p = np.zeros((n_tets_slice, 4), dtype=np.float64)
    slab_phi_c = np.zeros((n_tets_slice, 4), dtype=np.float64)
    slab_phi_eff_buf = np.zeros((n_tets_slice, 4), dtype=np.float64)


    for k in range(nz):
        fill_slice_buffers(
            k, nx, ny, simplex.tet_offsets,
            Xg, Yg, Zg, phi_prop, phi_case,
            slab_coords, slab_phi_p, slab_phi_c
        )

        metrics = integrator.compute_slice_metrics(slab_coords, slab_phi_p, slab_phi_c, slab_phi_eff_buf)

        perimeters[k] = metrics["perimeter"]
        hydraulic_perimeters[k] = metrics["hydraulic_perimeter"]
        flow_areas[k] = metrics["flow_area"]
        casing_areas[k] = metrics["casing_area"]
        propellant_areas[k] = metrics["propellant_area"]

        z_coords[k] = slab_coords[0, 0, 2]

    return z_coords, perimeters, hydraulic_perimeters, flow_areas, casing_areas, propellant_areas