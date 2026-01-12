import numpy as np
from numba import njit, prange

# --- Constants for Marching Squares ---
BIT_TL = 1
BIT_TR = 2
BIT_BR = 4
BIT_BL = 8

save_cache = False

@njit(cache=save_cache)
def get_intersection(p1_phi, p2_phi, p1_x, p1_y, p2_x, p2_y):
    """Finds the (x, y) coordinate of the zero-crossing using linear interpolation."""
    if p2_phi == p1_phi:
        return np.array([p1_x, p1_y])
    t = -p1_phi / (p2_phi - p1_phi)
    t = max(0.0, min(1.0, t))
    x = p1_x + t * (p2_x - p1_x)
    y = p1_y + t * (p2_y - p1_y)
    return np.array([x, y])


@njit(cache=save_cache)
def marching_cubes(phi_slice, x_coords, y_coords):
    """Extracts the contour segments (as a Numba List) of the zero-level set."""
    contour_segments = []
    n_r, n_t = phi_slice.shape
    p_top = np.zeros(2)
    p_bot = np.zeros(2)
    p_left = np.zeros(2)
    p_right = np.zeros(2)


    for j in range(n_t - 1): # Outer loop: Slow dimension
        for i in range(n_r - 1): # Inner loop: Fast dimension (contiguous)
            phi_A = phi_slice[i, j]
            phi_B = phi_slice[i, j + 1]
            phi_C = phi_slice[i + 1, j + 1]
            phi_D = phi_slice[i + 1, j]

            case = 0
            if phi_A > 0: case |= BIT_TL
            if phi_B > 0: case |= BIT_TR
            if phi_C > 0: case |= BIT_BR
            if phi_D > 0: case |= BIT_BL

            if case == 0 or case == 15:
                continue

            x_A, y_A = x_coords[i, j], y_coords[i, j]
            x_B, y_B = x_coords[i, j + 1], y_coords[i, j + 1]
            x_C, y_C = x_coords[i + 1, j + 1], y_coords[i + 1, j + 1]
            x_D, y_D = x_coords[i + 1, j], y_coords[i + 1, j]

            if (phi_A > 0) != (phi_B > 0):
                p_top = get_intersection(phi_A, phi_B, x_A, y_A, x_B, y_B)
            if (phi_B > 0) != (phi_C > 0):
                p_right = get_intersection(phi_B, phi_C, x_B, y_B, x_C, y_C)
            if (phi_D > 0) != (phi_C > 0):
                p_bot = get_intersection(phi_D, phi_C, x_D, y_D, x_C, y_C)
            if (phi_A > 0) != (phi_D > 0):
                p_left = get_intersection(phi_A, phi_D, x_A, y_A, x_D, y_D)

            if case == 1 or case == 14:
                contour_segments.append(np.array([p_left[0], p_left[1], p_top[0], p_top[1]]))
            elif case == 2 or case == 13:
                contour_segments.append(np.array([p_top[0], p_top[1], p_right[0], p_right[1]]))
            elif case == 3 or case == 12:
                contour_segments.append(np.array([p_left[0], p_left[1], p_right[0], p_right[1]]))
            elif case == 4 or case == 11:
                contour_segments.append(np.array([p_right[0], p_right[1], p_bot[0], p_bot[1]]))
            elif case == 5 or case == 10:
                phi_center = (phi_A + phi_B + phi_C + phi_D) / 4.0
                if (case == 5 and phi_center > 0) or (case == 10 and phi_center <= 0):
                    contour_segments.append(np.array([p_left[0], p_left[1], p_bot[0], p_bot[1]]))
                    contour_segments.append(np.array([p_top[0], p_top[1], p_right[0], p_right[1]]))
                else:
                    contour_segments.append(np.array([p_left[0], p_left[1], p_top[0], p_top[1]]))
                    contour_segments.append(np.array([p_right[0], p_right[1], p_bot[0], p_bot[1]]))
            elif case == 6 or case == 9:
                contour_segments.append(np.array([p_top[0], p_top[1], p_bot[0], p_bot[1]]))
            elif case == 7 or case == 8:
                contour_segments.append(np.array([p_left[0], p_left[1], p_bot[0], p_bot[1]]))
    return contour_segments


@njit(cache=save_cache)
def build_graph(segments_array):
    """Constructs an adjacency graph from a list of segments."""
    n_segs = len(segments_array)
    points_list = []
    point_to_idx = {}

    for seg in segments_array:
        for i in range(2):
            x, y = seg[i * 2], seg[i * 2 + 1]
            key = (x, y)
            if key not in point_to_idx:
                point_to_idx[key] = len(points_list)
                points_list.append(key)

    n_points = len(points_list)
    max_neighbors = 2 * n_segs
    adjacency = np.full((n_points, max_neighbors), -1, dtype=np.int32)
    degree = np.zeros(n_points, dtype=np.int32)

    for seg in segments_array:
        p1, p2 = (seg[0], seg[1]), (seg[2], seg[3])
        idx1, idx2 = point_to_idx[p1], point_to_idx[p2]
        adjacency[idx1, degree[idx1]] = idx2
        degree[idx1] += 1
        adjacency[idx2, degree[idx2]] = idx1
        degree[idx2] += 1

    return adjacency, degree, points_list


@njit(cache=save_cache)
def traverse_graph_path(adjacency, edge_used, start_node):
    """Finds a single continuous path using Hierholzer's algorithm logic."""
    stack = [start_node]
    path = []
    while len(stack) > 0:
        u = stack[-1]
        if edge_used[u] > 0:
            edge_used[u] -= 1
            v = adjacency[u, edge_used[u]]
            for i in range(edge_used[v]):
                if adjacency[v, i] == u:
                    adjacency[v, i] = adjacency[v, edge_used[v] - 1]
                    edge_used[v] -= 1
                    break
            stack.append(v)
        else:
            path.append(stack.pop())
    return path


@njit(cache=save_cache)
def make_continuous_contour(segments_array):
    """Finds all disconnected curves and returns them as a list of (segments, points)."""
    # Defensive check removed per request; empty arrays handle naturally
    adjacency, degree, points_list = build_graph(segments_array)
    n_points = len(points_list)
    edge_used = degree.copy()
    all_curves = []

    while np.sum(edge_used) > 0:
        start_node = -1
        for i in range(n_points):
            if edge_used[i] > 0 and degree[i] % 2 != 0:
                start_node = i
                break
        if start_node == -1:
            for i in range(n_points):
                if edge_used[i] > 0:
                    start_node = i
                    break
        if start_node == -1: break

        path = traverse_graph_path(adjacency, edge_used, start_node)
        path_ptr = len(path)

        ordered_segments = np.empty((path_ptr - 1, 4), dtype=np.float64)
        ordered_points = np.empty((path_ptr, 2), dtype=np.float64)

        for i in range(path_ptr):
            pt = points_list[path[i]]
            ordered_points[i, 0], ordered_points[i, 1] = pt[0], pt[1]
            if i < path_ptr - 1:
                pt_next = points_list[path[i + 1]]
                ordered_segments[i, 0], ordered_segments[i, 1] = pt[0], pt[1]
                ordered_segments[i, 2], ordered_segments[i, 3] = pt_next[0], pt_next[1]

        all_curves.append((ordered_segments, ordered_points))
    return all_curves


@njit(cache=save_cache)
def contour_bounded_area(contour_points):
    """Calculate area bounded by contour using shoelace formula (closed via origin)."""
    n = len(contour_points)
    if n < 2: return 0.0
    area = 0.0
    area += 0.0 * contour_points[0, 1] - contour_points[0, 0] * 0.0
    for i in range(n - 1):
        area += contour_points[i, 0] * contour_points[i + 1, 1] - contour_points[i + 1, 0] * contour_points[i, 1]
    area += contour_points[n - 1, 0] * 0.0 - 0.0 * contour_points[n - 1, 1]
    return abs(area) / 2.0


@njit(cache=save_cache)
def filter_curves(curves_list):
    """Selects the curve with the largest bounded area."""
    valid_curves = []
    if len(curves_list) == 0:
        return valid_curves

    max_area = -1.0
    best_idx = 0

    for i in range(len(curves_list)):
        _, pts = curves_list[i]
        area = contour_bounded_area(pts)
        if area > max_area:
            max_area = area
            best_idx = i

    valid_curves.append(curves_list[best_idx])
    return valid_curves


@njit(cache=save_cache)
def clip_contour(contour_segments, casing_segments):
    """Clips contour segments that share endpoints with casing segments."""
    n_casing = len(casing_segments)
    casing_points = {}
    for i in range(n_casing):
        casing_points[(casing_segments[i, 0], casing_segments[i, 1])] = True
        casing_points[(casing_segments[i, 2], casing_segments[i, 3])] = True

    n_contour = len(contour_segments)
    keep_mask = np.ones(n_contour, dtype=np.bool_)
    for i in range(n_contour):
        if (contour_segments[i, 0], contour_segments[i, 1]) in casing_points or \
                (contour_segments[i, 2], contour_segments[i, 3]) in casing_points:
            keep_mask[i] = False

    n_kept = np.sum(keep_mask)
    clip_segments = np.empty((n_kept, 4), dtype=np.float64)
    points_dict = {}
    idx = 0
    for i in range(n_contour):
        if keep_mask[i]:
            clip_segments[idx] = contour_segments[i]
            points_dict[(contour_segments[i, 0], contour_segments[i, 1])] = True
            points_dict[(contour_segments[i, 2], contour_segments[i, 3])] = True
            idx += 1

    clip_points = np.empty((len(points_dict), 2), dtype=np.float64)
    for i, pt in enumerate(points_dict.keys()):
        clip_points[i, 0], clip_points[i, 1] = pt[0], pt[1]

    return clip_segments, clip_points


@njit(cache=save_cache)
def contour_length(contour_segments):
    """Calculate total Euclidean length of all segments in the contour."""
    total_length = 0.0
    for i in range(len(contour_segments)):
        dx = contour_segments[i, 2] - contour_segments[i, 0]
        dy = contour_segments[i, 3] - contour_segments[i, 1]
        total_length += np.sqrt(dx * dx + dy * dy)
    return total_length


@njit(cache=save_cache)
def segments_list_to_array(segment_list):
    """Helper to convert Numba list of segments to fixed-size array."""
    n = len(segment_list)
    arr = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        arr[i] = segment_list[i]
    return arr


@njit(cache=save_cache)
def process_slice_geometry(phi_slice, phicas_slice, x_slice, y_slice):
    """Processes a 2D slice to calculate the primary contour area and perimeter."""
    contour_list = marching_cubes(phi_slice, x_slice, y_slice)
    if len(contour_list) == 0: return 0.0, 0.0, 0.0

    raw_segments = segments_list_to_array(contour_list)
    all_curves = make_continuous_contour(raw_segments)
    valid_curves = filter_curves(all_curves)

    if len(valid_curves) == 0: return 0.0, 0.0, 0.0

    ordered_segments, ordered_points = valid_curves[0]
    area = contour_bounded_area(ordered_points)

    wetted_perimeter = contour_length(ordered_segments)

    perimeter = 0.0
    casing_list = marching_cubes(phicas_slice, x_slice, y_slice)
    if len(casing_list) > 0:
        cas_raw = segments_list_to_array(casing_list)
        cas_curves = make_continuous_contour(cas_raw)
        # Select casing curve with max area as well
        valid_cas = filter_curves(cas_curves)
        if len(valid_cas) > 0:
            casing_segs, _ = valid_cas[0]
            clip_segs, _ = clip_contour(ordered_segments, casing_segs)
            perimeter = contour_length(clip_segs) if len(clip_segs) > 0 else 0.0
    else:
        perimeter = contour_length(ordered_segments)

    return area, perimeter, wetted_perimeter


@njit(parallel=True, cache=save_cache)
def calculate_axial_distributions(phi, phi_cas, cart_coords):
    """Calculate axial distributions of area and perimeter for all z-slices."""
    phi_combined = np.maximum(phi, phi_cas)
    _, _, n_z = phi_combined.shape
    x_coords, y_coords, z_coords = cart_coords

    z_distances = np.zeros(n_z, dtype=np.float64)
    areas = np.zeros(n_z, dtype=np.float64)
    burning_perimeters = np.zeros(n_z, dtype=np.float64)
    wetted_perimeters = np.zeros(n_z, dtype=np.float64)

    for k in prange(n_z):
        z_distances[k] = z_coords[0, 0, k]
        areas[k], burning_perimeters[k], wetted_perimeters[k] = process_slice_geometry(
            phi_combined[..., k], phi_cas[..., k],
            x_coords[..., k], y_coords[..., k]
        )

    return z_distances, areas, burning_perimeters, wetted_perimeters