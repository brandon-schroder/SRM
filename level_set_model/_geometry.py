import numpy as np
from numba import njit

# --- Constants for Marching Squares ---
BIT_TL = 1  # Top-Left
BIT_TR = 2  # Top-Right
BIT_BR = 4  # Bottom-Right
BIT_BL = 8  # Bottom-Left

@njit(cache=True)
def get_intersection(p1_phi, p2_phi, p1_x, p1_y, p2_x, p2_y):
    """Finds the (x, y) coordinate of the zero-crossing using linear interpolation."""
    if p2_phi == p1_phi:
        return np.array([p1_x, p1_y])
    t = -p1_phi / (p2_phi - p1_phi)
    t = max(0.0, min(1.0, t))
    x = p1_x + t * (p2_x - p1_x)
    y = p1_y + t * (p2_y - p1_y)
    return np.array([x, y])


@njit(cache=True)
def segments_list_to_array(segment_list):
    """Helper to convert Numba list of segments to fixed-size array."""
    n = len(segment_list)
    arr = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        arr[i] = segment_list[i]
    return arr


@njit(cache=True)
def marching_cubes(phi_slice, x_coords, y_coords):
    """
    Extracts the contour segments (as a Numba List) of the zero-level set.
    """
    contour_segments = []
    n_r, n_t = phi_slice.shape
    p_top = np.zeros(2)
    p_bot = np.zeros(2)
    p_left = np.zeros(2)
    p_right = np.zeros(2)

    for i in range(n_r - 1):
        for j in range(n_t - 1):
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


@njit
def build_graph(segments_array):
    """
    Constructs an adjacency graph from a list of segments.
    """
    n_segs = len(segments_array)

    # Extract unique points and build adjacency using indices
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

    # Build adjacency list using arrays (max degree = 2 * n_segs per point)
    max_neighbors = 2 * n_segs
    adjacency = np.full((n_points, max_neighbors), -1, dtype=np.int32)
    degree = np.zeros(n_points, dtype=np.int32)

    for seg in segments_array:
        p1 = (seg[0], seg[1])
        p2 = (seg[2], seg[3])
        idx1 = point_to_idx[p1]
        idx2 = point_to_idx[p2]

        # Add bidirectional edge
        adjacency[idx1, degree[idx1]] = idx2
        degree[idx1] += 1
        adjacency[idx2, degree[idx2]] = idx1
        degree[idx2] += 1

    return adjacency, degree, points_list


@njit
def traverse_graph_path(adjacency, degree, start_node, n_edges):
    """
    Finds a path through the graph using Hierholzer's algorithm.
    Returns the raw path array and the number of points in the path.
    """
    # Stack for backtracking
    stack = np.empty(n_edges * 2, dtype=np.int32)
    stack[0] = start_node
    stack_ptr = 1

    # Output path
    path = np.empty(n_edges * 2, dtype=np.int32)
    path_ptr = 0

    # Copy degree array to track used edges
    edge_used = degree.copy()

    while stack_ptr > 0:
        u = stack[stack_ptr - 1]
        if edge_used[u] > 0:
            edge_used[u] -= 1
            v = adjacency[u, edge_used[u]]

            # Remove reverse edge to ensure we don't traverse it back immediately
            for i in range(edge_used[v]):
                if adjacency[v, i] == u:
                    adjacency[v, i] = adjacency[v, edge_used[v] - 1]
                    edge_used[v] -= 1
                    break

            stack[stack_ptr] = v
            stack_ptr += 1
        else:
            stack_ptr -= 1
            path[path_ptr] = u
            path_ptr += 1

    return path, path_ptr


@njit
def make_continuous_contour(segments_array):
    n_segs = len(segments_array)

    # 1. Build the graph
    adjacency, degree, points_list = build_graph(segments_array)
    n_points = len(points_list)

    # 2. Find start node (odd degree or first node)
    start = 0
    for i in range(n_points):
        if degree[i] % 2 == 1:
            start = i
            break

    # 3. Run Traversal Algorithm
    path, path_ptr = traverse_graph_path(adjacency, degree, start, n_segs)

    # 4. Reverse path (Hierholzer's builds it backwards)
    for i in range(path_ptr // 2):
        path[i], path[path_ptr - 1 - i] = path[path_ptr - 1 - i], path[i]

    # 5. Format Output
    ordered_segments = np.empty((path_ptr - 1, 4), dtype=np.float64)
    ordered_points = np.empty((path_ptr, 2), dtype=np.float64)

    for i in range(path_ptr):
        pt = points_list[path[i]]
        ordered_points[i, 0] = pt[0]
        ordered_points[i, 1] = pt[1]

        if i < path_ptr - 1:
            pt_next = points_list[path[i + 1]]
            ordered_segments[i, 0] = pt[0]
            ordered_segments[i, 1] = pt[1]
            ordered_segments[i, 2] = pt_next[0]
            ordered_segments[i, 3] = pt_next[1]

    return ordered_segments, ordered_points


@njit
def clip_contour(contour_segments, casing_segments):
    # Build set of points from casing segments
    n_casing = len(casing_segments)
    casing_points = {}

    for i in range(n_casing):
        p1 = (casing_segments[i, 0], casing_segments[i, 1])
        p2 = (casing_segments[i, 2], casing_segments[i, 3])
        casing_points[p1] = True
        casing_points[p2] = True

    # Filter contour segments - keep only those with no casing points
    n_contour = len(contour_segments)
    keep_mask = np.ones(n_contour, dtype=np.bool_)

    for i in range(n_contour):
        p1 = (contour_segments[i, 0], contour_segments[i, 1])
        p2 = (contour_segments[i, 2], contour_segments[i, 3])

        # Remove segment if either endpoint is in casing
        if p1 in casing_points or p2 in casing_points:
            keep_mask[i] = False

    # Count kept segments
    n_kept = 0
    for i in range(n_contour):
        if keep_mask[i]:
            n_kept += 1

    # Build output arrays
    clip_segments = np.empty((n_kept, 4), dtype=np.float64)

    # Extract unique points from kept segments
    points_dict = {}
    idx = 0
    for i in range(n_contour):
        if keep_mask[i]:
            clip_segments[idx] = contour_segments[i]

            p1 = (contour_segments[i, 0], contour_segments[i, 1])
            p2 = (contour_segments[i, 2], contour_segments[i, 3])

            if p1 not in points_dict:
                points_dict[p1] = True
            if p2 not in points_dict:
                points_dict[p2] = True

            idx += 1

    # Convert points dict to array
    n_points = len(points_dict)
    clip_points = np.empty((n_points, 2), dtype=np.float64)

    idx = 0
    for point in points_dict.keys():
        clip_points[idx, 0] = point[0]
        clip_points[idx, 1] = point[1]
        idx += 1

    return clip_segments, clip_points


@njit
def contour_bounded_area(contour_points):
    """
    Calculate area bounded by contour using shoelace formula.
    The contour forms a closed loop: origin -> contour -> origin
    """
    n = len(contour_points)

    if n < 2:
        return 0.0

    area = 0.0

    # Edge from origin to first point
    x1, y1 = 0.0, 0.0
    x2, y2 = contour_points[0, 0], contour_points[0, 1]
    area += x1 * y2 - x2 * y1

    # Edges along the contour
    for i in range(n - 1):
        x1, y1 = contour_points[i, 0], contour_points[i, 1]
        x2, y2 = contour_points[i + 1, 0], contour_points[i + 1, 1]
        area += x1 * y2 - x2 * y1

    # Edge from last point back to origin
    x1, y1 = contour_points[n - 1, 0], contour_points[n - 1, 1]
    x2, y2 = 0.0, 0.0
    area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


@njit
def contour_length(contour_segments):
    """
    Calculate total length of all segments in the contour.
    """
    n = len(contour_segments)
    total_length = 0.0

    for i in range(n):
        x1 = contour_segments[i, 0]
        y1 = contour_segments[i, 1]
        x2 = contour_segments[i, 2]
        y2 = contour_segments[i, 3]

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx * dx + dy * dy)
        total_length += length

    return total_length


@njit
def process_slice_geometry(phi_slice, phicas_slice, x_slice, y_slice):
    """
    Processes a single 2D slice to calculate area and perimeter.
    """
    # 1. Generate Raw Contours
    contour_list = marching_cubes(phi_slice, x_slice, y_slice)
    casing_list = marching_cubes(phicas_slice, x_slice, y_slice)

    # Handle an empty propellant case early
    if len(contour_list) == 0:
        return 0.0, 0.0

    # 2. Convert to Arrays
    contour_segments_arr = segments_list_to_array(contour_list)

    # 3. Build Continuous Contour & Calculate Area
    contour_segments_ordered, contour_points = make_continuous_contour(contour_segments_arr)
    area = contour_bounded_area(contour_points)

    # 4. Handle Perimeter (with Casing Clip)
    perimeter = 0.0

    if len(casing_list) > 0:
        casing_segments_arr = segments_list_to_array(casing_list)
        casing_segments, _ = make_continuous_contour(casing_segments_arr)

        # Clip contour using the ordered segments for the propellant
        clip_segments, _ = clip_contour(contour_segments_ordered, casing_segments)

        if len(clip_segments) > 0:
            perimeter = contour_length(clip_segments)
    else:
        # No casing, use full contour length
        perimeter = contour_length(contour_segments_ordered)

    return area, perimeter


@njit
def calculate_axial_distributions(phi, phi_cas, cart_coords):
    """
    Calculate axial distributions of area and perimeter for all z-slices.

    Parameters:
    -----------
    phi : array of shape (n_r, n_t, n_z)
        Level set field
    phi_cas : array of shape (n_r, n_t, n_z)
        Casing level set field
    cart_coords : tuple of arrays
        (x_coords, y_coords, z_coords) each of shape (n_r, n_t, n_z)

    Returns:
    --------
    z_distances : array of shape (n_z,)
        Z coordinate at each slice
    areas : array of shape (n_z,)
        Bounded area at each z-slice
    perimeters : array of shape (n_z,)
        Interface perimeter at each z-slice
    """
    phi = np.maximum(phi, phi_cas)

    n_r, n_t, n_z = phi.shape
    x_coords, y_coords, z_coords = cart_coords

    z_distances = np.zeros(n_z, dtype=np.float64)
    areas = np.zeros(n_z, dtype=np.float64)
    perimeters = np.zeros(n_z, dtype=np.float64)

    for k in range(n_z):
        # 1. Update Z-distance (assuming constant across r, t at each z-slice)
        z_distances[k] = z_coords[0, 0, k]

        # 2. Extract Slices
        phi_slice = phi[:, :, k]
        phicas_slice = phi_cas[:, :, k]
        x_slice = x_coords[:, :, k]
        y_slice = y_coords[:, :, k]

        # 3. Delegate Processing
        areas[k], perimeters[k] = process_slice_geometry(
            phi_slice, phicas_slice, x_slice, y_slice
        )

    return z_distances, areas, perimeters