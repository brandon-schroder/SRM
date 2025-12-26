import numpy as np
from numba import njit


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
            if phi_A > 0: case |= 1
            if phi_B > 0: case |= 2
            if phi_C > 0: case |= 4
            if phi_D > 0: case |= 8

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
def make_continuous_contour(segments_array):
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

    # Find start node (odd degree or first node)
    start = 0
    for i in range(n_points):
        if degree[i] % 2 == 1:
            start = i
            break

    # Hierholzer's algorithm using arrays
    stack = np.empty(n_segs * 2, dtype=np.int32)
    stack[0] = start
    stack_ptr = 1

    path = np.empty(n_segs * 2, dtype=np.int32)
    path_ptr = 0

    edge_used = degree.copy()

    while stack_ptr > 0:
        u = stack[stack_ptr - 1]
        if edge_used[u] > 0:
            edge_used[u] -= 1
            v = adjacency[u, edge_used[u]]

            # Remove reverse edge
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

    # Reverse path
    for i in range(path_ptr // 2):
        path[i], path[path_ptr - 1 - i] = path[path_ptr - 1 - i], path[i]

    # Convert to output format
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
def calculate_axial_distributions(phi, phi_cas, cart_coords):
    """
    Calculate axial distributions of area and perimeter for all z-slices.

    Parameters:
    -----------
    phi : array of shape (n_r, n_t, n_z)
        Level set field
    phi_cas : array of shape (n_r, n_t, n_z)
        Casing level set field
    x_coords : array of shape (n_r, n_t, n_z)
        Cartesian x coordinates
    y_coords : array of shape (n_r, n_t, n_z)
        Cartesian y coordinates
    z_coords : array of shape (n_r, n_t, n_z)
        Cartesian z coordinates

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
        # Get z coordinate (assuming constant across r, t at each z-slice)
        z_distances[k] = z_coords[0, 0, k]

        # Extract slices
        phi_slice = phi[:, :, k]
        phicas_slice = phi_cas[:, :, k]
        x_slice = x_coords[:, :, k]
        y_slice = y_coords[:, :, k]

        # Extract contour segments
        contour_segments_list = marching_cubes(phi_slice, x_slice, y_slice)
        casing_segments_list = marching_cubes(phicas_slice, x_slice, y_slice)

        # Handle empty contours
        if len(contour_segments_list) == 0:
            areas[k] = 0.0
            perimeters[k] = 0.0
            continue

        # Convert to arrays
        n_contour = len(contour_segments_list)
        contour_segments_array = np.empty((n_contour, 4), dtype=np.float64)
        for i in range(n_contour):
            contour_segments_array[i] = contour_segments_list[i]

        # Make continuous contour
        contour_segments, contour_points = make_continuous_contour(contour_segments_array)

        # Calculate area
        areas[k] = contour_bounded_area(contour_points)

        # Handle casing clipping if casing contour exists
        if len(casing_segments_list) > 0:
            n_casing = len(casing_segments_list)
            casing_segments_array = np.empty((n_casing, 4), dtype=np.float64)
            for i in range(n_casing):
                casing_segments_array[i] = casing_segments_list[i]

            casing_segments, casing_points = make_continuous_contour(casing_segments_array)

            # Clip contour by casing
            clip_segments, clip_points = clip_contour(contour_segments, casing_segments)

            # Calculate perimeter from clipped segments
            if len(clip_segments) > 0:
                perimeters[k] = contour_length(clip_segments)
            else:
                perimeters[k] = 0.0
        else:
            # No casing, use full contour length
            perimeters[k] = contour_length(contour_segments)

    return z_distances, areas, perimeters

