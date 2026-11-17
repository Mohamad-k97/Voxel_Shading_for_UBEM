# footprint_geometry.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np


# ============================================================
# Optional Shapely support
# ============================================================
#
# This module works without Shapely.
# If Shapely is available, pairwise convex-piece merging becomes more robust,
# especially when one piece shares only part of another piece's edge.

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except Exception:
    ShapelyPolygon = None
    SHAPELY_AVAILABLE = False


EPS = 1e-10


# ============================================================
# Basic helpers
# ============================================================

def as_xy(points):
    """
    Convert an input footprint to an Nx2 float64 array.

    Accepts:
        Nx2 array/list
        Nx3 array/list, using x,y and ignoring z

    Removes repeated closing vertex if present.
    """
    p = np.asarray(points, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError("Footprint points must be a 2D array.")

    if p.shape[1] < 2:
        raise ValueError("Footprint points must have at least x,y columns.")

    p = p[:, :2].copy()

    if len(p) >= 2:
        if np.linalg.norm(p[0] - p[-1]) <= EPS:
            p = p[:-1].copy()

    return p


def cross2(a, b):
    """
    2D cross product scalar.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_signed_area_xy(points_xy):
    """
    Signed polygon area.

    Positive = CCW
    Negative = CW
    """
    p = as_xy(points_xy)

    if len(p) < 3:
        return 0.0

    x = p[:, 0]
    y = p[:, 1]

    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area_xy(points_xy):
    return abs(polygon_signed_area_xy(points_xy))


def ensure_ccw_xy(points_xy):
    """
    Return polygon vertices in CCW order.
    """
    p = as_xy(points_xy)

    if polygon_signed_area_xy(p) < 0.0:
        p = p[::-1].copy()

    return p


def ensure_cw_xy(points_xy):
    """
    Return polygon vertices in CW order.
    """
    p = as_xy(points_xy)

    if polygon_signed_area_xy(p) > 0.0:
        p = p[::-1].copy()

    return p


def edge_lengths_xy(points_xy):
    p = as_xy(points_xy)

    if len(p) < 2:
        return np.array([], dtype=np.float64)

    q = np.roll(p, -1, axis=0)

    return np.linalg.norm(q - p, axis=1)


def polygon_perimeter_xy(points_xy):
    return float(np.sum(edge_lengths_xy(points_xy)))


def bbox_xy(points_xy):
    p = as_xy(points_xy)

    return (
        float(np.min(p[:, 0])),
        float(np.min(p[:, 1])),
        float(np.max(p[:, 0])),
        float(np.max(p[:, 1])),
    )


def polygon_centroid_xy(points_xy):
    """
    Area-weighted centroid for a simple polygon.

    Falls back to mean point if polygon area is tiny.
    """
    p = as_xy(points_xy)

    if len(p) == 0:
        raise ValueError("Cannot compute centroid of empty polygon.")

    if len(p) < 3:
        return np.mean(p, axis=0)

    x = p[:, 0]
    y = p[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area2 = float(np.sum(cross))

    if abs(area2) <= EPS:
        return np.mean(p, axis=0)

    cx = float(np.sum((x + x_next) * cross) / (3.0 * area2))
    cy = float(np.sum((y + y_next) * cross) / (3.0 * area2))

    return np.array([cx, cy], dtype=np.float64)


# ============================================================
# Cleaning
# ============================================================

def remove_near_duplicate_vertices_xy(points_xy, tol=1e-8):
    """
    Remove consecutive duplicate or near-duplicate vertices.
    Also removes a repeated closing vertex.
    """
    p = as_xy(points_xy)

    if len(p) == 0:
        return p

    kept = [p[0]]

    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - kept[-1]) > tol:
            kept.append(p[i])

    p2 = np.asarray(kept, dtype=np.float64)

    if len(p2) >= 2:
        if np.linalg.norm(p2[0] - p2[-1]) <= tol:
            p2 = p2[:-1].copy()

    return p2


def is_collinear_triplet(a, b, c, tol=1e-10):
    ab = b - a
    bc = c - b

    scale = max(1.0, np.linalg.norm(ab) * np.linalg.norm(bc))

    return abs(cross2(ab, bc)) <= tol * scale


def remove_collinear_vertices_xy(points_xy, tol=1e-10, max_iter=20):
    """
    Remove vertices that are nearly collinear with their neighbors.

    Important:
        This is useful before barrier generation.
        During convex-piece merging, Shapely handles shared partial edges better.
    """
    p = as_xy(points_xy)

    if len(p) <= 3:
        return p

    for _ in range(max_iter):
        n = len(p)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            a = p[(i - 1) % n]
            b = p[i]
            c = p[(i + 1) % n]

            if is_collinear_triplet(a, b, c, tol=tol):
                keep[i] = False

        if np.all(keep):
            break

        p = p[keep].copy()

        if len(p) <= 3:
            break

    return p


def clean_footprint_xy(
    points_xy,
    duplicate_tol=1e-8,
    collinear_tol=1e-10,
    remove_collinear=True,
    ensure_ccw=True,
):
    """
    Clean a footprint for geometric processing.

    Steps:
        1. convert to Nx2
        2. remove repeated closing vertex
        3. remove consecutive duplicate vertices
        4. optionally remove collinear vertices
        5. optionally orient CCW

    Assumptions:
        - simple polygon
        - no holes

    Holes/courtyards should be handled later as separate rings or separate
    solid regions.
    """
    p = as_xy(points_xy)
    p = remove_near_duplicate_vertices_xy(p, tol=duplicate_tol)

    if remove_collinear:
        p = remove_collinear_vertices_xy(p, tol=collinear_tol)

    if len(p) < 3:
        raise ValueError("Cleaned footprint has fewer than 3 vertices.")

    area = polygon_signed_area_xy(p)

    if abs(area) <= EPS:
        raise ValueError("Cleaned footprint has near-zero area.")

    if ensure_ccw:
        p = ensure_ccw_xy(p)

    return p


# ============================================================
# Segment / polygon validity helpers
# ============================================================

def orientation(a, b, c, tol=1e-12):
    """
    Return:
        +1 if a,b,c make a left turn
        -1 if right turn
         0 if nearly collinear
    """
    val = cross2(b - a, c - a)

    if val > tol:
        return 1

    if val < -tol:
        return -1

    return 0


def point_on_segment_xy(p, a, b, tol=1e-10):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if abs(cross2(b - a, p - a)) > tol:
        return False

    dot1 = float(np.dot(p - a, b - a))
    dot2 = float(np.dot(p - b, a - b))

    return dot1 >= -tol and dot2 >= -tol


def segments_intersect_xy(a, b, c, d, tol=1e-10):
    """
    Closed segment intersection.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    o1 = orientation(a, b, c, tol=tol)
    o2 = orientation(a, b, d, tol=tol)
    o3 = orientation(c, d, a, tol=tol)
    o4 = orientation(c, d, b, tol=tol)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and point_on_segment_xy(c, a, b, tol=tol):
        return True

    if o2 == 0 and point_on_segment_xy(d, a, b, tol=tol):
        return True

    if o3 == 0 and point_on_segment_xy(a, c, d, tol=tol):
        return True

    if o4 == 0 and point_on_segment_xy(b, c, d, tol=tol):
        return True

    return False


def is_simple_polygon_xy(points_xy, tol=1e-10):
    """
    Check if polygon edges self-intersect.

    Adjacent edges are allowed to share endpoints.
    First and last edges are also adjacent.
    """
    p = as_xy(points_xy)
    n = len(p)

    if n < 3:
        return False

    for i in range(n):
        a = p[i]
        b = p[(i + 1) % n]

        for j in range(i + 1, n):
            # Adjacent edges.
            if (i + 1) % n == j:
                continue

            if i == (j + 1) % n:
                continue

            # First and last edge are adjacent.
            if i == 0 and j == n - 1:
                continue

            c = p[j]
            d = p[(j + 1) % n]

            if segments_intersect_xy(a, b, c, d, tol=tol):
                return False

    return True


# ============================================================
# Convexity
# ============================================================

def is_convex_polygon_xy(points_xy, tol=1e-10):
    """
    True if polygon is convex.

    Assumes simple polygon. Collinear vertices may exist, but they are ignored.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    n = len(p)

    if n < 3:
        return False

    if n == 3:
        return True

    sign = 0

    for i in range(n):
        a = p[i]
        b = p[(i + 1) % n]
        c = p[(i + 2) % n]

        turn = cross2(b - a, c - b)

        if abs(turn) <= tol:
            continue

        this_sign = 1 if turn > 0.0 else -1

        if sign == 0:
            sign = this_sign

        elif this_sign != sign:
            return False

    return sign != 0


def convex_vertices_mask_xy(points_xy, tol=1e-10):
    """
    For a CCW polygon, returns True for convex vertices and False for reflex vertices.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    n = len(p)

    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        a = p[(i - 1) % n]
        b = p[i]
        c = p[(i + 1) % n]

        mask[i] = cross2(b - a, c - b) >= -tol

    return mask


def reflex_vertices_indices_xy(points_xy, tol=1e-10):
    mask = convex_vertices_mask_xy(points_xy, tol=tol)

    return np.where(~mask)[0].astype(int)


# ============================================================
# Point-in-polygon and point-in-triangle
# ============================================================

def point_in_triangle_xy(p, a, b, c, tol=1e-10):
    """
    Inclusive point-in-triangle test.
    Works for CCW or CW triangle.
    """
    p = np.asarray(p, dtype=np.float64)

    c1 = cross2(b - a, p - a)
    c2 = cross2(c - b, p - b)
    c3 = cross2(a - c, p - c)

    has_neg = (c1 < -tol) or (c2 < -tol) or (c3 < -tol)
    has_pos = (c1 > tol) or (c2 > tol) or (c3 > tol)

    return not (has_neg and has_pos)


def point_in_polygon_xy(point, polygon_xy, tol=1e-12):
    """
    Ray-casting point-in-polygon test.

    Returns True for points inside or on boundary.
    """
    p = np.asarray(point, dtype=np.float64)
    poly = as_xy(polygon_xy)
    n = len(poly)

    # Boundary check first.
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if point_on_segment_xy(p, a, b, tol=tol):
            return True

    inside = False

    x = p[0]
    y = p[1]

    j = n - 1

    for i in range(n):
        xi = poly[i, 0]
        yi = poly[i, 1]
        xj = poly[j, 0]
        yj = poly[j, 1]

        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi

            if x < x_cross:
                inside = not inside

        j = i

    return inside


# ============================================================
# Ear clipping triangulation
# ============================================================

def _is_ear_at_index(vertices, local_index, tol=1e-10):
    """
    Ear test for a CCW simple polygon.
    """
    n = len(vertices)

    prev_i = (local_index - 1) % n
    next_i = (local_index + 1) % n

    a = vertices[prev_i]
    b = vertices[local_index]
    c = vertices[next_i]

    # For CCW polygon, convex corner has positive cross.
    if cross2(b - a, c - b) <= tol:
        return False

    # No other vertex should lie inside the ear triangle.
    # Inclusive test is conservative. It avoids diagonals through vertices.
    for k in range(n):
        if k == prev_i or k == local_index or k == next_i:
            continue

        q = vertices[k]

        if point_in_triangle_xy(q, a, b, c, tol=tol):
            return False

    return True


def triangulate_polygon_xy(points_xy, tol=1e-10, max_guard=100000):
    """
    Triangulate a simple polygon using ear clipping.

    Returns:
        list of triangle arrays, each shape (3,2), CCW.

    Assumptions:
        - simple polygon
        - no holes

    Notes:
        This is intentionally dependency-free.
        It is used only as the first step before greedy convex merging.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if len(p) == 3:
        return [p.copy()]

    if not is_simple_polygon_xy(p, tol=tol):
        raise ValueError("Cannot triangulate a self-intersecting footprint.")

    vertices = [p[i].copy() for i in range(len(p))]
    triangles = []

    guard = 0

    while len(vertices) > 3:
        guard += 1

        if guard > max_guard:
            raise RuntimeError("Ear clipping failed: exceeded guard limit.")

        arr = np.asarray(vertices, dtype=np.float64)
        n = len(vertices)
        ear_found = False

        for i in range(n):
            if _is_ear_at_index(arr, i, tol=tol):
                prev_i = (i - 1) % n
                next_i = (i + 1) % n

                tri = np.asarray(
                    [
                        vertices[prev_i],
                        vertices[i],
                        vertices[next_i],
                    ],
                    dtype=np.float64,
                )

                tri = ensure_ccw_xy(tri)
                triangles.append(tri)

                del vertices[i]

                ear_found = True
                break

        if not ear_found:
            raise RuntimeError(
                "Ear clipping failed: no ear found. "
                "Check for invalid, duplicate, or near-collinear vertices."
            )

    last_tri = np.asarray(vertices, dtype=np.float64)
    last_tri = ensure_ccw_xy(last_tri)
    triangles.append(last_tri)

    return triangles


# Backward-compatible alias.
def triangulate_concave_polygon_xy(points_xy, tol=1e-10, max_guard=100000):
    return triangulate_polygon_xy(points_xy, tol=tol, max_guard=max_guard)


# ============================================================
# Convex-piece merging: Shapely path
# ============================================================

def _try_merge_two_pieces_shapely(piece_a, piece_b, tol=1e-10):
    """
    Robust merge attempt using Shapely if available.

    Returns merged convex polygon or None.
    """
    if not SHAPELY_AVAILABLE:
        return None

    a = clean_footprint_xy(
        piece_a,
        collinear_tol=tol,
        remove_collinear=False,
        ensure_ccw=True,
    )
    b = clean_footprint_xy(
        piece_b,
        collinear_tol=tol,
        remove_collinear=False,
        ensure_ccw=True,
    )

    pa = ShapelyPolygon(a)
    pb = ShapelyPolygon(b)

    if not pa.is_valid or not pb.is_valid:
        return None

    union = pa.union(pb)

    if union.geom_type != "Polygon":
        return None

    if len(union.interiors) > 0:
        return None

    # If they only touch at a point, union area still equals sum,
    # but the union is not a useful merged convex piece.
    intersection = pa.intersection(pb)

    if intersection.is_empty:
        return None

    if intersection.geom_type not in ("LineString", "MultiLineString", "Polygon"):
        return None

    if intersection.length <= tol and intersection.area <= tol:
        return None

    coords = np.asarray(union.exterior.coords[:-1], dtype=np.float64)

    try:
        merged = clean_footprint_xy(
            coords,
            collinear_tol=tol,
            remove_collinear=True,
            ensure_ccw=True,
        )
    except Exception:
        return None

    if not is_simple_polygon_xy(merged, tol=tol):
        return None

    if not is_convex_polygon_xy(merged, tol=tol):
        return None

    area_in = polygon_area_xy(a) + polygon_area_xy(b)
    area_out = polygon_area_xy(merged)

    if abs(area_in - area_out) > max(1e-8, 1e-8 * area_in):
        return None

    return merged


# ============================================================
# Convex-piece merging: dependency-free fallback
# ============================================================

def _point_key(p, decimals=10):
    """
    Rounded coordinate key for dependency-free edge matching.
    """
    return (
        round(float(p[0]), decimals),
        round(float(p[1]), decimals),
    )


def _undirected_edge_key(edge):
    a, b = edge

    if a <= b:
        return a, b

    return b, a


def _polygon_oriented_edges(poly, decimals=10):
    p = clean_footprint_xy(
        poly,
        remove_collinear=False,
        ensure_ccw=True,
    )

    edges = []
    coord_map = {}

    for i in range(len(p)):
        a_key = _point_key(p[i], decimals=decimals)
        b_key = _point_key(p[(i + 1) % len(p)], decimals=decimals)

        coord_map[a_key] = p[i].copy()
        coord_map[b_key] = p[(i + 1) % len(p)].copy()

        edges.append((a_key, b_key))

    return edges, coord_map


def _reconstruct_cycle_from_oriented_edges(boundary_edges, coord_map):
    """
    Reconstruct one polygon cycle from oriented boundary edges.
    Returns Nx2 array or None.
    """
    if not boundary_edges:
        return None

    outgoing = {}

    for a, b in boundary_edges:
        if a in outgoing:
            return None

        outgoing[a] = b

    start = min(outgoing.keys())
    cycle = [start]
    current = start

    for _ in range(len(boundary_edges)):
        if current not in outgoing:
            return None

        nxt = outgoing[current]

        if nxt == start:
            if len(cycle) == len(boundary_edges):
                return np.asarray([coord_map[k] for k in cycle], dtype=np.float64)

            return None

        if nxt in cycle:
            return None

        cycle.append(nxt)
        current = nxt

    return None


def _try_merge_two_pieces_edge_based(piece_a, piece_b, tol=1e-10):
    """
    Dependency-free merge attempt.

    This works well when the two pieces share exactly matching whole edges.
    It is less robust than the Shapely path for partial shared edges.
    """
    a = clean_footprint_xy(
        piece_a,
        collinear_tol=tol,
        remove_collinear=False,
        ensure_ccw=True,
    )
    b = clean_footprint_xy(
        piece_b,
        collinear_tol=tol,
        remove_collinear=False,
        ensure_ccw=True,
    )

    edges_a, coord_a = _polygon_oriented_edges(a)
    edges_b, coord_b = _polygon_oriented_edges(b)

    coord_map = {}
    coord_map.update(coord_a)
    coord_map.update(coord_b)

    all_edges = edges_a + edges_b

    edge_counts = {}

    for edge in all_edges:
        key = _undirected_edge_key(edge)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    shared_edges = [key for key, count in edge_counts.items() if count == 2]

    if len(shared_edges) < 1:
        return None

    # Reject strange overlaps.
    for count in edge_counts.values():
        if count > 2:
            return None

    boundary_edges = [
        edge for edge in all_edges
        if edge_counts[_undirected_edge_key(edge)] == 1
    ]

    candidate = _reconstruct_cycle_from_oriented_edges(boundary_edges, coord_map)

    if candidate is None:
        return None

    try:
        merged = clean_footprint_xy(
            candidate,
            collinear_tol=tol,
            remove_collinear=True,
            ensure_ccw=True,
        )
    except Exception:
        return None

    if not is_simple_polygon_xy(merged, tol=tol):
        return None

    if not is_convex_polygon_xy(merged, tol=tol):
        return None

    area_in = polygon_area_xy(a) + polygon_area_xy(b)
    area_out = polygon_area_xy(merged)

    if abs(area_in - area_out) > max(1e-8, 1e-8 * area_in):
        return None

    return merged


def try_merge_two_convex_pieces_xy(piece_a, piece_b, tol=1e-10):
    """
    Try to merge two convex pieces.

    Returns:
        merged convex polygon if merge succeeds
        None otherwise

    The Shapely path is preferred if available because it handles shared
    partial edges better. The fallback handles exact shared whole edges.
    """
    merged = _try_merge_two_pieces_shapely(piece_a, piece_b, tol=tol)

    if merged is not None:
        return merged

    return _try_merge_two_pieces_edge_based(piece_a, piece_b, tol=tol)


def convex_piece_cost_for_barrier_xy(piece, tol=1e-10):
    """
    Number of barrier segments needed for one convex polygon.

    For a convex n-vertex polygon:
        minimum segment count = ceil(n / 2)
    """
    p = clean_footprint_xy(
        piece,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    return int(math.ceil(len(p) / 2.0))


def total_barrier_cost_xy(pieces, tol=1e-10):
    """
    Total number of 2D barrier segments for a list of convex pieces.
    """
    return int(sum(convex_piece_cost_for_barrier_xy(piece, tol=tol) for piece in pieces))


def merge_score_for_barrier_xy(piece_a, piece_b, merged, tol=1e-10):
    """
    Score candidate merge.

    Primary target:
        reduce final number of barrier segments.

    Secondary target:
        create larger pieces, because they usually reduce later overhead.
    """
    old_cost = (
        convex_piece_cost_for_barrier_xy(piece_a, tol=tol)
        + convex_piece_cost_for_barrier_xy(piece_b, tol=tol)
    )

    new_cost = convex_piece_cost_for_barrier_xy(merged, tol=tol)

    cost_gain = old_cost - new_cost
    area_gain = polygon_area_xy(merged)

    # Big weight on reducing barrier count, small weight on merged area.
    return float(1000000.0 * cost_gain + area_gain)


def merge_convex_pieces_greedy_xy(
    pieces,
    tol=1e-10,
    prefer_barrier_cost=True,
    max_passes=100000,
):
    """
    Greedily merge adjacent convex pieces when the union remains convex.

    Input:
        pieces: list of convex polygons, usually triangles from ear clipping.

    Output:
        list of larger convex polygons.

    This is a practical Hertel-Mehlhorn-style decomposition:
        triangulate first, then remove unnecessary internal edges by merging.

    It is not guaranteed to be the exact minimum convex partition, but it is
    usually much better than raw triangulation and is the right default here.
    """
    clean_pieces = []

    for piece in pieces:
        p = clean_footprint_xy(
            piece,
            collinear_tol=tol,
            remove_collinear=True,
            ensure_ccw=True,
        )

        if not is_convex_polygon_xy(p, tol=tol):
            raise ValueError("merge_convex_pieces_greedy_xy received a non-convex piece.")

        clean_pieces.append(p)

    pieces = clean_pieces

    passes = 0

    while True:
        passes += 1

        if passes > max_passes:
            raise RuntimeError("Greedy convex merging exceeded max_passes.")

        best_i = None
        best_j = None
        best_merged = None
        best_score = None

        n = len(pieces)

        for i in range(n):
            for j in range(i + 1, n):
                merged = try_merge_two_convex_pieces_xy(
                    pieces[i],
                    pieces[j],
                    tol=tol,
                )

                if merged is None:
                    continue

                if prefer_barrier_cost:
                    score = merge_score_for_barrier_xy(
                        pieces[i],
                        pieces[j],
                        merged,
                        tol=tol,
                    )
                else:
                    score = polygon_area_xy(merged)

                if best_score is None or score > best_score:
                    best_i = i
                    best_j = j
                    best_merged = merged
                    best_score = score

        if best_merged is None:
            break

        new_pieces = []

        for k, piece in enumerate(pieces):
            if k != best_i and k != best_j:
                new_pieces.append(piece)

        new_pieces.append(best_merged)
        pieces = new_pieces

    return pieces


# ============================================================
# Convex decomposition
# ============================================================

def convex_decompose_polygon_triangles_xy(points_xy, tol=1e-10):
    """
    Safe fallback:
        convex polygon -> itself
        concave polygon -> triangles
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if not is_simple_polygon_xy(p, tol=tol):
        raise ValueError("Footprint is not simple; cannot convex-decompose.")

    if is_convex_polygon_xy(p, tol=tol):
        return [p]

    return triangulate_polygon_xy(p, tol=tol)


def convex_decompose_polygon_merged_xy(
    points_xy,
    tol=1e-10,
    prefer_barrier_cost=True,
):
    """
    Default decomposition for the shading proxy.

    Algorithm:
        1. If footprint is already convex, return it.
        2. Otherwise triangulate using ear clipping.
        3. Greedily merge adjacent pieces whenever the merged piece is convex.
        4. Prefer merges that reduce the final barrier-segment count.

    Returns:
        list of convex polygons, each CCW.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if not is_simple_polygon_xy(p, tol=tol):
        raise ValueError("Footprint is not simple; cannot convex-decompose.")

    if is_convex_polygon_xy(p, tol=tol):
        return [p]

    triangles = triangulate_polygon_xy(p, tol=tol)

    if len(triangles) <= 1:
        return triangles

    return merge_convex_pieces_greedy_xy(
        triangles,
        tol=tol,
        prefer_barrier_cost=prefer_barrier_cost,
    )


def convex_decompose_polygon_xy(points_xy, tol=1e-10):
    """
    Main public function.

    This intentionally returns the merged convex decomposition, not raw triangles.
    Use convex_decompose_polygon_triangles_xy(...) if you explicitly want triangles.
    """
    return convex_decompose_polygon_merged_xy(
        points_xy,
        tol=tol,
        prefer_barrier_cost=True,
    )


# ============================================================
# Diagnostics
# ============================================================

def decomposition_summary_xy(points_xy, pieces, tol=1e-10):
    """
    Summarize a convex decomposition.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    piece_areas = [polygon_area_xy(piece) for piece in pieces]
    piece_vertex_counts = [
        int(len(clean_footprint_xy(
            piece,
            collinear_tol=tol,
            remove_collinear=True,
            ensure_ccw=True,
        )))
        for piece in pieces
    ]

    input_area = polygon_area_xy(p)
    pieces_area = float(sum(piece_areas))

    return {
        "input_vertices": int(len(p)),
        "input_area": float(input_area),
        "input_perimeter": float(polygon_perimeter_xy(p)),
        "input_is_simple": bool(is_simple_polygon_xy(p, tol=tol)),
        "input_is_convex": bool(is_convex_polygon_xy(p, tol=tol)),
        "n_pieces": int(len(pieces)),
        "piece_vertex_counts": piece_vertex_counts,
        "piece_areas": [float(a) for a in piece_areas],
        "pieces_area_sum": float(pieces_area),
        "area_error_abs": float(abs(input_area - pieces_area)),
        "barrier_segment_cost": int(total_barrier_cost_xy(pieces, tol=tol)),
        "shapely_available": bool(SHAPELY_AVAILABLE),
    }


def footprint_summary_xy(points_xy, tol=1e-10):
    """
    Return a small summary dictionary useful for diagnostics.
    """
    p = clean_footprint_xy(
        points_xy,
        collinear_tol=tol,
        remove_collinear=True,
        ensure_ccw=True,
    )

    triangle_pieces = convex_decompose_polygon_triangles_xy(p, tol=tol)
    merged_pieces = convex_decompose_polygon_merged_xy(p, tol=tol)

    return {
        "n_vertices": int(len(p)),
        "area": float(polygon_area_xy(p)),
        "perimeter": float(polygon_perimeter_xy(p)),
        "bbox": bbox_xy(p),
        "is_simple": bool(is_simple_polygon_xy(p, tol=tol)),
        "is_convex": bool(is_convex_polygon_xy(p, tol=tol)),
        "n_triangle_pieces": int(len(triangle_pieces)),
        "n_merged_convex_pieces": int(len(merged_pieces)),
        "triangle_barrier_cost": int(total_barrier_cost_xy(triangle_pieces, tol=tol)),
        "merged_barrier_cost": int(total_barrier_cost_xy(merged_pieces, tol=tol)),
        "merged_piece_vertex_counts": [
            int(len(clean_footprint_xy(
                piece,
                collinear_tol=tol,
                remove_collinear=True,
                ensure_ccw=True,
            )))
            for piece in merged_pieces
        ],
        "shapely_available": bool(SHAPELY_AVAILABLE),
    }


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    rect = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 2.0],
        [0.0, 2.0],
    ])

    l_shape = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.0],
        [1.5, 1.0],
        [1.5, 3.0],
        [0.0, 3.0],
    ])

    u_shape = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 4.0],
        [3.5, 4.0],
        [3.5, 1.2],
        [1.5, 1.2],
        [1.5, 4.0],
        [0.0, 4.0],
    ])

    examples = [
        ("rectangle", rect),
        ("L-shape", l_shape),
        ("U-shape", u_shape),
    ]

    for name, poly in examples:
        print()
        print("=" * 80)
        print(name)
        print("=" * 80)

        print("summary:")
        print(footprint_summary_xy(poly))

        triangles = convex_decompose_polygon_triangles_xy(poly)
        merged = convex_decompose_polygon_merged_xy(poly)

        print()
        print("triangles:", len(triangles))
        for i, piece in enumerate(triangles):
            print("  tri", i, piece.tolist())

        print()
        print("merged convex pieces:", len(merged))
        for i, piece in enumerate(merged):
            print("  piece", i, piece.tolist())

        print()
        print("decomposition summary:")
        print(decomposition_summary_xy(poly, merged))