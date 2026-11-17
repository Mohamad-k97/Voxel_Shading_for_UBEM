# barrier_proxy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from footprint_geometry import (
    EPS,
    as_xy,
    clean_footprint_xy,
    convex_decompose_polygon_xy,
    convex_decompose_polygon_triangles_xy,
    ensure_ccw_xy,
    is_convex_polygon_xy,
    polygon_area_xy,
    polygon_centroid_xy,
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class BarrierSegment:
    """
    One 2D barrier segment inside a convex footprint piece.

    p0, p1:
        2D endpoints, shape (2,)

    piece_index:
        index of the convex piece this segment belongs to

    segment_index:
        local segment index inside that convex piece

    object_id:
        building/object id if known

    source:
        diagnostic string
    """

    p0: np.ndarray
    p1: np.ndarray
    piece_index: int = -1
    segment_index: int = -1
    object_id: int = -1
    source: str = "barrier_proxy"


@dataclass
class BarrierSurfaceSpec:
    """
    One vertical 3D proxy caster surface created by extruding a 2D barrier segment.

    This is intentionally independent from generator.UrbanPolygon to avoid circular imports.
    shading_profiles.py can convert this spec into UrbanPolygon objects.
    """

    vertices: np.ndarray      # shape (4, 3)
    normal: np.ndarray        # shape (3,)
    center: np.ndarray        # shape (3,)
    area: float

    object_id: int = -1
    surface_kind: str = "barrier_wall_proxy"
    object_kind: str = "building"

    can_cast_shadow: bool = True
    can_receive_shadow: bool = False
    opacity: float = 1.0
    two_sided: bool = True

    piece_index: int = -1
    segment_index: int = -1
    source: str = "barrier_proxy"


@dataclass
class BarrierProfile:
    """
    Full barrier profile for one footprint/building.
    """

    object_id: int
    footprint_xy: np.ndarray
    convex_pieces_xy: List[np.ndarray]
    segments_xy: List[BarrierSegment]
    surfaces: List[BarrierSurfaceSpec]

    base_z: float
    top_z: float

    n_input_vertices: int
    n_convex_pieces: int
    n_segments: int
    n_surfaces: int
    total_proxy_area: float
    decomposition_mode: str


# ============================================================
# Basic helpers
# ============================================================

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n <= EPS:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / n


def segment_length_xy(p0, p1):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    return float(np.linalg.norm(p1 - p0))


def segment_midpoint_xy(p0, p1):
    return 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))


def barrier_segment_length(segment):
    return segment_length_xy(segment.p0, segment.p1)


def _segment_key_xy(p0, p1, decimals=9):
    """
    Hash key for deduplicating identical/reversed barrier segments.
    """
    a = (
        round(float(p0[0]), decimals),
        round(float(p0[1]), decimals),
    )
    b = (
        round(float(p1[0]), decimals),
        round(float(p1[1]), decimals),
    )

    if a <= b:
        return a, b

    return b, a


def deduplicate_barrier_segments_xy(segments, decimals=9):
    """
    Remove exactly duplicated segments, including reversed duplicates.

    This does not merge partially overlapping segments.
    """
    seen = set()
    result = []

    for seg in segments:
        key = _segment_key_xy(seg.p0, seg.p1, decimals=decimals)

        if key in seen:
            continue

        seen.add(key)
        result.append(seg)

    return result


# ============================================================
# Minimum-cardinality barrier for convex polygons
# ============================================================

def _make_convex_barrier_segments_for_start_xy(
    convex_xy,
    start_index,
    piece_index=-1,
    object_id=-1,
):
    """
    Build the ceil(n/2) opposite-chord barrier for one start index.

    For n = 2k:
        vi -> v(i+k), i = 0 ... k-1

    For n = 2k+1:
        vi -> v(i+k), i = 0 ... k
    """
    p = clean_footprint_xy(
        convex_xy,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if not is_convex_polygon_xy(p):
        raise ValueError("Convex barrier construction received a non-convex polygon.")

    n = len(p)

    if n < 3:
        raise ValueError("Convex barrier construction needs at least 3 vertices.")

    d = n // 2
    m = (n + 1) // 2

    segments = []

    for local_i in range(m):
        i = (start_index + local_i) % n
        j = (i + d) % n

        p0 = p[i].copy()
        p1 = p[j].copy()

        if segment_length_xy(p0, p1) <= EPS:
            continue

        segments.append(
            BarrierSegment(
                p0=p0,
                p1=p1,
                piece_index=int(piece_index),
                segment_index=int(local_i),
                object_id=int(object_id),
                source="convex_opposite_chord",
            )
        )

    return segments


def _barrier_segments_cost_length(segments):
    return float(sum(barrier_segment_length(seg) for seg in segments))


def _barrier_segments_cost_max_length(segments):
    if not segments:
        return 0.0

    return float(max(barrier_segment_length(seg) for seg in segments))


def make_convex_barrier_segments_xy(
    convex_xy,
    piece_index=-1,
    object_id=-1,
    start_index=None,
    optimize_rotation=True,
    rotation_cost_mode="total_length",
):
    """
    Create a minimum-cardinality exact barrier for a convex polygon.

    Output count:
        ceil(n / 2)

    Parameters
    ----------
    convex_xy:
        Convex polygon footprint piece, Nx2.

    start_index:
        If given, use this cyclic start vertex.
        If None and optimize_rotation=True, try all starts and keep the cheapest.

    optimize_rotation:
        For odd polygons, different rotations duplicate a different vertex.
        All are exact barriers, but some have shorter total length.

    rotation_cost_mode:
        "total_length" -> minimize sum of segment lengths
        "max_length"   -> minimize longest segment
        "none"         -> use start_index or 0

    Returns
    -------
    list[BarrierSegment]
    """
    p = clean_footprint_xy(
        convex_xy,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if not is_convex_polygon_xy(p):
        raise ValueError("make_convex_barrier_segments_xy expects a convex polygon.")

    n = len(p)

    if start_index is not None:
        starts = [int(start_index) % n]

    elif not optimize_rotation or rotation_cost_mode == "none":
        starts = [0]

    else:
        starts = list(range(n))

    best_segments = None
    best_cost = None

    for s in starts:
        segments = _make_convex_barrier_segments_for_start_xy(
            p,
            start_index=s,
            piece_index=piece_index,
            object_id=object_id,
        )

        if rotation_cost_mode == "max_length":
            cost = _barrier_segments_cost_max_length(segments)

        else:
            cost = _barrier_segments_cost_length(segments)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_segments = segments

    if best_segments is None:
        raise RuntimeError("Failed to create convex barrier segments.")

    # Re-number local segment indices after choosing rotation.
    for i, seg in enumerate(best_segments):
        seg.segment_index = int(i)

    return best_segments


# ============================================================
# Barrier for concave/simple footprints
# ============================================================

def make_barrier_segments_for_footprint_xy(
    footprint_xy,
    object_id=-1,
    decomposition_mode="merged",
    optimize_rotation=True,
    rotation_cost_mode="total_length",
    deduplicate=True,
):
    """
    Create barrier segments for a possibly concave footprint.

    Algorithm
    ---------
    If footprint is convex:
        one convex piece
        make ceil(n/2) opposite-chord segments

    If footprint is concave:
        decompose into convex pieces
        make a convex barrier for each piece
        optionally deduplicate exact repeated segments

    Parameters
    ----------
    decomposition_mode:
        "merged"    -> triangulate, then greedily merge convex pieces
        "triangles" -> raw ear-clipping triangles only

    Returns
    -------
    convex_pieces, segments
    """
    footprint = clean_footprint_xy(
        footprint_xy,
        remove_collinear=True,
        ensure_ccw=True,
    )

    if decomposition_mode == "triangles":
        convex_pieces = convex_decompose_polygon_triangles_xy(footprint)

    elif decomposition_mode == "merged":
        convex_pieces = convex_decompose_polygon_xy(footprint)

    else:
        raise ValueError(
            "decomposition_mode must be 'merged' or 'triangles'."
        )

    all_segments = []

    for piece_index, piece in enumerate(convex_pieces):
        piece = clean_footprint_xy(
            piece,
            remove_collinear=True,
            ensure_ccw=True,
        )

        if not is_convex_polygon_xy(piece):
            raise RuntimeError("Convex decomposition returned a non-convex piece.")

        piece_segments = make_convex_barrier_segments_xy(
            piece,
            piece_index=piece_index,
            object_id=object_id,
            start_index=None,
            optimize_rotation=optimize_rotation,
            rotation_cost_mode=rotation_cost_mode,
        )

        all_segments.extend(piece_segments)

    if deduplicate:
        all_segments = deduplicate_barrier_segments_xy(all_segments)

    # Re-number globally after deduplication.
    for i, seg in enumerate(all_segments):
        seg.segment_index = int(i)

    return convex_pieces, all_segments


# ============================================================
# 2D barrier validation by projected shadow intervals
# ============================================================

def _projection_interval_points(points_xy, axis):
    p = as_xy(points_xy)
    axis = unit(np.asarray(axis, dtype=np.float64))

    values = p @ axis

    return float(np.min(values)), float(np.max(values))


def _projection_interval_segment(segment, axis):
    axis = unit(np.asarray(axis, dtype=np.float64))

    a = float(np.dot(segment.p0, axis))
    b = float(np.dot(segment.p1, axis))

    if a <= b:
        return a, b

    return b, a


def _merge_intervals(intervals, tol=1e-10):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [list(intervals[0])]

    for a, b in intervals[1:]:
        last = merged[-1]

        if a <= last[1] + tol:
            if b > last[1]:
                last[1] = b
        else:
            merged.append([a, b])

    return [(float(a), float(b)) for a, b in merged]


def _uncovered_interval_length(target_interval, cover_intervals, tol=1e-10):
    """
    Length of target interval not covered by cover intervals.
    """
    a0, b0 = target_interval

    if b0 <= a0 + tol:
        return 0.0

    clipped = []

    for a, b in cover_intervals:
        aa = max(a, a0)
        bb = min(b, b0)

        if bb > aa + tol:
            clipped.append((aa, bb))

    merged = _merge_intervals(clipped, tol=tol)

    covered = 0.0

    for a, b in merged:
        covered += max(0.0, b - a)

    return max(0.0, (b0 - a0) - covered)


def validate_barrier_segments_by_projection_xy(
    footprint_xy,
    segments,
    n_directions=720,
    tol=1e-8,
):
    """
    Sampled validation of a 2D barrier by projection intervals.

    For a convex polygon:
        This is a strong and useful sampled test.

    For a concave polygon:
        This is conservative, because a concave footprint's projection interval
        may include offsets whose lines pass through empty notches. The actual
        per-piece construction is still valid because each convex piece is
        protected separately.

    Returns
    -------
    dict with:
        ok
        max_uncovered_length
        max_relative_error
        worst_angle_deg
    """
    footprint = clean_footprint_xy(
        footprint_xy,
        remove_collinear=True,
        ensure_ccw=True,
    )

    max_uncovered = 0.0
    max_rel = 0.0
    worst_angle = None

    for k in range(n_directions):
        angle = math.pi * float(k) / float(n_directions)

        # Line direction.
        line_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        # Offset axis normal to the line.
        axis = np.array([-line_dir[1], line_dir[0]], dtype=np.float64)

        target = _projection_interval_points(footprint, axis)
        target_length = max(EPS, target[1] - target[0])

        cover_intervals = [
            _projection_interval_segment(seg, axis)
            for seg in segments
        ]

        uncovered = _uncovered_interval_length(
            target,
            cover_intervals,
            tol=tol,
        )

        rel = uncovered / target_length

        if uncovered > max_uncovered:
            max_uncovered = float(uncovered)
            max_rel = float(rel)
            worst_angle = math.degrees(angle)

    return {
        "ok": bool(max_uncovered <= tol),
        "max_uncovered_length": float(max_uncovered),
        "max_relative_error": float(max_rel),
        "worst_angle_deg": None if worst_angle is None else float(worst_angle),
        "n_directions": int(n_directions),
    }


def validate_barrier_segments_piecewise_xy(
    convex_pieces,
    segments,
    n_directions=720,
    tol=1e-8,
):
    """
    Validate a decomposed footprint piece by piece.

    This is the better validation for concave footprints:
        every convex piece should be covered by its own barrier segments.
    """
    results = []

    for piece_index, piece in enumerate(convex_pieces):
        piece_segments = [
            seg for seg in segments
            if int(seg.piece_index) == int(piece_index)
        ]

        result = validate_barrier_segments_by_projection_xy(
            piece,
            piece_segments,
            n_directions=n_directions,
            tol=tol,
        )

        result["piece_index"] = int(piece_index)
        result["n_piece_segments"] = int(len(piece_segments))
        result["n_piece_vertices"] = int(len(clean_footprint_xy(piece)))

        results.append(result)

    ok = all(r["ok"] for r in results)

    max_uncovered = 0.0
    max_rel = 0.0

    for r in results:
        max_uncovered = max(max_uncovered, float(r["max_uncovered_length"]))
        max_rel = max(max_rel, float(r["max_relative_error"]))

    return {
        "ok": bool(ok),
        "max_uncovered_length": float(max_uncovered),
        "max_relative_error": float(max_rel),
        "piece_results": results,
    }


# ============================================================
# Extrusion to 3D vertical proxy surfaces
# ============================================================

def _quad_area_3d(vertices):
    """
    Area of a quad by two triangles.
    """
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) != 4:
        raise ValueError("_quad_area_3d expects exactly 4 vertices.")

    a0 = 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
    a1 = 0.5 * np.linalg.norm(np.cross(v[2] - v[0], v[3] - v[0]))

    return float(a0 + a1)


def _polygon_normal_3d(vertices):
    """
    Normal from the first non-degenerate triangle in the polygon.
    """
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) < 3:
        raise ValueError("Need at least 3 vertices to compute normal.")

    for i in range(1, len(v) - 1):
        n = np.cross(v[i] - v[0], v[i + 1] - v[0])
        norm = np.linalg.norm(n)

        if norm > EPS:
            return n / norm

    raise ValueError("Cannot compute normal for degenerate polygon.")


def extrude_barrier_segment_to_surface(
    segment,
    base_z,
    top_z,
    object_id=-1,
    surface_kind="barrier_wall_proxy",
    object_kind="building",
    opacity=1.0,
):
    """
    Extrude a 2D barrier segment vertically into one 3D quad surface.

    The surface is intended as a two-sided opaque caster.
    """
    z0 = float(base_z)
    z1 = float(top_z)

    if abs(z1 - z0) <= EPS:
        raise ValueError("Cannot extrude barrier surface with zero height.")

    if z1 < z0:
        z0, z1 = z1, z0

    p0 = np.asarray(segment.p0, dtype=np.float64)
    p1 = np.asarray(segment.p1, dtype=np.float64)

    if segment_length_xy(p0, p1) <= EPS:
        raise ValueError("Cannot extrude zero-length barrier segment.")

    vertices = np.array(
        [
            [p0[0], p0[1], z0],
            [p1[0], p1[1], z0],
            [p1[0], p1[1], z1],
            [p0[0], p0[1], z1],
        ],
        dtype=np.float64,
    )

    normal = _polygon_normal_3d(vertices)
    center = np.mean(vertices, axis=0)
    area = _quad_area_3d(vertices)

    return BarrierSurfaceSpec(
        vertices=vertices,
        normal=normal,
        center=center,
        area=float(area),
        object_id=int(object_id),
        surface_kind=str(surface_kind),
        object_kind=str(object_kind),
        can_cast_shadow=True,
        can_receive_shadow=False,
        opacity=float(opacity),
        two_sided=True,
        piece_index=int(segment.piece_index),
        segment_index=int(segment.segment_index),
        source=str(segment.source),
    )


def extrude_barrier_segments_to_surfaces(
    segments,
    base_z,
    top_z,
    object_id=-1,
    surface_kind="barrier_wall_proxy",
    object_kind="building",
    opacity=1.0,
):
    surfaces = []

    for seg in segments:
        surfaces.append(
            extrude_barrier_segment_to_surface(
                segment=seg,
                base_z=base_z,
                top_z=top_z,
                object_id=object_id,
                surface_kind=surface_kind,
                object_kind=object_kind,
                opacity=opacity,
            )
        )

    return surfaces


# ============================================================
# Full profile construction
# ============================================================

def build_barrier_profile_for_footprint(
    footprint_xy,
    base_z,
    top_z,
    object_id=-1,
    decomposition_mode="merged",
    optimize_rotation=True,
    rotation_cost_mode="total_length",
    deduplicate=True,
    surface_kind="barrier_wall_proxy",
    object_kind="building",
    opacity=1.0,
):
    """
    Build the full vertical barrier caster profile for one building footprint.

    This creates only the vertical proxy caster surfaces.
    Rooftop surfaces are handled separately in roof_profiles.py / shading_profiles.py.
    """
    footprint = clean_footprint_xy(
        footprint_xy,
        remove_collinear=True,
        ensure_ccw=True,
    )

    convex_pieces, segments = make_barrier_segments_for_footprint_xy(
        footprint,
        object_id=object_id,
        decomposition_mode=decomposition_mode,
        optimize_rotation=optimize_rotation,
        rotation_cost_mode=rotation_cost_mode,
        deduplicate=deduplicate,
    )

    surfaces = extrude_barrier_segments_to_surfaces(
        segments=segments,
        base_z=base_z,
        top_z=top_z,
        object_id=object_id,
        surface_kind=surface_kind,
        object_kind=object_kind,
        opacity=opacity,
    )

    total_proxy_area = float(sum(s.area for s in surfaces))

    return BarrierProfile(
        object_id=int(object_id),
        footprint_xy=footprint,
        convex_pieces_xy=convex_pieces,
        segments_xy=segments,
        surfaces=surfaces,
        base_z=float(base_z),
        top_z=float(top_z),
        n_input_vertices=int(len(footprint)),
        n_convex_pieces=int(len(convex_pieces)),
        n_segments=int(len(segments)),
        n_surfaces=int(len(surfaces)),
        total_proxy_area=total_proxy_area,
        decomposition_mode=str(decomposition_mode),
    )


# Backward-readable alias.
def make_barrier_profile_for_building(
    footprint_xy,
    base_z,
    top_z,
    object_id=-1,
    decomposition_mode="merged",
):
    return build_barrier_profile_for_footprint(
        footprint_xy=footprint_xy,
        base_z=base_z,
        top_z=top_z,
        object_id=object_id,
        decomposition_mode=decomposition_mode,
    )


# ============================================================
# Diagnostics and summaries
# ============================================================

def barrier_profile_summary(profile):
    """
    Convert a BarrierProfile to a plain dictionary.
    """
    piece_vertex_counts = [int(len(clean_footprint_xy(p))) for p in profile.convex_pieces_xy]
    piece_areas = [float(polygon_area_xy(p)) for p in profile.convex_pieces_xy]
    segment_lengths = [float(barrier_segment_length(s)) for s in profile.segments_xy]

    return {
        "object_id": int(profile.object_id),
        "n_input_vertices": int(profile.n_input_vertices),
        "n_convex_pieces": int(profile.n_convex_pieces),
        "piece_vertex_counts": piece_vertex_counts,
        "piece_areas": piece_areas,
        "n_segments": int(profile.n_segments),
        "n_surfaces": int(profile.n_surfaces),
        "segment_lengths": segment_lengths,
        "total_segment_length": float(sum(segment_lengths)),
        "total_proxy_area": float(profile.total_proxy_area),
        "base_z": float(profile.base_z),
        "top_z": float(profile.top_z),
        "height": float(abs(profile.top_z - profile.base_z)),
        "decomposition_mode": str(profile.decomposition_mode),
    }


def print_barrier_profile_summary(profile):
    s = barrier_profile_summary(profile)

    print()
    print("=" * 80)
    print("BARRIER PROFILE SUMMARY")
    print("=" * 80)
    print("object_id:", s["object_id"])
    print("input vertices:", s["n_input_vertices"])
    print("convex pieces:", s["n_convex_pieces"])
    print("piece vertex counts:", s["piece_vertex_counts"])
    print("segments/surfaces:", s["n_segments"])
    print("height:", s["height"])
    print("total segment length:", round(s["total_segment_length"], 6))
    print("total proxy area:", round(s["total_proxy_area"], 6))
    print("decomposition mode:", s["decomposition_mode"])
    print("=" * 80)


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

    pentagon = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [4.0, 1.5],
        [2.0, 3.0],
        [-0.5, 1.5],
    ])

    l_shape = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.0],
        [1.5, 1.0],
        [1.5, 3.0],
        [0.0, 3.0],
    ])

    examples = [
        ("rectangle", rect),
        ("pentagon", pentagon),
        ("L-shape", l_shape),
    ]

    for name, footprint in examples:
        print()
        print("#" * 80)
        print(name)
        print("#" * 80)

        profile = build_barrier_profile_for_footprint(
            footprint_xy=footprint,
            base_z=0.0,
            top_z=10.0,
            object_id=1,
            decomposition_mode="merged",
        )

        print_barrier_profile_summary(profile)

        validation = validate_barrier_segments_piecewise_xy(
            convex_pieces=profile.convex_pieces_xy,
            segments=profile.segments_xy,
            n_directions=720,
            tol=1e-8,
        )

        print("piecewise validation:", validation["ok"])
        print("max uncovered length:", validation["max_uncovered_length"])
        print("max relative error:", validation["max_relative_error"])

        print()
        print("segments:")
        for seg in profile.segments_xy:
            print(
                "  piece",
                seg.piece_index,
                "seg",
                seg.segment_index,
                "p0",
                seg.p0.tolist(),
                "p1",
                seg.p1.tolist(),
                "length",
                round(barrier_segment_length(seg), 6),
            )