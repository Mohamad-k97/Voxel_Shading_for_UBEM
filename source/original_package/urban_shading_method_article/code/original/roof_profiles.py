# roof_profiles.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
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
class RoofSurfaceSpec:
    """
    One roof surface, independent from generator.UrbanPolygon.

    shading_profiles.py will convert this into UrbanPolygon objects later.
    """

    vertices: np.ndarray      # shape (N, 3)
    normal: np.ndarray        # shape (3,)
    center: np.ndarray        # shape (3,)
    area: float

    object_id: int = -1
    object_kind: str = "building"
    surface_kind: str = "flat_roof"  # "flat_roof" or "pitched_roof"

    can_cast_shadow: bool = True
    can_receive_shadow: bool = True
    opacity: float = 1.0

    roof_type: str = "flat"
    piece_index: int = -1
    surface_index: int = -1
    source: str = "roof_profile"


@dataclass
class RoofProfile:
    """
    Full roof profile for one building.
    """

    object_id: int
    footprint_xy: np.ndarray
    roof_type: str
    eave_z: float
    roof_height: float
    ridge_azimuth_deg: Optional[float]
    surfaces: List[RoofSurfaceSpec]

    n_input_vertices: int
    n_surfaces: int
    total_roof_area: float
    max_z: float
    min_z: float


# ============================================================
# Basic helpers
# ============================================================

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n <= EPS:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / n


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_normal_3d(vertices):
    """
    Compute polygon normal using Newell's method.

    More stable than using only the first triangle.
    """
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) < 3:
        raise ValueError("Need at least 3 vertices to compute polygon normal.")

    n = np.zeros(3, dtype=np.float64)

    for i in range(len(v)):
        p = v[i]
        q = v[(i + 1) % len(v)]

        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])

    norm = np.linalg.norm(n)

    if norm <= EPS:
        # Fallback to first non-degenerate triangle.
        for i in range(1, len(v) - 1):
            n = np.cross(v[i] - v[0], v[i + 1] - v[0])
            norm = np.linalg.norm(n)

            if norm > EPS:
                return n / norm

        raise ValueError("Cannot compute normal for degenerate polygon.")

    return n / norm


def polygon_area_3d(vertices):
    """
    Area of a planar polygon by fan triangulation.
    Works well for convex roof pieces.
    """
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) < 3:
        return 0.0

    area = 0.0
    v0 = v[0]

    for i in range(1, len(v) - 1):
        area += 0.5 * np.linalg.norm(np.cross(v[i] - v0, v[i + 1] - v0))

    return float(area)


def make_roof_surface_spec(
    vertices,
    object_id=-1,
    roof_type="flat",
    surface_kind=None,
    piece_index=-1,
    surface_index=-1,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
    source="roof_profile",
):
    """
    Create one roof surface spec and orient it upward where possible.
    """
    verts = np.asarray(vertices, dtype=np.float64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Roof surface vertices must be an Nx3 array.")

    if len(verts) < 3:
        raise ValueError("Roof surface needs at least 3 vertices.")

    normal = polygon_normal_3d(verts)

    # Roof normals should generally point upward.
    if normal[2] < 0.0:
        verts = verts[::-1].copy()
        normal = polygon_normal_3d(verts)

    area = polygon_area_3d(verts)
    center = np.mean(verts, axis=0)

    if area <= EPS:
        raise ValueError("Degenerate roof surface with near-zero area.")

    if surface_kind is None:
        if roof_type == "flat":
            surface_kind = "flat_roof"
        else:
            surface_kind = "pitched_roof"

    return RoofSurfaceSpec(
        vertices=verts,
        normal=normal,
        center=center,
        area=float(area),
        object_id=int(object_id),
        object_kind="building",
        surface_kind=str(surface_kind),
        can_cast_shadow=bool(can_cast_shadow),
        can_receive_shadow=bool(can_receive_shadow),
        opacity=float(opacity),
        roof_type=str(roof_type),
        piece_index=int(piece_index),
        surface_index=int(surface_index),
        source=str(source),
    )


def local_axes_from_angle_deg(angle_deg):
    """
    Return local u,v axes in x-y plane.

    angle convention:
        degrees counterclockwise from +x/east.

    u = ridge / long direction
    v = perpendicular direction
    """
    a = math.radians(float(angle_deg))

    u = np.array([math.cos(a), math.sin(a)], dtype=np.float64)
    v = np.array([-math.sin(a), math.cos(a)], dtype=np.float64)

    return u, v


def estimate_main_axis_angle_deg(footprint_xy):
    """
    Estimate a stable main axis angle from the longest footprint edge.

    This is usually enough for synthetic buildings, especially mostly
    rectangular ones.
    """
    p = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    best_len = -1.0
    best_vec = None

    for i in range(len(p)):
        a = p[i]
        b = p[(i + 1) % len(p)]
        vec = b - a
        length = np.linalg.norm(vec)

        if length > best_len:
            best_len = float(length)
            best_vec = vec

    if best_vec is None or np.linalg.norm(best_vec) <= EPS:
        return 0.0

    angle = math.degrees(math.atan2(best_vec[1], best_vec[0]))

    return float(angle)


def local_coordinates_xy(points_xy, origin_xy, u_axis, v_axis):
    p = as_xy(points_xy)
    origin = np.asarray(origin_xy, dtype=np.float64)

    rel = p - origin

    u = rel @ np.asarray(u_axis, dtype=np.float64)
    v = rel @ np.asarray(v_axis, dtype=np.float64)

    return u, v


def polygon_with_z_from_function(points_xy, z_function):
    """
    Convert Nx2 polygon to Nx3 using z = z_function(x,y).
    """
    p = as_xy(points_xy)

    verts = np.zeros((len(p), 3), dtype=np.float64)
    verts[:, :2] = p

    for i in range(len(p)):
        verts[i, 2] = float(z_function(p[i, 0], p[i, 1]))

    return verts


# ============================================================
# Half-plane clipping for gable roof pieces
# ============================================================

def clip_polygon_by_halfplane_xy(points_xy, origin_xy, normal_xy, keep_positive=True, tol=1e-10):
    """
    Clip polygon by a 2D half-plane.

    Half-plane:
        dot(p - origin, normal) >= 0     if keep_positive=True
        dot(p - origin, normal) <= 0     if keep_positive=False

    Works well for convex polygons. In this module it is applied after
    convex decomposition, so that is exactly what we need.
    """
    p = clean_footprint_xy(points_xy, ensure_ccw=True)

    origin = np.asarray(origin_xy, dtype=np.float64)
    normal = unit(np.asarray(normal_xy, dtype=np.float64))

    def signed_dist(q):
        d = float(np.dot(q - origin, normal))
        return d if keep_positive else -d

    output = []

    n = len(p)

    for i in range(n):
        current = p[i]
        nxt = p[(i + 1) % n]

        dc = signed_dist(current)
        dn = signed_dist(nxt)

        current_inside = dc >= -tol
        next_inside = dn >= -tol

        if current_inside and next_inside:
            output.append(nxt)

        elif current_inside and not next_inside:
            denom = dc - dn

            if abs(denom) > EPS:
                t = dc / denom
                intersection = current + t * (nxt - current)
                output.append(intersection)

        elif not current_inside and next_inside:
            denom = dc - dn

            if abs(denom) > EPS:
                t = dc / denom
                intersection = current + t * (nxt - current)
                output.append(intersection)

            output.append(nxt)

    if len(output) < 3:
        return None

    clipped = np.asarray(output, dtype=np.float64)

    try:
        clipped = clean_footprint_xy(
            clipped,
            collinear_tol=tol,
            remove_collinear=True,
            ensure_ccw=True,
        )
    except Exception:
        return None

    if len(clipped) < 3 or polygon_area_xy(clipped) <= EPS:
        return None

    return clipped


# ============================================================
# Flat roof
# ============================================================

def make_flat_roof_surfaces(
    footprint_xy,
    eave_z,
    object_id=-1,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
    decompose=True,
):
    """
    Create flat horizontal roof surfaces.

    For concave footprints, this returns convex pieces so later ray/array
    handling is safer.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    if decompose:
        pieces = convex_decompose_polygon_xy(footprint)
    else:
        pieces = [footprint]

    surfaces = []

    for piece_index, piece in enumerate(pieces):
        piece = clean_footprint_xy(piece, ensure_ccw=True)

        verts = np.zeros((len(piece), 3), dtype=np.float64)
        verts[:, :2] = piece
        verts[:, 2] = float(eave_z)

        surfaces.append(
            make_roof_surface_spec(
                verts,
                object_id=object_id,
                roof_type="flat",
                surface_kind="flat_roof",
                piece_index=piece_index,
                surface_index=len(surfaces),
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
                source="flat_roof",
            )
        )

    return surfaces


# ============================================================
# Shed roof
# ============================================================

def make_shed_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_height,
    object_id=-1,
    slope_azimuth_deg=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
):
    """
    Create one-direction sloped/shed roof.

    The roof is one global plane:
        z = eave_z + roof_height * normalized_coordinate

    For concave footprints, the footprint is decomposed into convex pieces, but
    all pieces lie on the same plane.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    if slope_azimuth_deg is None:
        slope_azimuth_deg = estimate_main_axis_angle_deg(footprint)

    u_axis, v_axis = local_axes_from_angle_deg(slope_azimuth_deg)

    origin = polygon_centroid_xy(footprint)
    u, _ = local_coordinates_xy(footprint, origin, u_axis, v_axis)

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    span = max(EPS, u_max - u_min)

    def z_function(x, y):
        q = np.array([x, y], dtype=np.float64)
        uq = float(np.dot(q - origin, u_axis))
        t = (uq - u_min) / span
        t = max(0.0, min(1.0, t))

        return float(eave_z + roof_height * t)

    pieces = convex_decompose_polygon_xy(footprint)
    surfaces = []

    for piece_index, piece in enumerate(pieces):
        verts = polygon_with_z_from_function(piece, z_function)

        surfaces.append(
            make_roof_surface_spec(
                verts,
                object_id=object_id,
                roof_type="shed",
                surface_kind="pitched_roof",
                piece_index=piece_index,
                surface_index=len(surfaces),
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
                source="shed_roof",
            )
        )

    return surfaces


# ============================================================
# Gable roof
# ============================================================

def make_gable_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_height,
    object_id=-1,
    ridge_azimuth_deg=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
):
    """
    Create a gable roof over a possibly complex footprint.

    Method:
        1. Choose a global ridge axis through the footprint centroid.
        2. Decompose footprint into convex pieces.
        3. Split each convex piece by the ridge line.
        4. Each side becomes one planar roof piece.

    Ridge:
        v = 0 in local coordinates.
        z = eave_z + roof_height at v = 0.
        z = eave_z at the furthest footprint extent on each side.

    This is robust for synthetic geometry and produces planar pieces.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    if ridge_azimuth_deg is None:
        ridge_azimuth_deg = estimate_main_axis_angle_deg(footprint)

    u_axis, v_axis = local_axes_from_angle_deg(ridge_azimuth_deg)
    origin = polygon_centroid_xy(footprint)

    _, v_all = local_coordinates_xy(footprint, origin, u_axis, v_axis)

    v_min = float(np.min(v_all))
    v_max = float(np.max(v_all))

    neg_span = max(EPS, abs(v_min))
    pos_span = max(EPS, abs(v_max))

    def z_function(x, y):
        q = np.array([x, y], dtype=np.float64)
        vq = float(np.dot(q - origin, v_axis))

        if vq >= 0.0:
            t = 1.0 - min(1.0, abs(vq) / pos_span)
        else:
            t = 1.0 - min(1.0, abs(vq) / neg_span)

        return float(eave_z + roof_height * t)

    convex_pieces = convex_decompose_polygon_xy(footprint)
    surfaces = []

    for piece_index, piece in enumerate(convex_pieces):
        # Positive side.
        pos_poly = clip_polygon_by_halfplane_xy(
            piece,
            origin_xy=origin,
            normal_xy=v_axis,
            keep_positive=True,
        )

        if pos_poly is not None:
            verts = polygon_with_z_from_function(pos_poly, z_function)

            surfaces.append(
                make_roof_surface_spec(
                    verts,
                    object_id=object_id,
                    roof_type="gable",
                    surface_kind="pitched_roof",
                    piece_index=piece_index,
                    surface_index=len(surfaces),
                    can_cast_shadow=can_cast_shadow,
                    can_receive_shadow=can_receive_shadow,
                    opacity=opacity,
                    source="gable_roof_positive",
                )
            )

        # Negative side.
        neg_poly = clip_polygon_by_halfplane_xy(
            piece,
            origin_xy=origin,
            normal_xy=v_axis,
            keep_positive=False,
        )

        if neg_poly is not None:
            verts = polygon_with_z_from_function(neg_poly, z_function)

            surfaces.append(
                make_roof_surface_spec(
                    verts,
                    object_id=object_id,
                    roof_type="gable",
                    surface_kind="pitched_roof",
                    piece_index=piece_index,
                    surface_index=len(surfaces),
                    can_cast_shadow=can_cast_shadow,
                    can_receive_shadow=can_receive_shadow,
                    opacity=opacity,
                    source="gable_roof_negative",
                )
            )

    return surfaces


# ============================================================
# Pyramid / generic hip roof
# ============================================================

def make_pyramid_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_height,
    object_id=-1,
    apex_xy=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
    roof_type="pyramid",
):
    """
    Create a robust generic pitched roof by connecting every footprint edge to
    one apex point.

    This is valid for convex and concave simple footprints, but for strongly
    concave footprints the apex should be inside the footprint. We use the
    polygon centroid by default; for normal building footprints this is fine.

    For arbitrary concave polygons, a safer future version can place one apex
    per convex decomposed piece.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    if apex_xy is None:
        apex_xy = polygon_centroid_xy(footprint)

    apex_xy = np.asarray(apex_xy, dtype=np.float64)
    apex_z = float(eave_z + roof_height)

    apex = np.array([apex_xy[0], apex_xy[1], apex_z], dtype=np.float64)

    surfaces = []

    for i in range(len(footprint)):
        p0 = footprint[i]
        p1 = footprint[(i + 1) % len(footprint)]

        verts = np.array(
            [
                [p0[0], p0[1], float(eave_z)],
                [p1[0], p1[1], float(eave_z)],
                apex,
            ],
            dtype=np.float64,
        )

        try:
            surface = make_roof_surface_spec(
                verts,
                object_id=object_id,
                roof_type=roof_type,
                surface_kind="pitched_roof",
                piece_index=i,
                surface_index=len(surfaces),
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
                source=str(roof_type) + "_roof",
            )
            surfaces.append(surface)

        except ValueError:
            # Degenerate edge/apex alignment. Skip.
            continue

    return surfaces


def make_piecewise_pyramid_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_height,
    object_id=-1,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
    roof_type="piecewise_pyramid",
):
    """
    Safer version for concave footprints:
        decompose into convex pieces
        create one small pyramid roof per piece

    This may create more surfaces, but avoids weird apex placement in a concave
    notch.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)
    pieces = convex_decompose_polygon_xy(footprint)

    surfaces = []

    for piece_index, piece in enumerate(pieces):
        apex_xy = polygon_centroid_xy(piece)

        piece_surfaces = make_pyramid_roof_surfaces(
            piece,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            apex_xy=apex_xy,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
            roof_type=roof_type,
        )

        for s in piece_surfaces:
            s.piece_index = int(piece_index)
            s.surface_index = int(len(surfaces))
            surfaces.append(s)

    return surfaces


# ============================================================
# True-ish rectangular hip roof
# ============================================================

def _is_four_vertex_convex_polygon(footprint_xy):
    p = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    return len(p) == 4 and is_convex_polygon_xy(p)


def _rectangle_local_corner_order(footprint_xy, angle_deg):
    """
    Return rectangle-ish corners by local coordinate quadrant.

    Output dictionary:
        (-1,-1), (1,-1), (1,1), (-1,1) -> xy point

    This assumes a mostly rectangular convex quadrilateral.
    """
    p = clean_footprint_xy(footprint_xy, ensure_ccw=True)
    origin = polygon_centroid_xy(p)
    u_axis, v_axis = local_axes_from_angle_deg(angle_deg)

    u, v = local_coordinates_xy(p, origin, u_axis, v_axis)

    keys = []
    for i in range(len(p)):
        su = -1 if u[i] < 0.0 else 1
        sv = -1 if v[i] < 0.0 else 1
        keys.append((su, sv))

    out = {}

    for key in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        candidates = [i for i, k in enumerate(keys) if k == key]

        if len(candidates) == 1:
            out[key] = p[candidates[0]].copy()
        else:
            # fallback: choose point nearest quadrant target
            target = np.array([key[0], key[1]], dtype=np.float64)
            uv = np.column_stack([u, v])
            uv_norm = uv.copy()

            if np.max(np.abs(uv_norm[:, 0])) > EPS:
                uv_norm[:, 0] /= np.max(np.abs(uv_norm[:, 0]))

            if np.max(np.abs(uv_norm[:, 1])) > EPS:
                uv_norm[:, 1] /= np.max(np.abs(uv_norm[:, 1]))

            idx = int(np.argmin(np.linalg.norm(uv_norm - target, axis=1)))
            out[key] = p[idx].copy()

    return out, origin, u_axis, v_axis


def make_rectangular_hip_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_height,
    object_id=-1,
    ridge_azimuth_deg=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
):
    """
    Create a rectangular hip roof for 4-vertex convex footprints.

    If the footprint is not a 4-vertex convex polygon, this function raises.
    Use make_roof_surfaces(..., roof_type="hip") for automatic fallback.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    if not _is_four_vertex_convex_polygon(footprint):
        raise ValueError("Rectangular hip roof needs a 4-vertex convex footprint.")

    if ridge_azimuth_deg is None:
        ridge_azimuth_deg = estimate_main_axis_angle_deg(footprint)

    corners, origin, u_axis, v_axis = _rectangle_local_corner_order(
        footprint,
        ridge_azimuth_deg,
    )

    # Compute rectangle extents.
    u_all, v_all = local_coordinates_xy(footprint, origin, u_axis, v_axis)

    u_min = float(np.min(u_all))
    u_max = float(np.max(u_all))
    v_min = float(np.min(v_all))
    v_max = float(np.max(v_all))

    length = u_max - u_min
    width = v_max - v_min

    # If the chosen axis is actually the short axis, rotate by 90 degrees.
    if width > length:
        ridge_azimuth_deg = ridge_azimuth_deg + 90.0
        corners, origin, u_axis, v_axis = _rectangle_local_corner_order(
            footprint,
            ridge_azimuth_deg,
        )
        u_all, v_all = local_coordinates_xy(footprint, origin, u_axis, v_axis)

        u_min = float(np.min(u_all))
        u_max = float(np.max(u_all))
        v_min = float(np.min(v_all))
        v_max = float(np.max(v_all))

        length = u_max - u_min
        width = v_max - v_min

    half_width = 0.5 * width

    # Hip ridge shrinks from both ends by approximately half the width.
    ridge_u0 = u_min + half_width
    ridge_u1 = u_max - half_width

    # If footprint is close to square, ridge collapses to an apex.
    ridge_z = float(eave_z + roof_height)

    def xy_from_local(u, v):
        return origin + u * u_axis + v * v_axis

    if ridge_u1 <= ridge_u0 + EPS:
        return make_pyramid_roof_surfaces(
            footprint,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            apex_xy=origin,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
            roof_type="hip",
        )

    r0_xy = xy_from_local(ridge_u0, 0.0)
    r1_xy = xy_from_local(ridge_u1, 0.0)

    r0 = np.array([r0_xy[0], r0_xy[1], ridge_z], dtype=np.float64)
    r1 = np.array([r1_xy[0], r1_xy[1], ridge_z], dtype=np.float64)

    c00 = corners[(-1, -1)]
    c10 = corners[(1, -1)]
    c11 = corners[(1, 1)]
    c01 = corners[(-1, 1)]

    p00 = np.array([c00[0], c00[1], float(eave_z)], dtype=np.float64)
    p10 = np.array([c10[0], c10[1], float(eave_z)], dtype=np.float64)
    p11 = np.array([c11[0], c11[1], float(eave_z)], dtype=np.float64)
    p01 = np.array([c01[0], c01[1], float(eave_z)], dtype=np.float64)

    raw_surfaces = [
        # lower side trapezoid
        [p00, p10, r1, r0],
        # right hip triangle
        [p10, p11, r1],
        # upper side trapezoid
        [p11, p01, r0, r1],
        # left hip triangle
        [p01, p00, r0],
    ]

    surfaces = []

    for raw in raw_surfaces:
        try:
            surface = make_roof_surface_spec(
                np.asarray(raw, dtype=np.float64),
                object_id=object_id,
                roof_type="hip",
                surface_kind="pitched_roof",
                piece_index=len(surfaces),
                surface_index=len(surfaces),
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
                source="rectangular_hip_roof",
            )
            surfaces.append(surface)

        except ValueError:
            continue

    return surfaces


# ============================================================
# Main roof factory
# ============================================================

def make_roof_surfaces(
    footprint_xy,
    eave_z,
    roof_type="flat",
    roof_height=None,
    object_id=-1,
    ridge_azimuth_deg=None,
    slope_azimuth_deg=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
):
    """
    Main roof surface factory.

    Supported roof_type:
        "flat"
        "shed"
        "gable"
        "hip"
        "pyramid"
        "piecewise_pyramid"

    Notes:
        - "hip" uses a rectangular hip roof if the footprint is a 4-vertex
          convex polygon; otherwise it falls back to piecewise pyramid.
        - For arbitrary concave footprints, "piecewise_pyramid" is safer than
          a single apex pyramid.
    """
    roof_type = str(roof_type).lower().strip()

    if roof_height is None:
        if roof_type == "flat":
            roof_height = 0.0
        else:
            roof_height = 0.25 * math.sqrt(max(EPS, polygon_area_xy(footprint_xy)))

    roof_height = float(roof_height)

    if roof_type == "flat" or roof_height <= EPS:
        return make_flat_roof_surfaces(
            footprint_xy=footprint_xy,
            eave_z=eave_z,
            object_id=object_id,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
            decompose=True,
        )

    if roof_type == "shed":
        return make_shed_roof_surfaces(
            footprint_xy=footprint_xy,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            slope_azimuth_deg=slope_azimuth_deg,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
        )

    if roof_type == "gable":
        return make_gable_roof_surfaces(
            footprint_xy=footprint_xy,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            ridge_azimuth_deg=ridge_azimuth_deg,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
        )

    if roof_type == "hip":
        try:
            return make_rectangular_hip_roof_surfaces(
                footprint_xy=footprint_xy,
                eave_z=eave_z,
                roof_height=roof_height,
                object_id=object_id,
                ridge_azimuth_deg=ridge_azimuth_deg,
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
            )
        except Exception:
            return make_piecewise_pyramid_roof_surfaces(
                footprint_xy=footprint_xy,
                eave_z=eave_z,
                roof_height=roof_height,
                object_id=object_id,
                can_cast_shadow=can_cast_shadow,
                can_receive_shadow=can_receive_shadow,
                opacity=opacity,
                roof_type="hip",
            )

    if roof_type == "pyramid":
        return make_pyramid_roof_surfaces(
            footprint_xy=footprint_xy,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            apex_xy=None,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
            roof_type="pyramid",
        )

    if roof_type == "piecewise_pyramid":
        return make_piecewise_pyramid_roof_surfaces(
            footprint_xy=footprint_xy,
            eave_z=eave_z,
            roof_height=roof_height,
            object_id=object_id,
            can_cast_shadow=can_cast_shadow,
            can_receive_shadow=can_receive_shadow,
            opacity=opacity,
            roof_type="piecewise_pyramid",
        )

    raise ValueError(
        "Unsupported roof_type: {}. Expected one of: "
        "'flat', 'shed', 'gable', 'hip', 'pyramid', 'piecewise_pyramid'.".format(
            roof_type
        )
    )


def build_roof_profile_for_footprint(
    footprint_xy,
    eave_z,
    roof_type="flat",
    roof_height=None,
    object_id=-1,
    ridge_azimuth_deg=None,
    slope_azimuth_deg=None,
    can_cast_shadow=True,
    can_receive_shadow=True,
    opacity=1.0,
):
    """
    Build a complete roof profile for one building footprint.
    """
    footprint = clean_footprint_xy(footprint_xy, ensure_ccw=True)

    surfaces = make_roof_surfaces(
        footprint_xy=footprint,
        eave_z=eave_z,
        roof_type=roof_type,
        roof_height=roof_height,
        object_id=object_id,
        ridge_azimuth_deg=ridge_azimuth_deg,
        slope_azimuth_deg=slope_azimuth_deg,
        can_cast_shadow=can_cast_shadow,
        can_receive_shadow=can_receive_shadow,
        opacity=opacity,
    )

    if roof_height is None:
        if roof_type == "flat":
            roof_height = 0.0
        else:
            roof_height = 0.25 * math.sqrt(max(EPS, polygon_area_xy(footprint)))

    all_z = []

    for s in surfaces:
        all_z.extend(list(s.vertices[:, 2]))

    if all_z:
        min_z = float(np.min(all_z))
        max_z = float(np.max(all_z))
    else:
        min_z = float(eave_z)
        max_z = float(eave_z)

    total_area = float(sum(s.area for s in surfaces))

    return RoofProfile(
        object_id=int(object_id),
        footprint_xy=footprint,
        roof_type=str(roof_type),
        eave_z=float(eave_z),
        roof_height=float(roof_height),
        ridge_azimuth_deg=None if ridge_azimuth_deg is None else float(ridge_azimuth_deg),
        surfaces=surfaces,
        n_input_vertices=int(len(footprint)),
        n_surfaces=int(len(surfaces)),
        total_roof_area=total_area,
        max_z=max_z,
        min_z=min_z,
    )


# Backward-readable alias.
def make_roof_profile_for_building(
    footprint_xy,
    eave_z,
    roof_type="flat",
    roof_height=None,
    object_id=-1,
):
    return build_roof_profile_for_footprint(
        footprint_xy=footprint_xy,
        eave_z=eave_z,
        roof_type=roof_type,
        roof_height=roof_height,
        object_id=object_id,
    )


# ============================================================
# Diagnostics
# ============================================================

def roof_profile_summary(profile):
    surface_areas = [float(s.area) for s in profile.surfaces]
    surface_kinds = [str(s.surface_kind) for s in profile.surfaces]
    sources = [str(s.source) for s in profile.surfaces]

    return {
        "object_id": int(profile.object_id),
        "roof_type": str(profile.roof_type),
        "n_input_vertices": int(profile.n_input_vertices),
        "n_surfaces": int(profile.n_surfaces),
        "surface_kinds": surface_kinds,
        "sources": sources,
        "surface_areas": surface_areas,
        "total_roof_area": float(profile.total_roof_area),
        "eave_z": float(profile.eave_z),
        "roof_height": float(profile.roof_height),
        "min_z": float(profile.min_z),
        "max_z": float(profile.max_z),
        "ridge_azimuth_deg": profile.ridge_azimuth_deg,
    }


def print_roof_profile_summary(profile):
    s = roof_profile_summary(profile)

    print()
    print("=" * 80)
    print("ROOF PROFILE SUMMARY")
    print("=" * 80)
    print("object_id:", s["object_id"])
    print("roof_type:", s["roof_type"])
    print("input vertices:", s["n_input_vertices"])
    print("surfaces:", s["n_surfaces"])
    print("sources:", s["sources"])
    print("total roof area:", round(s["total_roof_area"], 6))
    print("eave_z:", s["eave_z"])
    print("roof_height:", s["roof_height"])
    print("min_z/max_z:", s["min_z"], s["max_z"])
    print("=" * 80)


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    rect = np.array([
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 4.0],
        [0.0, 4.0],
    ])

    l_shape = np.array([
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 2.0],
        [2.5, 2.0],
        [2.5, 5.0],
        [0.0, 5.0],
    ])

    examples = [
        ("rectangle", rect),
        ("L-shape", l_shape),
    ]

    roof_types = [
        "flat",
        "shed",
        "gable",
        "hip",
        "pyramid",
        "piecewise_pyramid",
    ]

    for name, footprint in examples:
        print()
        print("#" * 80)
        print(name)
        print("#" * 80)

        for rt in roof_types:
            profile = build_roof_profile_for_footprint(
                footprint_xy=footprint,
                eave_z=10.0,
                roof_type=rt,
                roof_height=2.0,
                object_id=1,
            )

            print_roof_profile_summary(profile)

            for i, surface in enumerate(profile.surfaces):
                print(
                    "  surface",
                    i,
                    "kind",
                    surface.surface_kind,
                    "source",
                    surface.source,
                    "nverts",
                    len(surface.vertices),
                    "area",
                    round(surface.area, 6),
                    "normal",
                    np.round(surface.normal, 4).tolist(),
                )