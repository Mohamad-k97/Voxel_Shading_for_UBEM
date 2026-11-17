# surface_array_builder.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import time

import numpy as np

from generator import UrbanPolygon
from shading_profiles import ShadingProfile


# ============================================================
# Codes
# ============================================================

EPS = 1e-10

OBJECT_KIND_CODE = {
    "unknown": 0,
    "building": 1,
    "tree": 2,
    "ground": 3,
    "urban_object": 4,
}

CODE_OBJECT_KIND = dict((v, k) for k, v in OBJECT_KIND_CODE.items())


SURFACE_KIND_CODE = {
    "unknown": 0,

    # Physical building surfaces.
    "wall": 1,
    "flat_roof": 2,
    "pitched_roof": 3,

    # Ground receivers.
    "ground": 4,
    "paved_ground": 5,

    # Tree proxies.
    "tree_crown": 6,
    "tree_crown_top": 7,
    "tree_trunk": 8,

    # New shading proxy caster surface.
    "barrier_wall_proxy": 9,
    "barrier_caster": 10,
}

CODE_SURFACE_KIND = dict((v, k) for k, v in SURFACE_KIND_CODE.items())


# ============================================================
# Data containers
# ============================================================

@dataclass
class SurfaceArrays:
    """
    Compact array representation of shading-profile polygons.

    The first block mirrors the old array_gen.py SurfaceArrays structure:

        surface_id
        object_id
        object_kind
        surface_kind
        center
        normal
        area
        radius_xy
        radius_z
        radius_full
        bbox_xy
        opacity
        can_cast
        can_receive
        normal_azimuth_deg

    Additional optional geometry arrays:

        vertex_start
        vertex_count
        vertices

    These let us reconstruct polygon geometry later without touching Python
    UrbanPolygon objects.
    """

    surface_id: np.ndarray        # uint32, (N,)
    object_id: np.ndarray         # uint32, (N,)
    object_kind: np.ndarray       # uint8,  (N,)
    surface_kind: np.ndarray      # uint8,  (N,)

    center: np.ndarray            # float32, (N, 3)
    normal: np.ndarray            # float32, (N, 3)
    area: np.ndarray              # float32, (N,)

    radius_xy: np.ndarray         # float32, (N,)
    radius_z: np.ndarray          # float32, (N,)
    radius_full: np.ndarray       # float32, (N,)

    bbox_xy: np.ndarray           # float32, (N, 4), xmin,ymin,xmax,ymax

    opacity: np.ndarray           # float32, (N,)
    can_cast: np.ndarray          # bool,    (N,)
    can_receive: np.ndarray       # bool,    (N,)

    normal_azimuth_deg: np.ndarray  # float32, (N,)

    # Extra compact polygon geometry.
    vertex_start: Optional[np.ndarray] = None  # uint32, (N,)
    vertex_count: Optional[np.ndarray] = None  # uint16, (N,)
    vertices: Optional[np.ndarray] = None      # float32, (M, 3)

    # Extra flags useful later.
    two_sided: Optional[np.ndarray] = None      # bool, (N,)

    # Metadata as plain Python dictionary.
    metadata: Optional[Dict] = None


# ============================================================
# Basic helpers
# ============================================================

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n <= EPS:
        return v * 0.0

    return v / n


def wrap_angle_360(angle_deg):
    return float(angle_deg) % 360.0


def vectorized_normal_azimuth(normals):
    """
    Azimuth convention:
        0   = North / +y
        90  = East  / +x
        180 = South / -y
        270 = West  / -x

    For near-horizontal normals, azimuth is set to 0.
    """
    n = np.asarray(normals, dtype=np.float64)

    nx = n[:, 0]
    ny = n[:, 1]

    h = np.sqrt(nx * nx + ny * ny)

    az = np.degrees(np.arctan2(nx, ny))
    az = np.mod(az, 360.0)

    az[h < EPS] = 0.0

    return az.astype(np.float32)


def polygon_area_3d(vertices):
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) < 3:
        return 0.0

    v0 = v[0]
    area = 0.0

    for i in range(1, len(v) - 1):
        area += 0.5 * np.linalg.norm(np.cross(v[i] - v0, v[i + 1] - v0))

    return float(area)


def polygon_normal_3d(vertices):
    """
    Newell normal.
    """
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    n = np.zeros(3, dtype=np.float64)

    for i in range(len(v)):
        p = v[i]
        q = v[(i + 1) % len(v)]

        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])

    norm = np.linalg.norm(n)

    if norm > EPS:
        return n / norm

    # Fallback to first valid triangle.
    for i in range(1, len(v) - 1):
        n = np.cross(v[i] - v[0], v[i + 1] - v[0])
        norm = np.linalg.norm(n)

        if norm > EPS:
            return n / norm

    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def polygon_center(vertices):
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) == 0:
        return np.zeros(3, dtype=np.float64)

    return np.mean(v, axis=0)


def polygon_radii_from_vertices(vertices, center):
    """
    Fast approximate size descriptors.

    radius_xy:
        maximum horizontal distance from center

    radius_z:
        maximum vertical distance from center

    radius_full:
        maximum 3D distance from center
    """
    verts = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    if len(verts) == 0:
        return 0.0, 0.0, 0.0

    rel = verts - c

    radius_xy = float(np.max(np.linalg.norm(rel[:, :2], axis=1)))
    radius_z = float(np.max(np.abs(rel[:, 2])))
    radius_full = float(np.max(np.linalg.norm(rel, axis=1)))

    return radius_xy, radius_z, radius_full


def bbox_xy_from_vertices(vertices):
    v = np.asarray(vertices, dtype=np.float64)

    if len(v) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return np.array(
        [
            float(np.min(v[:, 0])),
            float(np.min(v[:, 1])),
            float(np.max(v[:, 0])),
            float(np.max(v[:, 1])),
        ],
        dtype=np.float32,
    )


def object_kind_to_code(object_kind):
    return np.uint8(OBJECT_KIND_CODE.get(str(object_kind), OBJECT_KIND_CODE["unknown"]))


def surface_kind_to_code(surface_kind):
    return np.uint8(SURFACE_KIND_CODE.get(str(surface_kind), SURFACE_KIND_CODE["unknown"]))


def get_polygon_two_sided(poly):
    """
    Barrier proxy surfaces are intended to be two-sided opaque casters.

    The flag may be stored in poly.metadata["two_sided"] by shading_profiles.py.
    For ordinary physical surfaces, default is False.
    """
    if poly.metadata is not None:
        if "two_sided" in poly.metadata:
            return bool(poly.metadata["two_sided"])

    if poly.surface_kind in ("barrier_wall_proxy", "barrier_caster"):
        return True

    return False


# ============================================================
# Main builder
# ============================================================

def build_surface_arrays_from_polygons(
    polygons,
    profile_metadata=None,
    include_vertices=True,
    force_float32=True,
):
    """
    Convert a list of UrbanPolygon-like objects into compact arrays.

    Input:
        polygons are still object-based.

    Output:
        SurfaceArrays, suitable for the lobe/NJIT-style calculation.

    This is intentionally the only place where we touch polygon objects during
    the fast pipeline.
    """
    t0 = time.perf_counter()

    polygons = list(polygons)
    n = len(polygons)

    surface_id = np.zeros(n, dtype=np.uint32)
    object_id = np.zeros(n, dtype=np.uint32)
    object_kind = np.zeros(n, dtype=np.uint8)
    surface_kind = np.zeros(n, dtype=np.uint8)

    center = np.zeros((n, 3), dtype=np.float32)
    normal = np.zeros((n, 3), dtype=np.float32)
    area = np.zeros(n, dtype=np.float32)

    radius_xy = np.zeros(n, dtype=np.float32)
    radius_z = np.zeros(n, dtype=np.float32)
    radius_full = np.zeros(n, dtype=np.float32)

    bbox_xy = np.zeros((n, 4), dtype=np.float32)

    opacity = np.ones(n, dtype=np.float32)
    can_cast = np.zeros(n, dtype=bool)
    can_receive = np.zeros(n, dtype=bool)

    two_sided = np.zeros(n, dtype=bool)

    if include_vertices:
        vertex_count = np.zeros(n, dtype=np.uint16)
        vertex_start = np.zeros(n, dtype=np.uint32)
        vertices_list = []
    else:
        vertex_count = None
        vertex_start = None
        vertices_list = None

    for i, poly in enumerate(polygons):
        verts = np.asarray(poly.vertices, dtype=np.float64)

        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                "Polygon sid={} has invalid vertices shape {}. Expected Nx3.".format(
                    getattr(poly, "sid", -1),
                    verts.shape,
                )
            )

        if len(verts) < 3:
            raise ValueError(
                "Polygon sid={} has fewer than 3 vertices.".format(
                    getattr(poly, "sid", -1)
                )
            )

        # Core identity/codes.
        surface_id[i] = np.uint32(int(poly.sid))
        object_id[i] = np.uint32(max(0, int(poly.object_id)))
        object_kind[i] = object_kind_to_code(poly.object_kind)
        surface_kind[i] = surface_kind_to_code(poly.surface_kind)

        # Geometry. Prefer polygon fields but recompute if degenerate/missing.
        c = np.asarray(poly.center, dtype=np.float64)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            c = polygon_center(verts)

        nrm = np.asarray(poly.normal, dtype=np.float64)
        if nrm.shape != (3,) or np.linalg.norm(nrm) <= EPS or not np.all(np.isfinite(nrm)):
            nrm = polygon_normal_3d(verts)
        else:
            nrm = unit(nrm)

        a = float(poly.area)
        if a <= EPS or not np.isfinite(a):
            a = polygon_area_3d(verts)

        rxy, rz, rfull = polygon_radii_from_vertices(verts, c)

        center[i, :] = c.astype(np.float32)
        normal[i, :] = nrm.astype(np.float32)
        area[i] = np.float32(max(0.0, a))

        radius_xy[i] = np.float32(max(0.0, rxy))
        radius_z[i] = np.float32(max(0.0, rz))
        radius_full[i] = np.float32(max(0.0, rfull))

        bbox_xy[i, :] = bbox_xy_from_vertices(verts)

        opacity[i] = np.float32(max(0.0, min(1.0, float(poly.opacity))))
        can_cast[i] = bool(poly.can_cast_shadow)
        can_receive[i] = bool(poly.can_receive_shadow)
        two_sided[i] = bool(get_polygon_two_sided(poly))

        if include_vertices:
            if len(verts) > np.iinfo(np.uint16).max:
                raise ValueError(
                    "Polygon sid={} has too many vertices for uint16 vertex_count.".format(
                        int(poly.sid)
                    )
                )

            vertex_start[i] = np.uint32(len(vertices_list))
            vertex_count[i] = np.uint16(len(verts))

            for v in verts:
                vertices_list.append(v.astype(np.float32))

    normal_azimuth_deg = vectorized_normal_azimuth(normal)

    if include_vertices:
        if len(vertices_list) > 0:
            vertices = np.asarray(vertices_list, dtype=np.float32)
        else:
            vertices = np.zeros((0, 3), dtype=np.float32)
    else:
        vertices = None

    elapsed_s = time.perf_counter() - t0

    metadata = {
        "array_format": "surface_arrays_v2",
        "n_surfaces": int(n),
        "n_vertices": 0 if vertices is None else int(len(vertices)),
        "include_vertices": bool(include_vertices),
        "build_elapsed_s": float(elapsed_s),
        "object_kind_code": OBJECT_KIND_CODE,
        "surface_kind_code": SURFACE_KIND_CODE,
    }

    if profile_metadata is not None:
        metadata["profile_metadata"] = profile_metadata

    arrays = SurfaceArrays(
        surface_id=surface_id,
        object_id=object_id,
        object_kind=object_kind,
        surface_kind=surface_kind,
        center=center,
        normal=normal,
        area=area,
        radius_xy=radius_xy,
        radius_z=radius_z,
        radius_full=radius_full,
        bbox_xy=bbox_xy,
        opacity=opacity,
        can_cast=can_cast,
        can_receive=can_receive,
        normal_azimuth_deg=normal_azimuth_deg,
        vertex_start=vertex_start,
        vertex_count=vertex_count,
        vertices=vertices,
        two_sided=two_sided,
        metadata=metadata,
    )

    return arrays


def build_surface_arrays_from_profile(
    shading_profile,
    include_vertices=True,
):
    """
    Convert a ShadingProfile into SurfaceArrays.
    """
    if not isinstance(shading_profile, ShadingProfile):
        # Do not hard-fail on duck-typed profile objects.
        if not hasattr(shading_profile, "polygons"):
            raise TypeError("Expected ShadingProfile or object with .polygons.")

    profile_metadata = {}

    if hasattr(shading_profile, "metadata"):
        profile_metadata.update(shading_profile.metadata)

    if hasattr(shading_profile, "stats"):
        profile_metadata["profile_stats"] = shading_profile.stats

    return build_surface_arrays_from_polygons(
        polygons=shading_profile.polygons,
        profile_metadata=profile_metadata,
        include_vertices=include_vertices,
    )


# ============================================================
# Filtering and indexing helpers
# ============================================================

def get_caster_indices(arrays):
    return np.where(arrays.can_cast)[0].astype(np.int64)


def get_receiver_indices(arrays):
    return np.where(arrays.can_receive)[0].astype(np.int64)


def get_surface_indices_by_kind(arrays, surface_kind_name):
    code = SURFACE_KIND_CODE.get(str(surface_kind_name), SURFACE_KIND_CODE["unknown"])

    return np.where(arrays.surface_kind == np.uint8(code))[0].astype(np.int64)


def get_surface_indices_by_object_kind(arrays, object_kind_name):
    code = OBJECT_KIND_CODE.get(str(object_kind_name), OBJECT_KIND_CODE["unknown"])

    return np.where(arrays.object_kind == np.uint8(code))[0].astype(np.int64)


def build_sid_to_index(arrays):
    """
    Build dictionary:
        surface_id -> row index
    """
    out = {}

    for i, sid in enumerate(arrays.surface_id):
        out[int(sid)] = int(i)

    return out


def build_object_to_surface_indices(arrays):
    """
    Build dictionary:
        object_id -> list of row indices
    """
    out = {}

    for i, oid in enumerate(arrays.object_id):
        oid = int(oid)
        out.setdefault(oid, []).append(int(i))

    return out


# ============================================================
# Geometry reconstruction helpers
# ============================================================

def get_surface_vertices(arrays, row_index):
    """
    Return vertices for one surface row.

    Requires include_vertices=True at build time.
    """
    if arrays.vertices is None or arrays.vertex_start is None or arrays.vertex_count is None:
        raise ValueError("This SurfaceArrays object does not contain compact vertices.")

    i = int(row_index)
    start = int(arrays.vertex_start[i])
    count = int(arrays.vertex_count[i])

    return arrays.vertices[start:start + count].copy()


def iter_surface_vertices(arrays):
    """
    Yield:
        row_index, vertices
    """
    if arrays.vertices is None or arrays.vertex_start is None or arrays.vertex_count is None:
        raise ValueError("This SurfaceArrays object does not contain compact vertices.")

    for i in range(len(arrays.surface_id)):
        yield i, get_surface_vertices(arrays, i)


# ============================================================
# Memory / summary
# ============================================================

def array_nbytes(arr):
    if arr is None:
        return 0

    return int(arr.nbytes)


def surface_arrays_nbytes(arrays):
    total = 0

    total += array_nbytes(arrays.surface_id)
    total += array_nbytes(arrays.object_id)
    total += array_nbytes(arrays.object_kind)
    total += array_nbytes(arrays.surface_kind)
    total += array_nbytes(arrays.center)
    total += array_nbytes(arrays.normal)
    total += array_nbytes(arrays.area)
    total += array_nbytes(arrays.radius_xy)
    total += array_nbytes(arrays.radius_z)
    total += array_nbytes(arrays.radius_full)
    total += array_nbytes(arrays.bbox_xy)
    total += array_nbytes(arrays.opacity)
    total += array_nbytes(arrays.can_cast)
    total += array_nbytes(arrays.can_receive)
    total += array_nbytes(arrays.normal_azimuth_deg)
    total += array_nbytes(arrays.vertex_start)
    total += array_nbytes(arrays.vertex_count)
    total += array_nbytes(arrays.vertices)
    total += array_nbytes(arrays.two_sided)

    return int(total)


def mb(nbytes):
    return float(nbytes) / (1024.0 * 1024.0)


def _counts_from_codes(values):
    counts = {}

    for v in values:
        v_int = int(v)
        counts[v_int] = counts.get(v_int, 0) + 1

    return counts


def _decode_counts(counts, decoder):
    out = {}

    for code, count in counts.items():
        name = decoder.get(int(code), "unknown")
        out[name] = int(count)

    return out


def surface_arrays_summary(arrays):
    n = int(len(arrays.surface_id))

    n_vertices = 0
    if arrays.vertices is not None:
        n_vertices = int(len(arrays.vertices))

    object_counts = _decode_counts(
        _counts_from_codes(arrays.object_kind),
        CODE_OBJECT_KIND,
    )

    surface_counts = _decode_counts(
        _counts_from_codes(arrays.surface_kind),
        CODE_SURFACE_KIND,
    )

    caster_indices = get_caster_indices(arrays)
    receiver_indices = get_receiver_indices(arrays)

    summary = {
        "n_surfaces": n,
        "n_vertices": n_vertices,

        "n_casters": int(len(caster_indices)),
        "n_receivers": int(len(receiver_indices)),

        "n_two_sided": 0 if arrays.two_sided is None else int(np.sum(arrays.two_sided)),

        "total_area": float(np.sum(arrays.area)),
        "caster_area": float(np.sum(arrays.area[caster_indices])) if len(caster_indices) > 0 else 0.0,
        "receiver_area": float(np.sum(arrays.area[receiver_indices])) if len(receiver_indices) > 0 else 0.0,

        "object_kind_counts": object_counts,
        "surface_kind_counts": surface_counts,

        "memory_bytes": int(surface_arrays_nbytes(arrays)),
        "memory_mb": float(mb(surface_arrays_nbytes(arrays))),
    }

    if arrays.metadata is not None:
        summary["metadata"] = arrays.metadata

    return summary


def print_surface_arrays_summary(arrays):
    s = surface_arrays_summary(arrays)

    print()
    print("=" * 90)
    print("SURFACE ARRAYS SUMMARY")
    print("=" * 90)
    print("n_surfaces:", s["n_surfaces"])
    print("n_vertices:", s["n_vertices"])
    print("n_casters:", s["n_casters"])
    print("n_receivers:", s["n_receivers"])
    print("n_two_sided:", s["n_two_sided"])
    print("total_area:", round(s["total_area"], 3))
    print("caster_area:", round(s["caster_area"], 3))
    print("receiver_area:", round(s["receiver_area"], 3))
    print("memory_mb:", round(s["memory_mb"], 4))
    print()
    print("object_kind_counts:")
    print(s["object_kind_counts"])
    print()
    print("surface_kind_counts:")
    print(s["surface_kind_counts"])
    print("=" * 90)
    print()


# ============================================================
# Save/load
# ============================================================

def _metadata_to_json(metadata):
    if metadata is None:
        return "{}"

    try:
        return json.dumps(metadata)
    except TypeError:
        # Fallback: stringify non-JSON values.
        return json.dumps(_make_json_safe(metadata))


def _make_json_safe(value):
    if isinstance(value, dict):
        out = {}

        for k, v in value.items():
            out[str(k)] = _make_json_safe(v)

        return out

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def save_surface_arrays(arrays, path, compressed=True):
    """
    Save SurfaceArrays to .npz.

    The file is readable with numpy.load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "surface_id": arrays.surface_id,
        "object_id": arrays.object_id,
        "object_kind": arrays.object_kind,
        "surface_kind": arrays.surface_kind,

        "center": arrays.center,
        "normal": arrays.normal,
        "area": arrays.area,

        "radius_xy": arrays.radius_xy,
        "radius_z": arrays.radius_z,
        "radius_full": arrays.radius_full,

        "bbox_xy": arrays.bbox_xy,

        "opacity": arrays.opacity,
        "can_cast": arrays.can_cast,
        "can_receive": arrays.can_receive,

        "normal_azimuth_deg": arrays.normal_azimuth_deg,

        "metadata_json": np.array(_metadata_to_json(arrays.metadata)),
    }

    if arrays.vertex_start is not None:
        payload["vertex_start"] = arrays.vertex_start

    if arrays.vertex_count is not None:
        payload["vertex_count"] = arrays.vertex_count

    if arrays.vertices is not None:
        payload["vertices"] = arrays.vertices

    if arrays.two_sided is not None:
        payload["two_sided"] = arrays.two_sided

    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)

    return path


def load_surface_arrays(path):
    """
    Load SurfaceArrays from .npz.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        metadata = {}

        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))

        vertex_start = data["vertex_start"] if "vertex_start" in data else None
        vertex_count = data["vertex_count"] if "vertex_count" in data else None
        vertices = data["vertices"] if "vertices" in data else None
        two_sided = data["two_sided"] if "two_sided" in data else None

        arrays = SurfaceArrays(
            surface_id=data["surface_id"],
            object_id=data["object_id"],
            object_kind=data["object_kind"],
            surface_kind=data["surface_kind"],

            center=data["center"],
            normal=data["normal"],
            area=data["area"],

            radius_xy=data["radius_xy"],
            radius_z=data["radius_z"],
            radius_full=data["radius_full"],

            bbox_xy=data["bbox_xy"],

            opacity=data["opacity"],
            can_cast=data["can_cast"],
            can_receive=data["can_receive"],

            normal_azimuth_deg=data["normal_azimuth_deg"],

            vertex_start=vertex_start,
            vertex_count=vertex_count,
            vertices=vertices,
            two_sided=two_sided,

            metadata=metadata,
        )

    return arrays


# ============================================================
# Compatibility helper for old array_gen.py-style code
# ============================================================

def surface_arrays_to_legacy_dict(arrays):
    """
    Return a plain dictionary with the core arrays expected by old code.

    Use this when a function wants individual arrays instead of the dataclass.
    """
    return {
        "surface_id": arrays.surface_id,
        "object_id": arrays.object_id,
        "object_kind": arrays.object_kind,
        "surface_kind": arrays.surface_kind,

        "center": arrays.center,
        "normal": arrays.normal,
        "area": arrays.area,

        "radius_xy": arrays.radius_xy,
        "radius_z": arrays.radius_z,
        "radius_full": arrays.radius_full,

        "bbox_xy": arrays.bbox_xy,

        "opacity": arrays.opacity,
        "can_cast": arrays.can_cast,
        "can_receive": arrays.can_receive,

        "normal_azimuth_deg": arrays.normal_azimuth_deg,
    }


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    from generator import generate_urban_receiver_and_casters
    from shading_profiles import build_shading_profile

    scene = generate_urban_receiver_and_casters(
        n_polygons=120,
        terrain_type="small_town",
        seed=42,
        built_density=0.30,
        green_area_fraction=0.25,
        average_building_height=12.0,
        height_lognorm_sigma=0.30,
        elevation_type="undulating",
        verbose=True,
    )

    profile = build_shading_profile(
        scene=scene,
        profile_mode="barrier_proxy",
        decomposition_mode="merged",
        barrier_vertical_top_mode="eave",
        verbose=True,
    )

    arrays = build_surface_arrays_from_profile(
        shading_profile=profile,
        include_vertices=True,
    )

    print_surface_arrays_summary(arrays)

    out_path = save_surface_arrays(
        arrays,
        Path("surface_array_builder_selftest") / "surface_arrays.npz",
        compressed=True,
    )

    print("saved:", out_path)

    loaded = load_surface_arrays(out_path)
    print_surface_arrays_summary(loaded)

    # Tiny geometry reconstruction check.
    if loaded.vertices is not None and len(loaded.surface_id) > 0:
        verts0 = get_surface_vertices(loaded, 0)
        print("first surface vertices shape:", verts0.shape)