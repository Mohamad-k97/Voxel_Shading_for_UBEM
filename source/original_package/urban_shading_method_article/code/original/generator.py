# generator.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from footprint_geometry import (
    EPS,
    clean_footprint_xy,
    convex_decompose_polygon_xy,
    ensure_ccw_xy,
    is_convex_polygon_xy,
    polygon_area_xy,
    polygon_centroid_xy,
)

from roof_profiles import build_roof_profile_for_footprint


# ============================================================
# Data structures
# ============================================================

@dataclass
class UrbanPolygon:
    sid: int
    object_id: int
    object_kind: str          # "building", "tree", "ground", "urban_object"
    surface_kind: str         # "wall", "flat_roof", "pitched_roof", "ground", ...
    vertices: np.ndarray      # shape (N, 3)
    normal: np.ndarray        # unit normal; defines front side
    center: np.ndarray
    area: float
    can_receive_shadow: bool
    can_cast_shadow: bool
    opacity: float = 1.0

    # New optional metadata. Old scripts can ignore this.
    footprint_xy: Optional[np.ndarray] = None
    roof_type: Optional[str] = None
    base_z: Optional[float] = None
    eave_z: Optional[float] = None
    top_z: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class BuildingRecord:
    object_id: int
    footprint_xy: np.ndarray
    center_xy: np.ndarray
    base_z: float
    eave_z: float
    top_z: float
    height: float
    roof_type: str
    roof_height: float
    rotation_rad: float
    footprint_kind: str
    wall_sids: List[int]
    roof_sids: List[int]
    footprint_area: float
    radius_xy: float


@dataclass
class UrbanMorphologyConfig:
    """
    Explicit controls for synthetic urban morphology.

    built_density:
        Approximate plan-area fraction covered by building footprints.

    green_area_fraction:
        Approximate plan-area fraction occupied by trees and green patches.

    average_building_height:
        Mean building height used to parameterize the lognormal height distribution.

    elevation_type:
        One of:
            "flat"
            "gentle_slope"
            "steep_slope"
            "valley"
            "ridge"
            "terraced"
            "undulating"

    New footprint controls:
        rectangular_footprint_probability:
            Most synthetic buildings remain rectangular.

        convex_footprint_probability:
            Fraction of non-rectangular buildings that become convex polygons.

        concave_footprint_probability:
            Fraction of non-rectangular buildings that become concave simple polygons.

    New roof controls:
        roof_type_probabilities:
            Probabilities for "flat", "gable", "hip", "shed", "pyramid",
            and "piecewise_pyramid".
    """

    terrain_type: str = "custom"

    area_size: float = 400.0

    built_density: float = 0.30
    green_area_fraction: float = 0.20

    average_building_height: float = 15.0
    height_lognorm_sigma: float = 0.35
    min_building_height: float = 3.0
    max_building_height: float = 120.0

    footprint_w_range: Tuple[float, float] = (10.0, 35.0)
    footprint_d_range: Tuple[float, float] = (10.0, 35.0)

    min_clearance: float = 8.0

    # Backward-compatible old setting. Still used to build default roof probs.
    flat_roof_probability: float = 0.60

    # New footprint probabilities.
    rectangular_footprint_probability: float = 0.78
    convex_footprint_probability: float = 0.15
    concave_footprint_probability: float = 0.07
    convex_vertex_range: Tuple[int, int] = (5, 8)

    # New roof distribution. If None, it is created from flat_roof_probability.
    roof_type_probabilities: Optional[Dict[str, float]] = None
    roof_height_fraction_range: Tuple[float, float] = (0.12, 0.35)

    tree_share_of_green: float = 0.65
    tree_crown_radius_range: Tuple[float, float] = (2.0, 8.0)
    tree_crown_height_range: Tuple[float, float] = (3.0, 8.0)
    tree_trunk_height_range: Tuple[float, float] = (1.5, 5.0)

    ground_patch_size_range: Tuple[float, float] = (8.0, 35.0)

    grid_alignment_prob: float = 0.70
    rotation_noise_deg: float = 15.0

    elevation_type: str = "flat"
    elevation_amplitude: float = 8.0
    elevation_slope_deg: float = 3.0
    elevation_aspect_deg: float = 180.0
    terrace_step_height: float = 2.0
    terrace_width: float = 40.0

    receiver_weights: Optional[Dict[str, float]] = None


@dataclass
class UrbanScene:
    terrain_type: str
    receiver: UrbanPolygon
    casters: List[UrbanPolygon]
    all_polygons: List[UrbanPolygon]
    area_size: float

    config: Optional[UrbanMorphologyConfig] = None
    actual_built_density: Optional[float] = None
    actual_green_area_fraction: Optional[float] = None
    n_buildings: Optional[int] = None
    n_trees: Optional[int] = None

    # New: building-level metadata for preprocessing.
    buildings: Optional[List[BuildingRecord]] = None


# ============================================================
# Basic geometry helpers
# ============================================================

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < EPS:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / n


def polygon_normal(vertices):
    """
    Polygon normal by Newell's method.
    More robust for roof polygons than using only the first triangle.
    """
    v = np.asarray(vertices, dtype=float)

    if len(v) < 3:
        raise ValueError("Need at least three vertices to compute a normal.")

    n = np.zeros(3, dtype=float)

    for i in range(len(v)):
        p = v[i]
        q = v[(i + 1) % len(v)]

        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])

    norm = np.linalg.norm(n)

    if norm > EPS:
        return n / norm

    # Fallback to first non-degenerate triangle.
    for i in range(1, len(v) - 1):
        n = np.cross(v[i] - v[0], v[i + 1] - v[0])
        norm = np.linalg.norm(n)

        if norm > EPS:
            return n / norm

    raise ValueError("Cannot compute normal for degenerate polygon.")


def polygon_area(vertices):
    """
    Polygon area by fan triangulation.
    Intended for planar convex-ish pieces.
    """
    v = np.asarray(vertices, dtype=float)

    if len(v) < 3:
        return 0.0

    v0 = v[0]
    area = 0.0

    for i in range(1, len(v) - 1):
        area += 0.5 * np.linalg.norm(np.cross(v[i] - v0, v[i + 1] - v0))

    return float(area)


def polygon_centroid(vertices):
    return np.mean(np.asarray(vertices, dtype=float), axis=0)


def orient_polygon(vertices, desired_normal):
    """
    Ensure polygon winding creates a normal aligned with desired_normal.
    """
    vertices = np.asarray(vertices, dtype=float)
    desired_normal = unit(desired_normal)

    n = polygon_normal(vertices)

    if np.dot(n, desired_normal) < 0:
        vertices = vertices[::-1].copy()

    return vertices


def make_polygon(
    sid,
    object_id,
    object_kind,
    surface_kind,
    vertices,
    desired_normal,
    can_receive_shadow,
    can_cast_shadow,
    opacity=1.0,
    footprint_xy=None,
    roof_type=None,
    base_z=None,
    eave_z=None,
    top_z=None,
    metadata=None,
):
    vertices = orient_polygon(vertices, desired_normal)
    normal = polygon_normal(vertices)
    center = polygon_centroid(vertices)
    area = polygon_area(vertices)

    return UrbanPolygon(
        sid=int(sid),
        object_id=int(object_id),
        object_kind=str(object_kind),
        surface_kind=str(surface_kind),
        vertices=vertices,
        normal=normal,
        center=center,
        area=float(area),
        can_receive_shadow=bool(can_receive_shadow),
        can_cast_shadow=bool(can_cast_shadow),
        opacity=float(opacity),
        footprint_xy=None if footprint_xy is None else np.asarray(footprint_xy, dtype=float),
        roof_type=roof_type,
        base_z=None if base_z is None else float(base_z),
        eave_z=None if eave_z is None else float(eave_z),
        top_z=None if top_z is None else float(top_z),
        metadata=metadata,
    )


# ============================================================
# Default terrain presets
# ============================================================

def _default_roof_probs(flat_probability):
    flat = float(flat_probability)
    flat = max(0.0, min(1.0, flat))
    rest = 1.0 - flat

    return {
        "flat": flat,
        "gable": 0.45 * rest,
        "hip": 0.25 * rest,
        "shed": 0.15 * rest,
        "pyramid": 0.10 * rest,
        "piecewise_pyramid": 0.05 * rest,
    }


def normalize_probability_dict(d):
    if d is None:
        raise ValueError("Probability dictionary cannot be None here.")

    total = 0.0

    for value in d.values():
        total += max(0.0, float(value))

    if total <= EPS:
        raise ValueError("Probability dictionary has zero total probability.")

    out = {}

    for key, value in d.items():
        out[str(key)] = max(0.0, float(value)) / total

    return out


def default_morphology_config(terrain_type):
    """
    Terrain presets. These are defaults, not hard constraints.
    You can override built_density, green_area_fraction, average height, etc.
    """

    if terrain_type == "tall_buildings":
        config = UrbanMorphologyConfig(
            terrain_type="tall_buildings",
            area_size=250.0,
            built_density=0.55,
            green_area_fraction=0.08,
            average_building_height=55.0,
            height_lognorm_sigma=0.45,
            min_building_height=20.0,
            max_building_height=150.0,
            footprint_w_range=(16.0, 42.0),
            footprint_d_range=(16.0, 42.0),
            min_clearance=4.0,
            flat_roof_probability=0.86,
            rectangular_footprint_probability=0.82,
            convex_footprint_probability=0.13,
            concave_footprint_probability=0.05,
            roof_height_fraction_range=(0.06, 0.20),
            tree_share_of_green=0.55,
            tree_crown_radius_range=(2.0, 7.0),
            tree_crown_height_range=(3.0, 7.0),
            tree_trunk_height_range=(1.5, 4.0),
            ground_patch_size_range=(6.0, 22.0),
            grid_alignment_prob=0.90,
            rotation_noise_deg=6.0,
            elevation_type="flat",
            elevation_amplitude=4.0,
            elevation_slope_deg=1.0,
            receiver_weights={
                "wall": 0.70,
                "flat_roof": 0.15,
                "pitched_roof": 0.00,
                "ground": 0.08,
                "paved_ground": 0.07,
            },
        )
        config.roof_type_probabilities = {
            "flat": 0.86,
            "gable": 0.06,
            "hip": 0.04,
            "shed": 0.03,
            "pyramid": 0.01,
            "piecewise_pyramid": 0.00,
        }
        return config

    if terrain_type == "horizontal_city":
        config = UrbanMorphologyConfig(
            terrain_type="horizontal_city",
            area_size=450.0,
            built_density=0.38,
            green_area_fraction=0.18,
            average_building_height=16.0,
            height_lognorm_sigma=0.30,
            min_building_height=6.0,
            max_building_height=35.0,
            footprint_w_range=(25.0, 80.0),
            footprint_d_range=(18.0, 65.0),
            min_clearance=12.0,
            flat_roof_probability=0.70,
            rectangular_footprint_probability=0.74,
            convex_footprint_probability=0.17,
            concave_footprint_probability=0.09,
            roof_height_fraction_range=(0.10, 0.28),
            tree_share_of_green=0.60,
            tree_crown_radius_range=(2.5, 9.0),
            tree_crown_height_range=(3.0, 8.0),
            tree_trunk_height_range=(1.5, 5.0),
            ground_patch_size_range=(8.0, 35.0),
            grid_alignment_prob=0.75,
            rotation_noise_deg=14.0,
            elevation_type="flat",
            elevation_amplitude=5.0,
            elevation_slope_deg=1.5,
            receiver_weights={
                "wall": 0.45,
                "flat_roof": 0.30,
                "pitched_roof": 0.08,
                "ground": 0.10,
                "paved_ground": 0.07,
            },
        )
        config.roof_type_probabilities = {
            "flat": 0.70,
            "gable": 0.12,
            "hip": 0.08,
            "shed": 0.06,
            "pyramid": 0.03,
            "piecewise_pyramid": 0.01,
        }
        return config

    if terrain_type == "small_town":
        config = UrbanMorphologyConfig(
            terrain_type="small_town",
            area_size=350.0,
            built_density=0.25,
            green_area_fraction=0.25,
            average_building_height=10.0,
            height_lognorm_sigma=0.25,
            min_building_height=3.0,
            max_building_height=24.0,
            footprint_w_range=(8.0, 28.0),
            footprint_d_range=(8.0, 30.0),
            min_clearance=9.0,
            flat_roof_probability=0.35,
            rectangular_footprint_probability=0.76,
            convex_footprint_probability=0.14,
            concave_footprint_probability=0.10,
            roof_height_fraction_range=(0.16, 0.36),
            tree_share_of_green=0.70,
            tree_crown_radius_range=(2.0, 8.0),
            tree_crown_height_range=(3.0, 8.0),
            tree_trunk_height_range=(1.5, 5.0),
            ground_patch_size_range=(8.0, 30.0),
            grid_alignment_prob=0.55,
            rotation_noise_deg=24.0,
            elevation_type="gentle_slope",
            elevation_amplitude=7.0,
            elevation_slope_deg=2.0,
            receiver_weights={
                "wall": 0.45,
                "flat_roof": 0.12,
                "pitched_roof": 0.25,
                "ground": 0.10,
                "paved_ground": 0.08,
            },
        )
        config.roof_type_probabilities = {
            "flat": 0.35,
            "gable": 0.32,
            "hip": 0.20,
            "shed": 0.05,
            "pyramid": 0.06,
            "piecewise_pyramid": 0.02,
        }
        return config

    if terrain_type == "suburban":
        config = UrbanMorphologyConfig(
            terrain_type="suburban",
            area_size=500.0,
            built_density=0.18,
            green_area_fraction=0.38,
            average_building_height=7.5,
            height_lognorm_sigma=0.20,
            min_building_height=3.0,
            max_building_height=18.0,
            footprint_w_range=(8.0, 24.0),
            footprint_d_range=(8.0, 26.0),
            min_clearance=14.0,
            flat_roof_probability=0.20,
            rectangular_footprint_probability=0.70,
            convex_footprint_probability=0.15,
            concave_footprint_probability=0.15,
            roof_height_fraction_range=(0.18, 0.40),
            tree_share_of_green=0.80,
            tree_crown_radius_range=(2.5, 9.5),
            tree_crown_height_range=(3.0, 9.0),
            tree_trunk_height_range=(1.5, 5.0),
            ground_patch_size_range=(10.0, 35.0),
            grid_alignment_prob=0.45,
            rotation_noise_deg=30.0,
            elevation_type="undulating",
            elevation_amplitude=8.0,
            elevation_slope_deg=2.0,
            receiver_weights={
                "wall": 0.38,
                "flat_roof": 0.07,
                "pitched_roof": 0.32,
                "ground": 0.15,
                "paved_ground": 0.08,
            },
        )
        config.roof_type_probabilities = {
            "flat": 0.20,
            "gable": 0.40,
            "hip": 0.25,
            "shed": 0.04,
            "pyramid": 0.08,
            "piecewise_pyramid": 0.03,
        }
        return config

    # Generic fallback.
    config = UrbanMorphologyConfig(terrain_type=terrain_type)
    config.roof_type_probabilities = _default_roof_probs(config.flat_roof_probability)
    return config


# ============================================================
# Elevation
# ============================================================

def terrain_elevation_xy(x, y, config):
    """
    Synthetic terrain elevation.
    """
    t = str(config.elevation_type)

    area = float(config.area_size)
    amp = float(config.elevation_amplitude)

    xn = float(x) / max(area, EPS)
    yn = float(y) / max(area, EPS)

    if t == "flat":
        return 0.0

    if t == "gentle_slope" or t == "steep_slope":
        slope_deg = float(config.elevation_slope_deg)

        if t == "steep_slope":
            slope_deg = max(slope_deg, 8.0)

        aspect = math.radians(float(config.elevation_aspect_deg))
        direction = np.array([math.sin(aspect), math.cos(aspect)], dtype=float)
        rel = np.array([x - 0.5 * area, y - 0.5 * area], dtype=float)
        dist = float(np.dot(rel, direction))

        return math.tan(math.radians(slope_deg)) * dist

    if t == "valley":
        d = abs(yn - 0.5)
        return amp * (2.0 * d) ** 1.5

    if t == "ridge":
        d = abs(yn - 0.5)
        return amp * (1.0 - min(1.0, 2.0 * d))

    if t == "terraced":
        width = max(EPS, float(config.terrace_width))
        step = float(config.terrace_step_height)
        return step * math.floor(y / width)

    if t == "undulating":
        return amp * (
            0.45 * math.sin(2.0 * math.pi * xn)
            + 0.30 * math.sin(2.0 * math.pi * yn + 0.8)
            + 0.25 * math.sin(2.0 * math.pi * (xn + yn))
        )

    return 0.0


# ============================================================
# Random morphology helpers
# ============================================================

def sample_lognormal_clipped(rng, mean, sigma, min_value, max_value):
    value = rng.lognormal(mean=math.log(max(EPS, mean)), sigma=float(sigma))
    value = float(np.clip(value, min_value, max_value))

    return value


def choose_weighted(rng, weights_dict):
    weights_dict = normalize_probability_dict(weights_dict)

    keys = list(weights_dict.keys())
    weights = np.array([weights_dict[k] for k in keys], dtype=float)
    weights = weights / np.sum(weights)

    idx = int(rng.choice(len(keys), p=weights))

    return keys[idx]


def sample_rotation_rad(rng, config):
    """
    Grid-aligned most of the time, with optional rotation noise.
    """
    if rng.random() < config.grid_alignment_prob:
        base_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
        noise = rng.normal(0.0, float(config.rotation_noise_deg))
        return math.radians(base_deg + noise)

    return float(rng.uniform(0.0, 2.0 * math.pi))


def rotate_translate_xy(local_xy, center_xy, rotation_rad):
    p = np.asarray(local_xy, dtype=float)
    c = np.asarray(center_xy, dtype=float)

    ca = math.cos(rotation_rad)
    sa = math.sin(rotation_rad)

    R = np.array(
        [
            [ca, -sa],
            [sa, ca],
        ],
        dtype=float,
    )

    return p @ R.T + c


def footprint_radius_xy(footprint_xy, center_xy=None):
    p = np.asarray(footprint_xy, dtype=float)

    if center_xy is None:
        center_xy = polygon_centroid_xy(p)

    c = np.asarray(center_xy, dtype=float)

    return float(np.max(np.linalg.norm(p - c, axis=1)))


def footprint_inside_area(footprint_xy, area_size, margin=0.0):
    p = np.asarray(footprint_xy, dtype=float)

    return (
        np.min(p[:, 0]) >= margin
        and np.min(p[:, 1]) >= margin
        and np.max(p[:, 0]) <= area_size - margin
        and np.max(p[:, 1]) <= area_size - margin
    )


def make_rectangle_footprint_local(width, depth):
    w = 0.5 * float(width)
    d = 0.5 * float(depth)

    return np.array(
        [
            [-w, -d],
            [w, -d],
            [w, d],
            [-w, d],
        ],
        dtype=float,
    )


def make_convex_footprint_local(width, depth, rng, n_vertices_range):
    """
    Generate a convex polygon inside a width/depth box.

    The polygon is roughly ellipse-like with randomized radii.
    """
    n_min, n_max = n_vertices_range
    n_min = max(5, int(n_min))
    n_max = max(n_min, int(n_max))

    n = int(rng.integers(n_min, n_max + 1))

    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    angles += rng.normal(0.0, 0.10, size=n)
    angles = np.sort(angles)

    rx = 0.5 * float(width)
    ry = 0.5 * float(depth)

    pts = []

    for a in angles:
        radial = rng.uniform(0.75, 1.0)
        x = radial * rx * math.cos(a)
        y = radial * ry * math.sin(a)
        pts.append([x, y])

    pts = np.asarray(pts, dtype=float)
    pts = ensure_ccw_xy(pts)

    # Ensure the noisy polygon is convex. If the randomization made a bad one,
    # fall back to a regular-ish ellipse polygon.
    try:
        if not is_convex_polygon_xy(pts):
            pts = []
            for a in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
                pts.append([rx * math.cos(a), ry * math.sin(a)])
            pts = ensure_ccw_xy(np.asarray(pts, dtype=float))
    except Exception:
        pts = []
        for a in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
            pts.append([rx * math.cos(a), ry * math.sin(a)])
        pts = ensure_ccw_xy(np.asarray(pts, dtype=float))

    return clean_footprint_xy(pts)


def make_l_footprint_local(width, depth, rng):
    """
    Simple concave L-shaped footprint inside width/depth box.
    """
    w = float(width)
    d = float(depth)

    x0 = -0.5 * w
    x1 = 0.5 * w
    y0 = -0.5 * d
    y1 = 0.5 * d

    notch_x = x0 + rng.uniform(0.42, 0.68) * w
    notch_y = y0 + rng.uniform(0.38, 0.68) * d

    # CCW L shape: upper-right corner is cut out.
    pts = np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, notch_y],
            [notch_x, notch_y],
            [notch_x, y1],
            [x0, y1],
        ],
        dtype=float,
    )

    return clean_footprint_xy(pts)


def make_u_footprint_local(width, depth, rng):
    """
    Simple U-shaped concave footprint.
    """
    w = float(width)
    d = float(depth)

    x0 = -0.5 * w
    x1 = 0.5 * w
    y0 = -0.5 * d
    y1 = 0.5 * d

    leg = rng.uniform(0.20, 0.32) * w
    slot_depth = rng.uniform(0.45, 0.72) * d

    xl = x0 + leg
    xr = x1 - leg
    yslot = y1 - slot_depth

    if xr <= xl + EPS or yslot <= y0 + EPS:
        return make_l_footprint_local(width, depth, rng)

    pts = np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [xr, y1],
            [xr, yslot],
            [xl, yslot],
            [xl, y1],
            [x0, y1],
        ],
        dtype=float,
    )

    return clean_footprint_xy(pts)


def make_building_footprint_xy(center_xy, width, depth, rotation_rad, rng, config):
    """
    Generate rectangle / convex / concave building footprint.
    """
    p_rect = float(config.rectangular_footprint_probability)
    p_convex = float(config.convex_footprint_probability)
    p_concave = float(config.concave_footprint_probability)

    total = max(EPS, p_rect + p_convex + p_concave)

    p_rect /= total
    p_convex /= total
    p_concave /= total

    r = rng.random()

    if r < p_rect:
        kind = "rectangle"
        local = make_rectangle_footprint_local(width, depth)

    elif r < p_rect + p_convex:
        kind = "convex"
        local = make_convex_footprint_local(
            width=width,
            depth=depth,
            rng=rng,
            n_vertices_range=config.convex_vertex_range,
        )

    else:
        if rng.random() < 0.75:
            kind = "concave_l"
            local = make_l_footprint_local(width, depth, rng)
        else:
            kind = "concave_u"
            local = make_u_footprint_local(width, depth, rng)

    footprint = rotate_translate_xy(
        local_xy=local,
        center_xy=center_xy,
        rotation_rad=rotation_rad,
    )

    footprint = clean_footprint_xy(footprint)

    return footprint, kind


# ============================================================
# Building surfaces
# ============================================================

def make_wall_surfaces_for_footprint(
    sid_start,
    object_id,
    footprint_xy,
    base_z,
    eave_z,
    roof_type,
):
    """
    Create vertical wall surfaces along all footprint edges.
    """
    footprint = clean_footprint_xy(footprint_xy)
    polygons = []
    sid = int(sid_start)

    wall_sids = []

    for i in range(len(footprint)):
        p0 = footprint[i]
        p1 = footprint[(i + 1) % len(footprint)]

        bottom0 = np.array([p0[0], p0[1], base_z], dtype=float)
        bottom1 = np.array([p1[0], p1[1], base_z], dtype=float)
        top1 = np.array([p1[0], p1[1], eave_z], dtype=float)
        top0 = np.array([p0[0], p0[1], eave_z], dtype=float)

        verts = np.array([bottom0, bottom1, top1, top0], dtype=float)

        edge = p1 - p0
        desired_normal = unit(np.array([edge[1], -edge[0], 0.0], dtype=float))

        poly = make_polygon(
            sid=sid,
            object_id=object_id,
            object_kind="building",
            surface_kind="wall",
            vertices=verts,
            desired_normal=desired_normal,
            can_receive_shadow=True,
            can_cast_shadow=True,
            opacity=1.0,
            footprint_xy=footprint,
            roof_type=roof_type,
            base_z=base_z,
            eave_z=eave_z,
            top_z=eave_z,
            metadata={
                "role": "physical_wall",
                "edge_index": int(i),
            },
        )

        polygons.append(poly)
        wall_sids.append(sid)
        sid += 1

    return polygons, sid, wall_sids


def make_roof_polygons_from_profile(
    sid_start,
    object_id,
    footprint_xy,
    base_z,
    eave_z,
    roof_profile,
):
    polygons = []
    sid = int(sid_start)
    roof_sids = []

    for surface_index, spec in enumerate(roof_profile.surfaces):
        poly = make_polygon(
            sid=sid,
            object_id=object_id,
            object_kind="building",
            surface_kind=spec.surface_kind,
            vertices=spec.vertices,
            desired_normal=spec.normal,
            can_receive_shadow=spec.can_receive_shadow,
            can_cast_shadow=spec.can_cast_shadow,
            opacity=spec.opacity,
            footprint_xy=footprint_xy,
            roof_type=roof_profile.roof_type,
            base_z=base_z,
            eave_z=eave_z,
            top_z=roof_profile.max_z,
            metadata={
                "role": "physical_roof",
                "roof_type": roof_profile.roof_type,
                "roof_source": spec.source,
                "roof_piece_index": int(spec.piece_index),
                "roof_surface_index": int(surface_index),
            },
        )

        polygons.append(poly)
        roof_sids.append(sid)
        sid += 1

    return polygons, sid, roof_sids


def choose_roof_type(rng, config):
    if config.roof_type_probabilities is None:
        config.roof_type_probabilities = _default_roof_probs(config.flat_roof_probability)

    return choose_weighted(rng, config.roof_type_probabilities)


def make_building_polygons(
    sid_start,
    object_id,
    center_xy,
    width,
    depth,
    height,
    rotation_rad,
    rng,
    config,
):
    """
    Create full physical building polygons:
        - arbitrary footprint walls
        - roof surfaces from roof_profiles.py

    The later shading preprocessing will replace wall casters with barrier
    proxy surfaces, but these physical surfaces remain as receivers and as
    raw-mode casters.
    """
    center_xy = np.asarray(center_xy, dtype=float)

    footprint_xy, footprint_kind = make_building_footprint_xy(
        center_xy=center_xy,
        width=width,
        depth=depth,
        rotation_rad=rotation_rad,
        rng=rng,
        config=config,
    )

    cxy = polygon_centroid_xy(footprint_xy)
    base_z = terrain_elevation_xy(cxy[0], cxy[1], config)
    eave_z = base_z + float(height)

    roof_type = choose_roof_type(rng, config)

    if roof_type == "flat":
        roof_height = 0.0
    else:
        min_dim = max(EPS, min(float(width), float(depth)))
        f0, f1 = config.roof_height_fraction_range
        roof_height = float(rng.uniform(f0, f1) * min_dim)

    ridge_azimuth_deg = math.degrees(rotation_rad)

    roof_profile = build_roof_profile_for_footprint(
        footprint_xy=footprint_xy,
        eave_z=eave_z,
        roof_type=roof_type,
        roof_height=roof_height,
        object_id=object_id,
        ridge_azimuth_deg=ridge_azimuth_deg,
        slope_azimuth_deg=ridge_azimuth_deg,
        can_cast_shadow=True,
        can_receive_shadow=True,
        opacity=1.0,
    )

    polygons = []
    sid = int(sid_start)

    wall_polys, sid, wall_sids = make_wall_surfaces_for_footprint(
        sid_start=sid,
        object_id=object_id,
        footprint_xy=footprint_xy,
        base_z=base_z,
        eave_z=eave_z,
        roof_type=roof_type,
    )
    polygons.extend(wall_polys)

    roof_polys, sid, roof_sids = make_roof_polygons_from_profile(
        sid_start=sid,
        object_id=object_id,
        footprint_xy=footprint_xy,
        base_z=base_z,
        eave_z=eave_z,
        roof_profile=roof_profile,
    )
    polygons.extend(roof_polys)

    record = BuildingRecord(
        object_id=int(object_id),
        footprint_xy=footprint_xy,
        center_xy=cxy,
        base_z=float(base_z),
        eave_z=float(eave_z),
        top_z=float(roof_profile.max_z),
        height=float(height),
        roof_type=str(roof_type),
        roof_height=float(roof_height),
        rotation_rad=float(rotation_rad),
        footprint_kind=str(footprint_kind),
        wall_sids=wall_sids,
        roof_sids=roof_sids,
        footprint_area=float(polygon_area_xy(footprint_xy)),
        radius_xy=float(footprint_radius_xy(footprint_xy, cxy)),
    )

    return polygons, sid, record


# ============================================================
# Trees and ground
# ============================================================

def regular_ngon_xy(center_xy, radius, n=8, rotation=0.0):
    center = np.asarray(center_xy, dtype=float)
    pts = []

    for i in range(n):
        a = rotation + 2.0 * math.pi * float(i) / float(n)
        pts.append([
            center[0] + radius * math.cos(a),
            center[1] + radius * math.sin(a),
        ])

    return ensure_ccw_xy(np.asarray(pts, dtype=float))


def make_tree_polygons(
    sid_start,
    object_id,
    center_xy,
    ground_z,
    crown_radius,
    trunk_height,
    crown_height,
    rng,
):
    """
    Simple tree proxy:
        - one horizontal crown top
        - four vertical crown billboard/crown side quads
        - one trunk vertical quad proxy

    Trees cast shadow but are not receivers.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])

    sid = int(sid_start)
    polygons = []

    trunk_top_z = float(ground_z + trunk_height)
    crown_top_z = float(trunk_top_z + crown_height)

    opacity_crown = 0.55

    # Crown top.
    top_xy = regular_ngon_xy(
        center_xy=np.array([cx, cy], dtype=float),
        radius=float(crown_radius),
        n=8,
        rotation=float(rng.uniform(0.0, 2.0 * math.pi)),
    )

    top_verts = np.zeros((len(top_xy), 3), dtype=float)
    top_verts[:, :2] = top_xy
    top_verts[:, 2] = crown_top_z

    polygons.append(
        make_polygon(
            sid=sid,
            object_id=object_id,
            object_kind="tree",
            surface_kind="tree_crown_top",
            vertices=top_verts,
            desired_normal=np.array([0.0, 0.0, 1.0]),
            can_receive_shadow=False,
            can_cast_shadow=True,
            opacity=opacity_crown,
            base_z=ground_z,
            eave_z=trunk_top_z,
            top_z=crown_top_z,
            metadata={"role": "tree_crown_top"},
        )
    )
    sid += 1

    # Four vertical crown proxy planes.
    for k in range(4):
        angle = 0.25 * math.pi * float(k) + float(rng.uniform(-0.05, 0.05))
        dx = math.cos(angle)
        dy = math.sin(angle)

        p0 = np.array([cx - crown_radius * dx, cy - crown_radius * dy], dtype=float)
        p1 = np.array([cx + crown_radius * dx, cy + crown_radius * dy], dtype=float)

        z0 = trunk_top_z
        z1 = crown_top_z

        verts = np.array(
            [
                [p0[0], p0[1], z0],
                [p1[0], p1[1], z0],
                [p1[0], p1[1], z1],
                [p0[0], p0[1], z1],
            ],
            dtype=float,
        )

        normal = unit(np.array([dy, -dx, 0.0], dtype=float))

        polygons.append(
            make_polygon(
                sid=sid,
                object_id=object_id,
                object_kind="tree",
                surface_kind="tree_crown",
                vertices=verts,
                desired_normal=normal,
                can_receive_shadow=False,
                can_cast_shadow=True,
                opacity=opacity_crown,
                base_z=ground_z,
                eave_z=trunk_top_z,
                top_z=crown_top_z,
                metadata={"role": "tree_crown_billboard", "billboard_index": k},
            )
        )
        sid += 1

    # Trunk proxy.
    trunk_radius = max(0.08 * crown_radius, 0.15)
    angle = float(rng.uniform(0.0, math.pi))
    dx = math.cos(angle)
    dy = math.sin(angle)

    p0 = np.array([cx - trunk_radius * dx, cy - trunk_radius * dy], dtype=float)
    p1 = np.array([cx + trunk_radius * dx, cy + trunk_radius * dy], dtype=float)

    verts = np.array(
        [
            [p0[0], p0[1], ground_z],
            [p1[0], p1[1], ground_z],
            [p1[0], p1[1], trunk_top_z],
            [p0[0], p0[1], trunk_top_z],
        ],
        dtype=float,
    )

    normal = unit(np.array([dy, -dx, 0.0], dtype=float))

    polygons.append(
        make_polygon(
            sid=sid,
            object_id=object_id,
            object_kind="tree",
            surface_kind="tree_trunk",
            vertices=verts,
            desired_normal=normal,
            can_receive_shadow=False,
            can_cast_shadow=True,
            opacity=1.0,
            base_z=ground_z,
            eave_z=trunk_top_z,
            top_z=trunk_top_z,
            metadata={"role": "tree_trunk"},
        )
    )
    sid += 1

    return polygons, sid


def make_ground_patch_polygon(
    sid,
    object_id,
    center_xy,
    size,
    rotation_rad,
    config,
    surface_kind="ground",
):
    local = make_rectangle_footprint_local(size, size)
    footprint = rotate_translate_xy(local, center_xy, rotation_rad)

    z_values = np.array(
        [
            terrain_elevation_xy(p[0], p[1], config)
            for p in footprint
        ],
        dtype=float,
    )

    # Keep ground patches planar enough for the current polygon system:
    # use mean z instead of terrain-corner z.
    z = float(np.mean(z_values))

    verts = np.zeros((len(footprint), 3), dtype=float)
    verts[:, :2] = footprint
    verts[:, 2] = z

    return make_polygon(
        sid=sid,
        object_id=object_id,
        object_kind="ground",
        surface_kind=surface_kind,
        vertices=verts,
        desired_normal=np.array([0.0, 0.0, 1.0]),
        can_receive_shadow=True,
        can_cast_shadow=False,
        opacity=1.0,
        base_z=z,
        eave_z=z,
        top_z=z,
        metadata={"role": "ground_receiver"},
    )


# ============================================================
# Placement helpers
# ============================================================

def point_clear_of_buildings(
    center_xy,
    radius,
    placed_buildings,
    extra_clearance=0.0,
):
    for other_center_xy, other_radius in placed_buildings:
        dist = np.linalg.norm(center_xy - other_center_xy)

        if dist < radius + other_radius + extra_clearance:
            return False

    return True


def count_castable(polygons):
    return sum(1 for p in polygons if p.can_cast_shadow)


def count_receivable(polygons):
    return sum(1 for p in polygons if p.can_receive_shadow)


def random_point_in_area(rng, area_size, margin=0.0):
    return np.array(
        [
            rng.uniform(margin, area_size - margin),
            rng.uniform(margin, area_size - margin),
        ],
        dtype=float,
    )


# ============================================================
# Main configurable generator
# ============================================================

def generate_urban_receiver_and_casters(
    n_polygons,
    terrain_type="small_town",
    seed=None,
    config=None,
    built_density=None,
    green_area_fraction=None,
    average_building_height=None,
    height_lognorm_sigma=None,
    elevation_type=None,
    receiver_surface_kinds=None,
    include_same_building_casters=True,
    verbose=False,
):
    """
    Generate one receiver polygon and n_polygons nearest caster polygons.

    Backward-compatible with the old scripts.

    New behavior:
        - buildings may have rectangular, convex, or concave footprints
        - buildings may have flat/gable/hip/shed/pyramid roofs
        - building metadata stores the source footprint and roof type
        - physical surfaces remain available for raw-mode shading
        - later preprocessing can build barrier-proxy caster profiles from
          BuildingRecord.footprint_xy and roof polygons
    """
    if n_polygons <= 0:
        raise ValueError("n_polygons must be positive.")

    if config is None:
        config = default_morphology_config(terrain_type)

    # --------------------------------------------------------
    # Apply explicit overrides.
    # --------------------------------------------------------
    if built_density is not None:
        config.built_density = float(built_density)

    if green_area_fraction is not None:
        config.green_area_fraction = float(green_area_fraction)

    if average_building_height is not None:
        config.average_building_height = float(average_building_height)

    if height_lognorm_sigma is not None:
        config.height_lognorm_sigma = float(height_lognorm_sigma)

    if elevation_type is not None:
        config.elevation_type = str(elevation_type)

    if config.roof_type_probabilities is None:
        config.roof_type_probabilities = _default_roof_probs(config.flat_roof_probability)

    config.roof_type_probabilities = normalize_probability_dict(config.roof_type_probabilities)

    if not (0.0 <= config.built_density <= 0.95):
        raise ValueError("built_density should be between 0 and 0.95.")

    if not (0.0 <= config.green_area_fraction <= 0.95):
        raise ValueError("green_area_fraction should be between 0 and 0.95.")

    if receiver_surface_kinds is None:
        receiver_surface_kinds = (
            "wall",
            "flat_roof",
            "pitched_roof",
            "ground",
            "paved_ground",
        )

    rng = np.random.default_rng(seed)

    area_size = float(config.area_size)
    total_land_area = area_size * area_size

    all_polygons = []
    buildings = []

    sid = 1
    object_id = 1

    # ========================================================
    # Buildings
    # ========================================================

    target_built_area = config.built_density * total_land_area
    built_area = 0.0

    placed_buildings = []

    max_attempts = max(500, int(80 * max(1, n_polygons)))
    attempts = 0

    # We keep generating while target area is not reached.
    # Also try to make enough caster polygons for old n_polygons-based scripts.
    while attempts < max_attempts:
        attempts += 1

        enough_area = built_area >= target_built_area
        enough_casters = count_castable(all_polygons) >= n_polygons + 20

        if enough_area and enough_casters:
            break

        width = rng.uniform(
            float(config.footprint_w_range[0]),
            float(config.footprint_w_range[1]),
        )
        depth = rng.uniform(
            float(config.footprint_d_range[0]),
            float(config.footprint_d_range[1]),
        )

        height = sample_lognormal_clipped(
            rng=rng,
            mean=config.average_building_height,
            sigma=config.height_lognorm_sigma,
            min_value=config.min_building_height,
            max_value=config.max_building_height,
        )

        rotation_rad = sample_rotation_rad(rng, config)

        rough_radius = 0.5 * math.sqrt(width * width + depth * depth)
        margin = rough_radius + 2.0

        if margin >= 0.45 * area_size:
            continue

        center_xy = random_point_in_area(rng, area_size, margin=margin)

        try:
            footprint_xy, footprint_kind = make_building_footprint_xy(
                center_xy=center_xy,
                width=width,
                depth=depth,
                rotation_rad=rotation_rad,
                rng=rng,
                config=config,
            )
        except Exception:
            continue

        if not footprint_inside_area(footprint_xy, area_size, margin=0.0):
            continue

        radius = footprint_radius_xy(footprint_xy, polygon_centroid_xy(footprint_xy))

        if not point_clear_of_buildings(
            center_xy=polygon_centroid_xy(footprint_xy),
            radius=radius,
            placed_buildings=placed_buildings,
            extra_clearance=float(config.min_clearance),
        ):
            continue

        # Reuse the exact footprint from this validation by constructing the
        # building through the normal function. To avoid sampling a different
        # footprint inside that function, we temporarily build directly here.
        cxy = polygon_centroid_xy(footprint_xy)
        base_z = terrain_elevation_xy(cxy[0], cxy[1], config)
        eave_z = base_z + float(height)

        roof_type = choose_roof_type(rng, config)

        if roof_type == "flat":
            roof_height = 0.0
        else:
            min_dim = max(EPS, min(width, depth))
            f0, f1 = config.roof_height_fraction_range
            roof_height = float(rng.uniform(f0, f1) * min_dim)

        ridge_azimuth_deg = math.degrees(rotation_rad)

        try:
            roof_profile = build_roof_profile_for_footprint(
                footprint_xy=footprint_xy,
                eave_z=eave_z,
                roof_type=roof_type,
                roof_height=roof_height,
                object_id=object_id,
                ridge_azimuth_deg=ridge_azimuth_deg,
                slope_azimuth_deg=ridge_azimuth_deg,
                can_cast_shadow=True,
                can_receive_shadow=True,
                opacity=1.0,
            )

            building_polygons = []

            wall_polys, sid_new, wall_sids = make_wall_surfaces_for_footprint(
                sid_start=sid,
                object_id=object_id,
                footprint_xy=footprint_xy,
                base_z=base_z,
                eave_z=eave_z,
                roof_type=roof_type,
            )
            building_polygons.extend(wall_polys)

            roof_polys, sid_new, roof_sids = make_roof_polygons_from_profile(
                sid_start=sid_new,
                object_id=object_id,
                footprint_xy=footprint_xy,
                base_z=base_z,
                eave_z=eave_z,
                roof_profile=roof_profile,
            )
            building_polygons.extend(roof_polys)

            sid = sid_new

        except Exception:
            continue

        all_polygons.extend(building_polygons)

        footprint_area = polygon_area_xy(footprint_xy)
        built_area += footprint_area

        placed_buildings.append((cxy, radius))

        buildings.append(
            BuildingRecord(
                object_id=int(object_id),
                footprint_xy=footprint_xy,
                center_xy=cxy,
                base_z=float(base_z),
                eave_z=float(eave_z),
                top_z=float(roof_profile.max_z),
                height=float(height),
                roof_type=str(roof_type),
                roof_height=float(roof_height),
                rotation_rad=float(rotation_rad),
                footprint_kind=str(footprint_kind),
                wall_sids=wall_sids,
                roof_sids=roof_sids,
                footprint_area=float(footprint_area),
                radius_xy=float(radius),
            )
        )

        object_id += 1

    n_buildings = len(buildings)

    if n_buildings == 0:
        raise RuntimeError("No buildings could be generated. Check morphology settings.")

    # ========================================================
    # Ground / paved ground patches
    # ========================================================

    green_area_target = config.green_area_fraction * total_land_area
    green_area = 0.0

    # Add a set of ground receiver patches, some green and some paved.
    n_ground_target = max(8, int(0.00035 * total_land_area))

    for _ in range(n_ground_target):
        size = rng.uniform(
            float(config.ground_patch_size_range[0]),
            float(config.ground_patch_size_range[1]),
        )

        margin = 0.5 * math.sqrt(2.0) * size + 1.0

        if margin >= 0.45 * area_size:
            continue

        center_xy = random_point_in_area(rng, area_size, margin=margin)
        rotation_rad = rng.uniform(0.0, 2.0 * math.pi)

        surface_kind = "ground" if rng.random() < 0.55 else "paved_ground"

        try:
            gp = make_ground_patch_polygon(
                sid=sid,
                object_id=object_id,
                center_xy=center_xy,
                size=size,
                rotation_rad=rotation_rad,
                config=config,
                surface_kind=surface_kind,
            )
        except Exception:
            continue

        all_polygons.append(gp)
        sid += 1
        object_id += 1

        if surface_kind == "ground":
            green_area += size * size

    # ========================================================
    # Trees
    # ========================================================

    tree_area_target = max(0.0, green_area_target - green_area) * float(config.tree_share_of_green)
    tree_area = 0.0
    n_trees = 0

    max_tree_attempts = max(200, int(0.002 * total_land_area))

    for _ in range(max_tree_attempts):
        if tree_area >= tree_area_target:
            break

        crown_radius = rng.uniform(
            float(config.tree_crown_radius_range[0]),
            float(config.tree_crown_radius_range[1]),
        )

        margin = crown_radius + 1.0

        if margin >= 0.45 * area_size:
            continue

        center_xy = random_point_in_area(rng, area_size, margin=margin)

        # Keep trees away from buildings roughly.
        if not point_clear_of_buildings(
            center_xy=center_xy,
            radius=crown_radius,
            placed_buildings=placed_buildings,
            extra_clearance=1.0,
        ):
            continue

        ground_z = terrain_elevation_xy(center_xy[0], center_xy[1], config)

        trunk_height = rng.uniform(
            float(config.tree_trunk_height_range[0]),
            float(config.tree_trunk_height_range[1]),
        )
        crown_height = rng.uniform(
            float(config.tree_crown_height_range[0]),
            float(config.tree_crown_height_range[1]),
        )

        try:
            tree_polys, sid = make_tree_polygons(
                sid_start=sid,
                object_id=object_id,
                center_xy=center_xy,
                ground_z=ground_z,
                crown_radius=crown_radius,
                trunk_height=trunk_height,
                crown_height=crown_height,
                rng=rng,
            )
        except Exception:
            continue

        all_polygons.extend(tree_polys)

        tree_area += math.pi * crown_radius * crown_radius
        n_trees += 1
        object_id += 1

    actual_built_density = built_area / total_land_area
    actual_green_area_fraction = (green_area + tree_area) / total_land_area

    # ========================================================
    # Choose receiver.
    # ========================================================

    receiver_candidates = [
        p for p in all_polygons
        if p.can_receive_shadow
        and p.surface_kind in receiver_surface_kinds
    ]

    if not receiver_candidates:
        raise RuntimeError(
            "No receiver candidates found. Check receiver_surface_kinds."
        )

    if config.receiver_weights is None:
        receiver = receiver_candidates[int(rng.integers(0, len(receiver_candidates)))]
    else:
        weights = []

        for p in receiver_candidates:
            weights.append(float(config.receiver_weights.get(p.surface_kind, 0.0)))

        weights = np.asarray(weights, dtype=float)

        if np.sum(weights) <= EPS:
            weights = np.ones(len(receiver_candidates), dtype=float)

        weights = weights / np.sum(weights)

        receiver = receiver_candidates[int(rng.choice(len(receiver_candidates), p=weights))]

    # ========================================================
    # Choose nearest caster polygons.
    # ========================================================

    caster_pool = []

    for p in all_polygons:
        if p.sid == receiver.sid:
            continue

        if not p.can_cast_shadow:
            continue

        if not include_same_building_casters:
            if p.object_kind == "building" and p.object_id == receiver.object_id:
                continue

        caster_pool.append(p)

    caster_pool.sort(key=lambda p: float(np.linalg.norm(p.center - receiver.center)))

    if len(caster_pool) < n_polygons:
        raise RuntimeError(
            "Only {} caster polygons available, but {} requested. "
            "Try reducing n_polygons, increasing built_density, reducing clearance, "
            "or increasing area_size.".format(len(caster_pool), n_polygons)
        )

    casters = caster_pool[:int(n_polygons)]

    scene = UrbanScene(
        terrain_type=config.terrain_type,
        receiver=receiver,
        casters=casters,
        all_polygons=all_polygons,
        area_size=area_size,
        config=config,
        actual_built_density=actual_built_density,
        actual_green_area_fraction=actual_green_area_fraction,
        n_buildings=n_buildings,
        n_trees=n_trees,
        buildings=buildings,
    )

    if verbose:
        print()
        print("=" * 72)
        print("Generated urban scene")
        print("-" * 72)
        print("terrain_type:", scene.terrain_type)
        print("area_size:", scene.area_size)
        print("target built_density:", config.built_density)
        print("actual built_density:", round(scene.actual_built_density, 4))
        print("target green_area_fraction:", config.green_area_fraction)
        print("actual green_area_fraction:", round(scene.actual_green_area_fraction, 4))
        print("average_building_height target:", config.average_building_height)
        print("height_lognorm_sigma:", config.height_lognorm_sigma)
        print("elevation_type:", config.elevation_type)
        print("n_buildings:", scene.n_buildings)
        print("n_trees:", scene.n_trees)
        print("n_all_polygons:", len(scene.all_polygons))
        print("n_casters:", len(scene.casters))
        print("receiver sid:", scene.receiver.sid)
        print("receiver surface_kind:", scene.receiver.surface_kind)
        print("receiver object_kind:", scene.receiver.object_kind)

        footprint_counts = {}
        roof_counts = {}
        surface_counts = {}

        for b in scene.buildings:
            footprint_counts[b.footprint_kind] = footprint_counts.get(b.footprint_kind, 0) + 1
            roof_counts[b.roof_type] = roof_counts.get(b.roof_type, 0) + 1

        for p in scene.all_polygons:
            surface_counts[p.surface_kind] = surface_counts.get(p.surface_kind, 0) + 1

        print("footprint kinds:", footprint_counts)
        print("roof types:", roof_counts)
        print("surface kinds:", surface_counts)
        print("=" * 72)
        print()

    return scene


# ============================================================
# Convenience function for generating many scenes
# ============================================================

def generate_many_urban_scenes(
    n_scenes,
    n_polygons,
    terrain_type="small_town",
    seed=None,
    config=None,
    **kwargs
):
    """
    Generate multiple independent scenes.

    Extra keyword arguments are forwarded to generate_urban_receiver_and_casters.
    """
    rng = np.random.default_rng(seed)

    scenes = []

    for _ in range(int(n_scenes)):
        scene_seed = int(rng.integers(0, 2**31 - 1))

        scene = generate_urban_receiver_and_casters(
            n_polygons=n_polygons,
            terrain_type=terrain_type,
            seed=scene_seed,
            config=config,
            **kwargs
        )

        scenes.append(scene)

    return scenes


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    scene = generate_urban_receiver_and_casters(
        n_polygons=80,
        terrain_type="small_town",
        seed=42,
        built_density=0.30,
        green_area_fraction=0.25,
        average_building_height=12.0,
        height_lognorm_sigma=0.30,
        elevation_type="undulating",
        receiver_surface_kinds=(
            "wall",
            "flat_roof",
            "pitched_roof",
            "ground",
            "paved_ground",
        ),
        include_same_building_casters=True,
        verbose=True,
    )

    print("Receiver:", scene.receiver.sid, scene.receiver.surface_kind)
    print("Casters:", len(scene.casters))
    print("Buildings:", len(scene.buildings))

    print()
    print("First 5 buildings:")
    for b in scene.buildings[:5]:
        print(
            "object_id",
            b.object_id,
            "| footprint",
            b.footprint_kind,
            "| roof",
            b.roof_type,
            "| vertices",
            len(b.footprint_xy),
            "| walls",
            len(b.wall_sids),
            "| roofs",
            len(b.roof_sids),
        )