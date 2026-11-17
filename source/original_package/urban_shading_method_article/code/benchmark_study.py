#!/usr/bin/env python3
"""
Restartable validation study for the compact urban-shading lobe method.

The script intentionally keeps the original scene generator and barrier-proxy
implementation separate from the study code.  The exact reference projects
each caster polygon onto the receiver plane and clips it with the ordinary
Sutherland-Hodgman algorithm recovered from profiler.py.

Every scene/receiver phase is checkpointed.  A stopped run can therefore be
continued by launching this file again with the same output directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


EPS = 1.0e-10
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path(
    os.environ.get(
        "URBAN_SHADING_SOURCE_DIR",
        str(
            SCRIPT_DIR / "recovered_sources"
            if (SCRIPT_DIR / "recovered_sources").exists()
            else SCRIPT_DIR / "original"
        ),
    )
).resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "study_output"

TERRAINS = (
    "tall_buildings",
    "horizontal_city",
    "small_town",
    "suburban",
)

CALIBRATION_SEEDS = {
    "tall_buildings": 101,
    "horizontal_city": 103,
    "small_town": 107,
    "suburban": 109,
}

VALIDATION_SEEDS = {
    "tall_buildings": (211, 307),
    "horizontal_city": (223, 311),
    "small_town": (227, 313),
    "suburban": (229, 317),
}

RADIUS_LEVELS_M = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 500.0)
TOP_K_LEVELS = (16, 32, 48, 64, 80, 112, 160, 224, 320)
KERNELS = (
    "gaussian",
    "supergaussian",
    "supergaussian6",
    "supergaussian8",
    "logistic",
    "biweight",
    "cosine",
    "wendland",
)
GAMMA_LEVELS = (
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
)

RADIUS_MAE_LIMIT = 0.005
RADIUS_P95_LIMIT = 0.020
TOPK_MAE_LIMIT = 0.005
TOPK_P95_LIMIT = 0.020

FIT_GRID_N = 5
MIN_ACTIVE_SHADE = 0.01
FIT_GLOBAL_AZIMUTHS_DEG = tuple(np.arange(11.25, 360.0, 22.5))
FIT_GLOBAL_ELEVATIONS_DEG = (7.0, 17.0, 27.0, 38.0, 52.0, 67.0, 82.0)


def _load_as(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_original_modules(source_dir: Path):
    source_dir = Path(source_dir).resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    # Accept both the original Library names and the normalised package names.
    generator_path = source_dir / "generator.py"
    if not generator_path.exists():
        generator_path = source_dir / "generator(1).py"
    if "generator" not in sys.modules:
        _load_as("generator", generator_path)

    import generator
    import shading_profiles
    import profiler
    import caster_lobe_vector_and_raster

    return generator, shading_profiles, profiler, caster_lobe_vector_and_raster


GENERATOR, SHADING_PROFILES, PROFILER, LOBE_ORIGINAL = load_original_modules(SOURCE_DIR)


@dataclass
class ReceiverCache:
    receiver: object
    origin: np.ndarray
    normal: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    polygon_2d: np.ndarray
    area: float
    normal_azimuth_deg: float


@dataclass
class SceneBundle:
    split: str
    terrain: str
    seed: int
    scene: object
    raw_profile: object
    proxy_profile: object
    receivers: Dict[str, object]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {}".format(type(value)))


def atomic_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        stream.write("\n")
    os.replace(str(temporary), str(path))


def atomic_pickle(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(temporary), str(path))


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def atomic_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if fieldnames is None:
        names = set()
        for row in rows:
            names.update(row.keys())
        fieldnames = sorted(names)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(temporary), str(path))


def wrap_180(angle_deg):
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def wrap_360(angle_deg):
    return np.asarray(angle_deg, dtype=float) % 360.0


def horizontal_azimuth(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float)
    horizontal = math.hypot(float(vector[0]), float(vector[1]))
    if horizontal < EPS:
        return 0.0
    return float(math.degrees(math.atan2(float(vector[0]), float(vector[1]))) % 360.0)


def vector_azimuth_elevation(vector: np.ndarray) -> Tuple[float, float]:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return 0.0, 0.0
    azimuth = horizontal_azimuth(vector)
    value = max(-1.0, min(1.0, float(vector[2]) / norm))
    elevation = math.degrees(math.asin(value))
    return azimuth, elevation


def sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    horizontal = math.cos(elevation)
    return np.array(
        [
            horizontal * math.sin(azimuth),
            horizontal * math.cos(azimuth),
            math.sin(elevation),
        ],
        dtype=float,
    )


def receiver_cache(receiver) -> ReceiverCache:
    normal = PROFILER.unit(receiver.normal)
    basis_u, basis_v = PROFILER.make_plane_basis(normal)
    polygon_2d = PROFILER.points_3d_to_plane_2d(
        receiver.vertices,
        receiver.center,
        basis_u,
        basis_v,
    )
    polygon_2d = PROFILER.ensure_ccw(polygon_2d)
    area = abs(float(PROFILER.polygon_area_2d(polygon_2d)))
    return ReceiverCache(
        receiver=receiver,
        origin=np.asarray(receiver.center, dtype=float),
        normal=np.asarray(normal, dtype=float),
        basis_u=np.asarray(basis_u, dtype=float),
        basis_v=np.asarray(basis_v, dtype=float),
        polygon_2d=np.asarray(polygon_2d, dtype=float),
        area=float(area),
        normal_azimuth_deg=horizontal_azimuth(normal),
    )


def horizontal_distance(receiver, caster) -> float:
    delta = np.asarray(caster.center, dtype=float) - np.asarray(receiver.center, dtype=float)
    return float(math.hypot(float(delta[0]), float(delta[1])))


def building_casters(profile, receiver) -> List[object]:
    casters = []
    for caster in profile.casters:
        if caster.sid == receiver.sid:
            continue
        if caster.object_kind != "building":
            continue
        if caster.object_id == receiver.object_id:
            continue
        if not caster.can_cast_shadow:
            continue
        casters.append(caster)
    return casters


def clip_caster_to_receiver(
    cache: ReceiverCache,
    caster,
    direction: np.ndarray,
) -> Optional[ShapelyPolygon]:
    relative = np.asarray(caster.center, dtype=float) - cache.origin
    if float(np.dot(relative, direction)) <= EPS:
        return None

    matrix = PROFILER.projection_matrix_to_plane(
        receiver_point=cache.origin,
        receiver_normal=cache.normal,
        ray_direction=-np.asarray(direction, dtype=float),
    )
    if matrix is None:
        return None

    projected_3d = PROFILER.project_points(caster.vertices, matrix)
    projected_2d = PROFILER.points_3d_to_plane_2d(
        projected_3d,
        cache.origin,
        cache.basis_u,
        cache.basis_v,
    )
    if len(projected_2d) < 3:
        return None

    projected_2d = PROFILER.ensure_ccw(projected_2d)
    clipped = PROFILER.sutherland_hodgman_clip(
        subject_polygon=projected_2d,
        clip_polygon=cache.polygon_2d,
    )
    if len(clipped) < 3:
        return None

    area = abs(float(PROFILER.polygon_area_2d(clipped)))
    if area <= EPS:
        return None

    polygon = ShapelyPolygon(clipped)
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0.0)
    if polygon.is_empty or float(polygon.area) <= EPS:
        return None
    return polygon


def direction_parts(
    cache: ReceiverCache,
    casters: Sequence[object],
    azimuth_deg: float,
    elevation_deg: float,
) -> Tuple[List[Tuple[int, float, ShapelyPolygon]], float]:
    direction = sun_vector(azimuth_deg, elevation_deg)
    if float(np.dot(cache.normal, direction)) <= EPS:
        return [], 0.0

    started = time.perf_counter()
    parts = []
    for caster in casters:
        polygon = clip_caster_to_receiver(cache, caster, direction)
        if polygon is None:
            continue
        parts.append(
            (
                int(caster.sid),
                horizontal_distance(cache.receiver, caster),
                polygon,
            )
        )
    elapsed = time.perf_counter() - started
    return parts, elapsed


def union_fraction(parts: Iterable[ShapelyPolygon], receiver_area: float) -> float:
    polygons = list(parts)
    if not polygons or receiver_area <= EPS:
        return 0.0
    union = unary_union(polygons)
    fraction = float(union.area) / float(receiver_area)
    return max(0.0, min(1.0, fraction))


def single_caster_fraction(
    cache: ReceiverCache,
    caster,
    azimuth_deg: float,
    elevation_deg: float,
) -> float:
    direction = sun_vector(azimuth_deg, elevation_deg)
    if float(np.dot(cache.normal, direction)) <= EPS:
        return 0.0
    polygon = clip_caster_to_receiver(cache, caster, direction)
    if polygon is None or cache.area <= EPS:
        return 0.0
    return max(0.0, min(1.0, float(polygon.area) / cache.area))


def make_direction_grid(cache: ReceiverCache, fine: bool = False) -> List[Tuple[float, float]]:
    if fine:
        azimuths = np.arange(0.0, 360.0, 10.0)
        elevations = np.arange(5.0, 86.0, 5.0)
    else:
        azimuths = np.arange(0.0, 360.0, 30.0)
        elevations = np.asarray((10.0, 20.0, 30.0, 45.0, 60.0, 75.0))

    directions = []
    for elevation in elevations:
        for azimuth in azimuths:
            direction = sun_vector(float(azimuth), float(elevation))
            if float(np.dot(cache.normal, direction)) > 0.02:
                directions.append((float(azimuth), float(elevation)))
    return directions


def contextual_receiver_score(receiver, casters: Sequence[object], area_size: float) -> float:
    center = np.asarray(receiver.center, dtype=float)
    normal = np.asarray(receiver.normal, dtype=float)
    score = 0.0
    for caster in casters:
        if caster.object_id == receiver.object_id:
            continue
        delta = np.asarray(caster.center, dtype=float) - center
        distance_xy = math.hypot(float(delta[0]), float(delta[1]))
        if distance_xy > 175.0 or distance_xy < EPS:
            continue
        norm = float(np.linalg.norm(delta))
        if norm < EPS:
            continue
        front = max(0.0, float(np.dot(normal, delta / norm)))
        if np.linalg.norm(normal[:2]) < 0.05:
            front = max(front, 0.25)
        score += float(caster.area) * (0.15 + front) / (25.0 + distance_xy * distance_xy)

    city_center = np.array([0.5 * area_size, 0.5 * area_size], dtype=float)
    radial = float(np.linalg.norm(center[:2] - city_center))
    centrality = max(0.25, 1.0 - radial / max(EPS, 0.72 * area_size))
    return score * centrality


def select_receivers(proxy_profile, scene) -> Dict[str, object]:
    receivers = list(proxy_profile.receivers)
    casters = [p for p in proxy_profile.casters if p.object_kind == "building"]

    groups = {
        "wall": [p for p in receivers if p.surface_kind == "wall"],
        "roof": [
            p
            for p in receivers
            if p.surface_kind in ("flat_roof", "pitched_roof")
        ],
    }

    selected = {}
    used_sids = set()
    for label in ("wall", "roof"):
        candidates = groups[label]
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda p: (
                -contextual_receiver_score(p, casters, float(scene.area_size)),
                int(p.sid),
            ),
        )
        for candidate in ranked:
            if int(candidate.sid) not in used_sids:
                selected[label] = candidate
                used_sids.add(int(candidate.sid))
                break

    if not selected:
        raise RuntimeError("No wall or roof receiver was available.")
    return selected


def scene_key(split: str, terrain: str, seed: int) -> str:
    return "{}__{}__{}".format(split, terrain, int(seed))


def generate_scene_with_retry(terrain: str, seed: int):
    errors = []
    for requested in (200, 160, 120, 80):
        try:
            return GENERATOR.generate_urban_receiver_and_casters(
                n_polygons=int(requested),
                terrain_type=str(terrain),
                seed=int(seed),
                receiver_surface_kinds=("wall", "flat_roof", "pitched_roof"),
                include_same_building_casters=False,
                verbose=False,
            )
        except Exception as error:
            errors.append(
                {
                    "requested_n_polygons": int(requested),
                    "error": repr(error),
                }
            )
    raise RuntimeError("Scene generation failed after retries: {}".format(errors))


def load_or_build_scene(
    output_dir: Path,
    split: str,
    terrain: str,
    seed: int,
) -> SceneBundle:
    key = scene_key(split, terrain, seed)
    scene_dir = output_dir / "scenes" / key
    checkpoint = scene_dir / "scene_bundle.pkl"
    metadata_path = scene_dir / "scene_metadata.json"

    if checkpoint.exists():
        try:
            with checkpoint.open("rb") as stream:
                bundle = pickle.load(stream)
            return bundle
        except Exception:
            pass

    scene_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        scene = generate_scene_with_retry(terrain, seed)
        raw = SHADING_PROFILES.build_shading_profile(
            scene=scene,
            profile_mode="raw",
            receiver_surface_kinds=("wall", "flat_roof", "pitched_roof"),
            verbose=False,
        )
        proxy = SHADING_PROFILES.build_shading_profile(
            scene=scene,
            profile_mode="barrier_proxy",
            receiver_surface_kinds=("wall", "flat_roof", "pitched_roof"),
            decomposition_mode="merged",
            barrier_vertical_top_mode="eave",
            include_roof_casters=True,
            include_tree_casters=False,
            include_original_building_wall_casters=False,
            include_nonbuilding_casters=False,
            optimize_rotation=True,
            rotation_cost_mode="total_length",
            deduplicate_barriers=True,
            verbose=False,
        )
        receivers = select_receivers(proxy, scene)
        bundle = SceneBundle(
            split=str(split),
            terrain=str(terrain),
            seed=int(seed),
            scene=scene,
            raw_profile=raw,
            proxy_profile=proxy,
            receivers=receivers,
        )
        atomic_pickle(checkpoint, bundle)
        atomic_json(
            metadata_path,
            {
                "status": "complete",
                "created_at": now_iso(),
                "split": split,
                "terrain": terrain,
                "seed": int(seed),
                "area_size_m": float(scene.area_size),
                "n_buildings": int(scene.n_buildings),
                "n_scene_polygons": int(len(scene.all_polygons)),
                "raw_n_casters": int(len(raw.casters)),
                "proxy_n_casters": int(len(proxy.casters)),
                "raw_n_receivers": int(len(raw.receivers)),
                "proxy_n_receivers": int(len(proxy.receivers)),
                "receiver_sids": {
                    label: int(receiver.sid)
                    for label, receiver in receivers.items()
                },
                "receiver_kinds": {
                    label: str(receiver.surface_kind)
                    for label, receiver in receivers.items()
                },
                "elapsed_s": float(time.perf_counter() - started),
            },
        )
        return bundle
    except Exception as error:
        atomic_json(
            metadata_path,
            {
                "status": "failed",
                "created_at": now_iso(),
                "split": split,
                "terrain": terrain,
                "seed": int(seed),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def receiver_dir(output_dir: Path, bundle: SceneBundle, label: str) -> Path:
    return (
        output_dir
        / "scenes"
        / scene_key(bundle.split, bundle.terrain, bundle.seed)
        / "receivers"
        / label
    )


def metrics(reference: np.ndarray, estimate: np.ndarray, active_only: bool = False) -> Dict:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate)
    if active_only:
        mask &= reference >= MIN_ACTIVE_SHADE
    if not np.any(mask):
        return {
            "n": 0,
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
        }
    difference = estimate[mask] - reference[mask]
    absolute = np.abs(difference)
    return {
        "n": int(np.sum(mask)),
        "bias": float(np.mean(difference)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def radius_study_receiver(
    output_dir: Path,
    bundle: SceneBundle,
    label: str,
) -> List[Dict]:
    path = receiver_dir(output_dir, bundle, label) / "radius_rows.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    receiver = bundle.receivers[label]
    cache = receiver_cache(receiver)
    casters = building_casters(bundle.proxy_profile, receiver)
    directions = make_direction_grid(cache, fine=False)
    rows = []

    for direction_index, (azimuth, elevation) in enumerate(directions):
        parts, elapsed = direction_parts(cache, casters, azimuth, elevation)
        full = union_fraction((part[2] for part in parts), cache.area)
        for radius in RADIUS_LEVELS_M:
            estimate = union_fraction(
                (
                    polygon
                    for _, distance, polygon in parts
                    if distance <= float(radius)
                ),
                cache.area,
            )
            rows.append(
                {
                    "split": bundle.split,
                    "terrain": bundle.terrain,
                    "seed": int(bundle.seed),
                    "receiver_label": label,
                    "receiver_sid": int(receiver.sid),
                    "receiver_kind": str(receiver.surface_kind),
                    "direction_index": int(direction_index),
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": float(elevation),
                    "radius_m": float(radius),
                    "reference_full": float(full),
                    "estimate": float(estimate),
                    "n_clipped_full": int(len(parts)),
                    "exact_clip_elapsed_s": float(elapsed),
                    "n_candidates_full": int(len(casters)),
                }
            )

    atomic_csv(path, rows)
    return rows


def aggregate_level_metrics(
    rows: Sequence[Dict],
    level_key: str,
    reference_key: str,
    estimate_key: str,
) -> List[Dict]:
    levels = sorted(set(float(row[level_key]) for row in rows))
    result = []
    for level in levels:
        subset = [row for row in rows if float(row[level_key]) == level]
        reference = np.asarray([float(row[reference_key]) for row in subset])
        estimate = np.asarray([float(row[estimate_key]) for row in subset])
        all_metrics = metrics(reference, estimate, active_only=False)
        active_metrics = metrics(reference, estimate, active_only=True)
        result.append(
            {
                level_key: float(level),
                "all": all_metrics,
                "active": active_metrics,
            }
        )
    return result


def select_safety_level(
    level_metrics: Sequence[Dict],
    level_key: str,
    mae_limit: float,
    p95_limit: float,
) -> Tuple[float, float]:
    levels = [float(row[level_key]) for row in level_metrics]
    knee_index = None
    for index, row in enumerate(level_metrics):
        active = row["active"]
        if int(active["n"]) <= 0:
            continue
        if float(active["mae"]) <= mae_limit and float(active["p95_abs"]) <= p95_limit:
            knee_index = index
            break
    if knee_index is None:
        knee_index = len(levels) - 1
    retained_index = min(knee_index + 1, len(levels) - 1)
    return float(levels[knee_index]), float(levels[retained_index])


def angular_support(cache: ReceiverCache, caster) -> Dict[str, float]:
    receiver_vertices = np.asarray(cache.receiver.vertices, dtype=float)
    caster_vertices = np.asarray(caster.vertices, dtype=float)
    center_delta = np.asarray(caster.center, dtype=float) - cache.origin
    geometric_azimuth, geometric_elevation = vector_azimuth_elevation(center_delta)

    relative_azimuths = []
    elevations = []
    for receiver_vertex in receiver_vertices:
        for caster_vertex in caster_vertices:
            delta = caster_vertex - receiver_vertex
            if float(np.linalg.norm(delta)) <= EPS:
                continue
            azimuth, elevation = vector_azimuth_elevation(delta)
            relative_azimuths.append(float(wrap_180(azimuth - geometric_azimuth)))
            elevations.append(float(elevation))

    if not relative_azimuths:
        relative_azimuths = (-1.0, 1.0)
        elevations = (geometric_elevation - 1.0, geometric_elevation + 1.0)

    azimuth_min = float(max(-90.0, min(relative_azimuths)))
    azimuth_max = float(min(90.0, max(relative_azimuths)))
    elevation_min = float(max(2.0, min(elevations)))
    elevation_max = float(min(88.0, max(elevations)))

    if azimuth_max - azimuth_min < 0.75:
        midpoint = 0.5 * (azimuth_min + azimuth_max)
        azimuth_min = midpoint - 0.375
        azimuth_max = midpoint + 0.375
    if elevation_max - elevation_min < 0.75:
        midpoint = 0.5 * (elevation_min + elevation_max)
        elevation_min = max(2.0, midpoint - 0.375)
        elevation_max = min(88.0, midpoint + 0.375)

    return {
        "geometric_azimuth_deg": float(geometric_azimuth),
        "geometric_elevation_deg": float(geometric_elevation),
        "azimuth_min_rel_deg": float(azimuth_min),
        "azimuth_max_rel_deg": float(azimuth_max),
        "elevation_min_deg": float(elevation_min),
        "elevation_max_deg": float(elevation_max),
    }


def kernel_values(
    kernel: str,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    center_azimuth, center_elevation, sigma_azimuth, sigma_elevation, amplitude = [
        float(value) for value in params
    ]
    azimuth_difference = wrap_180(np.asarray(azimuth_deg, dtype=float) - center_azimuth)
    elevation_difference = np.asarray(elevation_deg, dtype=float) - center_elevation
    sigma_azimuth = max(0.05, abs(float(sigma_azimuth)))
    sigma_elevation = max(0.05, abs(float(sigma_elevation)))
    radius_squared = (
        (azimuth_difference / sigma_azimuth) ** 2
        + (elevation_difference / sigma_elevation) ** 2
    )

    if kernel == "gaussian":
        base = np.exp(-0.5 * radius_squared)
    elif kernel == "supergaussian":
        base = np.exp(-0.5 * radius_squared * radius_squared)
    elif kernel == "supergaussian6":
        base = np.exp(-0.5 * radius_squared * radius_squared * radius_squared)
    elif kernel == "supergaussian8":
        base = np.exp(
            -0.5
            * radius_squared
            * radius_squared
            * radius_squared
            * radius_squared
        )
    elif kernel == "logistic":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        normalizer = 1.0 / (1.0 + math.exp(-8.0))
        base = (1.0 / (1.0 + np.exp(np.clip(8.0 * (radius - 1.0), -60.0, 60.0)))) / normalizer
    elif kernel == "biweight":
        base = np.where(radius_squared < 1.0, (1.0 - radius_squared) ** 2, 0.0)
    elif kernel == "cosine":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        base = np.where(
            radius < 1.0,
            0.5 * (1.0 + np.cos(np.pi * radius)),
            0.0,
        )
    elif kernel == "wendland":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        base = np.where(
            radius < 1.0,
            (1.0 - radius) ** 4 * (4.0 * radius + 1.0),
            0.0,
        )
    else:
        raise ValueError("Unknown kernel: {}".format(kernel))
    return np.clip(float(amplitude) * base, 0.0, 1.0)


def kernel_values_many(
    kernel: str,
    azimuth_deg: float,
    elevation_deg: float,
    params: np.ndarray,
) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.size == 0:
        return np.zeros(0, dtype=float)
    azimuth_difference = wrap_180(float(azimuth_deg) - params[:, 0])
    elevation_difference = float(elevation_deg) - params[:, 1]
    sigma_azimuth = np.maximum(0.05, np.abs(params[:, 2]))
    sigma_elevation = np.maximum(0.05, np.abs(params[:, 3]))
    radius_squared = (
        (azimuth_difference / sigma_azimuth) ** 2
        + (elevation_difference / sigma_elevation) ** 2
    )

    if kernel == "gaussian":
        base = np.exp(-0.5 * radius_squared)
    elif kernel == "supergaussian":
        base = np.exp(-0.5 * radius_squared * radius_squared)
    elif kernel == "supergaussian6":
        base = np.exp(-0.5 * radius_squared * radius_squared * radius_squared)
    elif kernel == "supergaussian8":
        base = np.exp(
            -0.5
            * radius_squared
            * radius_squared
            * radius_squared
            * radius_squared
        )
    elif kernel == "logistic":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        normalizer = 1.0 / (1.0 + math.exp(-8.0))
        base = (1.0 / (1.0 + np.exp(np.clip(8.0 * (radius - 1.0), -60.0, 60.0)))) / normalizer
    elif kernel == "biweight":
        base = np.where(radius_squared < 1.0, (1.0 - radius_squared) ** 2, 0.0)
    elif kernel == "cosine":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        base = np.where(
            radius < 1.0,
            0.5 * (1.0 + np.cos(np.pi * radius)),
            0.0,
        )
    elif kernel == "wendland":
        radius = np.sqrt(np.maximum(0.0, radius_squared))
        base = np.where(
            radius < 1.0,
            (1.0 - radius) ** 4 * (4.0 * radius + 1.0),
            0.0,
        )
    else:
        raise ValueError("Unknown kernel: {}".format(kernel))
    return np.clip(params[:, 4] * base, 0.0, 1.0)


def fit_kernel(
    kernel: str,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    values: np.ndarray,
    support: Dict[str, float],
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    geometric_azimuth = float(support["geometric_azimuth_deg"])
    geometric_elevation = float(support["geometric_elevation_deg"])
    azimuth_span = max(
        0.75,
        float(support["azimuth_max_rel_deg"] - support["azimuth_min_rel_deg"]),
    )
    elevation_span = max(
        0.75,
        float(support["elevation_max_deg"] - support["elevation_min_deg"]),
    )
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 1.0e-8:
        return np.asarray(
            [
                geometric_azimuth,
                geometric_elevation,
                max(0.5, 0.5 * azimuth_span),
                max(0.5, 0.5 * elevation_span),
                0.0,
            ],
            dtype=float,
        )

    center_low = geometric_azimuth - 180.0
    center_high = geometric_azimuth + 180.0
    elevation_low = 2.0
    elevation_high = 88.0

    active_weights = np.maximum(values, 0.0)
    if float(np.sum(active_weights)) > EPS:
        azimuth_radians = np.radians(np.asarray(azimuths, dtype=float))
        sine_mean = float(np.sum(active_weights * np.sin(azimuth_radians)))
        cosine_mean = float(np.sum(active_weights * np.cos(azimuth_radians)))
        moment_azimuth = math.degrees(math.atan2(sine_mean, cosine_mean)) % 360.0
        moment_elevation = float(
            np.sum(active_weights * np.asarray(elevations, dtype=float))
            / np.sum(active_weights)
        )
        active_mask = values > max(1.0e-8, 0.02 * peak)
        active_azimuth_span = float(
            np.max(
                np.abs(
                    wrap_180(
                        np.asarray(azimuths, dtype=float)[active_mask]
                        - moment_azimuth
                    )
                )
            )
        )
        active_elevation_span = float(
            np.max(
                np.abs(
                    np.asarray(elevations, dtype=float)[active_mask]
                    - moment_elevation
                )
            )
        )
    else:
        moment_azimuth = geometric_azimuth
        moment_elevation = geometric_elevation
        active_azimuth_span = 0.5 * azimuth_span
        active_elevation_span = 0.5 * elevation_span

    # Express the circular moment in the continuous interval used by the
    # optimiser around the geometric centre.
    moment_azimuth = geometric_azimuth + float(
        wrap_180(moment_azimuth - geometric_azimuth)
    )

    if kernel in ("biweight", "cosine", "wendland", "logistic"):
        initial_sigma_azimuth = max(
            0.75 * azimuth_span,
            1.55 * active_azimuth_span,
        )
        initial_sigma_elevation = max(
            0.75 * elevation_span,
            1.55 * active_elevation_span,
        )
    elif kernel in ("supergaussian", "supergaussian6", "supergaussian8"):
        initial_sigma_azimuth = max(
            0.55 * azimuth_span,
            1.15 * active_azimuth_span,
        )
        initial_sigma_elevation = max(
            0.55 * elevation_span,
            1.15 * active_elevation_span,
        )
    else:
        initial_sigma_azimuth = max(
            0.40 * azimuth_span,
            0.85 * active_azimuth_span,
        )
        initial_sigma_elevation = max(
            0.40 * elevation_span,
            0.85 * active_elevation_span,
        )

    x0 = np.asarray(
        [
            moment_azimuth,
            min(elevation_high, max(elevation_low, moment_elevation)),
            min(180.0, max(0.5, initial_sigma_azimuth)),
            min(90.0, max(0.5, initial_sigma_elevation)),
            peak,
        ],
        dtype=float,
    )
    lower = np.asarray(
        [
            center_low,
            elevation_low,
            0.20,
            0.20,
            0.0,
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            center_high,
            elevation_high,
            180.0,
            90.0,
            1.0,
        ],
        dtype=float,
    )
    x0 = np.minimum(upper - 1.0e-8, np.maximum(lower + 1.0e-8, x0))

    def residual(parameters):
        prediction = kernel_values(
            kernel,
            azimuths,
            elevations,
            parameters,
        )
        # Active shadow values are sparse on a hemisphere grid.  Without this
        # weighting, a least-squares fit can improve its total objective by
        # suppressing a real but narrow obstruction.
        weights = 1.0 + 4.0 * values
        return weights * (prediction - values)

    try:
        fit = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=120,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        parameters = np.asarray(fit.x, dtype=float)
    except Exception:
        parameters = x0

    parameters[0] = float(wrap_360(parameters[0]))
    parameters[4] = max(0.0, min(1.0, float(parameters[4])))
    return parameters


def sample_and_fit_caster(
    cache: ReceiverCache,
    caster,
    kernels: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    support = angular_support(cache, caster)
    geometric_azimuth = float(support["geometric_azimuth_deg"])
    azimuth_span = max(
        0.75,
        float(support["azimuth_max_rel_deg"] - support["azimuth_min_rel_deg"]),
    )
    elevation_span = max(
        0.75,
        float(support["elevation_max_deg"] - support["elevation_min_deg"]),
    )
    azimuth_margin = max(0.25, 0.15 * azimuth_span)
    elevation_margin = max(0.25, 0.15 * elevation_span)

    azimuth_relative = np.linspace(
        float(support["azimuth_min_rel_deg"]) - azimuth_margin,
        float(support["azimuth_max_rel_deg"]) + azimuth_margin,
        FIT_GRID_N,
    )
    elevations_axis = np.linspace(
        max(2.0, float(support["elevation_min_deg"]) - elevation_margin),
        min(88.0, float(support["elevation_max_deg"]) + elevation_margin),
        FIT_GRID_N,
    )
    sample_map = {}
    for elevation in elevations_axis:
        for relative in azimuth_relative:
            azimuth = float(wrap_360(geometric_azimuth + float(relative)))
            value = single_caster_fraction(cache, caster, azimuth, float(elevation))
            sample_map[(round(azimuth, 8), round(float(elevation), 8))] = float(value)

    # A centre-derived support box can miss high, close obstructions whose
    # projected shadow covers directions far away from the centre ray.  Add an
    # offset hemisphere grid.  Its angles are deliberately different from the
    # validation grid, so the reported query errors still test interpolation.
    global_values = []
    for elevation in FIT_GLOBAL_ELEVATIONS_DEG:
        for azimuth in FIT_GLOBAL_AZIMUTHS_DEG:
            direction = sun_vector(float(azimuth), float(elevation))
            if float(np.dot(cache.normal, direction)) <= 0.02:
                continue
            key = (round(float(azimuth), 8), round(float(elevation), 8))
            if key not in sample_map:
                sample_map[key] = single_caster_fraction(
                    cache,
                    caster,
                    float(azimuth),
                    float(elevation),
                )
            global_values.append(float(sample_map[key]))

    ordered_samples = sorted(sample_map.items())
    azimuths = [float(key[0]) for key, _ in ordered_samples]
    elevations = [float(key[1]) for key, _ in ordered_samples]
    values = [float(value) for _, value in ordered_samples]

    azimuths_array = np.asarray(azimuths, dtype=float)
    elevations_array = np.asarray(elevations, dtype=float)
    values_array = np.asarray(values, dtype=float)
    fitted = {}
    for kernel in kernels:
        fitted[kernel] = fit_kernel(
            kernel,
            azimuths_array,
            elevations_array,
            values_array,
            support,
        )

    delta = np.asarray(caster.center, dtype=float) - cache.origin
    distance_squared = max(EPS, float(np.dot(delta, delta)))
    geometric_score = float(caster.area) / distance_squared
    span_mass = float(np.sum(np.asarray(global_values, dtype=float)))
    summary = {
        "sample_peak": float(np.max(values_array)),
        "sample_mean": float(np.mean(values_array)),
        "support_azimuth_span_deg": float(azimuth_span),
        "support_elevation_span_deg": float(elevation_span),
        "support_mass": float(span_mass),
        "geometric_score": float(geometric_score),
    }
    return fitted, summary


def fit_lobes_receiver(
    output_dir: Path,
    bundle: SceneBundle,
    label: str,
    radius_m: float,
    kernels: Sequence[str],
) -> Dict[str, np.ndarray]:
    directory = receiver_dir(output_dir, bundle, label)
    path = directory / "fitted_lobes.npz"
    receiver = bundle.receivers[label]
    cache = receiver_cache(receiver)
    casters = [
        caster
        for caster in building_casters(bundle.proxy_profile, receiver)
        if horizontal_distance(receiver, caster) <= float(radius_m)
    ]
    casters = sorted(casters, key=lambda caster: int(caster.sid))

    n = len(casters)
    surface_ids = np.asarray([int(caster.sid) for caster in casters], dtype=np.uint32)
    kind_codes = np.asarray(
        [
            1 if caster.surface_kind == "barrier_wall_proxy" else 2
            for caster in casters
        ],
        dtype=np.uint8,
    )
    sample_peak = np.full(n, np.nan, dtype=np.float64)
    support_mass = np.full(n, np.nan, dtype=np.float64)
    geometric_score = np.full(n, np.nan, dtype=np.float64)
    fit_elapsed_s = np.full(n, np.nan, dtype=np.float64)
    parameters = {
        kernel: np.full((n, 5), np.nan, dtype=np.float64)
        for kernel in kernels
    }

    if path.exists():
        try:
            previous = np.load(str(path), allow_pickle=False)
            if np.array_equal(previous["surface_ids"], surface_ids):
                sample_peak[:] = previous["sample_peak"]
                support_mass[:] = previous["support_mass"]
                geometric_score[:] = previous["geometric_score"]
                fit_elapsed_s[:] = previous["fit_elapsed_s"]
                for kernel in kernels:
                    key = "params_{}".format(kernel)
                    if key in previous:
                        parameters[kernel][:] = previous[key]
        except Exception:
            pass

    def save_checkpoint() -> None:
        payload = {
            "surface_ids": surface_ids,
            "kind_codes": kind_codes,
            "sample_peak": sample_peak,
            "support_mass": support_mass,
            "geometric_score": geometric_score,
            "fit_elapsed_s": fit_elapsed_s,
            "receiver_sid": np.asarray([int(receiver.sid)], dtype=np.uint32),
            "receiver_normal_azimuth_deg": np.asarray(
                [cache.normal_azimuth_deg],
                dtype=np.float64,
            ),
            "radius_m": np.asarray([float(radius_m)], dtype=np.float64),
        }
        for kernel in kernels:
            payload["params_{}".format(kernel)] = parameters[kernel]
        atomic_npz(path, **payload)

    for index, caster in enumerate(casters):
        complete = np.isfinite(sample_peak[index])
        for kernel in kernels:
            complete = complete and bool(np.all(np.isfinite(parameters[kernel][index])))
        if complete:
            continue

        started = time.perf_counter()
        fitted, summary = sample_and_fit_caster(cache, caster, kernels)
        sample_peak[index] = float(summary["sample_peak"])
        support_mass[index] = float(summary["support_mass"])
        geometric_score[index] = float(summary["geometric_score"])
        fit_elapsed_s[index] = float(time.perf_counter() - started)
        for kernel in kernels:
            parameters[kernel][index, :] = fitted[kernel]

        if (index + 1) % 20 == 0:
            save_checkpoint()

    save_checkpoint()
    result = {
        "surface_ids": surface_ids,
        "kind_codes": kind_codes,
        "sample_peak": sample_peak,
        "support_mass": support_mass,
        "geometric_score": geometric_score,
        "fit_elapsed_s": fit_elapsed_s,
        "receiver_normal_azimuth_deg": np.asarray(
            [cache.normal_azimuth_deg],
            dtype=np.float64,
        ),
    }
    for kernel in kernels:
        result["params_{}".format(kernel)] = parameters[kernel]
    return result


def ranking_indices(fits: Dict[str, np.ndarray]) -> np.ndarray:
    surface_ids = np.asarray(fits["surface_ids"], dtype=np.uint32)
    sample_peak = np.nan_to_num(
        np.asarray(fits["sample_peak"], dtype=float),
        nan=0.0,
    )
    support_mass = np.nan_to_num(
        np.asarray(fits["support_mass"], dtype=float),
        nan=0.0,
    )
    geometric_score = np.nan_to_num(
        np.asarray(fits["geometric_score"], dtype=float),
        nan=0.0,
    )
    # np.lexsort uses the last key as the primary key.
    return np.lexsort(
        (
            surface_ids,
            -geometric_score,
            -support_mass,
            -sample_peak,
        )
    )


def topk_study_receiver(
    output_dir: Path,
    bundle: SceneBundle,
    label: str,
    radius_m: float,
    fits: Dict[str, np.ndarray],
) -> List[Dict]:
    path = receiver_dir(output_dir, bundle, label) / "topk_rows.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    receiver = bundle.receivers[label]
    cache = receiver_cache(receiver)
    caster_by_sid = {
        int(caster.sid): caster
        for caster in building_casters(bundle.proxy_profile, receiver)
        if horizontal_distance(receiver, caster) <= float(radius_m)
    }
    rank = ranking_indices(fits)
    ranked_sids = [int(fits["surface_ids"][index]) for index in rank]
    ranked_sids = [sid for sid in ranked_sids if sid in caster_by_sid]
    ranked_casters = [caster_by_sid[sid] for sid in ranked_sids]
    directions = make_direction_grid(cache, fine=False)
    rows = []

    for direction_index, (azimuth, elevation) in enumerate(directions):
        parts, elapsed = direction_parts(
            cache,
            ranked_casters,
            azimuth,
            elevation,
        )
        polygon_by_sid = {int(sid): polygon for sid, _, polygon in parts}
        full = union_fraction(polygon_by_sid.values(), cache.area)

        for top_k in TOP_K_LEVELS:
            retained_sids = ranked_sids[: min(int(top_k), len(ranked_sids))]
            estimate = union_fraction(
                (
                    polygon_by_sid[sid]
                    for sid in retained_sids
                    if sid in polygon_by_sid
                ),
                cache.area,
            )
            rows.append(
                {
                    "split": bundle.split,
                    "terrain": bundle.terrain,
                    "seed": int(bundle.seed),
                    "receiver_label": label,
                    "receiver_sid": int(receiver.sid),
                    "receiver_kind": str(receiver.surface_kind),
                    "direction_index": int(direction_index),
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": float(elevation),
                    "top_k": int(top_k),
                    "reference_within_radius": float(full),
                    "estimate": float(estimate),
                    "n_candidates_radius": int(len(ranked_sids)),
                    "n_clipped_full": int(len(parts)),
                    "exact_clip_elapsed_s": float(elapsed),
                }
            )

    atomic_csv(path, rows)
    return rows


def quantize_params(
    params: np.ndarray,
    receiver_normal_azimuth_deg: float,
) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    relative = np.empty_like(params)
    relative[:, 0] = wrap_180(params[:, 0] - float(receiver_normal_azimuth_deg))
    relative[:, 1:] = params[:, 1:]
    quantized = np.empty(params.shape, dtype=np.int16)
    quantized[:, 0] = np.rint(np.clip(relative[:, 0], -180.0, 180.0) * 10.0).astype(np.int16)
    quantized[:, 1] = np.rint(np.clip(relative[:, 1], -90.0, 90.0) * 10.0).astype(np.int16)
    quantized[:, 2] = np.rint(np.clip(relative[:, 2], 0.0, 180.0) * 10.0).astype(np.int16)
    quantized[:, 3] = np.rint(np.clip(relative[:, 3], 0.0, 90.0) * 10.0).astype(np.int16)
    quantized[:, 4] = np.rint(np.clip(relative[:, 4], 0.0, 1.0) * 1000.0).astype(np.int16)
    return quantized


def dequantize_params(
    params_i16: np.ndarray,
    receiver_normal_azimuth_deg: float,
) -> np.ndarray:
    quantized = np.asarray(params_i16, dtype=np.int16)
    params = np.empty(quantized.shape, dtype=float)
    params[:, 0] = wrap_360(
        quantized[:, 0].astype(float) / 10.0
        + float(receiver_normal_azimuth_deg)
    )
    params[:, 1] = quantized[:, 1].astype(float) / 10.0
    params[:, 2] = quantized[:, 2].astype(float) / 10.0
    params[:, 3] = quantized[:, 3].astype(float) / 10.0
    params[:, 4] = quantized[:, 4].astype(float) / 1000.0
    return params


def evaluate_lobe_union(
    kernel: str,
    params: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    gamma: float,
) -> float:
    if len(params) == 0:
        return 0.0
    values = kernel_values_many(
        kernel,
        float(azimuth_deg),
        float(elevation_deg),
        params,
    )
    values = np.clip(values, 0.0, 1.0 - 1.0e-12)
    log_unshaded = float(np.sum(np.log1p(-values)))
    combined = 1.0 - math.exp(float(gamma) * log_unshaded)
    return max(0.0, min(1.0, combined))


def kernel_calibration_receiver_rows(
    output_dir: Path,
    bundle: SceneBundle,
    label: str,
    radius_m: float,
    top_k: int,
    fits: Dict[str, np.ndarray],
) -> List[Dict]:
    path = receiver_dir(output_dir, bundle, label) / "kernel_calibration_rows.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    receiver = bundle.receivers[label]
    cache = receiver_cache(receiver)
    rank = ranking_indices(fits)[: min(int(top_k), len(fits["surface_ids"]))]
    selected_sids = [int(fits["surface_ids"][index]) for index in rank]
    caster_by_sid = {
        int(caster.sid): caster
        for caster in building_casters(bundle.proxy_profile, receiver)
        if horizontal_distance(receiver, caster) <= float(radius_m)
    }
    selected_casters = [
        caster_by_sid[sid]
        for sid in selected_sids
        if sid in caster_by_sid
    ]
    selected_indices = [
        int(index)
        for index in rank
        if int(fits["surface_ids"][index]) in caster_by_sid
    ]
    directions = make_direction_grid(cache, fine=False)
    rows = []

    for direction_index, (azimuth, elevation) in enumerate(directions):
        parts, _ = direction_parts(
            cache,
            selected_casters,
            azimuth,
            elevation,
        )
        reference = union_fraction((part[2] for part in parts), cache.area)
        for kernel in KERNELS:
            params = fits["params_{}".format(kernel)][selected_indices]
            for gamma in GAMMA_LEVELS:
                estimate = evaluate_lobe_union(
                    kernel,
                    params,
                    azimuth,
                    elevation,
                    gamma,
                )
                rows.append(
                    {
                        "split": bundle.split,
                        "terrain": bundle.terrain,
                        "seed": int(bundle.seed),
                        "receiver_label": label,
                        "receiver_sid": int(receiver.sid),
                        "receiver_kind": str(receiver.surface_kind),
                        "direction_index": int(direction_index),
                        "azimuth_deg": float(azimuth),
                        "elevation_deg": float(elevation),
                        "kernel": kernel,
                        "gamma": float(gamma),
                        "reference_topk": float(reference),
                        "estimate": float(estimate),
                    }
                )

    atomic_csv(path, rows)
    return rows


def aggregate_kernel_selection(rows: Sequence[Dict]) -> Tuple[str, float, List[Dict]]:
    summary = []
    for kernel in KERNELS:
        for gamma in GAMMA_LEVELS:
            subset = [
                row
                for row in rows
                if str(row["kernel"]) == kernel
                and abs(float(row["gamma"]) - float(gamma)) < 1.0e-9
            ]
            reference = np.asarray([float(row["reference_topk"]) for row in subset])
            estimate = np.asarray([float(row["estimate"]) for row in subset])
            summary.append(
                {
                    "kernel": kernel,
                    "gamma": float(gamma),
                    "all": metrics(reference, estimate, active_only=False),
                    "active": metrics(reference, estimate, active_only=True),
                }
            )

    viable = [row for row in summary if int(row["active"]["n"]) > 0]
    if not viable:
        raise RuntimeError("No active calibration direction was produced.")
    best = min(
        viable,
        key=lambda row: (
            float(row["active"]["mae"]),
            abs(float(row["gamma"]) - 1.0),
            KERNELS.index(str(row["kernel"])),
        ),
    )
    return str(best["kernel"]), float(best["gamma"]), summary


def run_calibration(output_dir: Path) -> Dict:
    calibration_dir = output_dir / "calibration"
    selection_path = calibration_dir / "selection.json"
    if selection_path.exists():
        with selection_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    bundles = []
    radius_rows = []
    for terrain in TERRAINS:
        bundle = load_or_build_scene(
            output_dir,
            "calibration",
            terrain,
            CALIBRATION_SEEDS[terrain],
        )
        bundles.append(bundle)
        for label in sorted(bundle.receivers):
            radius_rows.extend(
                radius_study_receiver(output_dir, bundle, label)
            )

    radius_summary = aggregate_level_metrics(
        radius_rows,
        level_key="radius_m",
        reference_key="reference_full",
        estimate_key="estimate",
    )
    radius_knee, retained_radius = select_safety_level(
        radius_summary,
        "radius_m",
        RADIUS_MAE_LIMIT,
        RADIUS_P95_LIMIT,
    )

    all_fits = {}
    topk_rows = []
    for bundle in bundles:
        for label in sorted(bundle.receivers):
            key = (bundle.terrain, int(bundle.seed), label)
            fits = fit_lobes_receiver(
                output_dir,
                bundle,
                label,
                retained_radius,
                KERNELS,
            )
            all_fits[key] = fits
            topk_rows.extend(
                topk_study_receiver(
                    output_dir,
                    bundle,
                    label,
                    retained_radius,
                    fits,
                )
            )

    topk_summary = aggregate_level_metrics(
        topk_rows,
        level_key="top_k",
        reference_key="reference_within_radius",
        estimate_key="estimate",
    )
    topk_knee_float, retained_topk_float = select_safety_level(
        topk_summary,
        "top_k",
        TOPK_MAE_LIMIT,
        TOPK_P95_LIMIT,
    )
    topk_knee = int(round(topk_knee_float))
    retained_topk = int(round(retained_topk_float))

    kernel_rows = []
    for bundle in bundles:
        for label in sorted(bundle.receivers):
            key = (bundle.terrain, int(bundle.seed), label)
            kernel_rows.extend(
                kernel_calibration_receiver_rows(
                    output_dir,
                    bundle,
                    label,
                    retained_radius,
                    retained_topk,
                    all_fits[key],
                )
            )
    selected_kernel, selected_gamma, kernel_summary = aggregate_kernel_selection(
        kernel_rows
    )

    payload = {
        "status": "complete",
        "created_at": now_iso(),
        "radius_levels_m": list(RADIUS_LEVELS_M),
        "radius_mae_limit": RADIUS_MAE_LIMIT,
        "radius_p95_limit": RADIUS_P95_LIMIT,
        "radius_summary": radius_summary,
        "radius_knee_m": float(radius_knee),
        "retained_radius_m": float(retained_radius),
        "topk_levels": list(TOP_K_LEVELS),
        "topk_mae_limit": TOPK_MAE_LIMIT,
        "topk_p95_limit": TOPK_P95_LIMIT,
        "topk_summary": topk_summary,
        "topk_knee": int(topk_knee),
        "retained_topk": int(retained_topk),
        "kernel_summary": kernel_summary,
        "selected_kernel": selected_kernel,
        "selected_gamma": float(selected_gamma),
        "n_calibration_scenes": int(len(bundles)),
        "n_calibration_receivers": int(
            sum(len(bundle.receivers) for bundle in bundles)
        ),
    }
    atomic_json(selection_path, payload)
    return payload


def find_raw_receiver(bundle: SceneBundle, proxy_receiver):
    candidates = [
        receiver
        for receiver in bundle.raw_profile.receivers
        if int(receiver.sid) == int(proxy_receiver.sid)
    ]
    if not candidates:
        raise RuntimeError(
            "Raw receiver sid {} was not found.".format(proxy_receiver.sid)
        )
    return candidates[0]


def validation_receiver_rows(
    output_dir: Path,
    bundle: SceneBundle,
    label: str,
    calibration: Dict,
) -> List[Dict]:
    path = receiver_dir(output_dir, bundle, label) / "validation_rows.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    radius_m = float(calibration["retained_radius_m"])
    top_k = int(calibration["retained_topk"])
    kernel = str(calibration["selected_kernel"])
    gamma = float(calibration["selected_gamma"])

    proxy_receiver = bundle.receivers[label]
    raw_receiver = find_raw_receiver(bundle, proxy_receiver)
    proxy_cache = receiver_cache(proxy_receiver)
    raw_cache = receiver_cache(raw_receiver)

    proxy_all = building_casters(bundle.proxy_profile, proxy_receiver)
    raw_all = building_casters(bundle.raw_profile, raw_receiver)
    proxy_radius = [
        caster
        for caster in proxy_all
        if horizontal_distance(proxy_receiver, caster) <= radius_m
    ]
    fits = fit_lobes_receiver(
        output_dir,
        bundle,
        label,
        radius_m,
        (kernel,),
    )
    rank = ranking_indices(fits)[: min(top_k, len(fits["surface_ids"]))]
    selected_indices = [int(index) for index in rank]
    selected_sids = [int(fits["surface_ids"][index]) for index in selected_indices]
    caster_by_sid = {int(caster.sid): caster for caster in proxy_radius}
    selected_indices = [
        index
        for index in selected_indices
        if int(fits["surface_ids"][index]) in caster_by_sid
    ]
    selected_sids = [int(fits["surface_ids"][index]) for index in selected_indices]
    selected_casters = [caster_by_sid[sid] for sid in selected_sids]
    params_float = np.asarray(
        fits["params_{}".format(kernel)][selected_indices],
        dtype=float,
    )
    params_i16 = quantize_params(
        params_float,
        proxy_cache.normal_azimuth_deg,
    )
    params_quantized = dequantize_params(
        params_i16,
        proxy_cache.normal_azimuth_deg,
    )

    directions = make_direction_grid(proxy_cache, fine=False)
    rows = []
    for direction_index, (azimuth, elevation) in enumerate(directions):
        raw_parts, raw_elapsed = direction_parts(
            raw_cache,
            raw_all,
            azimuth,
            elevation,
        )
        raw_full = union_fraction((part[2] for part in raw_parts), raw_cache.area)

        proxy_parts, proxy_elapsed = direction_parts(
            proxy_cache,
            proxy_all,
            azimuth,
            elevation,
        )
        proxy_full = union_fraction(
            (part[2] for part in proxy_parts),
            proxy_cache.area,
        )
        proxy_radius_value = union_fraction(
            (
                polygon
                for _, distance, polygon in proxy_parts
                if distance <= radius_m
            ),
            proxy_cache.area,
        )
        selected_sid_set = set(selected_sids)
        proxy_topk = union_fraction(
            (
                polygon
                for sid, _, polygon in proxy_parts
                if int(sid) in selected_sid_set
            ),
            proxy_cache.area,
        )

        lobe_started = time.perf_counter()
        lobe_float = evaluate_lobe_union(
            kernel,
            params_float,
            azimuth,
            elevation,
            gamma,
        )
        lobe_elapsed = time.perf_counter() - lobe_started
        lobe_quantized = evaluate_lobe_union(
            kernel,
            params_quantized,
            azimuth,
            elevation,
            gamma,
        )

        rows.append(
            {
                "split": bundle.split,
                "terrain": bundle.terrain,
                "seed": int(bundle.seed),
                "receiver_label": label,
                "receiver_sid": int(proxy_receiver.sid),
                "receiver_kind": str(proxy_receiver.surface_kind),
                "direction_index": int(direction_index),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "raw_full": float(raw_full),
                "proxy_full": float(proxy_full),
                "proxy_radius": float(proxy_radius_value),
                "proxy_topk": float(proxy_topk),
                "lobe_float": float(lobe_float),
                "lobe_quantized": float(lobe_quantized),
                "raw_exact_elapsed_s": float(raw_elapsed),
                "proxy_exact_elapsed_s": float(proxy_elapsed),
                "lobe_elapsed_s": float(lobe_elapsed),
                "n_raw_candidates": int(len(raw_all)),
                "n_proxy_candidates": int(len(proxy_all)),
                "n_radius_candidates": int(len(proxy_radius)),
                "n_retained_lobes": int(len(params_float)),
            }
        )

    atomic_csv(path, rows)

    archive_path = receiver_dir(output_dir, bundle, label) / "packed_lobes.npz"
    atomic_npz(
        archive_path,
        receiver_id=np.asarray([int(proxy_receiver.sid)], dtype=np.uint32),
        valid_count=np.asarray([len(params_i16)], dtype=np.uint16),
        caster_sids=np.asarray(selected_sids, dtype=np.uint32),
        kind_codes=np.asarray(fits["kind_codes"][selected_indices], dtype=np.uint8),
        params_i16=params_i16,
        receiver_normal_azimuth_deg=np.asarray(
            [proxy_cache.normal_azimuth_deg],
            dtype=np.float32,
        ),
    )
    return rows


def validation_metrics(rows: Sequence[Dict]) -> Dict:
    raw = np.asarray([float(row["raw_full"]) for row in rows])
    proxy = np.asarray([float(row["proxy_full"]) for row in rows])
    radius = np.asarray([float(row["proxy_radius"]) for row in rows])
    topk = np.asarray([float(row["proxy_topk"]) for row in rows])
    lobe_float = np.asarray([float(row["lobe_float"]) for row in rows])
    lobe_quantized = np.asarray([float(row["lobe_quantized"]) for row in rows])

    stages = {
        "proxy_geometry": (raw, proxy),
        "neighbour_radius": (proxy, radius),
        "topk_selection": (radius, topk),
        "lobe_representation": (topk, lobe_float),
        "quantisation": (lobe_float, lobe_quantized),
        "combined": (raw, lobe_quantized),
    }
    payload = {}
    for name, pair in stages.items():
        payload[name] = {
            "all": metrics(pair[0], pair[1], active_only=False),
            "active": metrics(pair[0], pair[1], active_only=True),
        }

    raw_times = np.asarray([float(row["raw_exact_elapsed_s"]) for row in rows])
    proxy_times = np.asarray([float(row["proxy_exact_elapsed_s"]) for row in rows])
    lobe_times = np.asarray([float(row["lobe_elapsed_s"]) for row in rows])
    payload["timing"] = {
        "raw_exact_median_ms": float(np.median(raw_times) * 1000.0),
        "proxy_exact_median_ms": float(np.median(proxy_times) * 1000.0),
        "lobe_median_ms": float(np.median(lobe_times) * 1000.0),
        "raw_to_lobe_speedup_median": float(
            np.median(raw_times) / max(EPS, np.median(lobe_times))
        ),
        "proxy_to_lobe_speedup_median": float(
            np.median(proxy_times) / max(EPS, np.median(lobe_times))
        ),
    }
    payload["counts"] = {
        "n_rows": int(len(rows)),
        "n_active_raw": int(np.sum(raw >= MIN_ACTIVE_SHADE)),
        "n_scenes": int(
            len(set((str(row["terrain"]), int(row["seed"])) for row in rows))
        ),
        "n_receivers": int(
            len(
                set(
                    (
                        str(row["terrain"]),
                        int(row["seed"]),
                        str(row["receiver_label"]),
                    )
                    for row in rows
                )
            )
        ),
    }
    return payload


def write_combined_archive(
    output_dir: Path,
    bundles: Sequence[SceneBundle],
    calibration: Dict,
) -> Dict:
    radius_m = float(calibration["retained_radius_m"])
    top_k = int(calibration["retained_topk"])
    kernel = str(calibration["selected_kernel"])

    receiver_ids = []
    valid_counts = []
    caster_sids_rows = []
    kind_codes_rows = []
    params_rows = []
    receiver_meta = []

    for bundle in bundles:
        for label in sorted(bundle.receivers):
            receiver = bundle.receivers[label]
            cache = receiver_cache(receiver)
            fits = fit_lobes_receiver(
                output_dir,
                bundle,
                label,
                radius_m,
                (kernel,),
            )
            rank = ranking_indices(fits)[: min(top_k, len(fits["surface_ids"]))]
            count = int(len(rank))
            caster_sids = np.zeros(top_k, dtype=np.uint32)
            kind_codes = np.zeros(top_k, dtype=np.uint8)
            params_i16 = np.zeros((top_k, 5), dtype=np.int16)
            if count:
                caster_sids[:count] = fits["surface_ids"][rank]
                kind_codes[:count] = fits["kind_codes"][rank]
                params_i16[:count, :] = quantize_params(
                    fits["params_{}".format(kernel)][rank],
                    cache.normal_azimuth_deg,
                )
            receiver_ids.append(int(receiver.sid))
            valid_counts.append(count)
            caster_sids_rows.append(caster_sids)
            kind_codes_rows.append(kind_codes)
            params_rows.append(params_i16)
            receiver_meta.append(
                (
                    float(receiver.center[0]),
                    float(receiver.center[1]),
                    float(receiver.center[2]),
                    float(cache.normal_azimuth_deg),
                )
            )

    archive_path = output_dir / "validation" / "lobe_archive.npz"
    atomic_npz(
        archive_path,
        receiver_ids=np.asarray(receiver_ids, dtype=np.uint32),
        valid_counts=np.asarray(valid_counts, dtype=np.uint16),
        caster_sids=np.asarray(caster_sids_rows, dtype=np.uint32),
        kind_codes=np.asarray(kind_codes_rows, dtype=np.uint8),
        params_i16=np.asarray(params_rows, dtype=np.int16),
        receiver_meta=np.asarray(receiver_meta, dtype=np.float32),
    )
    packed_payload_bytes = (
        np.asarray(receiver_ids, dtype=np.uint32).nbytes
        + np.asarray(valid_counts, dtype=np.uint16).nbytes
        + np.asarray(caster_sids_rows, dtype=np.uint32).nbytes
        + np.asarray(kind_codes_rows, dtype=np.uint8).nbytes
        + np.asarray(params_rows, dtype=np.int16).nbytes
        + np.asarray(receiver_meta, dtype=np.float32).nbytes
    )
    n_receivers = max(1, len(receiver_ids))
    return {
        "archive_path": str(archive_path),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "packed_payload_bytes": int(packed_payload_bytes),
        "n_receivers": int(len(receiver_ids)),
        "compressed_bytes_per_receiver": float(
            archive_path.stat().st_size / n_receivers
        ),
        "packed_bytes_per_receiver": float(packed_payload_bytes / n_receivers),
    }


def run_validation(output_dir: Path, calibration: Dict) -> Dict:
    validation_dir = output_dir / "validation"
    summary_path = validation_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    bundles = []
    rows = []
    for terrain in TERRAINS:
        for seed in VALIDATION_SEEDS[terrain]:
            bundle = load_or_build_scene(
                output_dir,
                "validation",
                terrain,
                int(seed),
            )
            bundles.append(bundle)
            for label in sorted(bundle.receivers):
                rows.extend(
                    validation_receiver_rows(
                        output_dir,
                        bundle,
                        label,
                        calibration,
                    )
                )

    validation_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(validation_dir / "validation_all_rows.csv", rows)
    summary = validation_metrics(rows)
    summary["storage"] = write_combined_archive(
        output_dir,
        bundles,
        calibration,
    )

    raw_counts = []
    proxy_counts = []
    for bundle in bundles:
        raw_counts.append(int(len(bundle.raw_profile.casters)))
        proxy_counts.append(int(len(bundle.proxy_profile.casters)))
    raw_counts_array = np.asarray(raw_counts, dtype=float)
    proxy_counts_array = np.asarray(proxy_counts, dtype=float)
    summary["profile_reduction"] = {
        "mean_raw_casters": float(np.mean(raw_counts_array)),
        "mean_proxy_casters": float(np.mean(proxy_counts_array)),
        "mean_reduction_fraction": float(
            1.0 - np.mean(proxy_counts_array / raw_counts_array)
        ),
        "median_reduction_fraction": float(
            np.median(1.0 - proxy_counts_array / raw_counts_array)
        ),
    }
    summary["status"] = "complete"
    summary["created_at"] = now_iso()
    summary["selected_radius_m"] = float(calibration["retained_radius_m"])
    summary["selected_topk"] = int(calibration["retained_topk"])
    summary["selected_kernel"] = str(calibration["selected_kernel"])
    summary["selected_gamma"] = float(calibration["selected_gamma"])
    atomic_json(summary_path, summary)
    return summary


def run_heatmap_example(
    output_dir: Path,
    calibration: Dict,
) -> List[Dict]:
    path = output_dir / "validation" / "heatmap_example.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    terrain = "small_town"
    seed = int(VALIDATION_SEEDS[terrain][0])
    bundle = load_or_build_scene(output_dir, "validation", terrain, seed)
    label = "wall" if "wall" in bundle.receivers else sorted(bundle.receivers)[0]
    receiver = bundle.receivers[label]
    cache = receiver_cache(receiver)
    radius_m = float(calibration["retained_radius_m"])
    top_k = int(calibration["retained_topk"])
    kernel = str(calibration["selected_kernel"])
    gamma = float(calibration["selected_gamma"])
    fits = fit_lobes_receiver(
        output_dir,
        bundle,
        label,
        radius_m,
        (kernel,),
    )
    rank = ranking_indices(fits)[: min(top_k, len(fits["surface_ids"]))]
    selected_sids = [int(fits["surface_ids"][index]) for index in rank]
    caster_by_sid = {
        int(caster.sid): caster
        for caster in building_casters(bundle.proxy_profile, receiver)
        if horizontal_distance(receiver, caster) <= radius_m
    }
    selected_indices = [
        int(index)
        for index in rank
        if int(fits["surface_ids"][index]) in caster_by_sid
    ]
    selected_casters = [
        caster_by_sid[int(fits["surface_ids"][index])]
        for index in selected_indices
    ]
    params = fits["params_{}".format(kernel)][selected_indices]

    rows = []
    for direction_index, (azimuth, elevation) in enumerate(
        make_direction_grid(cache, fine=True)
    ):
        parts, _ = direction_parts(
            cache,
            selected_casters,
            azimuth,
            elevation,
        )
        reference = union_fraction((part[2] for part in parts), cache.area)
        estimate = evaluate_lobe_union(
            kernel,
            params,
            azimuth,
            elevation,
            gamma,
        )
        rows.append(
            {
                "terrain": terrain,
                "seed": seed,
                "receiver_label": label,
                "receiver_sid": int(receiver.sid),
                "receiver_kind": str(receiver.surface_kind),
                "direction_index": int(direction_index),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "reference": float(reference),
                "lobe": float(estimate),
                "error": float(estimate - reference),
            }
        )
    atomic_csv(path, rows)
    return rows


def study_manifest(calibration: Dict, validation: Dict) -> Dict:
    return {
        "study_name": "compact_directional_lobe_urban_shading",
        "created_at": now_iso(),
        "reference": {
            "projection": "oblique projection onto receiver plane",
            "clipper": "Sutherland-Hodgman convex polygon clipping",
            "overlap": "polygon union after clipping",
        },
        "geometry": {
            "profile": "proxy_merged_eave",
            "concave_decomposition": "ear clipping followed by greedy convex merging",
            "vegetation_in_validation": False,
            "same_building_casters": False,
        },
        "sampling": {
            "calibration_scenes": 4,
            "validation_scenes": 8,
            "receiver_classes_per_scene": ("wall", "roof"),
            "fit_grid_n": FIT_GRID_N,
            "active_shadow_threshold": MIN_ACTIVE_SHADE,
        },
        "selection": calibration,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the compact urban-shading validation study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--phase",
        choices=("calibration", "validation", "heatmap", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": now_iso(),
        "source_dir": str(SOURCE_DIR),
        "radius_levels_m": list(RADIUS_LEVELS_M),
        "top_k_levels": list(TOP_K_LEVELS),
        "kernels": list(KERNELS),
        "gamma_levels": list(GAMMA_LEVELS),
        "calibration_seeds": CALIBRATION_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "fit_grid_n": FIT_GRID_N,
        "active_shadow_threshold": MIN_ACTIVE_SHADE,
        "python": sys.version,
    }
    atomic_json(output_dir / "study_config.json", config)

    calibration = run_calibration(output_dir)
    print(
        "CALIBRATION",
        "R={:.0f} m".format(calibration["retained_radius_m"]),
        "K={}".format(calibration["retained_topk"]),
        "kernel={}".format(calibration["selected_kernel"]),
        "gamma={:.2f}".format(calibration["selected_gamma"]),
        flush=True,
    )
    if args.phase == "calibration":
        return

    validation = run_validation(output_dir, calibration)
    print(
        "VALIDATION",
        "active combined MAE={:.6f}".format(
            validation["combined"]["active"]["mae"]
        ),
        "speedup={:.1f}x".format(
            validation["timing"]["raw_to_lobe_speedup_median"]
        ),
        flush=True,
    )
    if args.phase == "validation":
        return

    heatmap_rows = run_heatmap_example(output_dir, calibration)
    print("HEATMAP rows={}".format(len(heatmap_rows)), flush=True)

    manifest = study_manifest(calibration, validation)
    atomic_json(output_dir / "study_manifest.json", manifest)
    print("COMPLETE {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
