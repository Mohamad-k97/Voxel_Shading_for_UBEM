# caster_lobe_and_numeric_heatmap.py

from __future__ import annotations

from pathlib import Path
from collections import Counter
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from generator import generate_urban_receiver_and_casters
from matplotlib.path import Path as MplPath

# ============================================================
# USER SETTINGS
# ============================================================

TERRAIN_TYPE = "tall_buildings"
SEED = 42

N_SCENE_CASTERS = 180
N_CASTERS_TO_PLOT = 180

CASTER_SORT_MODE = "score"       # "score" or "distance"

# Numerical heatmap grid
P_AZIMUTH = 121
Q_SLOPE = 46

REL_AZIMUTH_MIN_DEG = -180.0
REL_AZIMUTH_MAX_DEG = 180.0

SLOPE_MIN_DEG = 0.0
SLOPE_MAX_DEG = 90.0

MIN_SUN_SLOPE_FOR_SHADING_DEG = 0.0
# Lobe-to-heatmap reconstruction settings
N_LOBES_FOR_SURROGATE = 40

# "constant", "score", or "distance"
LOBE_AMPLITUDE_MODE = "score"

# Soft edge around hull in grid units/degrees.
# 0.0 = hard polygon hull only.
LOBE_SOFT_SIGMA_DEG = 5.0

# If True, normalize lobe scores so strongest lobe has amplitude 1.
NORMALIZE_LOBE_AMPLITUDES = True
# Optional morphology overrides
BUILT_DENSITY = None
GREEN_AREA_FRACTION = None
AVERAGE_BUILDING_HEIGHT = None
HEIGHT_LOGNORM_SIGMA = None
ELEVATION_TYPE = None

RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)

INCLUDE_SAME_BUILDING_CASTERS = True
ONLY_FRONT_HEMISPHERE = True

OUTPUT_DIR = Path("caster_lobe_and_heatmap_output")

SAVE_OUTPUTS = True
SHOW_PLOTS = True
DPI = 180


# ============================================================
# Constants
# ============================================================

EPS = 1e-10


# ============================================================
# Basic helpers
# ============================================================
def points_in_hull_mask(rel_azimuths, slopes, hull_points):
    """
    Rasterize one angular hull onto the azimuth-slope grid.
    """

    A, S = np.meshgrid(rel_azimuths, slopes)
    points = np.column_stack([A.ravel(), S.ravel()])

    path = MplPath(hull_points)
    mask = path.contains_points(points)

    return mask.reshape(len(slopes), len(rel_azimuths))


def angular_distance_to_cloud_grid(rel_azimuths, slopes, cloud):
    """
    Approximate distance from every grid point to nearest angular cloud point.
    This creates a soft blur around the hull.
    """

    A, S = np.meshgrid(rel_azimuths, slopes)

    dist2_min = np.full(A.shape, np.inf, dtype=float)

    for p in cloud:
        da = A - p[0]
        ds = S - p[1]
        dist2 = da * da + ds * ds
        dist2_min = np.minimum(dist2_min, dist2)

    return np.sqrt(dist2_min)


def lobe_amplitudes_from_table(df_lobes):
    """
    Convert lobe table to per-caster amplitudes.
    """

    if len(df_lobes) == 0:
        return np.array([], dtype=float)

    if LOBE_AMPLITUDE_MODE == "constant":
        amp = np.ones(len(df_lobes), dtype=float)

    elif LOBE_AMPLITUDE_MODE == "distance":
        d = df_lobes["distance"].to_numpy(dtype=float)
        amp = 1.0 / np.maximum(d, EPS) ** 2

    elif LOBE_AMPLITUDE_MODE == "score":
        amp = df_lobes["score"].to_numpy(dtype=float)

    else:
        raise ValueError(
            "LOBE_AMPLITUDE_MODE must be 'constant', 'score', or 'distance'."
        )

    amp = np.maximum(amp, 0.0)

    if NORMALIZE_LOBE_AMPLITUDES and np.max(amp) > EPS:
        amp = amp / np.max(amp)

    amp = np.clip(amp, 0.0, 1.0)

    return amp


def generate_lobe_surrogate_heatmap(heatmap, df_lobes, clouds):
    """
    Reconstruct approximate heatmap from caster angular lobe hulls.

    Each caster lobe contributes a soft/hard angular footprint.
    Contributions are combined using probabilistic union:

        H = 1 - product(1 - contribution_i)
    """

    rel_azimuths = heatmap["rel_azimuths"]
    slopes = heatmap["slopes"]
    front_sun_mask = heatmap["front_sun_mask"]

    df_use = df_lobes.head(N_LOBES_FOR_SURROGATE).copy()
    amplitudes = lobe_amplitudes_from_table(df_use)

    reconstructed = np.zeros(
        (len(slopes), len(rel_azimuths)),
        dtype=float,
    )

    individual_lobes = []

    for idx, (_, row) in enumerate(df_use.iterrows()):
        sid = int(row["caster_sid"])

        if sid not in clouds:
            continue

        cloud = clouds[sid]

        if len(cloud) < 3:
            continue

        try:
            hull = ConvexHull(cloud)
            hull_points = cloud[hull.vertices]
        except Exception:
            continue

        inside = points_in_hull_mask(
            rel_azimuths=rel_azimuths,
            slopes=slopes,
            hull_points=hull_points,
        )

        amp = float(amplitudes[idx])

        if LOBE_SOFT_SIGMA_DEG <= 0:
            contribution = amp * inside.astype(float)

        else:
            # Hard value inside the hull.
            contribution = amp * inside.astype(float)

            # Soft halo outside, based on distance to angular cloud.
            dist = angular_distance_to_cloud_grid(
                rel_azimuths=rel_azimuths,
                slopes=slopes,
                cloud=cloud,
            )

            halo = amp * np.exp(-0.5 * (dist / LOBE_SOFT_SIGMA_DEG) ** 2)

            # Keep full value inside hull, soft value outside.
            contribution = np.maximum(contribution, halo)

        contribution[~front_sun_mask] = 0.0
        contribution = np.clip(contribution, 0.0, 1.0)

        # Probabilistic union:
        # new = 1 - (1-old)(1-contribution)
        reconstructed = 1.0 - (1.0 - reconstructed) * (1.0 - contribution)

        individual_lobes.append({
            "caster_sid": sid,
            "amplitude": amp,
            "max_contribution": float(np.max(contribution)),
            "sum_contribution": float(np.sum(contribution)),
        })

    reconstructed[~front_sun_mask] = np.nan

    individual_lobes_df = pd.DataFrame(individual_lobes)

    return reconstructed, individual_lobes_df
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < EPS:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / n


def wrap_angle_360(angle_deg):
    return angle_deg % 360.0


def wrap_angle_180(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def horizontal_azimuth_from_vector(v):
    v = np.asarray(v, dtype=float)
    vh = np.array([v[0], v[1], 0.0], dtype=float)

    n = np.linalg.norm(vh)

    if n < EPS:
        return None

    vh = vh / n

    az = math.degrees(math.atan2(vh[0], vh[1]))
    return wrap_angle_360(az)


def vector_to_azimuth_slope(v):
    v = unit(v)

    az = math.degrees(math.atan2(v[0], v[1]))
    az = wrap_angle_360(az)

    slope = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))

    return az, slope


def sun_vector_from_azimuth_slope(global_azimuth_deg, slope_deg):
    az = math.radians(float(global_azimuth_deg))
    el = math.radians(float(slope_deg))

    x_east = math.cos(el) * math.sin(az)
    y_north = math.cos(el) * math.cos(az)
    z_up = math.sin(el)

    return unit(np.array([x_east, y_north, z_up], dtype=float))


def polygon_area_2d(points):
    p = np.asarray(points, dtype=float)

    if len(p) < 3:
        return 0.0

    x = p[:, 0]
    y = p[:, 1]

    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(points):
    p = np.asarray(points, dtype=float)

    if len(p) < 3:
        return p

    if polygon_area_2d(p) < 0:
        return p[::-1].copy()

    return p


def make_plane_basis(normal):
    n = unit(normal)

    if abs(n[2]) < 0.9:
        temp = np.array([0.0, 0.0, 1.0])
    else:
        temp = np.array([1.0, 0.0, 0.0])

    u = unit(np.cross(temp, n))
    v = unit(np.cross(n, u))

    return u, v


def points_3d_to_plane_2d(points_3d, plane_origin, u, v):
    P = np.asarray(points_3d, dtype=float)
    q = np.asarray(plane_origin, dtype=float)

    rel = P - q

    x = rel @ u
    y = rel @ v

    return np.column_stack([x, y])


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


# ============================================================
# Sutherland-Hodgman clipping for numeric heatmap
# ============================================================

def line_intersection_2d(s, e, a, b):
    s = np.asarray(s, dtype=float)
    e = np.asarray(e, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    edge = b - a
    direction = e - s

    denom = cross2(edge, direction)

    if abs(denom) < EPS:
        return e.copy()

    t = -cross2(edge, s - a) / denom

    return s + t * direction


def is_inside_edge(p, a, b):
    edge = b - a
    rel = p - a

    return cross2(edge, rel) >= -1e-9


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    output = np.asarray(subject_polygon, dtype=float).copy()
    clip = ensure_ccw(np.asarray(clip_polygon, dtype=float))

    if len(output) < 3:
        return np.empty((0, 2), dtype=float)

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]

        input_poly = output
        output_list = []

        if len(input_poly) == 0:
            break

        s = input_poly[-1]

        for e in input_poly:
            e_inside = is_inside_edge(e, a, b)
            s_inside = is_inside_edge(s, a, b)

            if e_inside:
                if not s_inside:
                    output_list.append(line_intersection_2d(s, e, a, b))
                output_list.append(e)

            elif s_inside:
                output_list.append(line_intersection_2d(s, e, a, b))

            s = e

        output = np.asarray(output_list, dtype=float)

    if len(output) < 3:
        return np.empty((0, 2), dtype=float)

    return output


def projection_matrix_to_plane(receiver_point, receiver_normal, ray_direction):
    q = np.asarray(receiver_point, dtype=float)
    n = unit(receiver_normal)
    d = unit(ray_direction)

    denom = float(np.dot(n, d))

    if abs(denom) < EPS:
        return None

    A = np.eye(3) - np.outer(d, n) / denom
    b = d * np.dot(n, q) / denom

    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = b

    return M


def project_points(points_3d, M):
    P = np.asarray(points_3d, dtype=float)
    P_h = np.column_stack([P, np.ones(len(P))])
    P_projected_h = P_h @ M.T

    return P_projected_h[:, :3]


# ============================================================
# Scene
# ============================================================

def generate_scene():
    scene = generate_urban_receiver_and_casters(
        n_polygons=N_SCENE_CASTERS,
        terrain_type=TERRAIN_TYPE,
        seed=SEED,
        built_density=BUILT_DENSITY,
        green_area_fraction=GREEN_AREA_FRACTION,
        average_building_height=AVERAGE_BUILDING_HEIGHT,
        height_lognorm_sigma=HEIGHT_LOGNORM_SIGMA,
        elevation_type=ELEVATION_TYPE,
        receiver_surface_kinds=RECEIVER_SURFACE_KINDS,
        include_same_building_casters=INCLUDE_SAME_BUILDING_CASTERS,
        verbose=True,
    )

    casters = list(scene.casters)
    casters.sort(key=lambda p: float(np.linalg.norm(p.center - scene.receiver.center)))

    return scene, casters


# ============================================================
# Lobe surrogate / hull functions
# ============================================================

def receiver_control_points(receiver):
    verts = np.asarray(receiver.vertices, dtype=float)
    centroid = np.mean(verts, axis=0, keepdims=True)
    return np.vstack([verts, centroid])


def caster_control_points(caster):
    verts = np.asarray(caster.vertices, dtype=float)
    centroid = np.mean(verts, axis=0, keepdims=True)
    return np.vstack([verts, centroid])


def angular_cloud_for_caster(receiver, caster):
    r_normal = unit(receiver.normal)

    receiver_az = horizontal_azimuth_from_vector(r_normal)
    if receiver_az is None:
        receiver_az = 0.0

    r_center = np.asarray(receiver.center, dtype=float)
    c_center = np.asarray(caster.center, dtype=float)

    center_vec = c_center - r_center
    dist = float(np.linalg.norm(center_vec))

    if dist < EPS:
        return None

    center_dir = center_vec / dist

    center_global_az, center_slope = vector_to_azimuth_slope(center_dir)
    center_rel_az = wrap_angle_180(center_global_az - receiver_az)

    front_factor = max(0.0, float(np.dot(r_normal, center_dir)))

    if ONLY_FRONT_HEMISPHERE and front_factor <= 0.0:
        return None

    r_pts = receiver_control_points(receiver)
    c_pts = caster_control_points(caster)

    cloud = []

    for rp in r_pts:
        for cp in c_pts:
            vec = cp - rp
            n = np.linalg.norm(vec)

            if n < EPS:
                continue

            az_g, slope = vector_to_azimuth_slope(vec)
            az_r = wrap_angle_180(az_g - receiver_az)

            # unwrap around caster center
            az_unwrapped = center_rel_az + wrap_angle_180(az_r - center_rel_az)

            cloud.append([az_unwrapped, slope])

    cloud = np.asarray(cloud, dtype=float)

    if len(cloud) < 3:
        return None

    caster_area = float(getattr(caster, "area", 1.0))
    score = (caster_area / max(dist ** 2, EPS)) * front_factor

    return {
        "cloud": cloud,
        "center_rel_azimuth_deg": float(center_rel_az),
        "center_slope_deg": float(center_slope),
        "distance": dist,
        "front_factor": front_factor,
        "score": score,
    }


def build_lobe_table(scene, casters):
    rows = []
    clouds = {}

    for caster in casters:
        result = angular_cloud_for_caster(scene.receiver, caster)

        if result is None:
            continue

        cloud = result["cloud"]
        clouds[caster.sid] = cloud

        rows.append({
            "caster_sid": caster.sid,
            "caster_object_id": caster.object_id,
            "caster_object_kind": caster.object_kind,
            "caster_surface_kind": caster.surface_kind,
            "distance": result["distance"],
            "front_factor": result["front_factor"],
            "score": result["score"],
            "center_rel_azimuth_deg": result["center_rel_azimuth_deg"],
            "center_slope_deg": result["center_slope_deg"],
            "azimuth_min_deg": float(np.min(cloud[:, 0])),
            "azimuth_max_deg": float(np.max(cloud[:, 0])),
            "slope_min_deg": float(np.min(cloud[:, 1])),
            "slope_max_deg": float(np.max(cloud[:, 1])),
            "azimuth_width_deg": float(np.max(cloud[:, 0]) - np.min(cloud[:, 0])),
            "slope_height_deg": float(np.max(cloud[:, 1]) - np.min(cloud[:, 1])),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df, clouds

    if CASTER_SORT_MODE == "distance":
        df = df.sort_values("distance", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df, clouds


# ============================================================
# Numeric heatmap
# ============================================================

def make_receiver_cache(receiver):
    q_B = receiver.center
    n_B = unit(receiver.normal)
    u_B, v_B = make_plane_basis(n_B)

    B_2d = points_3d_to_plane_2d(receiver.vertices, q_B, u_B, v_B)
    B_2d = ensure_ccw(B_2d)

    receiver_area = abs(polygon_area_2d(B_2d))

    receiver_az = horizontal_azimuth_from_vector(n_B)
    if receiver_az is None:
        receiver_az = 0.0

    return {
        "q_B": q_B,
        "n_B": n_B,
        "u_B": u_B,
        "v_B": v_B,
        "B_2d": B_2d,
        "receiver_area": receiver_area,
        "receiver_azimuth_deg": receiver_az,
    }


def compute_shadow_fraction_for_direction(receiver, receiver_cache, casters, sun_vector, slope_deg):
    q_B = receiver_cache["q_B"]
    n_B = receiver_cache["n_B"]
    u_B = receiver_cache["u_B"]
    v_B = receiver_cache["v_B"]
    B_2d = receiver_cache["B_2d"]
    receiver_area = receiver_cache["receiver_area"]

    if receiver_area < EPS:
        return 0.0, 0, False

    front_sun = bool(np.dot(n_B, sun_vector) > EPS)

    if slope_deg <= MIN_SUN_SLOPE_FOR_SHADING_DEG or not front_sun:
        return 0.0, 0, front_sun

    ray_direction = -sun_vector

    M = projection_matrix_to_plane(
        receiver_point=q_B,
        receiver_normal=n_B,
        ray_direction=ray_direction,
    )

    if M is None:
        return 0.0, 0, front_sun

    clipped_parts = []

    for caster in casters:
        rel = caster.center - receiver.center
        sunward_distance = float(np.dot(rel, sun_vector))

        if sunward_distance <= EPS:
            continue

        projected_3d = project_points(caster.vertices, M)

        projected_2d = points_3d_to_plane_2d(
            projected_3d,
            q_B,
            u_B,
            v_B,
        )

        if len(projected_2d) < 3:
            continue

        projected_2d = ensure_ccw(projected_2d)

        clipped_2d = sutherland_hodgman_clip(
            subject_polygon=projected_2d,
            clip_polygon=B_2d,
        )

        if len(clipped_2d) < 3:
            continue

        area = abs(polygon_area_2d(clipped_2d))

        if area < EPS:
            continue

        part = ShapelyPolygon(clipped_2d)

        if part.is_valid and not part.is_empty:
            clipped_parts.append(part)

    if not clipped_parts:
        return 0.0, 0, front_sun

    shadow_union = unary_union(clipped_parts)
    shaded_area = float(shadow_union.area)

    shaded_fraction = shaded_area / receiver_area
    shaded_fraction = max(0.0, min(1.0, shaded_fraction))

    return shaded_fraction, len(clipped_parts), front_sun


def compute_numeric_heatmap(scene, casters):
    rel_azimuths = np.linspace(
        REL_AZIMUTH_MIN_DEG,
        REL_AZIMUTH_MAX_DEG,
        P_AZIMUTH,
    )

    slopes = np.linspace(
        SLOPE_MIN_DEG,
        SLOPE_MAX_DEG,
        Q_SLOPE,
    )

    shading = np.zeros((Q_SLOPE, P_AZIMUTH), dtype=float)
    n_parts = np.zeros((Q_SLOPE, P_AZIMUTH), dtype=int)
    front_sun_mask = np.zeros((Q_SLOPE, P_AZIMUTH), dtype=bool)

    receiver_cache = make_receiver_cache(scene.receiver)
    receiver_az = receiver_cache["receiver_azimuth_deg"]

    total = Q_SLOPE * P_AZIMUTH
    done = 0
    t0 = time.perf_counter()

    for i, slope_deg in enumerate(slopes):
        for j, rel_az_deg in enumerate(rel_azimuths):
            global_az_deg = wrap_angle_360(receiver_az + rel_az_deg)

            sun_vector = sun_vector_from_azimuth_slope(
                global_azimuth_deg=global_az_deg,
                slope_deg=slope_deg,
            )

            shaded_fraction, parts, front_sun = compute_shadow_fraction_for_direction(
                receiver=scene.receiver,
                receiver_cache=receiver_cache,
                casters=casters,
                sun_vector=sun_vector,
                slope_deg=slope_deg,
            )

            shading[i, j] = shaded_fraction
            n_parts[i, j] = parts
            front_sun_mask[i, j] = front_sun

            done += 1

        print(f"Numeric heatmap row {i + 1}/{Q_SLOPE} | {done}/{total}")

    elapsed = time.perf_counter() - t0

    return {
        "rel_azimuths": rel_azimuths,
        "slopes": slopes,
        "shading": shading,
        "n_parts": n_parts,
        "front_sun_mask": front_sun_mask,
        "receiver_azimuth_deg": receiver_az,
        "elapsed_s": elapsed,
    }


# ============================================================
# Plotting
# ============================================================
def plot_lobe_surrogate_heatmap(scene, heatmap, lobe_heatmap):
    rel_azimuths = heatmap["rel_azimuths"]
    slopes = heatmap["slopes"]

    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        lobe_heatmap,
        origin="lower",
        aspect="auto",
        extent=[
            rel_azimuths.min(),
            rel_azimuths.max(),
            slopes.min(),
            slopes.max(),
        ],
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Surrogate shaded fraction [-]")

    ax.set_xlabel("Relative sun azimuth [deg]\n0 = receiver front normal")
    ax.set_ylabel("Sun slope / elevation angle [deg]")
    ax.set_title(
        f"Lobe-reconstructed heatmap | {scene.terrain_type} | "
        f"receiver={scene.receiver.surface_kind} | lobes={N_LOBES_FOR_SURROGATE}"
    )

    plt.tight_layout()

    if SAVE_OUTPUTS:
        plt.savefig(
            OUTPUT_DIR / "lobe_reconstructed_heatmap.png",
            dpi=DPI,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_lobe_reconstruction_error(scene, heatmap, lobe_heatmap):
    rel_azimuths = heatmap["rel_azimuths"]
    slopes = heatmap["slopes"]
    numeric = heatmap["shading"].copy()
    front = heatmap["front_sun_mask"]

    numeric[~front] = np.nan

    error = lobe_heatmap - numeric

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        error,
        origin="lower",
        aspect="auto",
        extent=[
            rel_azimuths.min(),
            rel_azimuths.max(),
            slopes.min(),
            slopes.max(),
        ],
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lobe surrogate - numeric [-]")

    ax.set_xlabel("Relative sun azimuth [deg]\n0 = receiver front normal")
    ax.set_ylabel("Sun slope / elevation angle [deg]")
    ax.set_title(
        f"Lobe reconstruction error | {scene.terrain_type} | "
        f"receiver={scene.receiver.surface_kind}"
    )

    plt.tight_layout()

    if SAVE_OUTPUTS:
        plt.savefig(
            OUTPUT_DIR / "lobe_reconstruction_error.png",
            dpi=DPI,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_lobe_surrogate_outputs(heatmap, lobe_heatmap, individual_lobes_df):
    front = heatmap["front_sun_mask"]
    numeric = heatmap["shading"].copy()
    numeric[~front] = np.nan

    error = lobe_heatmap - numeric
    abs_error = np.abs(error)

    np.save(OUTPUT_DIR / "lobe_reconstructed_grid.npy", lobe_heatmap)
    np.save(OUTPUT_DIR / "lobe_reconstruction_error.npy", error)

    matrix_df = pd.DataFrame(
        lobe_heatmap,
        index=[round(x, 3) for x in heatmap["slopes"]],
        columns=[round(x, 3) for x in heatmap["rel_azimuths"]],
    )
    matrix_df.index.name = "slope_deg"
    matrix_df.to_csv(OUTPUT_DIR / "lobe_reconstructed_matrix.csv")

    error_df = pd.DataFrame(
        error,
        index=[round(x, 3) for x in heatmap["slopes"]],
        columns=[round(x, 3) for x in heatmap["rel_azimuths"]],
    )
    error_df.index.name = "slope_deg"
    error_df.to_csv(OUTPUT_DIR / "lobe_reconstruction_error_matrix.csv")

    individual_lobes_df.to_csv(
        OUTPUT_DIR / "lobe_surrogate_individual_contributions.csv",
        index=False,
    )

    metrics = {
        "mae": float(np.nanmean(abs_error)),
        "rmse": float(np.sqrt(np.nanmean(error ** 2))),
        "max_abs_error": float(np.nanmax(abs_error)),
        "bias": float(np.nanmean(error)),
        "numeric_mean": float(np.nanmean(numeric)),
        "lobe_mean": float(np.nanmean(lobe_heatmap)),
        "numeric_max": float(np.nanmax(numeric)),
        "lobe_max": float(np.nanmax(lobe_heatmap)),
        "n_lobes_used": int(min(N_LOBES_FOR_SURROGATE, len(individual_lobes_df))),
        "lobe_amplitude_mode": LOBE_AMPLITUDE_MODE,
        "lobe_soft_sigma_deg": float(LOBE_SOFT_SIGMA_DEG),
    }

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "lobe_reconstruction_metrics.csv",
        index=False,
    )

    return metrics
def plot_lobe_hulls(scene, df_plot, clouds):
    fig, ax = plt.subplots(figsize=(11, 8))

    if ONLY_FRONT_HEMISPHERE:
        ax.set_xlim(-90.0, 90.0)
    else:
        ax.set_xlim(-180.0, 180.0)

    ax.set_ylim(0.0, 90.0)

    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Relative sun azimuth [deg]\n0 = receiver front normal")
    ax.set_ylabel("Sun slope / elevation [deg]")
    ax.set_title(
        f"Caster angular footprints | {scene.terrain_type} | "
        f"receiver={scene.receiver.surface_kind}"
    )
    ax.grid(True, alpha=0.25)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(df_plot), 1)))

    for idx, (_, row) in enumerate(df_plot.iterrows()):
        color = colors[idx % len(colors)]

        sid = int(row["caster_sid"])
        cloud = clouds[sid]

        cx = float(row["center_rel_azimuth_deg"])
        cy = float(row["center_slope_deg"])

        ax.arrow(
            0.0,
            0.0,
            cx,
            cy,
            linewidth=1.7,
            color=color,
            alpha=0.95,
            length_includes_head=True,
            head_width=2.2,
            head_length=3.0,
        )

        try:
            hull = ConvexHull(cloud)
            hull_points = cloud[hull.vertices]

            patch = MplPolygon(
                hull_points,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.28,
                linewidth=1.5,
            )

            ax.add_patch(patch)

        except Exception:
            pass

        ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=color, alpha=0.35)

        ax.scatter(
            [cx],
            [cy],
            s=40,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        ax.text(
            cx,
            cy,
            str(idx + 1),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            zorder=6,
        )

    plt.tight_layout()

    if SAVE_OUTPUTS:
        plt.savefig(OUTPUT_DIR / "caster_lobe_hulls.png", dpi=DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_numeric_heatmap(scene, heatmap):
    rel_azimuths = heatmap["rel_azimuths"]
    slopes = heatmap["slopes"]
    shading = heatmap["shading"]
    front_sun_mask = heatmap["front_sun_mask"]

    display = shading.copy()
    display[~front_sun_mask] = np.nan

    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[
            rel_azimuths.min(),
            rel_azimuths.max(),
            slopes.min(),
            slopes.max(),
        ],
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Shaded fraction [-]")

    ax.set_xlabel("Relative sun azimuth [deg]\n0 = receiver front normal")
    ax.set_ylabel("Sun slope / elevation angle [deg]")
    ax.set_title(
        f"Numeric shading heatmap | {scene.terrain_type} | "
        f"receiver={scene.receiver.surface_kind} | casters={N_SCENE_CASTERS}"
    )

    plt.tight_layout()

    if SAVE_OUTPUTS:
        plt.savefig(OUTPUT_DIR / "numeric_shading_heatmap.png", dpi=DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ============================================================
# Saving
# ============================================================

def save_numeric_outputs(scene, heatmap):
    rel_azimuths = heatmap["rel_azimuths"]
    slopes = heatmap["slopes"]
    shading = heatmap["shading"]
    front_sun_mask = heatmap["front_sun_mask"]
    n_parts = heatmap["n_parts"]

    display = shading.copy()
    display[~front_sun_mask] = np.nan

    np.save(OUTPUT_DIR / "numeric_shading_grid.npy", shading)
    np.save(OUTPUT_DIR / "front_sun_mask.npy", front_sun_mask)
    np.save(OUTPUT_DIR / "numeric_display_grid.npy", display)
    np.save(OUTPUT_DIR / "n_shadow_parts_grid.npy", n_parts)
    np.save(OUTPUT_DIR / "relative_azimuths.npy", rel_azimuths)
    np.save(OUTPUT_DIR / "slopes.npy", slopes)

    matrix_df = pd.DataFrame(
        display,
        index=[round(x, 3) for x in slopes],
        columns=[round(x, 3) for x in rel_azimuths],
    )
    matrix_df.index.name = "slope_deg"
    matrix_df.to_csv(OUTPUT_DIR / "numeric_shading_matrix.csv")

    rows = []

    for i, slope in enumerate(slopes):
        for j, rel_az in enumerate(rel_azimuths):
            rows.append({
                "terrain_type": scene.terrain_type,
                "receiver_sid": scene.receiver.sid,
                "receiver_surface_kind": scene.receiver.surface_kind,
                "relative_azimuth_deg": rel_az,
                "slope_deg": slope,
                "front_sun": bool(front_sun_mask[i, j]),
                "shaded_fraction": shading[i, j],
                "n_shadow_parts": int(n_parts[i, j]),
            })

    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUTPUT_DIR / "numeric_shading_long.csv", index=False)


def write_summary(scene, df_lobes, heatmap):
    display = heatmap["shading"].copy()
    display[~heatmap["front_sun_mask"]] = np.nan

    lines = []
    lines.append("Caster lobe + numeric heatmap summary")
    lines.append("=" * 80)
    lines.append(f"terrain_type: {scene.terrain_type}")
    lines.append(f"receiver_sid: {scene.receiver.sid}")
    lines.append(f"receiver_kind: {scene.receiver.object_kind} / {scene.receiver.surface_kind}")
    lines.append(f"receiver_center: {np.round(scene.receiver.center, 3)}")
    lines.append(f"receiver_normal: {np.round(scene.receiver.normal, 3)}")
    lines.append(f"receiver_area: {scene.receiver.area:.3f}")
    lines.append(f"n_scene_casters: {N_SCENE_CASTERS}")
    lines.append(f"n_casters_to_plot: {N_CASTERS_TO_PLOT}")
    lines.append(f"grid_size: {Q_SLOPE} x {P_AZIMUTH}")
    lines.append(f"numeric_heatmap_elapsed_s: {heatmap['elapsed_s']:.2f}")
    lines.append(f"max_shaded_fraction: {float(np.nanmax(display)):.4f}")
    lines.append(f"mean_shaded_fraction_front_domain: {float(np.nanmean(display)):.4f}")
    lines.append("")
    lines.append("Caster counts:")
    lines.append(str(Counter(p.surface_kind for p in scene.casters)))
    lines.append("")
    lines.append("Top lobe table:")
    lines.append(df_lobes.head(N_CASTERS_TO_PLOT).to_string(index=False))

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("CASTER LOBES + NUMERIC HEATMAP")
    print("=" * 100)

    scene, casters = generate_scene()

    print()
    print("Receiver:")
    print("  sid:", scene.receiver.sid)
    print("  object kind:", scene.receiver.object_kind)
    print("  surface kind:", scene.receiver.surface_kind)
    print("  center:", np.round(scene.receiver.center, 2))
    print("  normal:", np.round(scene.receiver.normal, 3))
    print("  area:", round(scene.receiver.area, 2))

    print()
    print("Caster counts:")
    print(Counter(p.surface_kind for p in casters))

    df_lobes, clouds = build_lobe_table(scene, casters)

    if len(df_lobes) == 0:
        print("No valid caster lobes found.")
        return

    df_plot = df_lobes.head(N_CASTERS_TO_PLOT).copy()

    if SAVE_OUTPUTS:
        df_lobes.to_csv(OUTPUT_DIR / "all_caster_lobes.csv", index=False)
        df_plot.to_csv(OUTPUT_DIR / "plotted_caster_lobes.csv", index=False)

    print()
    print("Top plotted lobes:")
    print(df_plot[[
        "caster_sid",
        "caster_surface_kind",
        "distance",
        "score",
        "center_rel_azimuth_deg",
        "center_slope_deg",
        "azimuth_width_deg",
        "slope_height_deg",
    ]].to_string(index=False))

    plot_lobe_hulls(scene, df_plot, clouds)

    print()
    print("Computing numeric Sutherland-Hodgman heatmap...")
    heatmap = compute_numeric_heatmap(scene, casters)

    plot_numeric_heatmap(scene, heatmap)
    print()
    print("Generating lobe-surrogate heatmap from vector/hull lobes...")

    lobe_heatmap, individual_lobes_df = generate_lobe_surrogate_heatmap(
        heatmap=heatmap,
        df_lobes=df_lobes,
        clouds=clouds,
    )

    plot_lobe_surrogate_heatmap(
        scene=scene,
        heatmap=heatmap,
        lobe_heatmap=lobe_heatmap,
    )

    plot_lobe_reconstruction_error(
        scene=scene,
        heatmap=heatmap,
        lobe_heatmap=lobe_heatmap,
    )

    if SAVE_OUTPUTS:
        metrics = save_lobe_surrogate_outputs(
            heatmap=heatmap,
            lobe_heatmap=lobe_heatmap,
            individual_lobes_df=individual_lobes_df,
        )

        print()
        print("Lobe surrogate reconstruction metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    if SAVE_OUTPUTS:
        save_numeric_outputs(scene, heatmap)
        write_summary(scene, df_lobes, heatmap)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Output folder:", OUTPUT_DIR)
    print("Numeric grid:", OUTPUT_DIR / "numeric_shading_grid.npy")
    print("Numeric matrix:", OUTPUT_DIR / "numeric_shading_matrix.csv")
    print("Heatmap image:", OUTPUT_DIR / "numeric_shading_heatmap.png")
    print("Lobe image:", OUTPUT_DIR / "caster_lobe_hulls.png")
    print("=" * 100)


if __name__ == "__main__":
    main()