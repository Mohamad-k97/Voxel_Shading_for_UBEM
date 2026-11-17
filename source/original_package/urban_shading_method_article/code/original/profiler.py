# polygon_day_test.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from generator import generate_urban_receiver_and_casters


# ============================================================
# USER SETTINGS
# ============================================================

# Scene settings
TERRAIN_TYPE = "tall_buildings"
N_CASTERS = 800
SEED = 88

BUILT_DENSITY = 0.30
GREEN_AREA_FRACTION = 0.35
AVERAGE_BUILDING_HEIGHT = 12.0
HEIGHT_LOGNORM_SIGMA = 0.35
ELEVATION_TYPE = "undulating"

RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",

)

INCLUDE_SAME_BUILDING_CASTERS = True

# Sun/location settings
YEAR = 2026
RANDOM_DAY_SEED = 123

LATITUDE_DEG = 45.4064      # Padova approx.
LONGITUDE_DEG = 11.8768     # Padova approx.
TIMEZONE_NAME = "Europe/Rome"

TIMESTEP_MINUTES = 15

# Shadow settings
MIN_SUN_ELEVATION_DEG = 0.0
PLOT_SUN_ELEVATION_TOO = True
SAVE_CSV = True
CSV_PATH = "polygon_day_profile.csv"


# ============================================================
# Geometry helpers
# ============================================================

EPS = 1e-10


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < EPS:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / n


def polygon_area_2d(points):
    p = np.asarray(points, dtype=float)

    if len(p) < 3:
        return 0.0

    x = p[:, 0]
    y = p[:, 1]

    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(points):
    p = np.asarray(points, dtype=float)

    if polygon_area_2d(p) < 0:
        return p[::-1].copy()

    return p


def make_plane_basis(normal):
    """
    Return u, v basis vectors for a plane.

    cross(u, v) = normal
    """
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
# Sutherland-Hodgman clipping
# ============================================================

def line_intersection_2d(s, e, a, b):
    """
    Intersection of segment line S->E with infinite clip edge line A->B.
    """
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
    """
    For CCW clip polygon, inside means point is on the left side
    of each directed edge.
    """
    edge = b - a
    rel = p - a

    return cross2(edge, rel) >= -1e-9


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Clip a polygon by a convex clip polygon.

    Parameters
    ----------
    subject_polygon:
        Nx2 array

    clip_polygon:
        Mx2 array, must be CCW and convex.

    Returns
    -------
    clipped polygon as Kx2 array.
    Empty if no intersection.
    """

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


# ============================================================
# Projection
# ============================================================

def projection_matrix_to_plane(receiver_point, receiver_normal, ray_direction):
    """
    Project points onto receiver plane along ray_direction.

    Plane:
        n · (x - q) = 0

    Projection:
        p' = p + t d
    """

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
# Solar position
# ============================================================

def solar_vector_noaa(local_dt, latitude_deg, longitude_deg):
    """
    Approximate solar vector using standard NOAA-style equations.

    Coordinate system:
        x = East
        y = North
        z = Up

    Returns
    -------
    sun_vector:
        unit vector pointing from surface point toward sun

    azimuth_deg:
        clockwise from North

    elevation_deg:
        degrees above horizon
    """

    lat = np.deg2rad(latitude_deg)

    doy = local_dt.timetuple().tm_yday

    decimal_hour = (
        local_dt.hour
        + local_dt.minute / 60.0
        + local_dt.second / 3600.0
    )

    utc_offset_hours = local_dt.utcoffset().total_seconds() / 3600.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (decimal_hour - 12.0) / 24.0)

    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    time_offset = eq_time + 4.0 * longitude_deg - 60.0 * utc_offset_hours
    true_solar_minutes = decimal_hour * 60.0 + time_offset

    # Normalize to [0, 1440)
    true_solar_minutes = true_solar_minutes % 1440.0

    hour_angle_deg = true_solar_minutes / 4.0 - 180.0

    if hour_angle_deg < -180.0:
        hour_angle_deg += 360.0

    H = np.deg2rad(hour_angle_deg)

    # ENU solar vector.
    east = -np.cos(decl) * np.sin(H)
    north = np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(H)
    up = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(H)

    sun_vector = unit(np.array([east, north, up], dtype=float))

    elevation_deg = float(np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0))))
    azimuth_deg = float((np.rad2deg(np.arctan2(east, north)) + 360.0) % 360.0)

    return sun_vector, azimuth_deg, elevation_deg


def random_day_datetime(year, seed, timezone_name):
    rng = np.random.default_rng(seed)

    # Non-leap-year friendly. For first tests, 365 is fine.
    day_of_year = int(rng.integers(1, 366))

    tz = ZoneInfo(timezone_name)

    dt = datetime(year, 1, 1, 0, 0, tzinfo=tz) + timedelta(days=day_of_year - 1)

    return dt


def make_day_timesteps(day_start, timestep_minutes):
    n_steps = int(24 * 60 / timestep_minutes)

    return [
        day_start + timedelta(minutes=i * timestep_minutes)
        for i in range(n_steps)
    ]


# ============================================================
# Shadow calculation for one timestep
# ============================================================

@dataclass
class ShadowStepResult:
    shaded_fraction: float
    shaded_area: float
    receiver_area: float
    n_clipped_parts: int
    front_sun: bool
    sun_elevation_deg: float
    sun_azimuth_deg: float


def compute_shadow_fraction_sutherland_for_sun(
    receiver,
    casters,
    sun_vector,
    sun_azimuth_deg,
    sun_elevation_deg,
):
    """
    Compute shaded fraction on receiver for one sun vector.

    Uses:
        projection onto receiver plane
        Sutherland-Hodgman clipping
        Shapely unary union for overlap-safe shaded area
    """

    q_B = receiver.center
    n_B = unit(receiver.normal)

    # Receiver ignores rear side.
    front_sun = bool(np.dot(n_B, sun_vector) > EPS)

    u_B, v_B = make_plane_basis(n_B)

    B_2d = points_3d_to_plane_2d(
        receiver.vertices,
        q_B,
        u_B,
        v_B,
    )

    B_2d = ensure_ccw(B_2d)
    receiver_area = abs(polygon_area_2d(B_2d))

    if receiver_area < EPS:
        return ShadowStepResult(
            shaded_fraction=0.0,
            shaded_area=0.0,
            receiver_area=0.0,
            n_clipped_parts=0,
            front_sun=False,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
        )

    if sun_elevation_deg <= MIN_SUN_ELEVATION_DEG or not front_sun:
        return ShadowStepResult(
            shaded_fraction=0.0,
            shaded_area=0.0,
            receiver_area=receiver_area,
            n_clipped_parts=0,
            front_sun=front_sun,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
        )

    # Light travels from sun to ground/object.
    # If sun_vector points from receiver to sun, incoming ray is -sun_vector.
    ray_direction = -sun_vector

    M = projection_matrix_to_plane(
        receiver_point=q_B,
        receiver_normal=n_B,
        ray_direction=ray_direction,
    )

    if M is None:
        return ShadowStepResult(
            shaded_fraction=0.0,
            shaded_area=0.0,
            receiver_area=receiver_area,
            n_clipped_parts=0,
            front_sun=front_sun,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
        )

    clipped_parts = []

    for caster in casters:
        # Physical prefilter:
        # caster must be sunward of receiver.
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

        # Projected caster polygon can be clockwise or CCW.
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

        clipped_parts.append(ShapelyPolygon(clipped_2d))

    if len(clipped_parts) == 0:
        shaded_area = 0.0
        n_clipped_parts = 0

    else:
        union = unary_union(clipped_parts)
        shaded_area = float(union.area)
        n_clipped_parts = len(clipped_parts)

    shaded_fraction = shaded_area / receiver_area
    shaded_fraction = max(0.0, min(1.0, shaded_fraction))

    return ShadowStepResult(
        shaded_fraction=shaded_fraction,
        shaded_area=shaded_area,
        receiver_area=receiver_area,
        n_clipped_parts=n_clipped_parts,
        front_sun=front_sun,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
    )


# ============================================================
# Daily profile
# ============================================================

def compute_polygon_day_profile(scene, day_start):
    times = make_day_timesteps(day_start, TIMESTEP_MINUTES)

    rows = []

    for dt in times:
        sun_vector, azimuth_deg, elevation_deg = solar_vector_noaa(
            local_dt=dt,
            latitude_deg=LATITUDE_DEG,
            longitude_deg=LONGITUDE_DEG,
        )

        result = compute_shadow_fraction_sutherland_for_sun(
            receiver=scene.receiver,
            casters=scene.casters,
            sun_vector=sun_vector,
            sun_azimuth_deg=azimuth_deg,
            sun_elevation_deg=elevation_deg,
        )

        rows.append({
            "datetime": dt,
            "hour": dt.hour + dt.minute / 60.0,
            "sun_azimuth_deg": result.sun_azimuth_deg,
            "sun_elevation_deg": result.sun_elevation_deg,
            "front_sun": result.front_sun,
            "n_clipped_shadow_parts": result.n_clipped_parts,
            "receiver_area": result.receiver_area,
            "shaded_area": result.shaded_area,
            "shaded_fraction": result.shaded_fraction,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_daily_profile(df, scene, day_start):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        df["hour"],
        df["shaded_fraction"],
        marker="o",
        linewidth=1.6,
        markersize=3,
        label="shaded fraction",
    )

    ax1.set_xlabel("Local time [h]")
    ax1.set_ylabel("Shaded fraction on receiver [-]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, 24)
    ax1.grid(True, alpha=0.3)

    title = (
        f"Polygon-day shading profile | "
        f"{day_start.date()} | "
        f"{scene.terrain_type} | "
        f"receiver={scene.receiver.surface_kind}"
    )

    ax1.set_title(title)

    if PLOT_SUN_ELEVATION_TOO:
        ax2 = ax1.twinx()

        ax2.plot(
            df["hour"],
            df["sun_elevation_deg"],
            linestyle="--",
            linewidth=1.2,
            label="sun elevation",
        )

        ax2.set_ylabel("Sun elevation [deg]")
        ax2.set_ylim(
            min(-10, df["sun_elevation_deg"].min() - 5),
            max(60, df["sun_elevation_deg"].max() + 5),
        )

    # Shade no-front-sun periods lightly.
    no_front = ~df["front_sun"].astype(bool)

    if no_front.any():
        for _, row in df[no_front].iterrows():
            ax1.axvspan(
                row["hour"],
                row["hour"] + TIMESTEP_MINUTES / 60.0,
                color="gray",
                alpha=0.08,
                linewidth=0,
            )

    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def print_scene_and_day_summary(scene, day_start, df):
    print()
    print("=" * 90)
    print("POLYGON-DAY TEST")
    print("=" * 90)

    print("Date:", day_start.date())
    print("Timezone:", TIMEZONE_NAME)
    print("Latitude:", LATITUDE_DEG)
    print("Longitude:", LONGITUDE_DEG)
    print("Timestep minutes:", TIMESTEP_MINUTES)

    print()
    print("Scene:")
    print("  terrain_type:", scene.terrain_type)
    print("  built_density target:", scene.config.built_density)
    print("  built_density actual:", round(scene.actual_built_density, 4))
    print("  green_area_fraction target:", scene.config.green_area_fraction)
    print("  green_area_fraction actual:", round(scene.actual_green_area_fraction, 4))
    print("  average_building_height:", scene.config.average_building_height)
    print("  elevation_type:", scene.config.elevation_type)
    print("  n_buildings:", scene.n_buildings)
    print("  n_trees:", scene.n_trees)
    print("  n_all_polygons:", len(scene.all_polygons))
    print("  n_casters:", len(scene.casters))

    print()
    print("Receiver:")
    print("  sid:", scene.receiver.sid)
    print("  object_kind:", scene.receiver.object_kind)
    print("  surface_kind:", scene.receiver.surface_kind)
    print("  center:", np.round(scene.receiver.center, 2))
    print("  normal:", np.round(scene.receiver.normal, 3))
    print("  area:", round(scene.receiver.area, 2))

    print()
    print("Caster types:")
    print(Counter(p.surface_kind for p in scene.casters))

    print()
    print("Daily profile summary:")
    print("  max shaded fraction:", round(float(df["shaded_fraction"].max()), 4))
    print("  mean shaded fraction:", round(float(df["shaded_fraction"].mean()), 4))
    print("  timesteps with front sun:", int(df["front_sun"].sum()))
    print("  timesteps with clipped shadow parts:", int((df["n_clipped_shadow_parts"] > 0).sum()))
    print("  max sun elevation:", round(float(df["sun_elevation_deg"].max()), 2))

    print("=" * 90)
    print()


# ============================================================
# Main
# ============================================================

def main():
    scene = generate_urban_receiver_and_casters(
        n_polygons=N_CASTERS,
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

    day_start = random_day_datetime(
        year=YEAR,
        seed=RANDOM_DAY_SEED,
        timezone_name=TIMEZONE_NAME,
    )

    df = compute_polygon_day_profile(
        scene=scene,
        day_start=day_start,
    )

    print_scene_and_day_summary(scene, day_start, df)

    if SAVE_CSV:
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved daily profile to: {CSV_PATH}")

    plot_daily_profile(df, scene, day_start)


if __name__ == "__main__":
    main()