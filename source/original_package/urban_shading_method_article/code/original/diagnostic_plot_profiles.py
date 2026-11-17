# diagnostic_plot_profiles.py

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import shutil

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generator import generate_urban_receiver_and_casters
from shading_profiles import (
    build_shading_profile,
    compare_raw_and_barrier_profiles,
)
from surface_array_builder import (
    build_surface_arrays_from_profile,
    surface_arrays_summary,
)
from barrier_proxy import barrier_profile_summary


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTPUT_DIR = Path("array_city_lobe_output") / "diagnostic_plots"

DEFAULT_SEED = 42
DEFAULT_TERRAIN_TYPE = "small_town"
DEFAULT_GENERATOR_N_POLYGONS = 250

DEFAULT_BUILT_DENSITY = 0.30
DEFAULT_GREEN_AREA_FRACTION = 0.25
DEFAULT_AVERAGE_BUILDING_HEIGHT = 12.0
DEFAULT_HEIGHT_LOGNORM_SIGMA = 0.30
DEFAULT_ELEVATION_TYPE = "undulating"

DEFAULT_DECOMPOSITION_MODE = "merged"
DEFAULT_BARRIER_VERTICAL_TOP_MODE = "eave"

DEFAULT_RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)

# User preference: no figure titles by default.
ADD_TITLES = False

FIG_DPI = 220


# ============================================================
# Colors
# ============================================================

SURFACE_COLOR = {
    "wall": "#9A9A9A",
    "flat_roof": "#C7B299",
    "pitched_roof": "#B8845A",
    "ground": "#B8D8A8",
    "paved_ground": "#D0D0D0",

    "tree_crown": "#4C9A2A",
    "tree_crown_top": "#6AB04A",
    "tree_trunk": "#8B5A2B",

    "barrier_wall_proxy": "#C0392B",
    "barrier_caster": "#C0392B",

    "receiver": "#2E86DE",
    "caster": "#E67E22",
}

OBJECT_COLOR = {
    "building": "#9A9A9A",
    "tree": "#4C9A2A",
    "ground": "#CFCFCF",
    "urban_object": "#A569BD",
}


# ============================================================
# Basic helpers
# ============================================================

def ensure_output_dir(path, overwrite=False):
    path = Path(path)

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def json_safe(value):
    if isinstance(value, dict):
        return dict((str(k), json_safe(v)) for k, v in value.items())

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )

    return path


def set_equal_xy_axes(ax, xmin, xmax, ymin, ymax, pad_fraction=0.04):
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)

    pad = pad_fraction * max(dx, dy, 1.0)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")


def get_xy_bounds_from_polygons(polygons):
    xs = []
    ys = []

    for p in polygons:
        v = np.asarray(p.vertices, dtype=float)

        if len(v) == 0:
            continue

        xs.extend(v[:, 0].tolist())
        ys.extend(v[:, 1].tolist())

    if not xs:
        return 0.0, 1.0, 0.0, 1.0

    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def polygon_xy(poly):
    v = np.asarray(poly.vertices, dtype=float)

    return v[:, :2]


def polygon_z_mean(poly):
    v = np.asarray(poly.vertices, dtype=float)

    if len(v) == 0:
        return 0.0

    return float(np.mean(v[:, 2]))


def sort_polygons_for_plan(polygons):
    """
    Draw low/ground surfaces first, higher surfaces later.
    """
    return sorted(
        list(polygons),
        key=lambda p: (
            polygon_z_mean(p),
            str(p.object_kind),
            str(p.surface_kind),
            int(p.sid),
        ),
    )


def surface_face_color(poly, fallback="#BBBBBB"):
    if poly.surface_kind in SURFACE_COLOR:
        return SURFACE_COLOR[poly.surface_kind]

    if poly.object_kind in OBJECT_COLOR:
        return OBJECT_COLOR[poly.object_kind]

    return fallback


def metadata_value(poly, key, default=None):
    if poly.metadata is None:
        return default

    return poly.metadata.get(key, default)


def is_barrier_proxy(poly):
    if poly.surface_kind in ("barrier_wall_proxy", "barrier_caster"):
        return True

    if metadata_value(poly, "profile_source", None) == "barrier_proxy":
        return True

    return False


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return path


def add_plan_polygon(
    ax,
    poly,
    facecolor,
    edgecolor="#222222",
    alpha=0.55,
    linewidth=0.35,
    zorder=1,
):
    xy = polygon_xy(poly)

    if len(xy) < 3:
        return None

    patch = MplPolygon(
        xy,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_patch(patch)

    return patch


def add_legend(ax, handles, loc="upper right"):
    if not handles:
        return

    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=9,
        frameon=True,
        framealpha=0.92,
    )


# ============================================================
# Scene/profile creation
# ============================================================

def create_diagnostic_scene(args):
    scene = generate_urban_receiver_and_casters(
        n_polygons=int(args.generator_n_polygons),
        terrain_type=args.terrain_type,
        seed=int(args.seed),
        built_density=float(args.built_density),
        green_area_fraction=float(args.green_area_fraction),
        average_building_height=float(args.average_building_height),
        height_lognorm_sigma=float(args.height_lognorm_sigma),
        elevation_type=args.elevation_type,
        receiver_surface_kinds=DEFAULT_RECEIVER_SURFACE_KINDS,
        include_same_building_casters=True,
        verbose=True,
    )

    return scene


def create_profiles(scene, args):
    raw_profile = build_shading_profile(
        scene=scene,
        profile_mode="raw",
        receiver_surface_kinds=DEFAULT_RECEIVER_SURFACE_KINDS,
        verbose=True,
    )

    proxy_profile = build_shading_profile(
        scene=scene,
        profile_mode="barrier_proxy",
        receiver_surface_kinds=DEFAULT_RECEIVER_SURFACE_KINDS,
        decomposition_mode=args.decomposition_mode,
        barrier_vertical_top_mode=args.barrier_vertical_top_mode,
        include_roof_casters=True,
        include_tree_casters=True,
        include_original_building_wall_casters=False,
        include_nonbuilding_casters=True,
        optimize_rotation=True,
        rotation_cost_mode="total_length",
        deduplicate_barriers=True,
        verbose=True,
    )

    return raw_profile, proxy_profile


# ============================================================
# Plot 1: physical scene plan
# ============================================================

def plot_physical_scene_plan(scene, output_path, add_titles=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    polygons = sort_polygons_for_plan(scene.all_polygons)

    for p in polygons:
        if p.object_kind == "ground":
            alpha = 0.35
            zorder = 1
        elif p.object_kind == "tree":
            alpha = 0.45
            zorder = 3
        else:
            alpha = 0.48
            zorder = 2

        add_plan_polygon(
            ax,
            p,
            facecolor=surface_face_color(p),
            edgecolor="#222222",
            alpha=alpha,
            linewidth=0.25,
            zorder=zorder,
        )

    # Mark selected receiver.
    rx = polygon_xy(scene.receiver)
    add_plan_polygon(
        ax,
        scene.receiver,
        facecolor=SURFACE_COLOR["receiver"],
        edgecolor="#003A70",
        alpha=0.85,
        linewidth=1.2,
        zorder=20,
    )

    xmin, xmax, ymin, ymax = get_xy_bounds_from_polygons(scene.all_polygons)
    set_equal_xy_axes(ax, xmin, xmax, ymin, ymax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if add_titles:
        ax.set_title("Physical generated scene")

    handles = [
        Patch(facecolor=OBJECT_COLOR["building"], edgecolor="#222222", label="Buildings"),
        Patch(facecolor=OBJECT_COLOR["tree"], edgecolor="#222222", label="Trees"),
        Patch(facecolor=OBJECT_COLOR["ground"], edgecolor="#222222", label="Ground"),
        Patch(facecolor=SURFACE_COLOR["receiver"], edgecolor="#003A70", label="Selected receiver"),
    ]
    add_legend(ax, handles)

    return save_figure(fig, output_path)


# ============================================================
# Plot 2: profile plan
# ============================================================

def plot_profile_plan(profile, output_path, add_titles=False, max_polygons=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    polygons = list(profile.polygons)

    if max_polygons is not None and len(polygons) > int(max_polygons):
        polygons = polygons[:int(max_polygons)]

    polygons = sort_polygons_for_plan(polygons)

    for p in polygons:
        if is_barrier_proxy(p):
            face = SURFACE_COLOR["barrier_wall_proxy"]
            edge = "#7B241C"
            alpha = 0.72
            lw = 0.55
            zorder = 12

        elif p.can_cast_shadow and p.can_receive_shadow:
            face = surface_face_color(p)
            edge = "#111111"
            alpha = 0.58
            lw = 0.35
            zorder = 6

        elif p.can_cast_shadow:
            face = SURFACE_COLOR["caster"]
            edge = "#7D3C00"
            alpha = 0.38
            lw = 0.25
            zorder = 5

        elif p.can_receive_shadow:
            face = SURFACE_COLOR["receiver"]
            edge = "#003A70"
            alpha = 0.40
            lw = 0.25
            zorder = 4

        else:
            face = "#DDDDDD"
            edge = "#999999"
            alpha = 0.15
            lw = 0.10
            zorder = 1

        add_plan_polygon(
            ax,
            p,
            facecolor=face,
            edgecolor=edge,
            alpha=alpha,
            linewidth=lw,
            zorder=zorder,
        )

    xmin, xmax, ymin, ymax = get_xy_bounds_from_polygons(profile.polygons)
    set_equal_xy_axes(ax, xmin, xmax, ymin, ymax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if add_titles:
        ax.set_title("{} shading profile".format(profile.profile_mode))

    handles = [
        Patch(facecolor=SURFACE_COLOR["receiver"], edgecolor="#003A70", label="Receivers"),
        Patch(facecolor=SURFACE_COLOR["caster"], edgecolor="#7D3C00", label="Casters"),
        Patch(facecolor=SURFACE_COLOR["barrier_wall_proxy"], edgecolor="#7B241C", label="Barrier proxies"),
    ]
    add_legend(ax, handles)

    return save_figure(fig, output_path)


# ============================================================
# Plot 3: footprint barriers
# ============================================================

def plot_barrier_footprints(proxy_profile, output_path, max_buildings=25, add_titles=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    profiles = list(proxy_profile.building_barrier_profiles)

    if max_buildings is not None:
        profiles = profiles[:int(max_buildings)]

    for bp in profiles:
        footprint = np.asarray(bp.footprint_xy, dtype=float)

        patch = MplPolygon(
            footprint,
            closed=True,
            facecolor="#D6DBDF",
            edgecolor="#2C3E50",
            linewidth=0.7,
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(patch)

        for piece in bp.convex_pieces_xy:
            piece = np.asarray(piece, dtype=float)

            patch_piece = MplPolygon(
                piece,
                closed=True,
                facecolor="none",
                edgecolor="#5D6D7E",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )
            ax.add_patch(patch_piece)

        for seg in bp.segments_xy:
            ax.plot(
                [seg.p0[0], seg.p1[0]],
                [seg.p0[1], seg.p1[1]],
                linewidth=1.1,
                alpha=0.92,
                zorder=5,
            )

        c = np.mean(footprint, axis=0)
        ax.text(
            c[0],
            c[1],
            str(bp.object_id),
            fontsize=6,
            ha="center",
            va="center",
            zorder=10,
        )

    if profiles:
        all_polys = []
        for bp in profiles:
            class Dummy(object):
                pass

            d = Dummy()
            verts = np.zeros((len(bp.footprint_xy), 3), dtype=float)
            verts[:, :2] = bp.footprint_xy
            d.vertices = verts
            all_polys.append(d)

        xmin, xmax, ymin, ymax = get_xy_bounds_from_polygons(all_polys)
    else:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    set_equal_xy_axes(ax, xmin, xmax, ymin, ymax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if add_titles:
        ax.set_title("Footprint decomposition and barrier segments")

    handles = [
        Patch(facecolor="#D6DBDF", edgecolor="#2C3E50", label="Building footprint"),
        Patch(facecolor="none", edgecolor="#5D6D7E", label="Convex pieces"),
        Patch(facecolor="#C0392B", edgecolor="#C0392B", label="Barrier segments"),
    ]
    add_legend(ax, handles)

    return save_figure(fig, output_path)


# ============================================================
# Plot 4: selected building barrier detail
# ============================================================

def plot_single_barrier_profile(bp, output_path, add_titles=False):
    fig, ax = plt.subplots(figsize=(7, 7))

    footprint = np.asarray(bp.footprint_xy, dtype=float)

    patch = MplPolygon(
        footprint,
        closed=True,
        facecolor="#D6DBDF",
        edgecolor="#2C3E50",
        linewidth=1.2,
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(patch)

    for i, piece in enumerate(bp.convex_pieces_xy):
        piece = np.asarray(piece, dtype=float)

        patch_piece = MplPolygon(
            piece,
            closed=True,
            facecolor="none",
            edgecolor="#2874A6",
            linewidth=0.95,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(patch_piece)

        c = np.mean(piece, axis=0)
        ax.text(
            c[0],
            c[1],
            "P{}".format(i),
            fontsize=8,
            ha="center",
            va="center",
            zorder=8,
        )

    for i, seg in enumerate(bp.segments_xy):
        ax.plot(
            [seg.p0[0], seg.p1[0]],
            [seg.p0[1], seg.p1[1]],
            linewidth=1.8,
            alpha=0.95,
            zorder=5,
        )

        m = 0.5 * (np.asarray(seg.p0) + np.asarray(seg.p1))
        ax.text(
            m[0],
            m[1],
            "S{}".format(i),
            fontsize=7,
            ha="center",
            va="center",
            zorder=9,
        )

    xmin = float(np.min(footprint[:, 0]))
    xmax = float(np.max(footprint[:, 0]))
    ymin = float(np.min(footprint[:, 1]))
    ymax = float(np.max(footprint[:, 1]))

    set_equal_xy_axes(ax, xmin, xmax, ymin, ymax, pad_fraction=0.12)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if add_titles:
        ax.set_title("Building {} barrier profile".format(bp.object_id))

    return save_figure(fig, output_path)


# ============================================================
# Plot 5: 3D profile sample
# ============================================================

def poly3d_face_color(poly):
    if is_barrier_proxy(poly):
        return SURFACE_COLOR["barrier_wall_proxy"]

    return surface_face_color(poly)


def plot_profile_3d_sample(
    profile,
    output_path,
    max_objects=35,
    add_titles=False,
):
    """
    Small 3D sample for sanity check.

    This is not for huge scenes. It takes a subset of low object_id surfaces.
    """
    object_ids = sorted(set(int(p.object_id) for p in profile.polygons))
    keep_ids = set(object_ids[:int(max_objects)])

    polygons = [p for p in profile.polygons if int(p.object_id) in keep_ids]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    verts3d = []
    colors = []
    alphas = []

    for p in polygons:
        v = np.asarray(p.vertices, dtype=float)

        if len(v) < 3:
            continue

        verts3d.append(v)
        colors.append(poly3d_face_color(p))

        if is_barrier_proxy(p):
            alphas.append(0.78)
        elif p.can_receive_shadow and not p.can_cast_shadow:
            alphas.append(0.35)
        else:
            alphas.append(0.52)

    collection = Poly3DCollection(
        verts3d,
        linewidths=0.25,
        edgecolors="#222222",
    )

    collection.set_facecolor(colors)
    collection.set_alpha(0.58)
    ax.add_collection3d(collection)

    if verts3d:
        allv = np.vstack(verts3d)

        xmin = float(np.min(allv[:, 0]))
        xmax = float(np.max(allv[:, 0]))
        ymin = float(np.min(allv[:, 1]))
        ymax = float(np.max(allv[:, 1]))
        zmin = float(np.min(allv[:, 2]))
        zmax = float(np.max(allv[:, 2]))

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        span = max(dx, dy, dz, 1.0)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
        ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
        ax.set_zlim(max(0.0, cz - 0.35 * span), cz + 0.65 * span)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.view_init(elev=28, azim=-55)

    if add_titles:
        ax.set_title("{} profile, 3D sample".format(profile.profile_mode))

    return save_figure(fig, output_path)


# ============================================================
# Plot 6: count comparison
# ============================================================

def plot_profile_count_comparison(raw_profile, proxy_profile, output_path, add_titles=False):
    raw = raw_profile.stats
    proxy = proxy_profile.stats

    labels = [
        "Profile polygons",
        "Casters",
        "Receivers",
        "Barrier surfaces",
    ]

    raw_values = [
        raw["n_profile_polygons"],
        raw["n_casters"],
        raw["n_receivers"],
        0,
    ]

    proxy_values = [
        proxy["n_profile_polygons"],
        proxy["n_casters"],
        proxy["n_receivers"],
        proxy["n_barrier_surfaces"],
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(x - width / 2.0, raw_values, width, label="Raw")
    ax.bar(x + width / 2.0, proxy_values, width, label="Barrier proxy")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count")

    if add_titles:
        ax.set_title("Raw vs barrier-proxy profile counts")

    ax.legend(fontsize=9)

    fig.tight_layout()

    return save_figure(fig, output_path)


# ============================================================
# Diagnostics summary
# ============================================================

def build_diagnostic_summary(scene, raw_profile, proxy_profile):
    raw_arrays = build_surface_arrays_from_profile(raw_profile, include_vertices=False)
    proxy_arrays = build_surface_arrays_from_profile(proxy_profile, include_vertices=False)

    barrier_summaries = [
        barrier_profile_summary(bp)
        for bp in proxy_profile.building_barrier_profiles[:50]
    ]

    summary = {
        "scene": {
            "terrain_type": scene.terrain_type,
            "area_size": float(scene.area_size),
            "n_all_polygons": int(len(scene.all_polygons)),
            "n_buildings": 0 if scene.buildings is None else int(len(scene.buildings)),
            "n_trees": 0 if scene.n_trees is None else int(scene.n_trees),
            "actual_built_density": scene.actual_built_density,
            "actual_green_area_fraction": scene.actual_green_area_fraction,
            "receiver_sid": int(scene.receiver.sid),
            "receiver_surface_kind": str(scene.receiver.surface_kind),
            "receiver_object_kind": str(scene.receiver.object_kind),
        },
        "raw_profile": raw_profile.stats,
        "proxy_profile": proxy_profile.stats,
        "raw_arrays": surface_arrays_summary(raw_arrays),
        "proxy_arrays": surface_arrays_summary(proxy_arrays),
        "first_barrier_profiles": barrier_summaries,
    }

    return summary


def print_diagnostic_summary(summary):
    print()
    print("=" * 100)
    print("DIAGNOSTIC PROFILE SUMMARY")
    print("=" * 100)

    print("scene polygons:", summary["scene"]["n_all_polygons"])
    print("buildings:", summary["scene"]["n_buildings"])
    print("trees:", summary["scene"]["n_trees"])
    print("receiver:", summary["scene"]["receiver_sid"], summary["scene"]["receiver_surface_kind"])

    print()
    print("raw profile polygons:", summary["raw_profile"]["n_profile_polygons"])
    print("raw casters:", summary["raw_profile"]["n_casters"])
    print("raw receivers:", summary["raw_profile"]["n_receivers"])

    print()
    print("proxy profile polygons:", summary["proxy_profile"]["n_profile_polygons"])
    print("proxy casters:", summary["proxy_profile"]["n_casters"])
    print("proxy receivers:", summary["proxy_profile"]["n_receivers"])
    print("proxy barrier surfaces:", summary["proxy_profile"]["n_barrier_surfaces"])
    print("proxy barrier segments:", summary["proxy_profile"]["n_barrier_segments"])

    print()
    print("raw array memory MB:", round(summary["raw_arrays"]["memory_mb"], 4))
    print("proxy array memory MB:", round(summary["proxy_arrays"]["memory_mb"], 4))

    print("=" * 100)
    print()


# ============================================================
# Main run
# ============================================================

def run_diagnostics(args):
    output_dir = ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    print()
    print("=" * 100)
    print("DIAGNOSTIC PROFILE PLOTS")
    print("=" * 100)
    print("output_dir:", output_dir)
    print("=" * 100)
    print()

    scene = create_diagnostic_scene(args)
    raw_profile, proxy_profile = create_profiles(scene, args)

    output_paths = {}

    output_paths["physical_scene_plan"] = plot_physical_scene_plan(
        scene,
        output_dir / "01_physical_scene_plan.png",
        add_titles=args.add_titles,
    )

    output_paths["raw_profile_plan"] = plot_profile_plan(
        raw_profile,
        output_dir / "02_raw_profile_plan.png",
        add_titles=args.add_titles,
        max_polygons=args.max_plan_polygons,
    )

    output_paths["barrier_proxy_profile_plan"] = plot_profile_plan(
        proxy_profile,
        output_dir / "03_barrier_proxy_profile_plan.png",
        add_titles=args.add_titles,
        max_polygons=args.max_plan_polygons,
    )

    output_paths["barrier_footprints"] = plot_barrier_footprints(
        proxy_profile,
        output_dir / "04_barrier_footprints.png",
        max_buildings=args.max_barrier_buildings,
        add_titles=args.add_titles,
    )

    if proxy_profile.building_barrier_profiles:
        # Prefer a non-rectangular footprint if available.
        chosen_bp = proxy_profile.building_barrier_profiles[0]

        for bp in proxy_profile.building_barrier_profiles:
            if bp.n_input_vertices > 4 or bp.n_convex_pieces > 1:
                chosen_bp = bp
                break

        output_paths["single_barrier_profile"] = plot_single_barrier_profile(
            chosen_bp,
            output_dir / "05_single_barrier_profile.png",
            add_titles=args.add_titles,
        )

    output_paths["barrier_proxy_3d_sample"] = plot_profile_3d_sample(
        proxy_profile,
        output_dir / "06_barrier_proxy_3d_sample.png",
        max_objects=args.max_3d_objects,
        add_titles=args.add_titles,
    )

    output_paths["profile_count_comparison"] = plot_profile_count_comparison(
        raw_profile,
        proxy_profile,
        output_dir / "07_profile_count_comparison.png",
        add_titles=args.add_titles,
    )

    summary = build_diagnostic_summary(scene, raw_profile, proxy_profile)

    summary["output_paths"] = dict((k, str(v)) for k, v in output_paths.items())

    write_json(output_dir / "diagnostic_summary.json", summary)
    write_markdown_index(output_dir / "diagnostic_index.md", summary, output_paths)

    print_diagnostic_summary(summary)

    print("Wrote diagnostic plots:")
    for name, path in output_paths.items():
        print("  {}: {}".format(name, path))

    print()
    print("Wrote:", output_dir / "diagnostic_summary.json")
    print("Wrote:", output_dir / "diagnostic_index.md")
    print()

    return {
        "scene": scene,
        "raw_profile": raw_profile,
        "proxy_profile": proxy_profile,
        "summary": summary,
        "output_paths": output_paths,
    }


def write_markdown_index(path, summary, output_paths):
    path = Path(path)

    lines = []

    lines.append("# Diagnostic profile plots")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- scene polygons: `{}`".format(summary["scene"]["n_all_polygons"]))
    lines.append("- buildings: `{}`".format(summary["scene"]["n_buildings"]))
    lines.append("- trees: `{}`".format(summary["scene"]["n_trees"]))
    lines.append("- raw casters: `{}`".format(summary["raw_profile"]["n_casters"]))
    lines.append("- proxy casters: `{}`".format(summary["proxy_profile"]["n_casters"]))
    lines.append("- proxy barrier surfaces: `{}`".format(summary["proxy_profile"]["n_barrier_surfaces"]))
    lines.append("- proxy barrier segments: `{}`".format(summary["proxy_profile"]["n_barrier_segments"]))
    lines.append("")
    lines.append("## Figures")
    lines.append("")

    for name, p in output_paths.items():
        p = Path(p)
        rel = p.name
        lines.append("### {}".format(name.replace("_", " ")))
        lines.append("")
        lines.append("![]({})".format(rel))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


# ============================================================
# CLI
# ============================================================

def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).lower().strip()

    if value in ("1", "true", "yes", "y", "on"):
        return True

    if value in ("0", "false", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError("Expected true/false, yes/no, or 1/0.")


def none_or_int(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in ("none", "null", ""):
        return None

    return int(text)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create diagnostic plots for raw and barrier-proxy shading profiles.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--terrain-type",
        type=str,
        default=DEFAULT_TERRAIN_TYPE,
    )

    parser.add_argument(
        "--generator-n-polygons",
        type=int,
        default=DEFAULT_GENERATOR_N_POLYGONS,
    )

    parser.add_argument(
        "--built-density",
        type=float,
        default=DEFAULT_BUILT_DENSITY,
    )

    parser.add_argument(
        "--green-area-fraction",
        type=float,
        default=DEFAULT_GREEN_AREA_FRACTION,
    )

    parser.add_argument(
        "--average-building-height",
        type=float,
        default=DEFAULT_AVERAGE_BUILDING_HEIGHT,
    )

    parser.add_argument(
        "--height-lognorm-sigma",
        type=float,
        default=DEFAULT_HEIGHT_LOGNORM_SIGMA,
    )

    parser.add_argument(
        "--elevation-type",
        type=str,
        default=DEFAULT_ELEVATION_TYPE,
    )

    parser.add_argument(
        "--decomposition-mode",
        choices=("merged", "triangles"),
        default=DEFAULT_DECOMPOSITION_MODE,
    )

    parser.add_argument(
        "--barrier-vertical-top-mode",
        choices=("eave", "top", "max"),
        default=DEFAULT_BARRIER_VERTICAL_TOP_MODE,
    )

    parser.add_argument(
        "--max-plan-polygons",
        type=none_or_int,
        default=None,
        help="Limit drawn profile polygons in plan plots. None draws all.",
    )

    parser.add_argument(
        "--max-barrier-buildings",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--max-3d-objects",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--add-titles",
        type=str_to_bool,
        default=ADD_TITLES,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)

    return run_diagnostics(args)


if __name__ == "__main__":
    main()