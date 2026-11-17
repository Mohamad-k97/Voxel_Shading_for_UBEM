# shading_profiles.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from generator import (
    UrbanPolygon,
    UrbanScene,
    BuildingRecord,
    make_polygon,
)

from barrier_proxy import (
    BarrierProfile,
    BarrierSurfaceSpec,
    build_barrier_profile_for_footprint,
    barrier_profile_summary,
)


# ============================================================
# Constants / defaults
# ============================================================

EPS = 1e-10

BUILDING_ROOF_KINDS = (
    "flat_roof",
    "pitched_roof",
)

BUILDING_WALL_KINDS = (
    "wall",
)

TREE_CASTER_KINDS = (
    "tree_crown",
    "tree_crown_top",
    "tree_trunk",
)

DEFAULT_RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class ShadingProfile:
    """
    Output of shading-profile preprocessing.

    The key idea:
        receiver geometry = real physical surfaces
        caster geometry   = either raw physical surfaces or proxy caster surfaces

    polygons:
        all profile polygons, with profile-specific can_cast/can_receive flags

    casters:
        subset of polygons where can_cast_shadow is True

    receivers:
        subset of polygons where can_receive_shadow is True
    """

    profile_mode: str
    scene: UrbanScene

    polygons: List[UrbanPolygon]
    casters: List[UrbanPolygon]
    receivers: List[UrbanPolygon]

    building_barrier_profiles: List[BarrierProfile]

    stats: Dict
    metadata: Dict


# ============================================================
# Basic helpers
# ============================================================

def _max_sid(polygons):
    if not polygons:
        return 0

    return max(int(p.sid) for p in polygons)


def _copy_metadata(metadata):
    if metadata is None:
        return {}

    out = {}

    for k, v in metadata.items():
        out[k] = v

    return out


def clone_urban_polygon(
    poly,
    sid=None,
    object_id=None,
    object_kind=None,
    surface_kind=None,
    can_cast_shadow=None,
    can_receive_shadow=None,
    opacity=None,
    metadata_extra=None,
):
    """
    Clone an UrbanPolygon with profile-specific cast/receive flags.

    This avoids the dangerous case where a physical wall is kept as a receiver
    but accidentally still casts shadows in barrier-proxy mode.
    """
    if sid is None:
        sid = poly.sid

    if object_id is None:
        object_id = poly.object_id

    if object_kind is None:
        object_kind = poly.object_kind

    if surface_kind is None:
        surface_kind = poly.surface_kind

    if can_cast_shadow is None:
        can_cast_shadow = poly.can_cast_shadow

    if can_receive_shadow is None:
        can_receive_shadow = poly.can_receive_shadow

    if opacity is None:
        opacity = poly.opacity

    metadata = _copy_metadata(poly.metadata)
    metadata["profile_source"] = "original_surface"
    metadata["original_sid"] = int(poly.sid)

    if metadata_extra is not None:
        metadata.update(metadata_extra)

    return make_polygon(
        sid=int(sid),
        object_id=int(object_id),
        object_kind=str(object_kind),
        surface_kind=str(surface_kind),
        vertices=np.asarray(poly.vertices, dtype=float).copy(),
        desired_normal=np.asarray(poly.normal, dtype=float),
        can_receive_shadow=bool(can_receive_shadow),
        can_cast_shadow=bool(can_cast_shadow),
        opacity=float(opacity),
        footprint_xy=None if poly.footprint_xy is None else np.asarray(poly.footprint_xy, dtype=float).copy(),
        roof_type=poly.roof_type,
        base_z=poly.base_z,
        eave_z=poly.eave_z,
        top_z=poly.top_z,
        metadata=metadata,
    )


def barrier_surface_spec_to_urban_polygon(
    spec,
    sid,
    building_record=None,
):
    """
    Convert BarrierSurfaceSpec from barrier_proxy.py into UrbanPolygon.
    """
    metadata = {
        "profile_source": "barrier_proxy",
        "role": "barrier_wall_proxy",
        "two_sided": bool(spec.two_sided),
        "piece_index": int(spec.piece_index),
        "segment_index": int(spec.segment_index),
        "source": str(spec.source),
    }

    footprint_xy = None
    roof_type = None
    base_z = None
    eave_z = None
    top_z = None

    if building_record is not None:
        footprint_xy = building_record.footprint_xy
        roof_type = building_record.roof_type
        base_z = building_record.base_z
        eave_z = building_record.eave_z
        top_z = building_record.top_z

        metadata.update({
            "building_footprint_kind": building_record.footprint_kind,
            "building_roof_type": building_record.roof_type,
            "building_height": float(building_record.height),
            "building_roof_height": float(building_record.roof_height),
        })

    return make_polygon(
        sid=int(sid),
        object_id=int(spec.object_id),
        object_kind=str(spec.object_kind),
        surface_kind=str(spec.surface_kind),
        vertices=np.asarray(spec.vertices, dtype=float).copy(),
        desired_normal=np.asarray(spec.normal, dtype=float),
        can_receive_shadow=bool(spec.can_receive_shadow),
        can_cast_shadow=bool(spec.can_cast_shadow),
        opacity=float(spec.opacity),
        footprint_xy=footprint_xy,
        roof_type=roof_type,
        base_z=base_z,
        eave_z=eave_z,
        top_z=top_z,
        metadata=metadata,
    )


def is_receiver_surface(poly, receiver_surface_kinds):
    return (
        bool(poly.can_receive_shadow)
        and poly.surface_kind in receiver_surface_kinds
    )


def is_building_roof(poly):
    return (
        poly.object_kind == "building"
        and poly.surface_kind in BUILDING_ROOF_KINDS
    )


def is_building_wall(poly):
    return (
        poly.object_kind == "building"
        and poly.surface_kind in BUILDING_WALL_KINDS
    )


def is_tree_caster(poly):
    return (
        poly.object_kind == "tree"
        and poly.surface_kind in TREE_CASTER_KINDS
        and bool(poly.can_cast_shadow)
    )


def polygon_sort_key(poly):
    return (
        int(poly.object_id),
        str(poly.object_kind),
        str(poly.surface_kind),
        int(poly.sid),
    )


# ============================================================
# Building-record fallback
# ============================================================

def _infer_building_records_from_scene(scene):
    """
    Fallback for scenes that have building polygons but no scene.buildings list.

    New generator.py already writes scene.buildings, so normally this is not used.
    """
    if scene.buildings is not None and len(scene.buildings) > 0:
        return list(scene.buildings)

    groups = {}

    for p in scene.all_polygons:
        if p.object_kind != "building":
            continue

        groups.setdefault(int(p.object_id), []).append(p)

    records = []

    for object_id, polys in groups.items():
        footprint_xy = None

        for p in polys:
            if p.footprint_xy is not None:
                footprint_xy = np.asarray(p.footprint_xy, dtype=float).copy()
                break

        if footprint_xy is None:
            # Last-resort inference: collect x-y points at lower z values.
            all_vertices = np.vstack([p.vertices for p in polys])
            z_min = float(np.min(all_vertices[:, 2]))
            near_base = all_vertices[np.abs(all_vertices[:, 2] - z_min) < 1e-7]

            if len(near_base) < 3:
                continue

            # This is not a true convex hull implementation. It is only a
            # fallback for old files. New scenes should have footprint_xy.
            center = np.mean(near_base[:, :2], axis=0)
            angles = np.arctan2(near_base[:, 1] - center[1], near_base[:, 0] - center[0])
            order = np.argsort(angles)
            footprint_xy = near_base[order, :2]

        all_vertices = np.vstack([p.vertices for p in polys])
        base_z = float(np.min(all_vertices[:, 2]))
        top_z = float(np.max(all_vertices[:, 2]))

        roof_polys = [p for p in polys if p.surface_kind in BUILDING_ROOF_KINDS]
        wall_polys = [p for p in polys if p.surface_kind in BUILDING_WALL_KINDS]

        if wall_polys:
            wall_vertices = np.vstack([p.vertices for p in wall_polys])
            eave_z = float(np.max(wall_vertices[:, 2]))
        else:
            eave_z = top_z

        roof_type = None

        for p in roof_polys:
            if p.roof_type is not None:
                roof_type = p.roof_type
                break

        if roof_type is None:
            roof_type = "unknown"

        center_xy = np.mean(np.asarray(footprint_xy, dtype=float), axis=0)
        radius_xy = float(np.max(np.linalg.norm(np.asarray(footprint_xy) - center_xy, axis=1)))

        record = BuildingRecord(
            object_id=int(object_id),
            footprint_xy=np.asarray(footprint_xy, dtype=float),
            center_xy=center_xy,
            base_z=base_z,
            eave_z=eave_z,
            top_z=top_z,
            height=float(eave_z - base_z),
            roof_type=str(roof_type),
            roof_height=float(top_z - eave_z),
            rotation_rad=0.0,
            footprint_kind="inferred",
            wall_sids=[int(p.sid) for p in wall_polys],
            roof_sids=[int(p.sid) for p in roof_polys],
            footprint_area=0.0,
            radius_xy=radius_xy,
        )

        records.append(record)

    return records


def _building_record_by_object_id(scene):
    records = _infer_building_records_from_scene(scene)

    out = {}

    for b in records:
        out[int(b.object_id)] = b

    return out


# ============================================================
# Profile filtering
# ============================================================

def _build_raw_profile_polygons(
    scene,
    receiver_surface_kinds,
):
    """
    Raw mode:
        castable surfaces = original physical castable surfaces
        receivers         = original physical receivable surfaces
    """
    profile_polygons = []

    for p in scene.all_polygons:
        can_receive = is_receiver_surface(p, receiver_surface_kinds)
        can_cast = bool(p.can_cast_shadow)

        if not can_receive and not can_cast:
            continue

        profile_polygons.append(
            clone_urban_polygon(
                p,
                can_cast_shadow=can_cast,
                can_receive_shadow=can_receive,
                metadata_extra={
                    "profile_mode": "raw",
                },
            )
        )

    return profile_polygons


def _barrier_top_z_for_building(building, barrier_vertical_top_mode):
    """
    Decide how high the vertical footprint barrier should be.

    Options:
        "eave":
            vertical proxy from base_z to eave_z.
            Best physical interpretation when roof surfaces are also included.

        "top":
            vertical proxy from base_z to top_z.
            More conservative for complex pitched roof volumes.

        "max":
            alias for "top".
    """
    mode = str(barrier_vertical_top_mode).lower().strip()

    if mode == "eave":
        return float(building.eave_z)

    if mode == "top" or mode == "max":
        return float(building.top_z)

    raise ValueError(
        "barrier_vertical_top_mode must be 'eave', 'top', or 'max'."
    )


def _should_keep_original_in_barrier_mode(
    poly,
    receiver_surface_kinds,
    include_roof_casters,
    include_tree_casters,
    include_original_building_wall_casters,
    include_nonbuilding_casters,
):
    """
    Return profile-specific can_cast/can_receive flags for an original surface.
    """
    can_receive = is_receiver_surface(poly, receiver_surface_kinds)
    can_cast = False

    if poly.object_kind == "building":
        if is_building_roof(poly):
            can_cast = bool(include_roof_casters and poly.can_cast_shadow)

        elif is_building_wall(poly):
            can_cast = bool(include_original_building_wall_casters and poly.can_cast_shadow)

        else:
            can_cast = False

    elif poly.object_kind == "tree":
        can_cast = bool(include_tree_casters and poly.can_cast_shadow)

    else:
        can_cast = bool(include_nonbuilding_casters and poly.can_cast_shadow)

    return can_cast, can_receive


def _build_barrier_proxy_profile_polygons(
    scene,
    receiver_surface_kinds,
    decomposition_mode,
    barrier_vertical_top_mode,
    include_roof_casters,
    include_tree_casters,
    include_original_building_wall_casters,
    include_nonbuilding_casters,
    optimize_rotation,
    rotation_cost_mode,
    deduplicate_barriers,
):
    """
    Barrier-proxy mode:
        building casters =
            real roof surfaces
            + vertical barrier proxy surfaces from footprint

        building receivers =
            real physical receiver surfaces

        trees =
            original tree caster proxies, not receivers

        ground =
            receivers only, unless include_nonbuilding_casters says otherwise
    """
    profile_polygons = []
    building_barrier_profiles = []

    building_by_id = _building_record_by_object_id(scene)

    # 1. Keep original physical surfaces only with corrected profile flags.
    for p in scene.all_polygons:
        can_cast, can_receive = _should_keep_original_in_barrier_mode(
            poly=p,
            receiver_surface_kinds=receiver_surface_kinds,
            include_roof_casters=include_roof_casters,
            include_tree_casters=include_tree_casters,
            include_original_building_wall_casters=include_original_building_wall_casters,
            include_nonbuilding_casters=include_nonbuilding_casters,
        )

        if not can_cast and not can_receive:
            continue

        profile_polygons.append(
            clone_urban_polygon(
                p,
                can_cast_shadow=can_cast,
                can_receive_shadow=can_receive,
                metadata_extra={
                    "profile_mode": "barrier_proxy",
                    "barrier_role": "original_receiver_or_roof_caster",
                },
            )
        )

    # 2. Add barrier proxy surfaces for every building.
    next_sid = _max_sid(scene.all_polygons) + 1

    for object_id in sorted(building_by_id.keys()):
        b = building_by_id[object_id]

        top_z = _barrier_top_z_for_building(
            building=b,
            barrier_vertical_top_mode=barrier_vertical_top_mode,
        )

        if top_z <= float(b.base_z) + EPS:
            continue

        try:
            barrier_profile = build_barrier_profile_for_footprint(
                footprint_xy=b.footprint_xy,
                base_z=b.base_z,
                top_z=top_z,
                object_id=b.object_id,
                decomposition_mode=decomposition_mode,
                optimize_rotation=optimize_rotation,
                rotation_cost_mode=rotation_cost_mode,
                deduplicate=deduplicate_barriers,
                surface_kind="barrier_wall_proxy",
                object_kind="building",
                opacity=1.0,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to build barrier profile for building object_id={}: {}".format(
                    b.object_id,
                    e,
                )
            )

        building_barrier_profiles.append(barrier_profile)

        for spec in barrier_profile.surfaces:
            poly = barrier_surface_spec_to_urban_polygon(
                spec=spec,
                sid=next_sid,
                building_record=b,
            )

            profile_polygons.append(poly)
            next_sid += 1

    return profile_polygons, building_barrier_profiles


# ============================================================
# Deduplication / extraction
# ============================================================

def _deduplicate_profile_polygons(polygons):
    """
    Deduplicate by sid.

    If a proxy accidentally shares an sid with original geometry, keep the last.
    New proxy sids should normally be above original max sid, so no conflict.
    """
    by_sid = {}

    for p in polygons:
        by_sid[int(p.sid)] = p

    return [by_sid[k] for k in sorted(by_sid.keys())]


def _extract_casters_receivers(polygons):
    casters = [p for p in polygons if bool(p.can_cast_shadow)]
    receivers = [p for p in polygons if bool(p.can_receive_shadow)]

    casters.sort(key=polygon_sort_key)
    receivers.sort(key=polygon_sort_key)

    return casters, receivers


# ============================================================
# Statistics
# ============================================================

def _count_by_attr(polygons, attr):
    counts = {}

    for p in polygons:
        key = getattr(p, attr)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _sum_area(polygons):
    return float(sum(float(p.area) for p in polygons))


def _build_profile_stats(
    scene,
    profile_mode,
    polygons,
    casters,
    receivers,
    building_barrier_profiles,
    elapsed_s,
):
    original_polygons = list(scene.all_polygons)

    n_original_castable = sum(1 for p in original_polygons if p.can_cast_shadow)
    n_original_receivable = sum(1 for p in original_polygons if p.can_receive_shadow)

    n_barrier_surfaces = sum(
        int(profile.n_surfaces)
        for profile in building_barrier_profiles
    )

    n_barrier_segments = sum(
        int(profile.n_segments)
        for profile in building_barrier_profiles
    )

    total_barrier_proxy_area = sum(
        float(profile.total_proxy_area)
        for profile in building_barrier_profiles
    )

    stats = {
        "profile_mode": str(profile_mode),
        "elapsed_s": float(elapsed_s),

        "n_scene_polygons": int(len(original_polygons)),
        "n_profile_polygons": int(len(polygons)),

        "n_original_castable": int(n_original_castable),
        "n_original_receivable": int(n_original_receivable),

        "n_casters": int(len(casters)),
        "n_receivers": int(len(receivers)),

        "profile_area_total": _sum_area(polygons),
        "caster_area_total": _sum_area(casters),
        "receiver_area_total": _sum_area(receivers),

        "n_buildings": 0 if scene.buildings is None else int(len(scene.buildings)),
        "n_barrier_profiles": int(len(building_barrier_profiles)),
        "n_barrier_surfaces": int(n_barrier_surfaces),
        "n_barrier_segments": int(n_barrier_segments),
        "total_barrier_proxy_area": float(total_barrier_proxy_area),

        "surface_kind_counts_all": _count_by_attr(polygons, "surface_kind"),
        "surface_kind_counts_casters": _count_by_attr(casters, "surface_kind"),
        "surface_kind_counts_receivers": _count_by_attr(receivers, "surface_kind"),

        "object_kind_counts_all": _count_by_attr(polygons, "object_kind"),
        "object_kind_counts_casters": _count_by_attr(casters, "object_kind"),
        "object_kind_counts_receivers": _count_by_attr(receivers, "object_kind"),
    }

    return stats


def print_shading_profile_summary(profile):
    s = profile.stats

    print()
    print("=" * 90)
    print("SHADING PROFILE SUMMARY")
    print("=" * 90)
    print("profile_mode:", s["profile_mode"])
    print("elapsed_s:", round(s["elapsed_s"], 6))
    print()
    print("scene polygons:", s["n_scene_polygons"])
    print("profile polygons:", s["n_profile_polygons"])
    print("original castable:", s["n_original_castable"])
    print("original receivable:", s["n_original_receivable"])
    print()
    print("casters:", s["n_casters"])
    print("receivers:", s["n_receivers"])
    print()
    print("barrier profiles:", s["n_barrier_profiles"])
    print("barrier segments:", s["n_barrier_segments"])
    print("barrier surfaces:", s["n_barrier_surfaces"])
    print("total barrier proxy area:", round(s["total_barrier_proxy_area"], 3))
    print()
    print("surface kinds, all:")
    print(s["surface_kind_counts_all"])
    print("surface kinds, casters:")
    print(s["surface_kind_counts_casters"])
    print("surface kinds, receivers:")
    print(s["surface_kind_counts_receivers"])
    print()
    print("object kinds, casters:")
    print(s["object_kind_counts_casters"])
    print("object kinds, receivers:")
    print(s["object_kind_counts_receivers"])
    print("=" * 90)
    print()


# ============================================================
# Public API
# ============================================================

def build_shading_profile(
    scene,
    profile_mode="barrier_proxy",
    receiver_surface_kinds=None,
    decomposition_mode="merged",
    barrier_vertical_top_mode="eave",
    include_roof_casters=True,
    include_tree_casters=True,
    include_original_building_wall_casters=False,
    include_nonbuilding_casters=True,
    optimize_rotation=True,
    rotation_cost_mode="total_length",
    deduplicate_barriers=True,
    verbose=False,
):
    """
    Build caster/receiver profiles from a generated UrbanScene.

    Parameters
    ----------
    profile_mode:
        "raw":
            original physical surfaces are both casters and receivers according
            to their original flags.

        "barrier_proxy":
            building walls stop being caster surfaces.
            Instead, each building footprint is converted to barrier-proxy
            vertical caster surfaces. Roof surfaces remain real casters.

    receiver_surface_kinds:
        Which real physical surfaces are allowed as receivers.

    barrier_vertical_top_mode:
        "eave":
            barrier walls go from base_z to eave_z.
            This is the best physical match when roof surfaces are also casters.

        "top":
            barrier walls go from base_z to top_z.
            More conservative for complex roof volumes.
    """
    if receiver_surface_kinds is None:
        receiver_surface_kinds = DEFAULT_RECEIVER_SURFACE_KINDS

    receiver_surface_kinds = tuple(receiver_surface_kinds)

    profile_mode = str(profile_mode).lower().strip()

    t0 = time.perf_counter()

    if profile_mode == "raw":
        polygons = _build_raw_profile_polygons(
            scene=scene,
            receiver_surface_kinds=receiver_surface_kinds,
        )
        building_barrier_profiles = []

    elif profile_mode == "barrier_proxy":
        polygons, building_barrier_profiles = _build_barrier_proxy_profile_polygons(
            scene=scene,
            receiver_surface_kinds=receiver_surface_kinds,
            decomposition_mode=decomposition_mode,
            barrier_vertical_top_mode=barrier_vertical_top_mode,
            include_roof_casters=include_roof_casters,
            include_tree_casters=include_tree_casters,
            include_original_building_wall_casters=include_original_building_wall_casters,
            include_nonbuilding_casters=include_nonbuilding_casters,
            optimize_rotation=optimize_rotation,
            rotation_cost_mode=rotation_cost_mode,
            deduplicate_barriers=deduplicate_barriers,
        )

    else:
        raise ValueError(
            "profile_mode must be 'raw' or 'barrier_proxy'."
        )

    polygons = _deduplicate_profile_polygons(polygons)
    casters, receivers = _extract_casters_receivers(polygons)

    elapsed_s = time.perf_counter() - t0

    stats = _build_profile_stats(
        scene=scene,
        profile_mode=profile_mode,
        polygons=polygons,
        casters=casters,
        receivers=receivers,
        building_barrier_profiles=building_barrier_profiles,
        elapsed_s=elapsed_s,
    )

    metadata = {
        "profile_mode": profile_mode,
        "receiver_surface_kinds": list(receiver_surface_kinds),
        "decomposition_mode": decomposition_mode,
        "barrier_vertical_top_mode": barrier_vertical_top_mode,
        "include_roof_casters": bool(include_roof_casters),
        "include_tree_casters": bool(include_tree_casters),
        "include_original_building_wall_casters": bool(include_original_building_wall_casters),
        "include_nonbuilding_casters": bool(include_nonbuilding_casters),
        "optimize_rotation": bool(optimize_rotation),
        "rotation_cost_mode": str(rotation_cost_mode),
        "deduplicate_barriers": bool(deduplicate_barriers),
    }

    profile = ShadingProfile(
        profile_mode=profile_mode,
        scene=scene,
        polygons=polygons,
        casters=casters,
        receivers=receivers,
        building_barrier_profiles=building_barrier_profiles,
        stats=stats,
        metadata=metadata,
    )

    if verbose:
        print_shading_profile_summary(profile)

    return profile


def build_raw_shading_profile(
    scene,
    receiver_surface_kinds=None,
    verbose=False,
):
    return build_shading_profile(
        scene=scene,
        profile_mode="raw",
        receiver_surface_kinds=receiver_surface_kinds,
        verbose=verbose,
    )


def build_barrier_proxy_shading_profile(
    scene,
    receiver_surface_kinds=None,
    decomposition_mode="merged",
    barrier_vertical_top_mode="eave",
    verbose=False,
):
    return build_shading_profile(
        scene=scene,
        profile_mode="barrier_proxy",
        receiver_surface_kinds=receiver_surface_kinds,
        decomposition_mode=decomposition_mode,
        barrier_vertical_top_mode=barrier_vertical_top_mode,
        verbose=verbose,
    )


# ============================================================
# Comparison helper
# ============================================================

def compare_raw_and_barrier_profiles(
    scene,
    receiver_surface_kinds=None,
    decomposition_mode="merged",
    barrier_vertical_top_mode="eave",
    verbose=True,
):
    raw = build_raw_shading_profile(
        scene=scene,
        receiver_surface_kinds=receiver_surface_kinds,
        verbose=False,
    )

    proxy = build_barrier_proxy_shading_profile(
        scene=scene,
        receiver_surface_kinds=receiver_surface_kinds,
        decomposition_mode=decomposition_mode,
        barrier_vertical_top_mode=barrier_vertical_top_mode,
        verbose=False,
    )

    comparison = {
        "raw_n_polygons": raw.stats["n_profile_polygons"],
        "proxy_n_polygons": proxy.stats["n_profile_polygons"],

        "raw_n_casters": raw.stats["n_casters"],
        "proxy_n_casters": proxy.stats["n_casters"],

        "raw_n_receivers": raw.stats["n_receivers"],
        "proxy_n_receivers": proxy.stats["n_receivers"],

        "proxy_n_barrier_surfaces": proxy.stats["n_barrier_surfaces"],
        "proxy_n_barrier_segments": proxy.stats["n_barrier_segments"],

        "caster_count_delta": proxy.stats["n_casters"] - raw.stats["n_casters"],
        "caster_count_ratio": (
            float(proxy.stats["n_casters"]) / max(1.0, float(raw.stats["n_casters"]))
        ),

        "receiver_count_delta": proxy.stats["n_receivers"] - raw.stats["n_receivers"],
        "receiver_count_ratio": (
            float(proxy.stats["n_receivers"]) / max(1.0, float(raw.stats["n_receivers"]))
        ),

        "raw_elapsed_s": raw.stats["elapsed_s"],
        "proxy_elapsed_s": proxy.stats["elapsed_s"],
    }

    if verbose:
        print()
        print("=" * 90)
        print("RAW VS BARRIER-PROXY PROFILE COMPARISON")
        print("=" * 90)
        for k in sorted(comparison.keys()):
            v = comparison[k]

            if isinstance(v, float):
                print(k + ":", round(v, 6))
            else:
                print(k + ":", v)
        print("=" * 90)
        print()

    return raw, proxy, comparison


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    from generator import generate_urban_receiver_and_casters

    scene = generate_urban_receiver_and_casters(
        n_polygons=120,
        terrain_type="small_town",
        seed=42,
        built_density=0.30,
        green_area_fraction=0.25,
        average_building_height=12.0,
        height_lognorm_sigma=0.30,
        elevation_type="undulating",
        receiver_surface_kinds=DEFAULT_RECEIVER_SURFACE_KINDS,
        include_same_building_casters=True,
        verbose=True,
    )

    raw_profile, proxy_profile, comparison = compare_raw_and_barrier_profiles(
        scene=scene,
        receiver_surface_kinds=DEFAULT_RECEIVER_SURFACE_KINDS,
        decomposition_mode="merged",
        barrier_vertical_top_mode="eave",
        verbose=True,
    )

    print("First 5 proxy casters:")
    for p in proxy_profile.casters[:5]:
        print(
            "sid",
            p.sid,
            "object",
            p.object_id,
            "kind",
            p.object_kind,
            "surface",
            p.surface_kind,
            "can_cast",
            p.can_cast_shadow,
            "can_receive",
            p.can_receive_shadow,
        )

    print()
    print("First 3 barrier profile summaries:")
    for bp in proxy_profile.building_barrier_profiles[:3]:
        print(barrier_profile_summary(bp))