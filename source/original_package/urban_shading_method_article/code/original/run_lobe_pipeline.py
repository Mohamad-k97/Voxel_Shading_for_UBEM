# run_lobe_pipeline.py

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
import shutil

import numpy as np

import array_gen


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTPUT_ROOT = Path("array_city_lobe_runs")

DEFAULT_PROFILE_MODE = "barrier_proxy"  # "raw", "barrier_proxy", or "compare"

DEFAULT_TERRAIN_TYPE = "small_town"
DEFAULT_SEED = 42
DEFAULT_GENERATOR_N_POLYGONS = 1000

DEFAULT_BUILT_DENSITY = 0.30
DEFAULT_GREEN_AREA_FRACTION = 0.25
DEFAULT_AVERAGE_BUILDING_HEIGHT = 12.0
DEFAULT_HEIGHT_LOGNORM_SIGMA = 0.30
DEFAULT_ELEVATION_TYPE = "undulating"

DEFAULT_DECOMPOSITION_MODE = "merged"
DEFAULT_BARRIER_VERTICAL_TOP_MODE = "eave"

DEFAULT_TOP_K_LOBES = 40

DEFAULT_CORE_TILE_SIZE_M = 250.0
DEFAULT_HALO_RADIUS_M = 350.0
DEFAULT_GRID_CELL_SIZE_M = 250.0

DEFAULT_NUMBA_THREADS = 4

DEFAULT_RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)


# ============================================================
# Utility helpers
# ============================================================

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


def read_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_size_mb(path):
    path = Path(path)

    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024.0 * 1024.0)


def dir_size_mb(path):
    path = Path(path)

    if not path.exists():
        return 0.0

    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total / (1024.0 * 1024.0)


def ensure_clean_dir(path, overwrite=False):
    path = Path(path)

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


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


def none_or_float(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in ("none", "null", ""):
        return None

    return float(text)


# ============================================================
# Configure array_gen globals
# ============================================================

def apply_config_to_array_gen(config):
    """
    array_gen.py is the actual pipeline engine.
    This runner sets array_gen's module-level settings before calling it.

    This keeps array_gen.py runnable by itself, while this file becomes the
    cleaner command-line entry point.
    """
    array_gen.OUTPUT_DIR = Path(config["output_dir"])

    array_gen.PROFILE_MODE = str(config["profile_mode"])

    array_gen.TERRAIN_TYPE = str(config["terrain_type"])
    array_gen.SEED = int(config["seed"])
    array_gen.GENERATOR_N_POLYGONS = int(config["generator_n_polygons"])

    array_gen.BUILT_DENSITY = config["built_density"]
    array_gen.GREEN_AREA_FRACTION = config["green_area_fraction"]
    array_gen.AVERAGE_BUILDING_HEIGHT = config["average_building_height"]
    array_gen.HEIGHT_LOGNORM_SIGMA = config["height_lognorm_sigma"]
    array_gen.ELEVATION_TYPE = config["elevation_type"]

    array_gen.RECEIVER_SURFACE_KINDS = tuple(config["receiver_surface_kinds"])

    array_gen.DECOMPOSITION_MODE = str(config["decomposition_mode"])
    array_gen.BARRIER_VERTICAL_TOP_MODE = str(config["barrier_vertical_top_mode"])

    array_gen.INCLUDE_ROOF_CASTERS = bool(config["include_roof_casters"])
    array_gen.INCLUDE_TREE_CASTERS = bool(config["include_tree_casters"])
    array_gen.INCLUDE_ORIGINAL_BUILDING_WALL_CASTERS = bool(config["include_original_building_wall_casters"])
    array_gen.INCLUDE_NONBUILDING_CASTERS = bool(config["include_nonbuilding_casters"])

    array_gen.OPTIMIZE_BARRIER_ROTATION = bool(config["optimize_barrier_rotation"])
    array_gen.BARRIER_ROTATION_COST_MODE = str(config["barrier_rotation_cost_mode"])
    array_gen.DEDUPLICATE_BARRIERS = bool(config["deduplicate_barriers"])

    array_gen.SAVE_SURFACE_ARRAYS = bool(config["save_surface_arrays"])
    array_gen.SAVE_SURFACE_ARRAYS_COMPRESSED = bool(config["save_surface_arrays_compressed"])
    array_gen.INCLUDE_COMPACT_VERTICES = bool(config["include_compact_vertices"])

    array_gen.CORE_TILE_SIZE_M = float(config["core_tile_size_m"])
    array_gen.HALO_RADIUS_M = float(config["halo_radius_m"])
    array_gen.GRID_CELL_SIZE_M = float(config["grid_cell_size_m"])

    array_gen.TOP_K_LOBES = int(config["top_k_lobes"])

    array_gen.INCLUDE_SAME_BUILDING_CASTERS = bool(config["include_same_building_casters"])
    array_gen.ONLY_FRONT_HEMISPHERE = bool(config["only_front_hemisphere"])
    array_gen.USE_OPACITY_IN_SCORE = bool(config["use_opacity_in_score"])

    array_gen.NUMBA_REQUESTED_THREADS = config["numba_threads"]

    array_gen.RESUME_EXISTING_TILES = bool(config["resume_existing_tiles"])
    array_gen.OVERWRITE_OUTPUT_DIR = bool(config["overwrite"])
    array_gen.COMPRESS_TILE_NPZ = bool(config["compress_tile_npz"])

    array_gen.PRINT_EMPTY_TILES = bool(config["print_empty_tiles"])
    array_gen.PRINT_EVERY_TILE = bool(config["print_every_tile"])
    array_gen.WRITE_MANIFEST = bool(config["write_manifest"])


# ============================================================
# Run helpers
# ============================================================

def build_base_config(args):
    output_root = Path(args.output_root)

    config = {
        "output_root": str(output_root),
        "output_dir": str(output_root / args.profile_mode),

        "profile_mode": args.profile_mode,

        "terrain_type": args.terrain_type,
        "seed": args.seed,
        "generator_n_polygons": args.generator_n_polygons,

        "built_density": args.built_density,
        "green_area_fraction": args.green_area_fraction,
        "average_building_height": args.average_building_height,
        "height_lognorm_sigma": args.height_lognorm_sigma,
        "elevation_type": args.elevation_type,

        "receiver_surface_kinds": list(DEFAULT_RECEIVER_SURFACE_KINDS),

        "decomposition_mode": args.decomposition_mode,
        "barrier_vertical_top_mode": args.barrier_vertical_top_mode,

        "include_roof_casters": args.include_roof_casters,
        "include_tree_casters": args.include_tree_casters,
        "include_original_building_wall_casters": args.include_original_building_wall_casters,
        "include_nonbuilding_casters": args.include_nonbuilding_casters,

        "optimize_barrier_rotation": args.optimize_barrier_rotation,
        "barrier_rotation_cost_mode": args.barrier_rotation_cost_mode,
        "deduplicate_barriers": args.deduplicate_barriers,

        "save_surface_arrays": args.save_surface_arrays,
        "save_surface_arrays_compressed": args.save_surface_arrays_compressed,
        "include_compact_vertices": args.include_compact_vertices,

        "core_tile_size_m": args.core_tile_size_m,
        "halo_radius_m": args.halo_radius_m,
        "grid_cell_size_m": args.grid_cell_size_m,

        "top_k_lobes": args.top_k_lobes,

        "include_same_building_casters": args.include_same_building_casters,
        "only_front_hemisphere": args.only_front_hemisphere,
        "use_opacity_in_score": args.use_opacity_in_score,

        "numba_threads": args.numba_threads,

        "resume_existing_tiles": args.resume_existing_tiles,
        "overwrite": args.overwrite,
        "compress_tile_npz": args.compress_tile_npz,

        "print_empty_tiles": args.print_empty_tiles,
        "print_every_tile": args.print_every_tile,
        "write_manifest": args.write_manifest,
    }

    return config


def run_one_profile(config):
    profile_mode = str(config["profile_mode"])

    output_dir = Path(config["output_dir"])
    ensure_clean_dir(output_dir, overwrite=bool(config["overwrite"]))

    write_json(output_dir / "run_config.json", config)

    print()
    print("=" * 100)
    print("RUN LOBE PIPELINE")
    print("=" * 100)
    print("profile mode:", profile_mode)
    print("output dir:  ", output_dir)
    print("=" * 100)
    print()

    apply_config_to_array_gen(config)

    t0 = time.perf_counter()

    result = array_gen.run_full_pipeline()

    elapsed_s = time.perf_counter() - t0

    manifest_path = output_dir / "pipeline_manifest.json"

    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {}

    run_summary = build_run_summary(
        config=config,
        result=result,
        manifest=manifest,
        elapsed_s=elapsed_s,
    )

    write_json(output_dir / "run_summary.json", run_summary)

    print()
    print("=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)
    print("profile mode:      ", profile_mode)
    print("output dir:        ", output_dir)
    print("elapsed_s:         ", round(elapsed_s, 3))
    print("profile polygons:  ", run_summary["profile"].get("n_profile_polygons"))
    print("casters:           ", run_summary["profile"].get("n_casters"))
    print("receivers:         ", run_summary["profile"].get("n_receivers"))
    print("barrier surfaces:  ", run_summary["profile"].get("n_barrier_surfaces"))
    print("array memory MB:   ", run_summary["surface_arrays"].get("memory_mb"))
    print("tile size MB:      ", run_summary["lobe_cache"].get("total_tile_size_mb"))
    print("=" * 100)
    print()

    return run_summary


def build_run_summary(config, result, manifest, elapsed_s):
    profile_stats = {}

    if "profile" in manifest:
        profile_stats = manifest["profile"]
    elif "profile" in result:
        profile_stats = result["profile"].stats

    surface_arrays = {}

    if "surface_arrays" in manifest:
        surface_arrays = manifest["surface_arrays"]
    elif "arrays" in result:
        from surface_array_builder import surface_arrays_summary
        surface_arrays = surface_arrays_summary(result["arrays"])

    lobe_cache = {}

    if "lobe_cache" in manifest:
        lobe_cache = manifest["lobe_cache"]
    elif "lobe_result" in result:
        tile_index = result["lobe_result"].get("tile_index", {})
        lobe_cache = {
            "tile_output_dir": str(result["lobe_result"].get("tile_output_dir")),
            "n_tile_files": tile_index.get("n_tile_files"),
            "n_receiver_rows": tile_index.get("n_receiver_rows"),
            "total_compute_elapsed_s": tile_index.get("total_compute_elapsed_s"),
            "total_loop_elapsed_s": tile_index.get("total_loop_elapsed_s"),
            "total_tile_size_mb": tile_index.get("total_tile_size_mb"),
        }

    output_dir = Path(config["output_dir"])

    summary = {
        "profile_mode": str(config["profile_mode"]),
        "output_dir": str(output_dir),
        "elapsed_s": float(elapsed_s),
        "config": config,
        "profile": profile_stats,
        "surface_arrays": surface_arrays,
        "lobe_cache": lobe_cache,
        "file_sizes": {
            "output_dir_mb": dir_size_mb(output_dir),
            "surface_arrays_npz_mb": file_size_mb(output_dir / "surface_arrays.npz"),
            "pipeline_manifest_json_mb": file_size_mb(output_dir / "pipeline_manifest.json"),
            "pipeline_report_md_mb": file_size_mb(output_dir / "pipeline_report.md"),
        },
    }

    return summary


def run_compare(config):
    root = Path(config["output_root"])
    ensure_clean_dir(root, overwrite=bool(config["overwrite"]))

    raw_config = dict(config)
    raw_config["profile_mode"] = "raw"
    raw_config["output_dir"] = str(root / "raw")

    proxy_config = dict(config)
    proxy_config["profile_mode"] = "barrier_proxy"
    proxy_config["output_dir"] = str(root / "barrier_proxy")

    # When comparing, avoid deleting the root twice.
    raw_config["overwrite"] = bool(config["overwrite"])
    proxy_config["overwrite"] = bool(config["overwrite"])

    raw_summary = run_one_profile(raw_config)
    proxy_summary = run_one_profile(proxy_config)

    comparison = build_comparison_summary(
        raw_summary=raw_summary,
        proxy_summary=proxy_summary,
        output_root=root,
    )

    write_json(root / "comparison_summary.json", comparison)
    write_comparison_markdown(root / "comparison_report.md", comparison)

    print()
    print("=" * 100)
    print("RAW VS BARRIER-PROXY COMPARISON")
    print("=" * 100)
    print("raw casters:          ", comparison["caster_counts"]["raw"])
    print("proxy casters:        ", comparison["caster_counts"]["barrier_proxy"])
    print("caster ratio:         ", round(comparison["caster_counts"]["proxy_over_raw"], 4))
    print("raw receivers:        ", comparison["receiver_counts"]["raw"])
    print("proxy receivers:      ", comparison["receiver_counts"]["barrier_proxy"])
    print("receiver ratio:       ", round(comparison["receiver_counts"]["proxy_over_raw"], 4))
    print("raw total elapsed s:  ", round(comparison["timing"]["raw_elapsed_s"], 3))
    print("proxy total elapsed s:", round(comparison["timing"]["proxy_elapsed_s"], 3))
    print("raw tile size MB:     ", round(comparison["storage"]["raw_tile_size_mb"], 4))
    print("proxy tile size MB:   ", round(comparison["storage"]["proxy_tile_size_mb"], 4))
    print("wrote:                ", root / "comparison_summary.json")
    print("wrote:                ", root / "comparison_report.md")
    print("=" * 100)
    print()

    return comparison


def ratio_safe(a, b):
    b = float(b)

    if abs(b) <= 1e-12:
        return None

    return float(a) / b


def value_from(summary, section, key, default=0.0):
    try:
        value = summary.get(section, {}).get(key, default)

        if value is None:
            return default

        return value
    except Exception:
        return default


def build_comparison_summary(raw_summary, proxy_summary, output_root):
    raw_profile_polygons = value_from(raw_summary, "profile", "n_profile_polygons", 0)
    proxy_profile_polygons = value_from(proxy_summary, "profile", "n_profile_polygons", 0)

    raw_casters = value_from(raw_summary, "profile", "n_casters", 0)
    proxy_casters = value_from(proxy_summary, "profile", "n_casters", 0)

    raw_receivers = value_from(raw_summary, "profile", "n_receivers", 0)
    proxy_receivers = value_from(proxy_summary, "profile", "n_receivers", 0)

    raw_array_mb = value_from(raw_summary, "surface_arrays", "memory_mb", 0.0)
    proxy_array_mb = value_from(proxy_summary, "surface_arrays", "memory_mb", 0.0)

    raw_tile_size_mb = value_from(raw_summary, "lobe_cache", "total_tile_size_mb", 0.0)
    proxy_tile_size_mb = value_from(proxy_summary, "lobe_cache", "total_tile_size_mb", 0.0)

    raw_compute_s = value_from(raw_summary, "lobe_cache", "total_compute_elapsed_s", 0.0)
    proxy_compute_s = value_from(proxy_summary, "lobe_cache", "total_compute_elapsed_s", 0.0)

    raw_loop_s = value_from(raw_summary, "lobe_cache", "total_loop_elapsed_s", 0.0)
    proxy_loop_s = value_from(proxy_summary, "lobe_cache", "total_loop_elapsed_s", 0.0)

    comparison = {
        "output_root": str(output_root),

        "profile_polygon_counts": {
            "raw": int(raw_profile_polygons),
            "barrier_proxy": int(proxy_profile_polygons),
            "delta": int(proxy_profile_polygons) - int(raw_profile_polygons),
            "proxy_over_raw": ratio_safe(proxy_profile_polygons, raw_profile_polygons),
        },

        "caster_counts": {
            "raw": int(raw_casters),
            "barrier_proxy": int(proxy_casters),
            "delta": int(proxy_casters) - int(raw_casters),
            "proxy_over_raw": ratio_safe(proxy_casters, raw_casters),
        },

        "receiver_counts": {
            "raw": int(raw_receivers),
            "barrier_proxy": int(proxy_receivers),
            "delta": int(proxy_receivers) - int(raw_receivers),
            "proxy_over_raw": ratio_safe(proxy_receivers, raw_receivers),
        },

        "barrier_proxy": {
            "n_barrier_surfaces": int(value_from(proxy_summary, "profile", "n_barrier_surfaces", 0)),
            "n_barrier_segments": int(value_from(proxy_summary, "profile", "n_barrier_segments", 0)),
            "total_barrier_proxy_area": float(value_from(proxy_summary, "profile", "total_barrier_proxy_area", 0.0)),
        },

        "array_memory": {
            "raw_mb": float(raw_array_mb),
            "barrier_proxy_mb": float(proxy_array_mb),
            "delta_mb": float(proxy_array_mb) - float(raw_array_mb),
            "proxy_over_raw": ratio_safe(proxy_array_mb, raw_array_mb),
        },

        "timing": {
            "raw_elapsed_s": float(raw_summary.get("elapsed_s", 0.0)),
            "proxy_elapsed_s": float(proxy_summary.get("elapsed_s", 0.0)),
            "raw_lobe_compute_s": float(raw_compute_s),
            "proxy_lobe_compute_s": float(proxy_compute_s),
            "raw_lobe_loop_s": float(raw_loop_s),
            "proxy_lobe_loop_s": float(proxy_loop_s),
            "compute_proxy_over_raw": ratio_safe(proxy_compute_s, raw_compute_s),
            "loop_proxy_over_raw": ratio_safe(proxy_loop_s, raw_loop_s),
        },

        "storage": {
            "raw_output_dir_mb": float(raw_summary.get("file_sizes", {}).get("output_dir_mb", 0.0)),
            "proxy_output_dir_mb": float(proxy_summary.get("file_sizes", {}).get("output_dir_mb", 0.0)),
            "raw_tile_size_mb": float(raw_tile_size_mb),
            "proxy_tile_size_mb": float(proxy_tile_size_mb),
            "tile_size_proxy_over_raw": ratio_safe(proxy_tile_size_mb, raw_tile_size_mb),
        },

        "raw_summary": raw_summary,
        "proxy_summary": proxy_summary,
    }

    return comparison


def write_comparison_markdown(path, comparison):
    path = Path(path)

    lines = []

    lines.append("# Raw vs barrier-proxy lobe pipeline comparison")
    lines.append("")
    lines.append("## Geometry/profile counts")
    lines.append("")
    lines.append("| Metric | Raw | Barrier proxy | Proxy / Raw | Delta |")
    lines.append("|---|---:|---:|---:|---:|")

    rows = [
        (
            "Profile polygons",
            comparison["profile_polygon_counts"]["raw"],
            comparison["profile_polygon_counts"]["barrier_proxy"],
            comparison["profile_polygon_counts"]["proxy_over_raw"],
            comparison["profile_polygon_counts"]["delta"],
        ),
        (
            "Casters",
            comparison["caster_counts"]["raw"],
            comparison["caster_counts"]["barrier_proxy"],
            comparison["caster_counts"]["proxy_over_raw"],
            comparison["caster_counts"]["delta"],
        ),
        (
            "Receivers",
            comparison["receiver_counts"]["raw"],
            comparison["receiver_counts"]["barrier_proxy"],
            comparison["receiver_counts"]["proxy_over_raw"],
            comparison["receiver_counts"]["delta"],
        ),
    ]

    for label, raw, proxy, ratio, delta in rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                label,
                raw,
                proxy,
                "n/a" if ratio is None else round(float(ratio), 4),
                delta,
            )
        )

    lines.append("")
    lines.append("## Barrier proxy")
    lines.append("")
    lines.append("- barrier surfaces: `{}`".format(comparison["barrier_proxy"]["n_barrier_surfaces"]))
    lines.append("- barrier segments: `{}`".format(comparison["barrier_proxy"]["n_barrier_segments"]))
    lines.append("- total barrier proxy area: `{}`".format(round(comparison["barrier_proxy"]["total_barrier_proxy_area"], 4)))
    lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("| Metric | Raw | Barrier proxy | Proxy / Raw |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| Total elapsed s | {} | {} | {} |".format(
            round(comparison["timing"]["raw_elapsed_s"], 4),
            round(comparison["timing"]["proxy_elapsed_s"], 4),
            _ratio_text(comparison["timing"].get("proxy_elapsed_s"), comparison["timing"].get("raw_elapsed_s")),
        )
    )
    lines.append(
        "| Lobe compute s | {} | {} | {} |".format(
            round(comparison["timing"]["raw_lobe_compute_s"], 4),
            round(comparison["timing"]["proxy_lobe_compute_s"], 4),
            _ratio_value_text(comparison["timing"]["compute_proxy_over_raw"]),
        )
    )
    lines.append(
        "| Lobe loop s | {} | {} | {} |".format(
            round(comparison["timing"]["raw_lobe_loop_s"], 4),
            round(comparison["timing"]["proxy_lobe_loop_s"], 4),
            _ratio_value_text(comparison["timing"]["loop_proxy_over_raw"]),
        )
    )
    lines.append("")

    lines.append("## Memory/storage")
    lines.append("")
    lines.append("| Metric | Raw | Barrier proxy | Proxy / Raw |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| Array memory MB | {} | {} | {} |".format(
            round(comparison["array_memory"]["raw_mb"], 4),
            round(comparison["array_memory"]["barrier_proxy_mb"], 4),
            _ratio_value_text(comparison["array_memory"]["proxy_over_raw"]),
        )
    )
    lines.append(
        "| Tile size MB | {} | {} | {} |".format(
            round(comparison["storage"]["raw_tile_size_mb"], 4),
            round(comparison["storage"]["proxy_tile_size_mb"], 4),
            _ratio_value_text(comparison["storage"]["tile_size_proxy_over_raw"]),
        )
    )

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def _ratio_value_text(value):
    if value is None:
        return "n/a"

    return str(round(float(value), 4))


def _ratio_text(a, b):
    r = ratio_safe(a, b)

    if r is None:
        return "n/a"

    return str(round(float(r), 4))


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run raw or barrier-proxy lobe-vector shading pipeline.",
    )

    parser.add_argument(
        "--profile-mode",
        choices=("raw", "barrier_proxy", "compare"),
        default=DEFAULT_PROFILE_MODE,
        help="Run raw profile, barrier proxy profile, or both.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output folder.",
    )

    parser.add_argument(
        "--terrain-type",
        type=str,
        default=DEFAULT_TERRAIN_TYPE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--generator-n-polygons",
        type=int,
        default=DEFAULT_GENERATOR_N_POLYGONS,
    )

    parser.add_argument(
        "--built-density",
        type=none_or_float,
        default=DEFAULT_BUILT_DENSITY,
    )

    parser.add_argument(
        "--green-area-fraction",
        type=none_or_float,
        default=DEFAULT_GREEN_AREA_FRACTION,
    )

    parser.add_argument(
        "--average-building-height",
        type=none_or_float,
        default=DEFAULT_AVERAGE_BUILDING_HEIGHT,
    )

    parser.add_argument(
        "--height-lognorm-sigma",
        type=none_or_float,
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
        "--top-k-lobes",
        type=int,
        default=DEFAULT_TOP_K_LOBES,
    )

    parser.add_argument(
        "--core-tile-size-m",
        type=float,
        default=DEFAULT_CORE_TILE_SIZE_M,
    )

    parser.add_argument(
        "--halo-radius-m",
        type=float,
        default=DEFAULT_HALO_RADIUS_M,
    )

    parser.add_argument(
        "--grid-cell-size-m",
        type=float,
        default=DEFAULT_GRID_CELL_SIZE_M,
    )

    parser.add_argument(
        "--numba-threads",
        type=none_or_int,
        default=DEFAULT_NUMBA_THREADS,
        help="None, 1, or a positive integer.",
    )

    parser.add_argument(
        "--include-roof-casters",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--include-tree-casters",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--include-original-building-wall-casters",
        type=str_to_bool,
        default=False,
    )

    parser.add_argument(
        "--include-nonbuilding-casters",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--optimize-barrier-rotation",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--barrier-rotation-cost-mode",
        choices=("total_length", "max_length", "none"),
        default="total_length",
    )

    parser.add_argument(
        "--deduplicate-barriers",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--save-surface-arrays",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--save-surface-arrays-compressed",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--include-compact-vertices",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--include-same-building-casters",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--only-front-hemisphere",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--use-opacity-in-score",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--resume-existing-tiles",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output folder before running.",
    )

    parser.add_argument(
        "--compress-tile-npz",
        type=str_to_bool,
        default=False,
        help="Save tile npz files compressed directly. Slower. Usually leave false and run compress_lobe_tiles.py after.",
    )

    parser.add_argument(
        "--print-empty-tiles",
        type=str_to_bool,
        default=False,
    )

    parser.add_argument(
        "--print-every-tile",
        type=str_to_bool,
        default=True,
    )

    parser.add_argument(
        "--write-manifest",
        type=str_to_bool,
        default=True,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = build_base_config(args)

    if args.profile_mode == "compare":
        return run_compare(config)

    return run_one_profile(config)


if __name__ == "__main__":
    main()