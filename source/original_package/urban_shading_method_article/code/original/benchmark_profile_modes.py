# benchmark_profile_modes.py

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shutil
import time
import traceback

import numpy as np

import run_lobe_pipeline


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTPUT_ROOT = Path("array_city_lobe_benchmarks")

DEFAULT_SEEDS = [42]
DEFAULT_N_POLYGONS = [150, 400, 800]

DEFAULT_TERRAIN_TYPE = "small_town"

DEFAULT_BUILT_DENSITY = 0.30
DEFAULT_GREEN_AREA_FRACTION = 0.25
DEFAULT_AVERAGE_BUILDING_HEIGHT = 12.0
DEFAULT_HEIGHT_LOGNORM_SIGMA = 0.30
DEFAULT_ELEVATION_TYPE = "undulating"

DEFAULT_TOP_K_LOBES = 40

DEFAULT_CORE_TILE_SIZE_M = 250.0
DEFAULT_HALO_RADIUS_M = 350.0
DEFAULT_GRID_CELL_SIZE_M = 250.0

DEFAULT_NUMBA_THREADS = 4

DEFAULT_CASES = [
    "raw",
    "proxy_merged_eave",
    "proxy_triangles_eave",
    "proxy_merged_top",
]

DEFAULT_RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)


# ============================================================
# JSON / filesystem helpers
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


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def clean_dir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def file_size_mb(path):
    path = Path(path)

    if not path.exists():
        return 0.0

    return float(path.stat().st_size) / (1024.0 * 1024.0)


def dir_size_mb(path):
    path = Path(path)

    if not path.exists():
        return 0.0

    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return float(total) / (1024.0 * 1024.0)


# ============================================================
# Parsing helpers
# ============================================================

def parse_int_list(text):
    if isinstance(text, list):
        return [int(x) for x in text]

    parts = str(text).replace(";", ",").split(",")

    out = []

    for p in parts:
        p = p.strip()

        if p:
            out.append(int(p))

    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")

    return out


def parse_case_list(text):
    if isinstance(text, list):
        return [str(x) for x in text]

    parts = str(text).replace(";", ",").split(",")

    out = []

    for p in parts:
        p = p.strip()

        if p:
            out.append(p)

    if not out:
        raise argparse.ArgumentTypeError("Expected at least one case name.")

    return out


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


def safe_ratio(a, b):
    try:
        b = float(b)

        if abs(b) <= 1e-12:
            return None

        return float(a) / b

    except Exception:
        return None


def get_nested(d, keys, default=None):
    cur = d

    try:
        for k in keys:
            cur = cur[k]

        if cur is None:
            return default

        return cur

    except Exception:
        return default


# ============================================================
# Case definition
# ============================================================

def normalize_case_name(case_name):
    return str(case_name).lower().strip()


def case_to_profile_options(case_name):
    """
    Supported benchmark cases:

        raw
        proxy_merged_eave
        proxy_triangles_eave
        proxy_merged_top
        proxy_triangles_top

    You can add more cases here without touching the runner.
    """
    name = normalize_case_name(case_name)

    if name == "raw":
        return {
            "case_name": "raw",
            "profile_mode": "raw",
            "decomposition_mode": "merged",
            "barrier_vertical_top_mode": "eave",
        }

    if name == "proxy_merged_eave":
        return {
            "case_name": "proxy_merged_eave",
            "profile_mode": "barrier_proxy",
            "decomposition_mode": "merged",
            "barrier_vertical_top_mode": "eave",
        }

    if name == "proxy_triangles_eave":
        return {
            "case_name": "proxy_triangles_eave",
            "profile_mode": "barrier_proxy",
            "decomposition_mode": "triangles",
            "barrier_vertical_top_mode": "eave",
        }

    if name == "proxy_merged_top":
        return {
            "case_name": "proxy_merged_top",
            "profile_mode": "barrier_proxy",
            "decomposition_mode": "merged",
            "barrier_vertical_top_mode": "top",
        }

    if name == "proxy_triangles_top":
        return {
            "case_name": "proxy_triangles_top",
            "profile_mode": "barrier_proxy",
            "decomposition_mode": "triangles",
            "barrier_vertical_top_mode": "top",
        }

    raise ValueError(
        "Unknown benchmark case '{}'. Supported cases: {}".format(
            case_name,
            [
                "raw",
                "proxy_merged_eave",
                "proxy_triangles_eave",
                "proxy_merged_top",
                "proxy_triangles_top",
            ],
        )
    )


# ============================================================
# Config builder
# ============================================================

def make_run_config(args, case_name, seed, n_polygons):
    case_options = case_to_profile_options(case_name)

    output_root = Path(args.output_root)
    case_dir = output_root / "runs" / (
        "case__{}__seed__{}__n__{}".format(
            case_options["case_name"],
            int(seed),
            int(n_polygons),
        )
    )

    config = {
        "output_root": str(output_root),
        "output_dir": str(case_dir),

        "profile_mode": case_options["profile_mode"],

        "benchmark_case_name": case_options["case_name"],
        "terrain_type": args.terrain_type,
        "seed": int(seed),
        "generator_n_polygons": int(n_polygons),

        "built_density": float(args.built_density),
        "green_area_fraction": float(args.green_area_fraction),
        "average_building_height": float(args.average_building_height),
        "height_lognorm_sigma": float(args.height_lognorm_sigma),
        "elevation_type": str(args.elevation_type),

        "receiver_surface_kinds": list(DEFAULT_RECEIVER_SURFACE_KINDS),

        "decomposition_mode": case_options["decomposition_mode"],
        "barrier_vertical_top_mode": case_options["barrier_vertical_top_mode"],

        "include_roof_casters": bool(args.include_roof_casters),
        "include_tree_casters": bool(args.include_tree_casters),
        "include_original_building_wall_casters": bool(args.include_original_building_wall_casters),
        "include_nonbuilding_casters": bool(args.include_nonbuilding_casters),

        "optimize_barrier_rotation": bool(args.optimize_barrier_rotation),
        "barrier_rotation_cost_mode": str(args.barrier_rotation_cost_mode),
        "deduplicate_barriers": bool(args.deduplicate_barriers),

        "save_surface_arrays": bool(args.save_surface_arrays),
        "save_surface_arrays_compressed": bool(args.save_surface_arrays_compressed),
        "include_compact_vertices": bool(args.include_compact_vertices),

        "core_tile_size_m": float(args.core_tile_size_m),
        "halo_radius_m": float(args.halo_radius_m),
        "grid_cell_size_m": float(args.grid_cell_size_m),

        "top_k_lobes": int(args.top_k_lobes),

        "include_same_building_casters": bool(args.include_same_building_casters),
        "only_front_hemisphere": bool(args.only_front_hemisphere),
        "use_opacity_in_score": bool(args.use_opacity_in_score),

        "numba_threads": args.numba_threads,

        "resume_existing_tiles": bool(args.resume_existing_tiles),
        "overwrite": bool(args.overwrite_runs),
        "compress_tile_npz": bool(args.compress_tile_npz),

        "print_empty_tiles": bool(args.print_empty_tiles),
        "print_every_tile": bool(args.print_every_tile),
        "write_manifest": True,
    }

    return config


# ============================================================
# Result flattening
# ============================================================

def flatten_run_summary(run_summary, success=True, error_message=None, traceback_text=None):
    """
    Convert one run_summary.json-like dictionary into one flat CSV row.
    """
    config = run_summary.get("config", {})

    profile = run_summary.get("profile", {})
    surface_arrays = run_summary.get("surface_arrays", {})
    lobe_cache = run_summary.get("lobe_cache", {})
    file_sizes = run_summary.get("file_sizes", {})

    row = {
        "success": bool(success),
        "error_message": "" if error_message is None else str(error_message),

        "case_name": str(config.get("benchmark_case_name", run_summary.get("case_name", ""))),
        "profile_mode": str(run_summary.get("profile_mode", config.get("profile_mode", ""))),
        "seed": int(config.get("seed", -1)),
        "n_polygons_requested": int(config.get("generator_n_polygons", -1)),

        "terrain_type": str(config.get("terrain_type", "")),
        "built_density": float(config.get("built_density", 0.0)),
        "green_area_fraction": float(config.get("green_area_fraction", 0.0)),
        "average_building_height": float(config.get("average_building_height", 0.0)),
        "height_lognorm_sigma": float(config.get("height_lognorm_sigma", 0.0)),
        "elevation_type": str(config.get("elevation_type", "")),

        "decomposition_mode": str(config.get("decomposition_mode", "")),
        "barrier_vertical_top_mode": str(config.get("barrier_vertical_top_mode", "")),
        "top_k_lobes": int(config.get("top_k_lobes", -1)),
        "core_tile_size_m": float(config.get("core_tile_size_m", 0.0)),
        "halo_radius_m": float(config.get("halo_radius_m", 0.0)),
        "grid_cell_size_m": float(config.get("grid_cell_size_m", 0.0)),
        "numba_threads": "" if config.get("numba_threads", None) is None else int(config.get("numba_threads")),

        "run_elapsed_s": float(run_summary.get("elapsed_s", 0.0)),

        "n_profile_polygons": int(profile.get("n_profile_polygons", 0)),
        "n_scene_polygons": int(profile.get("n_scene_polygons", 0)),
        "n_casters": int(profile.get("n_casters", 0)),
        "n_receivers": int(profile.get("n_receivers", 0)),
        "n_buildings": int(profile.get("n_buildings", 0)),
        "n_barrier_profiles": int(profile.get("n_barrier_profiles", 0)),
        "n_barrier_surfaces": int(profile.get("n_barrier_surfaces", 0)),
        "n_barrier_segments": int(profile.get("n_barrier_segments", 0)),
        "total_barrier_proxy_area": float(profile.get("total_barrier_proxy_area", 0.0)),
        "profile_elapsed_s": float(profile.get("elapsed_s", 0.0)),

        "array_n_surfaces": int(surface_arrays.get("n_surfaces", 0)),
        "array_n_vertices": int(surface_arrays.get("n_vertices", 0)),
        "array_n_casters": int(surface_arrays.get("n_casters", 0)),
        "array_n_receivers": int(surface_arrays.get("n_receivers", 0)),
        "array_n_two_sided": int(surface_arrays.get("n_two_sided", 0)),
        "array_memory_mb": float(surface_arrays.get("memory_mb", 0.0)),
        "array_total_area": float(surface_arrays.get("total_area", 0.0)),
        "array_caster_area": float(surface_arrays.get("caster_area", 0.0)),
        "array_receiver_area": float(surface_arrays.get("receiver_area", 0.0)),

        "tile_files": int(lobe_cache.get("n_tile_files", 0)),
        "receiver_rows": int(lobe_cache.get("n_receiver_rows", 0)),
        "lobe_compute_elapsed_s": float(lobe_cache.get("total_compute_elapsed_s", 0.0)),
        "lobe_loop_elapsed_s": float(lobe_cache.get("total_loop_elapsed_s", 0.0)),
        "tile_size_mb": float(lobe_cache.get("total_tile_size_mb", 0.0)),

        "output_dir_mb": float(file_sizes.get("output_dir_mb", 0.0)),
        "surface_arrays_npz_mb": float(file_sizes.get("surface_arrays_npz_mb", 0.0)),

        "output_dir": str(run_summary.get("output_dir", config.get("output_dir", ""))),
    }

    if traceback_text is not None:
        row["traceback"] = str(traceback_text)
    else:
        row["traceback"] = ""

    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return path


# ============================================================
# Comparisons
# ============================================================

def build_pairwise_comparisons(rows):
    """
    Compare every non-raw case against raw with same seed and n_polygons.
    """
    by_group = {}

    for row in rows:
        if not row.get("success", False):
            continue

        key = (
            int(row["seed"]),
            int(row["n_polygons_requested"]),
        )

        by_group.setdefault(key, {})[row["case_name"]] = row

    comparisons = []

    for key, case_rows in sorted(by_group.items()):
        seed, n_polygons = key

        raw = case_rows.get("raw")

        if raw is None:
            continue

        for case_name, row in sorted(case_rows.items()):
            if case_name == "raw":
                continue

            comp = {
                "seed": int(seed),
                "n_polygons_requested": int(n_polygons),
                "case_name": str(case_name),
                "baseline_case": "raw",

                "raw_n_casters": int(raw["n_casters"]),
                "case_n_casters": int(row["n_casters"]),
                "caster_delta": int(row["n_casters"]) - int(raw["n_casters"]),
                "caster_ratio": safe_ratio(row["n_casters"], raw["n_casters"]),

                "raw_n_receivers": int(raw["n_receivers"]),
                "case_n_receivers": int(row["n_receivers"]),
                "receiver_delta": int(row["n_receivers"]) - int(raw["n_receivers"]),
                "receiver_ratio": safe_ratio(row["n_receivers"], raw["n_receivers"]),

                "raw_array_memory_mb": float(raw["array_memory_mb"]),
                "case_array_memory_mb": float(row["array_memory_mb"]),
                "array_memory_delta_mb": float(row["array_memory_mb"]) - float(raw["array_memory_mb"]),
                "array_memory_ratio": safe_ratio(row["array_memory_mb"], raw["array_memory_mb"]),

                "raw_lobe_compute_s": float(raw["lobe_compute_elapsed_s"]),
                "case_lobe_compute_s": float(row["lobe_compute_elapsed_s"]),
                "lobe_compute_delta_s": float(row["lobe_compute_elapsed_s"]) - float(raw["lobe_compute_elapsed_s"]),
                "lobe_compute_ratio": safe_ratio(row["lobe_compute_elapsed_s"], raw["lobe_compute_elapsed_s"]),

                "raw_lobe_loop_s": float(raw["lobe_loop_elapsed_s"]),
                "case_lobe_loop_s": float(row["lobe_loop_elapsed_s"]),
                "lobe_loop_delta_s": float(row["lobe_loop_elapsed_s"]) - float(raw["lobe_loop_elapsed_s"]),
                "lobe_loop_ratio": safe_ratio(row["lobe_loop_elapsed_s"], raw["lobe_loop_elapsed_s"]),

                "raw_run_elapsed_s": float(raw["run_elapsed_s"]),
                "case_run_elapsed_s": float(row["run_elapsed_s"]),
                "run_elapsed_delta_s": float(row["run_elapsed_s"]) - float(raw["run_elapsed_s"]),
                "run_elapsed_ratio": safe_ratio(row["run_elapsed_s"], raw["run_elapsed_s"]),

                "raw_tile_size_mb": float(raw["tile_size_mb"]),
                "case_tile_size_mb": float(row["tile_size_mb"]),
                "tile_size_delta_mb": float(row["tile_size_mb"]) - float(raw["tile_size_mb"]),
                "tile_size_ratio": safe_ratio(row["tile_size_mb"], raw["tile_size_mb"]),

                "case_barrier_surfaces": int(row["n_barrier_surfaces"]),
                "case_barrier_segments": int(row["n_barrier_segments"]),
            }

            comparisons.append(comp)

    return comparisons


def summarize_by_case(rows, comparisons):
    """
    Aggregate over seeds and requested n values.
    """
    by_case = {}

    for row in rows:
        if not row.get("success", False):
            continue

        case = str(row["case_name"])

        by_case.setdefault(case, []).append(row)

    summary_rows = []

    for case, case_rows in sorted(by_case.items()):
        summary = {
            "case_name": case,
            "n_runs": int(len(case_rows)),

            "mean_n_casters": mean_value(case_rows, "n_casters"),
            "mean_n_receivers": mean_value(case_rows, "n_receivers"),
            "mean_n_profile_polygons": mean_value(case_rows, "n_profile_polygons"),
            "mean_n_barrier_surfaces": mean_value(case_rows, "n_barrier_surfaces"),
            "mean_n_barrier_segments": mean_value(case_rows, "n_barrier_segments"),

            "mean_array_memory_mb": mean_value(case_rows, "array_memory_mb"),
            "mean_lobe_compute_s": mean_value(case_rows, "lobe_compute_elapsed_s"),
            "mean_lobe_loop_s": mean_value(case_rows, "lobe_loop_elapsed_s"),
            "mean_run_elapsed_s": mean_value(case_rows, "run_elapsed_s"),
            "mean_tile_size_mb": mean_value(case_rows, "tile_size_mb"),

            "median_array_memory_mb": median_value(case_rows, "array_memory_mb"),
            "median_lobe_compute_s": median_value(case_rows, "lobe_compute_elapsed_s"),
            "median_lobe_loop_s": median_value(case_rows, "lobe_loop_elapsed_s"),
            "median_run_elapsed_s": median_value(case_rows, "run_elapsed_s"),
            "median_tile_size_mb": median_value(case_rows, "tile_size_mb"),
        }

        summary_rows.append(summary)

    # Aggregate comparison ratios by case.
    by_comp_case = {}

    for comp in comparisons:
        by_comp_case.setdefault(comp["case_name"], []).append(comp)

    for summary in summary_rows:
        case = summary["case_name"]

        comps = by_comp_case.get(case, [])

        if not comps:
            summary["mean_caster_ratio_vs_raw"] = None
            summary["mean_lobe_compute_ratio_vs_raw"] = None
            summary["mean_run_elapsed_ratio_vs_raw"] = None
            summary["mean_tile_size_ratio_vs_raw"] = None
            continue

        summary["mean_caster_ratio_vs_raw"] = mean_value_nullable(comps, "caster_ratio")
        summary["mean_lobe_compute_ratio_vs_raw"] = mean_value_nullable(comps, "lobe_compute_ratio")
        summary["mean_run_elapsed_ratio_vs_raw"] = mean_value_nullable(comps, "run_elapsed_ratio")
        summary["mean_tile_size_ratio_vs_raw"] = mean_value_nullable(comps, "tile_size_ratio")

    return summary_rows


def mean_value(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key, None) is not None]

    if not vals:
        return None

    return float(np.mean(vals))


def median_value(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key, None) is not None]

    if not vals:
        return None

    return float(np.median(vals))


def mean_value_nullable(rows, key):
    vals = []

    for r in rows:
        v = r.get(key, None)

        if v is None:
            continue

        vals.append(float(v))

    if not vals:
        return None

    return float(np.mean(vals))


# ============================================================
# Markdown report
# ============================================================

def fmt(value, digits=4):
    if value is None:
        return "n/a"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    try:
        return str(round(float(value), digits))
    except Exception:
        return str(value)


def write_markdown_report(path, args, rows, comparisons, summary_rows):
    path = Path(path)

    successful = [r for r in rows if r.get("success", False)]
    failed = [r for r in rows if not r.get("success", False)]

    lines = []

    lines.append("# Profile mode benchmark report")
    lines.append("")
    lines.append("## Benchmark setup")
    lines.append("")
    lines.append("- terrain type: `{}`".format(args.terrain_type))
    lines.append("- seeds: `{}`".format(args.seeds))
    lines.append("- requested n-polygons: `{}`".format(args.n_polygons))
    lines.append("- cases: `{}`".format(args.cases))
    lines.append("- top K lobes: `{}`".format(args.top_k_lobes))
    lines.append("- numba threads: `{}`".format(args.numba_threads))
    lines.append("- successful runs: `{}`".format(len(successful)))
    lines.append("- failed runs: `{}`".format(len(failed)))
    lines.append("")

    lines.append("## Case-level averages")
    lines.append("")
    lines.append(
        "| Case | Runs | Mean casters | Mean receivers | Mean compute s | Mean total s | Compute ratio vs raw | Total ratio vs raw |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in summary_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["case_name"],
                row["n_runs"],
                fmt(row["mean_n_casters"], 2),
                fmt(row["mean_n_receivers"], 2),
                fmt(row["mean_lobe_compute_s"], 4),
                fmt(row["mean_run_elapsed_s"], 4),
                fmt(row.get("mean_lobe_compute_ratio_vs_raw"), 4),
                fmt(row.get("mean_run_elapsed_ratio_vs_raw"), 4),
            )
        )

    lines.append("")
    lines.append("## Raw-normalized comparisons")
    lines.append("")
    lines.append(
        "| Seed | N requested | Case | Caster ratio | Compute ratio | Total ratio | Tile-size ratio | Barrier surfaces |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")

    for comp in comparisons:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                comp["seed"],
                comp["n_polygons_requested"],
                comp["case_name"],
                fmt(comp["caster_ratio"], 4),
                fmt(comp["lobe_compute_ratio"], 4),
                fmt(comp["run_elapsed_ratio"], 4),
                fmt(comp["tile_size_ratio"], 4),
                comp["case_barrier_surfaces"],
            )
        )

    lines.append("")
    lines.append("## Individual runs")
    lines.append("")
    lines.append(
        "| Success | Case | Seed | N requested | Casters | Receivers | Compute s | Total s | Array MB | Tile MB |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["success"],
                row["case_name"],
                row["seed"],
                row["n_polygons_requested"],
                row["n_casters"],
                row["n_receivers"],
                fmt(row["lobe_compute_elapsed_s"], 4),
                fmt(row["run_elapsed_s"], 4),
                fmt(row["array_memory_mb"], 4),
                fmt(row["tile_size_mb"], 4),
            )
        )

    if failed:
        lines.append("")
        lines.append("## Failed runs")
        lines.append("")

        for row in failed:
            lines.append(
                "- case `{}`, seed `{}`, n `{}`: `{}`".format(
                    row["case_name"],
                    row["seed"],
                    row["n_polygons_requested"],
                    row["error_message"],
                )
            )

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


# ============================================================
# Main benchmark
# ============================================================

def run_one_benchmark_case(args, case_name, seed, n_polygons):
    config = make_run_config(
        args=args,
        case_name=case_name,
        seed=seed,
        n_polygons=n_polygons,
    )

    output_dir = Path(config["output_dir"])

    t0 = time.perf_counter()

    try:
        summary = run_lobe_pipeline.run_one_profile(config)
        elapsed = time.perf_counter() - t0

        summary["benchmark_case_name"] = config["benchmark_case_name"]
        summary["elapsed_s"] = float(elapsed)
        summary["config"] = config

        row = flatten_run_summary(
            summary,
            success=True,
            error_message=None,
            traceback_text=None,
        )

        return row, summary

    except Exception as e:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()

        print()
        print("=" * 100)
        print("BENCHMARK CASE FAILED")
        print("=" * 100)
        print("case:", case_name)
        print("seed:", seed)
        print("n_polygons:", n_polygons)
        print("elapsed_s:", round(elapsed, 3))
        print("error:", repr(e))
        print("=" * 100)
        print()

        failed_summary = {
            "profile_mode": case_to_profile_options(case_name)["profile_mode"],
            "case_name": case_name,
            "output_dir": str(output_dir),
            "elapsed_s": float(elapsed),
            "config": config,
            "profile": {},
            "surface_arrays": {},
            "lobe_cache": {},
            "file_sizes": {
                "output_dir_mb": dir_size_mb(output_dir),
                "surface_arrays_npz_mb": file_size_mb(output_dir / "surface_arrays.npz"),
            },
            "error": repr(e),
            "traceback": tb,
        }

        row = flatten_run_summary(
            failed_summary,
            success=False,
            error_message=repr(e),
            traceback_text=tb,
        )

        return row, failed_summary


def run_benchmark(args):
    output_root = Path(args.output_root)

    if args.overwrite_root:
        clean_dir(output_root)
    else:
        ensure_dir(output_root)

    seeds = parse_int_list(args.seeds)
    n_values = parse_int_list(args.n_polygons)
    cases = parse_case_list(args.cases)

    args.seeds = seeds
    args.n_polygons = n_values
    args.cases = cases

    benchmark_config = {
        "output_root": str(output_root),
        "seeds": seeds,
        "n_polygons": n_values,
        "cases": cases,
        "terrain_type": args.terrain_type,
        "top_k_lobes": args.top_k_lobes,
        "numba_threads": args.numba_threads,
        "started_at_unix_s": float(time.time()),
    }

    write_json(output_root / "benchmark_config.json", benchmark_config)

    rows = []
    summaries = []

    total_runs = len(seeds) * len(n_values) * len(cases)
    run_counter = 0

    t_all = time.perf_counter()

    for n_polygons in n_values:
        for seed in seeds:
            for case_name in cases:
                run_counter += 1

                print()
                print("#" * 100)
                print(
                    "BENCHMARK RUN {}/{} | case={} | seed={} | n_polygons={}".format(
                        run_counter,
                        total_runs,
                        case_name,
                        seed,
                        n_polygons,
                    )
                )
                print("#" * 100)
                print()

                row, summary = run_one_benchmark_case(
                    args=args,
                    case_name=case_name,
                    seed=seed,
                    n_polygons=n_polygons,
                )

                rows.append(row)
                summaries.append(summary)

                # Write partial outputs after every run.
                write_csv(output_root / "benchmark_results_partial.csv", rows)
                write_json(output_root / "benchmark_summaries_partial.json", summaries)

    total_elapsed_s = time.perf_counter() - t_all

    comparisons = build_pairwise_comparisons(rows)
    summary_rows = summarize_by_case(rows, comparisons)

    payload = {
        "benchmark_config": benchmark_config,
        "total_elapsed_s": float(total_elapsed_s),
        "rows": rows,
        "comparisons_vs_raw": comparisons,
        "summary_by_case": summary_rows,
        "summaries": summaries,
    }

    write_json(output_root / "benchmark_results.json", payload)
    write_csv(output_root / "benchmark_results.csv", rows)
    write_csv(output_root / "benchmark_comparisons_vs_raw.csv", comparisons)
    write_csv(output_root / "benchmark_summary_by_case.csv", summary_rows)
    write_markdown_report(
        output_root / "benchmark_report.md",
        args=args,
        rows=rows,
        comparisons=comparisons,
        summary_rows=summary_rows,
    )

    print()
    print("=" * 100)
    print("BENCHMARK DONE")
    print("=" * 100)
    print("output_root:", output_root)
    print("runs:", len(rows))
    print("successful:", sum(1 for r in rows if r.get("success", False)))
    print("failed:", sum(1 for r in rows if not r.get("success", False)))
    print("total_elapsed_s:", round(total_elapsed_s, 3))
    print("wrote:", output_root / "benchmark_results.csv")
    print("wrote:", output_root / "benchmark_comparisons_vs_raw.csv")
    print("wrote:", output_root / "benchmark_summary_by_case.csv")
    print("wrote:", output_root / "benchmark_report.md")
    print("=" * 100)
    print()

    return payload


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark raw vs barrier-proxy shading/lobe profile modes.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    parser.add_argument(
        "--overwrite-root",
        action="store_true",
        help="Delete the whole benchmark output root before running.",
    )

    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="Overwrite individual run folders.",
    )

    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(DEFAULT_CASES),
        help=(
            "Comma-separated cases. Supported: raw, proxy_merged_eave, "
            "proxy_triangles_eave, proxy_merged_top, proxy_triangles_top."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--n-polygons",
        type=str,
        default=",".join(str(x) for x in DEFAULT_N_POLYGONS),
        help="Comma-separated generator_n_polygons values.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Override cases/seeds/n-polygons with a small smoke benchmark.",
    )

    parser.add_argument(
        "--terrain-type",
        type=str,
        default=DEFAULT_TERRAIN_TYPE,
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
        help="None, 1, or positive integer.",
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
        "--compress-tile-npz",
        type=str_to_bool,
        default=False,
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

    return parser


def apply_quick_settings(args):
    if not args.quick:
        return args

    args.cases = "raw,proxy_merged_eave,proxy_triangles_eave"
    args.seeds = "42"
    args.n_polygons = "120"
    args.top_k_lobes = 20
    args.numba_threads = 1
    args.print_every_tile = True

    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_quick_settings(args)

    return run_benchmark(args)


if __name__ == "__main__":
    main()