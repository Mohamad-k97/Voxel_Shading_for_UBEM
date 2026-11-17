# array_gen.py

from __future__ import annotations

from pathlib import Path
import json
import shutil
import time

import numpy as np

from generator import generate_urban_receiver_and_casters
from shading_profiles import (
    build_shading_profile,
    print_shading_profile_summary,
)
from surface_array_builder import (
    SurfaceArrays,
    build_surface_arrays_from_profile,
    save_surface_arrays,
    load_surface_arrays,
    surface_arrays_summary,
    print_surface_arrays_summary,
    surface_arrays_nbytes,
    OBJECT_KIND_CODE,
    SURFACE_KIND_CODE,
)
from memory_timer import (
    PipelineProfiler,
    format_seconds,
    format_mb,
    get_process_memory_mb,
)

from njit_lobe_core import (
    build_surface_grid,
    compute_tile_lobes_safe,
    make_tiles,
    get_receiver_indices_for_tile,
    configure_numba_threads,
)


# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_DIR = Path("array_city_lobe_output")

# Pipeline mode:
#   "raw"           -> original physical surfaces cast shadows
#   "barrier_proxy" -> building wall casters replaced by barrier proxy surfaces
PROFILE_MODE = "barrier_proxy"

# Scene generation
TERRAIN_TYPE = "small_town"
SEED = 42

# This is still the old generator-style argument:
# number of nearest caster polygons requested in the generated scene.
# The full scene will usually contain more polygons than this.
GENERATOR_N_POLYGONS = 1200

BUILT_DENSITY = 0.30
GREEN_AREA_FRACTION = 0.25
AVERAGE_BUILDING_HEIGHT = 12.0
HEIGHT_LOGNORM_SIGMA = 0.30
ELEVATION_TYPE = "undulating"

RECEIVER_SURFACE_KINDS = (
    "wall",
    "flat_roof",
    "pitched_roof",
    "ground",
    "paved_ground",
)

# Shading-profile preprocessing
DECOMPOSITION_MODE = "merged"       # "merged" or "triangles"
BARRIER_VERTICAL_TOP_MODE = "eave"  # "eave" or "top"

INCLUDE_ROOF_CASTERS = True
INCLUDE_TREE_CASTERS = True
INCLUDE_ORIGINAL_BUILDING_WALL_CASTERS = False
INCLUDE_NONBUILDING_CASTERS = True

OPTIMIZE_BARRIER_ROTATION = True
BARRIER_ROTATION_COST_MODE = "total_length"
DEDUPLICATE_BARRIERS = True

# Array save
SAVE_SURFACE_ARRAYS = True
SAVE_SURFACE_ARRAYS_COMPRESSED = True
INCLUDE_COMPACT_VERTICES = True

# Tile/lobe calculation
CORE_TILE_SIZE_M = 250.0
HALO_RADIUS_M = 350.0
GRID_CELL_SIZE_M = 250.0

TOP_K_LOBES = 40

INCLUDE_SAME_BUILDING_CASTERS = True
ONLY_FRONT_HEMISPHERE = True
USE_OPACITY_IN_SCORE = True

# Numba
# None = automatic safe value
# 1    = serial
# >1   = requested number of threads
NUMBA_REQUESTED_THREADS = 4

# Save / resume
RESUME_EXISTING_TILES = True
OVERWRITE_OUTPUT_DIR = False

# Tile saving:
# False is faster. Later use compress_lobe_tiles.py.
COMPRESS_TILE_NPZ = False

# Diagnostics
PRINT_EMPTY_TILES = False
PRINT_EVERY_TILE = True
WRITE_MANIFEST = True


# ============================================================
# Constants
# ============================================================

EPS = 1e-10


# ============================================================
# Small helpers
# ============================================================

def ensure_output_dir(path, overwrite=False):
    path = Path(path)

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def mb_from_bytes(n_bytes):
    return float(n_bytes) / (1024.0 * 1024.0)


def file_size_mb(path):
    path = Path(path)

    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024.0 * 1024.0)


def json_safe(value):
    """
    Convert objects into JSON-safe basic Python values.
    """
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


def load_npz_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" in data:
            return json.loads(str(data["metadata_json"]))

    return {}


# ============================================================
# City/grid extent helpers
# ============================================================

def city_extent_from_scene_and_arrays(scene, arrays):
    """
    Determine city width/height for tiling.

    For generated scenes, scene.area_size is the cleanest source.
    For external/future profiles, we fall back to array bounding boxes.

    Important:
        njit_lobe_core.make_tiles starts from x=0, y=0.
        The current generator keeps generated geometry inside positive city
        coordinates, so this is okay.

    If you later feed real GIS coordinates, shift them before this stage.
    """
    if scene is not None and hasattr(scene, "area_size"):
        area = float(scene.area_size)

        if area > EPS:
            return area, area

    xmin = float(np.min(arrays.bbox_xy[:, 0]))
    ymin = float(np.min(arrays.bbox_xy[:, 1]))
    xmax = float(np.max(arrays.bbox_xy[:, 2]))
    ymax = float(np.max(arrays.bbox_xy[:, 3]))

    if xmin < -EPS or ymin < -EPS:
        raise ValueError(
            "Array coordinates include negative x/y. "
            "Shift real GIS coordinates to a local positive origin before tiling. "
            "xmin={}, ymin={}".format(xmin, ymin)
        )

    width = max(EPS, xmax)
    height = max(EPS, ymax)

    return float(width), float(height)


def receiver_count_in_tiles(arrays, city_width, city_height):
    tiles = make_tiles(
        city_width=float(city_width),
        city_height=float(city_height),
        core_size=float(CORE_TILE_SIZE_M),
        halo=float(HALO_RADIUS_M),
    )

    center_xy = arrays.center[:, :2]

    count = 0

    for _, core_bbox, _ in tiles:
        idx = get_receiver_indices_for_tile(
            center_xy=center_xy,
            can_receive=arrays.can_receive,
            core_bbox=core_bbox,
        )
        count += int(len(idx))

    return count


# ============================================================
# Tile output
# ============================================================

def build_receiver_meta(arrays, receiver_idx):
    """
    Receiver metadata table saved inside each tile.

    Columns:
        0  object_id
        1  object_kind_code
        2  surface_kind_code
        3  area
        4  receiver_horizontal_azimuth_deg
        5  center_x
        6  center_y
        7  center_z
        8  normal_x
        9  normal_y
        10 normal_z
    """
    receiver_idx = np.asarray(receiver_idx, dtype=np.int64)

    receiver_meta = np.column_stack([
        arrays.object_id[receiver_idx],
        arrays.object_kind[receiver_idx],
        arrays.surface_kind[receiver_idx],
        arrays.area[receiver_idx],
        arrays.normal_azimuth_deg[receiver_idx],
        arrays.center[receiver_idx, 0],
        arrays.center[receiver_idx, 1],
        arrays.center[receiver_idx, 2],
        arrays.normal[receiver_idx, 0],
        arrays.normal[receiver_idx, 1],
        arrays.normal[receiver_idx, 2],
    ]).astype(np.float32)

    return receiver_meta


def save_tile_lobe_cache(
    outpath,
    receiver_ids,
    valid_counts,
    caster_sids,
    kind_codes,
    params_i16,
    receiver_meta,
    metadata,
    compressed=False,
):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "receiver_ids": receiver_ids.astype(np.uint32),
        "valid_counts": valid_counts.astype(np.uint16),
        "caster_sids": caster_sids.astype(np.uint32),
        "kind_codes": kind_codes.astype(np.uint8),
        "params_i16": params_i16.astype(np.int16),
        "receiver_meta": receiver_meta.astype(np.float32),
        "metadata_json": np.array(json.dumps(json_safe(metadata))),
    }

    if compressed:
        np.savez_compressed(outpath, **payload)
    else:
        np.savez(outpath, **payload)

    return outpath


def process_tile(
    arrays,
    tile_id,
    core_bbox,
    halo_bbox,
    sorted_caster_indices,
    cell_start,
    cell_end,
    grid_nx,
    grid_ny,
    tile_output_dir,
    profile_metadata,
):
    """
    Process one tile and save its lobe cache.
    """
    outpath = Path(tile_output_dir) / "tile_{:05d}.npz".format(int(tile_id))

    if outpath.exists() and RESUME_EXISTING_TILES:
        if PRINT_EVERY_TILE:
            print("tile={:05d} | skip existing".format(int(tile_id)))

        return {
            "tile_id": int(tile_id),
            "path": outpath,
            "skipped": True,
            "n_receivers": int(load_npz_metadata(outpath).get("n_receivers", -1)),
            "elapsed_s": 0.0,
            "size_mb": file_size_mb(outpath),
        }

    center_xy = arrays.center[:, :2]

    receiver_idx = get_receiver_indices_for_tile(
        center_xy=center_xy,
        can_receive=arrays.can_receive,
        core_bbox=core_bbox,
    )

    if len(receiver_idx) == 0:
        if PRINT_EMPTY_TILES:
            print("tile={:05d} | empty".format(int(tile_id)))

        return {
            "tile_id": int(tile_id),
            "path": None,
            "skipped": False,
            "empty": True,
            "n_receivers": 0,
            "elapsed_s": 0.0,
            "size_mb": 0.0,
        }

    t0 = time.perf_counter()

    valid_counts, caster_sids, kind_codes, params_i16 = compute_tile_lobes_safe(
        receiver_idx.astype(np.int64),
        sorted_caster_indices.astype(np.int64),
        cell_start.astype(np.int64),
        cell_end.astype(np.int64),
        int(grid_nx),
        int(grid_ny),
        float(GRID_CELL_SIZE_M),
        np.array(halo_bbox, dtype=np.float64),

        arrays.surface_id.astype(np.uint32),
        arrays.object_id.astype(np.uint32),
        arrays.object_kind.astype(np.uint8),
        arrays.surface_kind.astype(np.uint8),
        arrays.center.astype(np.float32),
        arrays.normal.astype(np.float32),
        arrays.area.astype(np.float32),
        arrays.radius_xy.astype(np.float32),
        arrays.radius_z.astype(np.float32),
        arrays.radius_full.astype(np.float32),
        arrays.opacity.astype(np.float32),

        bool(INCLUDE_SAME_BUILDING_CASTERS),
        bool(ONLY_FRONT_HEMISPHERE),
        bool(USE_OPACITY_IN_SCORE),

        int(TOP_K_LOBES),

        requested_threads=NUMBA_REQUESTED_THREADS,
    )

    elapsed = time.perf_counter() - t0

    receiver_ids = arrays.surface_id[receiver_idx].astype(np.uint32)
    receiver_meta = build_receiver_meta(arrays, receiver_idx)

    metadata = {
        "tile_id": int(tile_id),
        "core_bbox": [float(x) for x in core_bbox],
        "halo_bbox": [float(x) for x in halo_bbox],
        "n_receivers": int(len(receiver_idx)),
        "n_casters_global": int(np.sum(arrays.can_cast)),
        "top_k_lobes": int(TOP_K_LOBES),
        "elapsed_s": float(elapsed),
        "compressed_npz": bool(COMPRESS_TILE_NPZ),

        "params_i16_columns": [
            "center_rel_azimuth_deg_x10",
            "center_slope_deg_x10",
            "sigma_azimuth_deg_x10",
            "sigma_slope_deg_x10",
            "amplitude_x1000",
        ],
        "receiver_meta_columns": [
            "object_id",
            "object_kind_code",
            "surface_kind_code",
            "area",
            "receiver_horizontal_azimuth_deg",
            "center_x",
            "center_y",
            "center_z",
            "normal_x",
            "normal_y",
            "normal_z",
        ],
        "object_kind_code": OBJECT_KIND_CODE,
        "surface_kind_code": SURFACE_KIND_CODE,
        "profile_metadata": profile_metadata,
    }

    save_tile_lobe_cache(
        outpath=outpath,
        receiver_ids=receiver_ids,
        valid_counts=valid_counts,
        caster_sids=caster_sids,
        kind_codes=kind_codes,
        params_i16=params_i16,
        receiver_meta=receiver_meta,
        metadata=metadata,
        compressed=COMPRESS_TILE_NPZ,
    )

    size_mb = file_size_mb(outpath)

    if PRINT_EVERY_TILE:
        mean_lobes = 0.0
        if len(valid_counts) > 0:
            mean_lobes = float(np.mean(valid_counts))

        print(
            "tile={:05d} | receivers={:,} | mean_lobes={:.2f} | time={} | size={:.3f} MB".format(
                int(tile_id),
                int(len(receiver_idx)),
                mean_lobes,
                format_seconds(elapsed),
                size_mb,
            )
        )

    return {
        "tile_id": int(tile_id),
        "path": outpath,
        "skipped": False,
        "empty": False,
        "n_receivers": int(len(receiver_idx)),
        "elapsed_s": float(elapsed),
        "size_mb": float(size_mb),
    }


# ============================================================
# Sliding-window lobe cache
# ============================================================

def run_sliding_window_lobe_cache(
    arrays,
    city_width,
    city_height,
    output_dir,
    profile_metadata=None,
):
    """
    Main tiled lobe-cache runner.

    Saves uncompressed tile_XXXXX.npz files by default.
    Use compress_lobe_tiles.py afterward for compact archive storage.
    """
    output_dir = Path(output_dir)
    tile_output_dir = output_dir / "tiles_njit"
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    center_xy = arrays.center[:, :2].astype(np.float32)

    n_casters = int(np.sum(arrays.can_cast))
    n_receivers = int(np.sum(arrays.can_receive))

    if n_casters == 0:
        raise RuntimeError("No castable surfaces found in SurfaceArrays.")

    if n_receivers == 0:
        raise RuntimeError("No receiver surfaces found in SurfaceArrays.")

    thread_info = configure_numba_threads(
        requested_threads=NUMBA_REQUESTED_THREADS,
        verbose=True,
    )

    print()
    print("=" * 90)
    print("BUILDING CASTER GRID")
    print("=" * 90)
    print("city_width_m:      ", city_width)
    print("city_height_m:     ", city_height)
    print("grid_cell_size_m:  ", GRID_CELL_SIZE_M)
    print("casters:           ", n_casters)
    print("receivers:         ", n_receivers)
    print("=" * 90)

    t_grid = time.perf_counter()

    sorted_caster_indices, cell_start, cell_end, grid_nx, grid_ny = build_surface_grid(
        center_xy=center_xy,
        can_cast=arrays.can_cast,
        city_width=float(city_width),
        city_height=float(city_height),
        cell_size=float(GRID_CELL_SIZE_M),
    )

    grid_elapsed = time.perf_counter() - t_grid

    tiles = make_tiles(
        city_width=float(city_width),
        city_height=float(city_height),
        core_size=float(CORE_TILE_SIZE_M),
        halo=float(HALO_RADIUS_M),
    )

    expected_receivers_in_tiles = receiver_count_in_tiles(
        arrays=arrays,
        city_width=city_width,
        city_height=city_height,
    )

    print()
    print("=" * 90)
    print("SLIDING WINDOW LOBE CACHE")
    print("=" * 90)
    print("tiles:             ", len(tiles))
    print("grid:              ", str(grid_nx) + " x " + str(grid_ny))
    print("grid build time:   ", format_seconds(grid_elapsed))
    print("core tile size m:  ", CORE_TILE_SIZE_M)
    print("halo radius m:     ", HALO_RADIUS_M)
    print("top K lobes:       ", TOP_K_LOBES)
    print("receiver coverage: ", expected_receivers_in_tiles, "/", n_receivers)
    print("output:            ", tile_output_dir)
    print("=" * 90)
    print()

    tile_records = []
    total_t0 = time.perf_counter()

    for tile_id, core_bbox, halo_bbox in tiles:
        record = process_tile(
            arrays=arrays,
            tile_id=tile_id,
            core_bbox=core_bbox,
            halo_bbox=halo_bbox,
            sorted_caster_indices=sorted_caster_indices,
            cell_start=cell_start,
            cell_end=cell_end,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            tile_output_dir=tile_output_dir,
            profile_metadata=profile_metadata,
        )

        tile_records.append(record)

    total_elapsed = time.perf_counter() - total_t0

    written_records = [r for r in tile_records if r.get("path") is not None]
    nonempty_records = [r for r in tile_records if int(r.get("n_receivers", 0)) > 0]

    tile_files = [
        Path(r["path"]).name
        for r in written_records
        if r.get("path") is not None
    ]

    total_tile_size_mb = float(sum(float(r.get("size_mb", 0.0)) for r in written_records))
    total_receiver_rows = int(sum(int(r.get("n_receivers", 0)) for r in nonempty_records))
    total_compute_s = float(sum(float(r.get("elapsed_s", 0.0)) for r in tile_records))

    index = {
        "tile_cache_format": "receiver_lobe_tile_cache_v2",
        "tile_dir": str(tile_output_dir),
        "n_tiles_total": int(len(tiles)),
        "n_tiles_with_receivers": int(len(nonempty_records)),
        "n_tile_files": int(len(tile_files)),
        "n_receiver_rows": int(total_receiver_rows),
        "n_receivers_global": int(n_receivers),
        "n_casters_global": int(n_casters),
        "city_width_m": float(city_width),
        "city_height_m": float(city_height),
        "core_tile_size_m": float(CORE_TILE_SIZE_M),
        "halo_radius_m": float(HALO_RADIUS_M),
        "grid_cell_size_m": float(GRID_CELL_SIZE_M),
        "grid_nx": int(grid_nx),
        "grid_ny": int(grid_ny),
        "top_k_lobes": int(TOP_K_LOBES),
        "total_loop_elapsed_s": float(total_elapsed),
        "total_compute_elapsed_s": float(total_compute_s),
        "total_tile_size_mb": float(total_tile_size_mb),
        "numba": thread_info,
        "profile_metadata": profile_metadata,
        "tile_files": tile_files,
        "tile_records": [
            {
                "tile_id": int(r.get("tile_id", -1)),
                "file": None if r.get("path") is None else Path(r["path"]).name,
                "n_receivers": int(r.get("n_receivers", 0)),
                "elapsed_s": float(r.get("elapsed_s", 0.0)),
                "size_mb": float(r.get("size_mb", 0.0)),
                "skipped": bool(r.get("skipped", False)),
                "empty": bool(r.get("empty", False)),
            }
            for r in tile_records
        ],
    }

    write_json(tile_output_dir / "tile_index.json", index)

    print()
    print("=" * 90)
    print("DONE SLIDING WINDOW LOBE CACHE")
    print("=" * 90)
    print("tile files:        ", len(tile_files))
    print("receiver rows:     ", total_receiver_rows)
    print("compute elapsed:   ", format_seconds(total_compute_s))
    print("loop elapsed:      ", format_seconds(total_elapsed))
    print("tile size total:   ", format_mb(total_tile_size_mb))
    print("output:            ", tile_output_dir)
    print("=" * 90)
    print()

    return {
        "tile_output_dir": tile_output_dir,
        "tile_records": tile_records,
        "tile_index": index,
        "grid_elapsed_s": float(grid_elapsed),
        "loop_elapsed_s": float(total_elapsed),
        "compute_elapsed_s": float(total_compute_s),
        "total_tile_size_mb": float(total_tile_size_mb),
    }


# ============================================================
# Tile cache reader
# ============================================================

def load_tile_lobe_cache(path):
    """
    Load one tile cache into a readable dictionary.

    This is mostly for debugging/inspection.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        receiver_ids = data["receiver_ids"]
        valid_counts = data["valid_counts"]
        caster_sids = data["caster_sids"]
        kind_codes = data["kind_codes"]
        params_i16 = data["params_i16"]
        receiver_meta = data["receiver_meta"]
        metadata = json.loads(str(data["metadata_json"]))

    cache_dict = {}

    for i, rid in enumerate(receiver_ids):
        n = int(valid_counts[i])

        params = params_i16[i, :n, :].astype(np.float32)

        # Decode quantized values.
        if n > 0:
            decoded = np.zeros((n, 5), dtype=np.float32)
            decoded[:, 0] = params[:, 0] / 10.0
            decoded[:, 1] = params[:, 1] / 10.0
            decoded[:, 2] = params[:, 2] / 10.0
            decoded[:, 3] = params[:, 3] / 10.0
            decoded[:, 4] = params[:, 4] / 1000.0
        else:
            decoded = np.zeros((0, 5), dtype=np.float32)

        cache_dict[int(rid)] = {
            "valid_count": n,
            "caster_sids": caster_sids[i, :n].astype(np.uint32),
            "kind_codes": kind_codes[i, :n].astype(np.uint8),
            "params": decoded,
            "receiver_meta": receiver_meta[i].astype(np.float32),
        }

    return {
        "metadata": metadata,
        "receiver_ids": receiver_ids,
        "valid_counts": valid_counts,
        "caster_sids": caster_sids,
        "kind_codes": kind_codes,
        "params_i16": params_i16,
        "receiver_meta": receiver_meta,
        "cache_by_receiver": cache_dict,
    }


# ============================================================
# Manifest / report
# ============================================================

def build_pipeline_manifest(
    scene,
    profile,
    arrays,
    lobe_result,
    profiler,
    output_dir,
):
    array_summary = surface_arrays_summary(arrays)

    tile_index = lobe_result["tile_index"]

    manifest = {
        "pipeline": "array_lobe_pipeline_v2",
        "output_dir": str(output_dir),

        "settings": {
            "profile_mode": PROFILE_MODE,
            "terrain_type": TERRAIN_TYPE,
            "seed": SEED,
            "generator_n_polygons": GENERATOR_N_POLYGONS,
            "built_density": BUILT_DENSITY,
            "green_area_fraction": GREEN_AREA_FRACTION,
            "average_building_height": AVERAGE_BUILDING_HEIGHT,
            "height_lognorm_sigma": HEIGHT_LOGNORM_SIGMA,
            "elevation_type": ELEVATION_TYPE,
            "receiver_surface_kinds": list(RECEIVER_SURFACE_KINDS),
            "decomposition_mode": DECOMPOSITION_MODE,
            "barrier_vertical_top_mode": BARRIER_VERTICAL_TOP_MODE,
            "core_tile_size_m": CORE_TILE_SIZE_M,
            "halo_radius_m": HALO_RADIUS_M,
            "grid_cell_size_m": GRID_CELL_SIZE_M,
            "top_k_lobes": TOP_K_LOBES,
            "include_same_building_casters": INCLUDE_SAME_BUILDING_CASTERS,
            "only_front_hemisphere": ONLY_FRONT_HEMISPHERE,
            "use_opacity_in_score": USE_OPACITY_IN_SCORE,
            "numba_requested_threads": NUMBA_REQUESTED_THREADS,
        },

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

        "profile": profile.stats,
        "profile_metadata": profile.metadata,

        "surface_arrays": array_summary,

        "lobe_cache": {
            "tile_output_dir": str(lobe_result["tile_output_dir"]),
            "n_tile_files": int(tile_index["n_tile_files"]),
            "n_receiver_rows": int(tile_index["n_receiver_rows"]),
            "total_compute_elapsed_s": float(tile_index["total_compute_elapsed_s"]),
            "total_loop_elapsed_s": float(tile_index["total_loop_elapsed_s"]),
            "total_tile_size_mb": float(tile_index["total_tile_size_mb"]),
        },

        "profiler": profiler.summary_dict(),
    }

    return manifest


def write_markdown_report(manifest, path):
    path = Path(path)

    lines = []

    lines.append("# Array lobe pipeline report")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append("- profile mode: `{}`".format(manifest["settings"]["profile_mode"]))
    lines.append("- terrain type: `{}`".format(manifest["settings"]["terrain_type"]))
    lines.append("- seed: `{}`".format(manifest["settings"]["seed"]))
    lines.append("- top K lobes: `{}`".format(manifest["settings"]["top_k_lobes"]))
    lines.append("- core tile size: `{}` m".format(manifest["settings"]["core_tile_size_m"]))
    lines.append("- halo radius: `{}` m".format(manifest["settings"]["halo_radius_m"]))
    lines.append("- grid cell size: `{}` m".format(manifest["settings"]["grid_cell_size_m"]))
    lines.append("")

    lines.append("## Scene")
    lines.append("")
    scene = manifest["scene"]
    lines.append("- buildings: `{}`".format(scene["n_buildings"]))
    lines.append("- trees: `{}`".format(scene["n_trees"]))
    lines.append("- all physical polygons: `{}`".format(scene["n_all_polygons"]))
    lines.append("- actual built density: `{}`".format(scene["actual_built_density"]))
    lines.append("- actual green fraction: `{}`".format(scene["actual_green_area_fraction"]))
    lines.append("")

    lines.append("## Shading profile")
    lines.append("")
    profile = manifest["profile"]
    lines.append("- profile polygons: `{}`".format(profile["n_profile_polygons"]))
    lines.append("- casters: `{}`".format(profile["n_casters"]))
    lines.append("- receivers: `{}`".format(profile["n_receivers"]))
    lines.append("- barrier surfaces: `{}`".format(profile["n_barrier_surfaces"]))
    lines.append("- barrier segments: `{}`".format(profile["n_barrier_segments"]))
    lines.append("- preprocessing elapsed: `{}` s".format(round(float(profile["elapsed_s"]), 4)))
    lines.append("")

    lines.append("## Arrays")
    lines.append("")
    arrays = manifest["surface_arrays"]
    lines.append("- surfaces: `{}`".format(arrays["n_surfaces"]))
    lines.append("- compact vertices: `{}`".format(arrays["n_vertices"]))
    lines.append("- memory: `{}` MB".format(round(float(arrays["memory_mb"]), 4)))
    lines.append("")

    lines.append("## Lobe cache")
    lines.append("")
    lobe = manifest["lobe_cache"]
    lines.append("- tile files: `{}`".format(lobe["n_tile_files"]))
    lines.append("- receiver rows: `{}`".format(lobe["n_receiver_rows"]))
    lines.append("- compute elapsed: `{}`".format(format_seconds(lobe["total_compute_elapsed_s"])))
    lines.append("- loop elapsed: `{}`".format(format_seconds(lobe["total_loop_elapsed_s"])))
    lines.append("- tile size total: `{}` MB".format(round(float(lobe["total_tile_size_mb"]), 4)))
    lines.append("")

    lines.append("## Timing")
    lines.append("")

    for r in manifest["profiler"]["records"]:
        lines.append(
            "- `{}`: `{}`".format(
                r["name"],
                r["elapsed_pretty"],
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


# ============================================================
# Full pipeline
# ============================================================

def generate_scene():
    scene = generate_urban_receiver_and_casters(
        n_polygons=int(GENERATOR_N_POLYGONS),
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

    return scene


def build_profile(scene):
    profile = build_shading_profile(
        scene=scene,
        profile_mode=PROFILE_MODE,
        receiver_surface_kinds=RECEIVER_SURFACE_KINDS,
        decomposition_mode=DECOMPOSITION_MODE,
        barrier_vertical_top_mode=BARRIER_VERTICAL_TOP_MODE,
        include_roof_casters=INCLUDE_ROOF_CASTERS,
        include_tree_casters=INCLUDE_TREE_CASTERS,
        include_original_building_wall_casters=INCLUDE_ORIGINAL_BUILDING_WALL_CASTERS,
        include_nonbuilding_casters=INCLUDE_NONBUILDING_CASTERS,
        optimize_rotation=OPTIMIZE_BARRIER_ROTATION,
        rotation_cost_mode=BARRIER_ROTATION_COST_MODE,
        deduplicate_barriers=DEDUPLICATE_BARRIERS,
        verbose=True,
    )

    return profile


def build_arrays(profile):
    arrays = build_surface_arrays_from_profile(
        shading_profile=profile,
        include_vertices=INCLUDE_COMPACT_VERTICES,
    )

    print_surface_arrays_summary(arrays)

    return arrays


def run_full_pipeline():
    output_dir = ensure_output_dir(
        OUTPUT_DIR,
        overwrite=OVERWRITE_OUTPUT_DIR,
    )

    profiler = PipelineProfiler("array_lobe_pipeline")

    print()
    print("=" * 100)
    print("ARRAY LOBE PIPELINE")
    print("=" * 100)
    print("output_dir:", output_dir)
    print("profile_mode:", PROFILE_MODE)
    print("=" * 100)
    print()

    with profiler.block("scene_generation"):
        scene = generate_scene()

    with profiler.block("shading_profile_preprocessing"):
        profile = build_profile(scene)

    with profiler.block("surface_array_build"):
        arrays = build_arrays(profile)

    if SAVE_SURFACE_ARRAYS:
        with profiler.block("surface_array_save"):
            surface_array_path = output_dir / "surface_arrays.npz"
            save_surface_arrays(
                arrays,
                surface_array_path,
                compressed=SAVE_SURFACE_ARRAYS_COMPRESSED,
            )
            print("saved surface arrays:", surface_array_path)
            print("surface arrays file size:", format_mb(file_size_mb(surface_array_path)))
    else:
        surface_array_path = None

    city_width, city_height = city_extent_from_scene_and_arrays(scene, arrays)

    profile_metadata = {
        "profile_mode": profile.profile_mode,
        "profile_stats": profile.stats,
        "profile_options": profile.metadata,
        "surface_arrays_memory_mb": mb_from_bytes(surface_arrays_nbytes(arrays)),
        "surface_array_path": None if surface_array_path is None else str(surface_array_path),
    }

    with profiler.block("lobe_cache_calculation"):
        lobe_result = run_sliding_window_lobe_cache(
            arrays=arrays,
            city_width=city_width,
            city_height=city_height,
            output_dir=output_dir,
            profile_metadata=profile_metadata,
        )

    profiler.finish()
    profiler.print_summary(include_metadata=False)

    if WRITE_MANIFEST:
        manifest = build_pipeline_manifest(
            scene=scene,
            profile=profile,
            arrays=arrays,
            lobe_result=lobe_result,
            profiler=profiler,
            output_dir=output_dir,
        )

        write_json(output_dir / "pipeline_manifest.json", manifest)
        write_markdown_report(manifest, output_dir / "pipeline_report.md")

        profiler.save_json(output_dir / "profiler_summary.json")

        print("wrote:", output_dir / "pipeline_manifest.json")
        print("wrote:", output_dir / "pipeline_report.md")
        print("wrote:", output_dir / "profiler_summary.json")

    print()
    print("=" * 100)
    print("PIPELINE DONE")
    print("=" * 100)
    print("profile mode:        ", PROFILE_MODE)
    print("profile polygons:    ", profile.stats["n_profile_polygons"])
    print("casters:             ", profile.stats["n_casters"])
    print("receivers:           ", profile.stats["n_receivers"])
    print("barrier surfaces:    ", profile.stats["n_barrier_surfaces"])
    print("array memory:        ", format_mb(mb_from_bytes(surface_arrays_nbytes(arrays))))
    print("tile cache dir:      ", lobe_result["tile_output_dir"])
    print("total tile size:     ", format_mb(lobe_result["total_tile_size_mb"]))
    print("total elapsed:       ", format_seconds(profiler.total_elapsed_s()))
    current_rss = get_process_memory_mb()
    if current_rss is not None:
        print("current RSS memory:  ", format_mb(current_rss))
    print("=" * 100)
    print()

    return {
        "scene": scene,
        "profile": profile,
        "arrays": arrays,
        "lobe_result": lobe_result,
        "profiler": profiler,
    }


# ============================================================
# Compatibility entry points
# ============================================================

def run_njit_sliding_lobe_cache(surfaces, city_width=None, city_height=None, output_dir=None):
    """
    Compatibility wrapper.

    Old scripts may call:
        run_njit_sliding_lobe_cache(surfaces, city_width, city_height, output_dir)

    If city dimensions are omitted, they are inferred from surface bboxes.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if city_width is None or city_height is None:
        city_width, city_height = city_extent_from_scene_and_arrays(None, surfaces)

    return run_sliding_window_lobe_cache(
        arrays=surfaces,
        city_width=float(city_width),
        city_height=float(city_height),
        output_dir=output_dir,
        profile_metadata={
            "profile_mode": "external_surface_arrays",
        },
    )


def run_pipeline_from_existing_surface_arrays(surface_array_path, output_dir=None):
    """
    Run lobe cache directly from a saved surface_arrays.npz file.
    Useful after debugging preprocessing separately.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    arrays = load_surface_arrays(surface_array_path)

    print_surface_arrays_summary(arrays)

    city_width, city_height = city_extent_from_scene_and_arrays(None, arrays)

    return run_sliding_window_lobe_cache(
        arrays=arrays,
        city_width=city_width,
        city_height=city_height,
        output_dir=output_dir,
        profile_metadata={
            "profile_mode": "loaded_surface_arrays",
            "surface_array_path": str(surface_array_path),
        },
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_full_pipeline()