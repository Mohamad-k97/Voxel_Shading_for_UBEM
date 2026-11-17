# njit_lobe_core.py

from __future__ import annotations

import os
import math
import time
import warnings

import numpy as np


# More stable on Windows/laptops than some OpenMP/TBB setups.
# Must be set before importing numba.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")


# ============================================================
# Optional Numba support
# ============================================================

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    prange = range

    def set_num_threads(n):
        return None

    def get_num_threads():
        return 1


# ============================================================
# USER SETTINGS / DEFAULTS
# ============================================================

EPS = 1e-9

TOP_K_LOBES = 40

CORE_TILE_SIZE_M = 250.0
HALO_RADIUS_M = 350.0
GRID_CELL_SIZE_M = 250.0

ONLY_FRONT_HEMISPHERE = True
INCLUDE_SAME_BUILDING_CASTERS = True
USE_OPACITY_IN_SCORE = True

MIN_SIGMA_AZ_DEG = 1.0
MIN_SIGMA_SLOPE_DEG = 1.0
MAX_SIGMA_AZ_DEG = 50.0
MAX_SIGMA_SLOPE_DEG = 50.0

SIGMA_AZ_FACTOR = 0.75
SIGMA_SLOPE_FACTOR = 0.75

# Object kind codes. Must match surface_array_builder.py.
OBJECT_KIND_UNKNOWN = 0
OBJECT_KIND_BUILDING = 1
OBJECT_KIND_TREE = 2
OBJECT_KIND_GROUND = 3
OBJECT_KIND_URBAN_OBJECT = 4

# Surface kind codes. Must match surface_array_builder.py.
SURFACE_KIND_UNKNOWN = 0
SURFACE_KIND_WALL = 1
SURFACE_KIND_FLAT_ROOF = 2
SURFACE_KIND_PITCHED_ROOF = 3
SURFACE_KIND_GROUND = 4
SURFACE_KIND_PAVED_GROUND = 5
SURFACE_KIND_TREE_CROWN = 6
SURFACE_KIND_TREE_CROWN_TOP = 7
SURFACE_KIND_TREE_TRUNK = 8
SURFACE_KIND_BARRIER_WALL_PROXY = 9
SURFACE_KIND_BARRIER_CASTER = 10

# Numba thread config.
NUMBA_REQUESTED_THREADS = None
NUMBA_MAX_THREADS_FRACTION = 0.75


# ============================================================
# Spatial grid
# ============================================================

def build_surface_grid(center_xy, can_cast, city_width, city_height, cell_size):
    """
    Build sorted grid index for caster lookup.

    Parameters
    ----------
    center_xy:
        float array, shape (N,2)

    can_cast:
        bool array, shape (N,)

    city_width, city_height:
        positive city dimensions

    cell_size:
        grid cell size

    Returns
    -------
    sorted_indices:
        caster row indices sorted by cell id

    cell_start, cell_end:
        arrays of length nx*ny.
        For each cell id, sorted_indices[cell_start:cell_end] are the casters.
        Empty cells have -1, -1.

    nx, ny:
        grid dimensions
    """
    center_xy = np.asarray(center_xy)
    can_cast = np.asarray(can_cast)

    nx = int(math.ceil(float(city_width) / float(cell_size)))
    ny = int(math.ceil(float(city_height) / float(cell_size)))

    nx = max(1, nx)
    ny = max(1, ny)

    n_cells = nx * ny

    x = center_xy[:, 0]
    y = center_xy[:, 1]

    cx = np.floor(x / float(cell_size)).astype(np.int64)
    cy = np.floor(y / float(cell_size)).astype(np.int64)

    cx = np.clip(cx, 0, nx - 1)
    cy = np.clip(cy, 0, ny - 1)

    cell_id = cy * nx + cx

    caster_indices = np.where(can_cast)[0].astype(np.int64)
    caster_cell_id = cell_id[caster_indices]

    order = np.argsort(caster_cell_id)
    sorted_indices = caster_indices[order]
    sorted_cell_id = caster_cell_id[order]

    cell_start = np.full(n_cells, -1, dtype=np.int64)
    cell_end = np.full(n_cells, -1, dtype=np.int64)

    if len(sorted_cell_id) > 0:
        unique_cells, starts = np.unique(sorted_cell_id, return_index=True)

        for c, s in zip(unique_cells, starts):
            cell_start[int(c)] = int(s)

        ends = np.r_[starts[1:], len(sorted_cell_id)]

        for c, e in zip(unique_cells, ends):
            cell_end[int(c)] = int(e)

    return sorted_indices, cell_start, cell_end, nx, ny


def make_tiles(city_width, city_height, core_size, halo):
    """
    Create core/halo tiles.

    Returns list of:
        (tile_id, core_bbox, halo_bbox)

    bbox format:
        xmin, ymin, xmax, ymax
    """
    tiles = []
    tile_id = 0

    city_width = float(city_width)
    city_height = float(city_height)
    core_size = float(core_size)
    halo = float(halo)

    xs = np.arange(0.0, city_width, core_size)
    ys = np.arange(0.0, city_height, core_size)

    for x0 in xs:
        for y0 in ys:
            x1 = min(x0 + core_size, city_width)
            y1 = min(y0 + core_size, city_height)

            core = (
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )

            halo_box = (
                float(x0 - halo),
                float(y0 - halo),
                float(x1 + halo),
                float(y1 + halo),
            )

            tiles.append((int(tile_id), core, halo_box))
            tile_id += 1

    return tiles


def get_receiver_indices_for_tile(center_xy, can_receive, core_bbox):
    """
    Return receiver row indices whose centers fall inside tile core bbox.
    """
    center_xy = np.asarray(center_xy)
    can_receive = np.asarray(can_receive)

    x0, y0, x1, y1 = core_bbox

    x = center_xy[:, 0]
    y = center_xy[:, 1]

    mask = (
        can_receive
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    )

    return np.where(mask)[0].astype(np.int64)


# ============================================================
# Safe Numba runtime
# ============================================================

def configure_numba_threads(requested_threads=None, verbose=True):
    """
    Configure Numba parallel threads safely.

    Returns dictionary:
        {
            "numba_available": bool,
            "threads": int,
            "mode": "parallel" / "serial" / "python"
        }
    """
    if not NUMBA_AVAILABLE:
        if verbose:
            print("Numba is not available. Falling back to Python mode.")

        return {
            "numba_available": False,
            "threads": 1,
            "mode": "python",
        }

    cpu_count = os.cpu_count() or 1

    if requested_threads is None:
        safe_threads = max(1, int(cpu_count * NUMBA_MAX_THREADS_FRACTION))
    else:
        safe_threads = int(requested_threads)

    safe_threads = max(1, min(safe_threads, cpu_count))

    try:
        set_num_threads(safe_threads)
        actual = int(get_num_threads())
    except Exception:
        actual = 1

    mode = "parallel" if actual > 1 else "serial"

    if verbose:
        print()
        print("=" * 80)
        print("NUMBA THREAD CONFIG")
        print("=" * 80)
        print("numba available:", NUMBA_AVAILABLE)
        print("cpu count:", cpu_count)
        print("requested threads:", requested_threads)
        print("actual threads:", actual)
        print("mode:", mode)
        print("=" * 80)

    return {
        "numba_available": True,
        "threads": actual,
        "mode": mode,
    }


# ============================================================
# Scalar helpers used by kernels
# ============================================================

@njit(cache=True)
def _wrap_360(angle):
    out = angle % 360.0

    if out < 0.0:
        out += 360.0

    return out


@njit(cache=True)
def _wrap_180(angle):
    out = (angle + 180.0) % 360.0 - 180.0

    return out


@njit(cache=True)
def _horizontal_azimuth_from_xy(dx, dy):
    """
    Azimuth convention:
        0   = North / +y
        90  = East  / +x
        180 = South / -y
        270 = West  / -x
    """
    h = math.sqrt(dx * dx + dy * dy)

    if h < EPS:
        return 0.0

    az = math.degrees(math.atan2(dx, dy))

    return _wrap_360(az)


@njit(cache=True)
def _slope_from_vector(dx, dy, dz):
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < EPS:
        return 0.0

    s = dz / dist

    if s < -1.0:
        s = -1.0
    elif s > 1.0:
        s = 1.0

    return math.degrees(math.asin(s))


@njit(cache=True)
def _surface_horizontal_azimuth(nx, ny):
    h = math.sqrt(nx * nx + ny * ny)

    if h < EPS:
        return 0.0

    az = math.degrees(math.atan2(nx, ny))

    return _wrap_360(az)


@njit(cache=True)
def _clamp(x, lo, hi):
    if x < lo:
        return lo

    if x > hi:
        return hi

    return x


@njit(cache=True)
def _caster_kind_weight(surface_kind_code):
    """
    Optional weak kind-based weighting.

    Keep almost neutral. The lobe score should mostly be area / distance^2.
    Barrier surfaces are deliberately treated like normal wall casters.
    """
    if surface_kind_code == SURFACE_KIND_TREE_CROWN:
        return 0.90

    if surface_kind_code == SURFACE_KIND_TREE_CROWN_TOP:
        return 0.75

    if surface_kind_code == SURFACE_KIND_TREE_TRUNK:
        return 0.55

    if surface_kind_code == SURFACE_KIND_GROUND:
        return 0.20

    if surface_kind_code == SURFACE_KIND_PAVED_GROUND:
        return 0.20

    return 1.0


@njit(cache=True)
def _insert_top_k(
    score,
    caster_sid,
    surface_kind_code,
    center_rel_az,
    center_slope,
    sigma_az,
    sigma_slope,
    top_scores,
    top_caster_sids,
    top_kind_codes,
    top_params,
):
    """
    Insert one candidate into descending top-K arrays.

    top_params columns:
        0 center_rel_az
        1 center_slope
        2 sigma_az
        3 sigma_slope
        4 raw_score
    """
    k_max = top_scores.shape[0]

    if score <= top_scores[k_max - 1]:
        return

    pos = k_max - 1

    while pos > 0 and score > top_scores[pos - 1]:
        top_scores[pos] = top_scores[pos - 1]
        top_caster_sids[pos] = top_caster_sids[pos - 1]
        top_kind_codes[pos] = top_kind_codes[pos - 1]

        top_params[pos, 0] = top_params[pos - 1, 0]
        top_params[pos, 1] = top_params[pos - 1, 1]
        top_params[pos, 2] = top_params[pos - 1, 2]
        top_params[pos, 3] = top_params[pos - 1, 3]
        top_params[pos, 4] = top_params[pos - 1, 4]

        pos -= 1

    top_scores[pos] = score
    top_caster_sids[pos] = caster_sid
    top_kind_codes[pos] = surface_kind_code

    top_params[pos, 0] = center_rel_az
    top_params[pos, 1] = center_slope
    top_params[pos, 2] = sigma_az
    top_params[pos, 3] = sigma_slope
    top_params[pos, 4] = score


@njit(cache=True)
def _quantize_and_store_topk(
    rr,
    top_k,
    top_scores,
    top_caster_sids,
    top_kind_codes,
    top_params,
    valid_counts,
    caster_sids_out,
    kind_codes_out,
    params_i16_out,
):
    count = 0

    for k in range(top_k):
        if top_scores[k] > EPS:
            count += 1

    valid_counts[rr] = count

    if count <= 0:
        return

    max_score = top_scores[0]

    if max_score <= EPS:
        return

    for k in range(count):
        amp = top_scores[k] / max_score

        az10 = int(round(top_params[k, 0] * 10.0))
        sl10 = int(round(top_params[k, 1] * 10.0))
        saz10 = int(round(top_params[k, 2] * 10.0))
        ssl10 = int(round(top_params[k, 3] * 10.0))
        amp1000 = int(round(amp * 1000.0))

        if az10 < -1800:
            az10 = -1800
        elif az10 > 1800:
            az10 = 1800

        if sl10 < -900:
            sl10 = -900
        elif sl10 > 900:
            sl10 = 900

        if saz10 < 0:
            saz10 = 0
        elif saz10 > 1800:
            saz10 = 1800

        if ssl10 < 0:
            ssl10 = 0
        elif ssl10 > 900:
            ssl10 = 900

        if amp1000 < 0:
            amp1000 = 0
        elif amp1000 > 1000:
            amp1000 = 1000

        caster_sids_out[rr, k] = top_caster_sids[k]
        kind_codes_out[rr, k] = top_kind_codes[k]

        params_i16_out[rr, k, 0] = az10
        params_i16_out[rr, k, 1] = sl10
        params_i16_out[rr, k, 2] = saz10
        params_i16_out[rr, k, 3] = ssl10
        params_i16_out[rr, k, 4] = amp1000


# ============================================================
# Numba kernel
# ============================================================

@njit(parallel=True, fastmath=True, cache=True)
def compute_tile_lobes_njit(
    receiver_idx,
    sorted_caster_indices,
    cell_start,
    cell_end,
    grid_nx,
    grid_ny,
    grid_cell_size,
    halo_bbox,

    surface_id,
    object_id,
    object_kind,
    surface_kind,
    center,
    normal,
    area,
    radius_xy,
    radius_z,
    radius_full,
    opacity,

    include_same_building,
    only_front_hemisphere,
    use_opacity_in_score,

    top_k,
):
    """
    Compute compact angular lobe vectors for all receivers in one tile.

    Output
    ------
    valid_counts:
        uint16, shape (R,)

    caster_sids_out:
        uint32, shape (R,K)

    kind_codes_out:
        uint8, shape (R,K)

    params_i16_out:
        int16, shape (R,K,5)

    params_i16 columns:
        0 center_rel_azimuth_deg * 10
        1 center_slope_deg * 10
        2 sigma_azimuth_deg * 10
        3 sigma_slope_deg * 10
        4 amplitude * 1000
    """
    R = receiver_idx.shape[0]

    valid_counts = np.zeros(R, dtype=np.uint16)
    caster_sids_out = np.zeros((R, top_k), dtype=np.uint32)
    kind_codes_out = np.zeros((R, top_k), dtype=np.uint8)
    params_i16_out = np.zeros((R, top_k, 5), dtype=np.int16)

    hx0 = halo_bbox[0]
    hy0 = halo_bbox[1]
    hx1 = halo_bbox[2]
    hy1 = halo_bbox[3]

    cx0 = int(math.floor(hx0 / grid_cell_size))
    cy0 = int(math.floor(hy0 / grid_cell_size))
    cx1 = int(math.floor(hx1 / grid_cell_size))
    cy1 = int(math.floor(hy1 / grid_cell_size))

    if cx0 < 0:
        cx0 = 0

    if cy0 < 0:
        cy0 = 0

    if cx1 >= grid_nx:
        cx1 = grid_nx - 1

    if cy1 >= grid_ny:
        cy1 = grid_ny - 1

    for rr in prange(R):
        r_idx = receiver_idx[rr]

        rx = float(center[r_idx, 0])
        ry = float(center[r_idx, 1])
        rz = float(center[r_idx, 2])

        rnx = float(normal[r_idx, 0])
        rny = float(normal[r_idx, 1])
        rnz = float(normal[r_idx, 2])

        receiver_az = _surface_horizontal_azimuth(rnx, rny)

        top_scores = np.zeros(top_k, dtype=np.float32)
        top_caster_sids = np.zeros(top_k, dtype=np.uint32)
        top_kind_codes = np.zeros(top_k, dtype=np.uint8)
        top_params = np.zeros((top_k, 5), dtype=np.float32)

        r_object_id = object_id[r_idx]

        for gy in range(cy0, cy1 + 1):
            for gx in range(cx0, cx1 + 1):
                cell_id = gy * grid_nx + gx

                start = cell_start[cell_id]
                end = cell_end[cell_id]

                if start < 0 or end <= start:
                    continue

                for pp in range(start, end):
                    c_idx = sorted_caster_indices[pp]

                    if c_idx == r_idx:
                        continue

                    if not include_same_building:
                        if object_kind[c_idx] == OBJECT_KIND_BUILDING:
                            if object_id[c_idx] == r_object_id:
                                continue

                    cx = float(center[c_idx, 0])
                    cy = float(center[c_idx, 1])
                    cz = float(center[c_idx, 2])

                    # Must be inside halo box by caster center.
                    # The grid can include cells that extend beyond halo.
                    if cx < hx0 or cx > hx1 or cy < hy0 or cy > hy1:
                        continue

                    dx = cx - rx
                    dy = cy - ry
                    dz = cz - rz

                    hdist2 = dx * dx + dy * dy
                    dist2 = hdist2 + dz * dz

                    if dist2 <= EPS:
                        continue

                    dist = math.sqrt(dist2)
                    hdist = math.sqrt(max(EPS, hdist2))

                    inv_dist = 1.0 / dist

                    ux = dx * inv_dist
                    uy = dy * inv_dist
                    uz = dz * inv_dist

                    # Receiver front factor.
                    # For vertical receivers, this means "in front of the wall".
                    # For roof/ground receivers, horizontal normal is weak;
                    # the dot product with 3D normal still keeps physically
                    # reasonable directions.
                    front_dot = rnx * ux + rny * uy + rnz * uz

                    if only_front_hemisphere and front_dot <= 0.0:
                        continue

                    front_factor = front_dot

                    if front_factor < 0.0:
                        front_factor = 0.0

                    if front_factor > 1.0:
                        front_factor = 1.0

                    # Horizontal angular position relative to receiver normal.
                    caster_az = _horizontal_azimuth_from_xy(dx, dy)
                    center_rel_az = _wrap_180(caster_az - receiver_az)

                    if only_front_hemisphere:
                        # For horizontal receivers receiver_az is arbitrary.
                        # The front_dot test above is the main physical filter.
                        # This extra azimuth filter should only matter for
                        # near-vertical surfaces.
                        rh = math.sqrt(rnx * rnx + rny * rny)

                        if rh > 0.05:
                            if center_rel_az < -90.0 or center_rel_az > 90.0:
                                continue

                    center_slope = _slope_from_vector(dx, dy, dz)

                    # Angular spread approximation.
                    # radius_xy controls azimuth width.
                    # radius_z controls elevation width.
                    rxy = float(radius_xy[c_idx])
                    rz_c = float(radius_z[c_idx])
                    rf = float(radius_full[c_idx])

                    sigma_az = math.degrees(math.atan2(max(EPS, rxy), max(EPS, hdist)))
                    sigma_slope = math.degrees(math.atan2(max(EPS, rz_c + 0.25 * rf), max(EPS, dist)))

                    sigma_az *= SIGMA_AZ_FACTOR
                    sigma_slope *= SIGMA_SLOPE_FACTOR

                    sigma_az = _clamp(sigma_az, MIN_SIGMA_AZ_DEG, MAX_SIGMA_AZ_DEG)
                    sigma_slope = _clamp(sigma_slope, MIN_SIGMA_SLOPE_DEG, MAX_SIGMA_SLOPE_DEG)

                    # Apparent angular importance.
                    a = float(area[c_idx])

                    if a <= EPS:
                        continue

                    score = a * max(EPS, front_factor) / max(EPS, dist2)

                    if use_opacity_in_score:
                        score *= float(opacity[c_idx])

                    score *= _caster_kind_weight(surface_kind[c_idx])

                    if score <= EPS:
                        continue

                    _insert_top_k(
                        float(score),
                        surface_id[c_idx],
                        surface_kind[c_idx],
                        float(center_rel_az),
                        float(center_slope),
                        float(sigma_az),
                        float(sigma_slope),
                        top_scores,
                        top_caster_sids,
                        top_kind_codes,
                        top_params,
                    )

        _quantize_and_store_topk(
            rr,
            top_k,
            top_scores,
            top_caster_sids,
            top_kind_codes,
            top_params,
            valid_counts,
            caster_sids_out,
            kind_codes_out,
            params_i16_out,
        )

    return valid_counts, caster_sids_out, kind_codes_out, params_i16_out


# ============================================================
# Pure Python fallback
# ============================================================

def compute_tile_lobes_python(
    receiver_idx,
    sorted_caster_indices,
    cell_start,
    cell_end,
    grid_nx,
    grid_ny,
    grid_cell_size,
    halo_bbox,

    surface_id,
    object_id,
    object_kind,
    surface_kind,
    center,
    normal,
    area,
    radius_xy,
    radius_z,
    radius_full,
    opacity,

    include_same_building,
    only_front_hemisphere,
    use_opacity_in_score,

    top_k,
):
    """
    Slow pure-Python fallback with the same output format.

    This is not meant for large production runs. It is here so the pipeline
    still works when Numba is unavailable or broken.
    """
    R = len(receiver_idx)

    valid_counts = np.zeros(R, dtype=np.uint16)
    caster_sids_out = np.zeros((R, top_k), dtype=np.uint32)
    kind_codes_out = np.zeros((R, top_k), dtype=np.uint8)
    params_i16_out = np.zeros((R, top_k, 5), dtype=np.int16)

    hx0, hy0, hx1, hy1 = [float(x) for x in halo_bbox]

    cx0 = max(0, int(math.floor(hx0 / grid_cell_size)))
    cy0 = max(0, int(math.floor(hy0 / grid_cell_size)))
    cx1 = min(int(grid_nx) - 1, int(math.floor(hx1 / grid_cell_size)))
    cy1 = min(int(grid_ny) - 1, int(math.floor(hy1 / grid_cell_size)))

    for rr, r_idx in enumerate(receiver_idx):
        r_idx = int(r_idx)

        rx, ry, rz = center[r_idx]
        rnx, rny, rnz = normal[r_idx]

        receiver_az = _py_surface_horizontal_azimuth(float(rnx), float(rny))

        candidates = []

        r_object_id = int(object_id[r_idx])

        for gy in range(cy0, cy1 + 1):
            for gx in range(cx0, cx1 + 1):
                cell_id = gy * int(grid_nx) + gx

                start = int(cell_start[cell_id])
                end = int(cell_end[cell_id])

                if start < 0 or end <= start:
                    continue

                for pp in range(start, end):
                    c_idx = int(sorted_caster_indices[pp])

                    if c_idx == r_idx:
                        continue

                    if not include_same_building:
                        if int(object_kind[c_idx]) == OBJECT_KIND_BUILDING:
                            if int(object_id[c_idx]) == r_object_id:
                                continue

                    cx, cy, cz = center[c_idx]

                    if cx < hx0 or cx > hx1 or cy < hy0 or cy > hy1:
                        continue

                    dx = float(cx - rx)
                    dy = float(cy - ry)
                    dz = float(cz - rz)

                    hdist2 = dx * dx + dy * dy
                    dist2 = hdist2 + dz * dz

                    if dist2 <= EPS:
                        continue

                    dist = math.sqrt(dist2)
                    hdist = math.sqrt(max(EPS, hdist2))

                    ux = dx / dist
                    uy = dy / dist
                    uz = dz / dist

                    front_dot = float(rnx) * ux + float(rny) * uy + float(rnz) * uz

                    if only_front_hemisphere and front_dot <= 0.0:
                        continue

                    front_factor = max(0.0, min(1.0, front_dot))

                    caster_az = _py_horizontal_azimuth_from_xy(dx, dy)
                    center_rel_az = _py_wrap_180(caster_az - receiver_az)

                    if only_front_hemisphere:
                        rh = math.sqrt(float(rnx) * float(rnx) + float(rny) * float(rny))

                        if rh > 0.05:
                            if center_rel_az < -90.0 or center_rel_az > 90.0:
                                continue

                    center_slope = _py_slope_from_vector(dx, dy, dz)

                    rxy = float(radius_xy[c_idx])
                    rz_c = float(radius_z[c_idx])
                    rf = float(radius_full[c_idx])

                    sigma_az = math.degrees(math.atan2(max(EPS, rxy), max(EPS, hdist)))
                    sigma_slope = math.degrees(math.atan2(max(EPS, rz_c + 0.25 * rf), max(EPS, dist)))

                    sigma_az *= SIGMA_AZ_FACTOR
                    sigma_slope *= SIGMA_SLOPE_FACTOR

                    sigma_az = max(MIN_SIGMA_AZ_DEG, min(MAX_SIGMA_AZ_DEG, sigma_az))
                    sigma_slope = max(MIN_SIGMA_SLOPE_DEG, min(MAX_SIGMA_SLOPE_DEG, sigma_slope))

                    a = float(area[c_idx])

                    if a <= EPS:
                        continue

                    score = a * max(EPS, front_factor) / max(EPS, dist2)

                    if use_opacity_in_score:
                        score *= float(opacity[c_idx])

                    score *= _py_caster_kind_weight(int(surface_kind[c_idx]))

                    if score <= EPS:
                        continue

                    candidates.append((
                        score,
                        int(surface_id[c_idx]),
                        int(surface_kind[c_idx]),
                        center_rel_az,
                        center_slope,
                        sigma_az,
                        sigma_slope,
                    ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[0])
        candidates = candidates[:top_k]

        max_score = candidates[0][0]

        valid_counts[rr] = len(candidates)

        for k, c in enumerate(candidates):
            score = c[0]
            amp = 0.0 if max_score <= EPS else score / max_score

            caster_sids_out[rr, k] = np.uint32(c[1])
            kind_codes_out[rr, k] = np.uint8(c[2])

            params_i16_out[rr, k, 0] = np.int16(round(max(-180.0, min(180.0, c[3])) * 10.0))
            params_i16_out[rr, k, 1] = np.int16(round(max(-90.0, min(90.0, c[4])) * 10.0))
            params_i16_out[rr, k, 2] = np.int16(round(max(0.0, min(180.0, c[5])) * 10.0))
            params_i16_out[rr, k, 3] = np.int16(round(max(0.0, min(90.0, c[6])) * 10.0))
            params_i16_out[rr, k, 4] = np.int16(round(max(0.0, min(1.0, amp)) * 1000.0))

    return valid_counts, caster_sids_out, kind_codes_out, params_i16_out


def _py_wrap_360(angle):
    out = float(angle) % 360.0

    if out < 0.0:
        out += 360.0

    return out


def _py_wrap_180(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def _py_horizontal_azimuth_from_xy(dx, dy):
    h = math.sqrt(dx * dx + dy * dy)

    if h < EPS:
        return 0.0

    return _py_wrap_360(math.degrees(math.atan2(dx, dy)))


def _py_surface_horizontal_azimuth(nx, ny):
    h = math.sqrt(nx * nx + ny * ny)

    if h < EPS:
        return 0.0

    return _py_wrap_360(math.degrees(math.atan2(nx, ny)))


def _py_slope_from_vector(dx, dy, dz):
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < EPS:
        return 0.0

    s = dz / dist
    s = max(-1.0, min(1.0, s))

    return math.degrees(math.asin(s))


def _py_caster_kind_weight(surface_kind_code):
    if surface_kind_code == SURFACE_KIND_TREE_CROWN:
        return 0.90

    if surface_kind_code == SURFACE_KIND_TREE_CROWN_TOP:
        return 0.75

    if surface_kind_code == SURFACE_KIND_TREE_TRUNK:
        return 0.55

    if surface_kind_code == SURFACE_KIND_GROUND:
        return 0.20

    if surface_kind_code == SURFACE_KIND_PAVED_GROUND:
        return 0.20

    return 1.0


# ============================================================
# Safe wrapper
# ============================================================

def compute_tile_lobes_safe(
    receiver_idx,
    sorted_caster_indices,
    cell_start,
    cell_end,
    grid_nx,
    grid_ny,
    grid_cell_size,
    halo_bbox,

    surface_id,
    object_id,
    object_kind,
    surface_kind,
    center,
    normal,
    area,
    radius_xy,
    radius_z,
    radius_full,
    opacity,

    include_same_building,
    only_front_hemisphere,
    use_opacity_in_score,

    top_k,

    requested_threads=None,
):
    """
    Safe wrapper.

    Attempts:
        1. parallel Numba
        2. serial Numba
        3. pure Python fallback

    This is intentionally defensive because some Windows/laptop Numba setups
    fail during parallel execution.
    """
    if not NUMBA_AVAILABLE:
        print()
        print("WARNING: Numba is not available. Using slow Python fallback.")

        return compute_tile_lobes_python(
            receiver_idx,
            sorted_caster_indices,
            cell_start,
            cell_end,
            grid_nx,
            grid_ny,
            grid_cell_size,
            halo_bbox,

            surface_id,
            object_id,
            object_kind,
            surface_kind,
            center,
            normal,
            area,
            radius_xy,
            radius_z,
            radius_full,
            opacity,

            include_same_building,
            only_front_hemisphere,
            use_opacity_in_score,

            top_k,
        )

    configure_numba_threads(
        requested_threads=requested_threads,
        verbose=False,
    )

    try:
        return compute_tile_lobes_njit(
            receiver_idx,
            sorted_caster_indices,
            cell_start,
            cell_end,
            grid_nx,
            grid_ny,
            grid_cell_size,
            halo_bbox,

            surface_id,
            object_id,
            object_kind,
            surface_kind,
            center,
            normal,
            area,
            radius_xy,
            radius_z,
            radius_full,
            opacity,

            include_same_building,
            only_front_hemisphere,
            use_opacity_in_score,

            top_k,
        )

    except Exception as e:
        print()
        print("WARNING: parallel Numba call failed.")
        print("Reason:", repr(e))
        print("Retrying with 1 Numba thread.")

        try:
            set_num_threads(1)
        except Exception:
            pass

    try:
        return compute_tile_lobes_njit(
            receiver_idx,
            sorted_caster_indices,
            cell_start,
            cell_end,
            grid_nx,
            grid_ny,
            grid_cell_size,
            halo_bbox,

            surface_id,
            object_id,
            object_kind,
            surface_kind,
            center,
            normal,
            area,
            radius_xy,
            radius_z,
            radius_full,
            opacity,

            include_same_building,
            only_front_hemisphere,
            use_opacity_in_score,

            top_k,
        )

    except Exception as e:
        print()
        print("WARNING: serial Numba call failed too.")
        print("Reason:", repr(e))
        print("Falling back to slow Python implementation.")

        return compute_tile_lobes_python(
            receiver_idx,
            sorted_caster_indices,
            cell_start,
            cell_end,
            grid_nx,
            grid_ny,
            grid_cell_size,
            halo_bbox,

            surface_id,
            object_id,
            object_kind,
            surface_kind,
            center,
            normal,
            area,
            radius_xy,
            radius_z,
            radius_full,
            opacity,

            include_same_building,
            only_front_hemisphere,
            use_opacity_in_score,

            top_k,
        )


# ============================================================
# Tiny self-test
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 80)
    print("njit_lobe_core self-test")
    print("=" * 80)

    configure_numba_threads(requested_threads=1, verbose=True)

    # One vertical wall receiver at origin, facing east.
    # Two caster surfaces in front of it.
    surface_id = np.array([1, 2, 3], dtype=np.uint32)
    object_id = np.array([1, 2, 3], dtype=np.uint32)
    object_kind = np.array([OBJECT_KIND_BUILDING, OBJECT_KIND_BUILDING, OBJECT_KIND_TREE], dtype=np.uint8)
    surface_kind = np.array([SURFACE_KIND_WALL, SURFACE_KIND_BARRIER_WALL_PROXY, SURFACE_KIND_TREE_CROWN], dtype=np.uint8)

    center = np.array(
        [
            [10.0, 10.0, 5.0],
            [25.0, 10.0, 6.0],
            [35.0, 14.0, 8.0],
        ],
        dtype=np.float32,
    )

    normal = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    area = np.array([20.0, 50.0, 25.0], dtype=np.float32)
    radius_xy = np.array([2.0, 5.0, 3.0], dtype=np.float32)
    radius_z = np.array([2.0, 5.0, 3.0], dtype=np.float32)
    radius_full = np.array([3.0, 7.0, 5.0], dtype=np.float32)
    opacity = np.array([1.0, 1.0, 0.6], dtype=np.float32)

    can_cast = np.array([False, True, True], dtype=bool)
    can_receive = np.array([True, False, False], dtype=bool)

    sorted_indices, cell_start, cell_end, nx, ny = build_surface_grid(
        center_xy=center[:, :2],
        can_cast=can_cast,
        city_width=100.0,
        city_height=100.0,
        cell_size=25.0,
    )

    receiver_idx = np.array([0], dtype=np.int64)
    halo_bbox = np.array([0.0, 0.0, 80.0, 80.0], dtype=np.float64)

    valid_counts, caster_sids, kind_codes, params_i16 = compute_tile_lobes_safe(
        receiver_idx=receiver_idx,
        sorted_caster_indices=sorted_indices,
        cell_start=cell_start,
        cell_end=cell_end,
        grid_nx=nx,
        grid_ny=ny,
        grid_cell_size=25.0,
        halo_bbox=halo_bbox,

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
        opacity=opacity,

        include_same_building=True,
        only_front_hemisphere=True,
        use_opacity_in_score=True,

        top_k=5,
        requested_threads=1,
    )

    print("valid_counts:", valid_counts)
    print("caster_sids:", caster_sids)
    print("kind_codes:", kind_codes)
    print("params_i16:", params_i16)
    print("=" * 80)