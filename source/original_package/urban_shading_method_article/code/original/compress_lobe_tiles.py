# compress_lobe_tiles.py

from __future__ import annotations

from pathlib import Path
import json
import time
import shutil

import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_TILE_DIR = Path("array_city_lobe_output/tiles_njit")
OUTPUT_ARCHIVE_DIR = Path("array_city_lobe_output/lobe_archive")

OVERWRITE = True

# If True, original uncompressed tiles are deleted after successful compression.
DELETE_UNCOMPRESSED_AFTER = False


# ============================================================
# Helpers
# ============================================================

def load_npz_metadata(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" in data:
            return json.loads(str(data["metadata_json"]))
    return {}


def compress_one_tile(src_path: Path, dst_path: Path):
    """
    Read one uncompressed .npz tile and write compressed .npz tile.

    Keeps the same array names.
    """

    with np.load(src_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(dst_path, **arrays)


def build_receiver_index(compressed_tile_paths):
    """
    Build global receiver lookup table.

    Output arrays:
        receiver_ids
        tile_ids
        local_rows

    Sorted by receiver_id for fast binary search.
    """

    receiver_ids_all = []
    tile_ids_all = []
    local_rows_all = []

    for tile_path in compressed_tile_paths:
        with np.load(tile_path, allow_pickle=False) as data:
            receiver_ids = data["receiver_ids"]

            metadata = json.loads(str(data["metadata_json"]))
            tile_id = int(metadata["tile_id"])

            n = len(receiver_ids)

            receiver_ids_all.append(receiver_ids.astype(np.uint32))
            tile_ids_all.append(np.full(n, tile_id, dtype=np.uint32))
            local_rows_all.append(np.arange(n, dtype=np.uint32))

    if not receiver_ids_all:
        return {
            "receiver_ids": np.array([], dtype=np.uint32),
            "tile_ids": np.array([], dtype=np.uint32),
            "local_rows": np.array([], dtype=np.uint32),
        }

    receiver_ids = np.concatenate(receiver_ids_all)
    tile_ids = np.concatenate(tile_ids_all)
    local_rows = np.concatenate(local_rows_all)

    order = np.argsort(receiver_ids)

    return {
        "receiver_ids": receiver_ids[order],
        "tile_ids": tile_ids[order],
        "local_rows": local_rows[order],
    }


def write_manifest(
    output_dir: Path,
    compressed_tile_paths,
    receiver_index,
    source_tile_dir: Path,
    elapsed_s: float,
):
    tile_files = [p.name for p in compressed_tile_paths]

    n_receivers = int(len(receiver_index["receiver_ids"]))

    tile_summaries = []

    total_size_mb = 0.0

    for p in compressed_tile_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        total_size_mb += size_mb

        meta = load_npz_metadata(p)

        tile_summaries.append({
            "file": p.name,
            "tile_id": int(meta.get("tile_id", -1)),
            "n_receivers": int(meta.get("n_receivers", -1)),
            "top_k_lobes": int(meta.get("top_k_lobes", -1)),
            "size_mb": round(size_mb, 4),
        })

    manifest = {
        "archive_format": "receiver_lobe_tile_archive_v1",
        "description": "Compressed tile-wise lobe cache. Readable with numpy.load and json.",
        "source_tile_dir": str(source_tile_dir),
        "n_tiles": len(tile_files),
        "n_receivers": n_receivers,
        "total_tile_size_mb": round(total_size_mb, 4),
        "compression_elapsed_s": round(elapsed_s, 3),
        "files": {
            "receiver_index": "receiver_index.npz",
            "tiles_dir": "tiles",
        },
        "tile_files": tile_files,
        "tile_summaries": tile_summaries,
        "receiver_index_columns": {
            "receiver_ids": "uint32 receiver surface id",
            "tile_ids": "uint32 tile id containing the receiver",
            "local_rows": "uint32 row of receiver inside tile arrays",
        },
        "tile_array_names": [
            "receiver_ids",
            "valid_counts",
            "caster_sids",
            "kind_codes",
            "params_i16",
            "receiver_meta",
            "metadata_json",
        ],
        "params_i16_columns": [
            "center_rel_azimuth_deg_x10",
            "center_slope_deg_x10",
            "sigma_azimuth_deg_x10",
            "sigma_slope_deg_x10",
            "amplitude_x1000",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


# ============================================================
# Main compression function
# ============================================================

def compress_lobe_tiles(
    input_tile_dir: Path = INPUT_TILE_DIR,
    output_archive_dir: Path = OUTPUT_ARCHIVE_DIR,
):
    t0 = time.perf_counter()

    if not input_tile_dir.exists():
        raise FileNotFoundError(f"Input tile folder does not exist: {input_tile_dir}")

    tile_paths = sorted(input_tile_dir.glob("tile_*.npz"))

    if not tile_paths:
        raise RuntimeError(f"No tile_*.npz files found in: {input_tile_dir}")

    if output_archive_dir.exists() and OVERWRITE:
        shutil.rmtree(output_archive_dir)

    output_archive_dir.mkdir(parents=True, exist_ok=True)

    out_tile_dir = output_archive_dir / "tiles"
    out_tile_dir.mkdir(parents=True, exist_ok=True)

    compressed_paths = []

    print()
    print("=" * 90)
    print("COMPRESSING LOBE TILE CACHE")
    print("=" * 90)
    print("input :", input_tile_dir)
    print("output:", output_archive_dir)
    print("tiles :", len(tile_paths))
    print("=" * 90)

    for i, src in enumerate(tile_paths):
        dst = out_tile_dir / src.name

        print(f"[{i + 1}/{len(tile_paths)}] {src.name}")

        compress_one_tile(src, dst)
        compressed_paths.append(dst)

        src_mb = src.stat().st_size / (1024 * 1024)
        dst_mb = dst.stat().st_size / (1024 * 1024)
        ratio = dst_mb / src_mb if src_mb > 0 else 0.0

        print(f"    {src_mb:.3f} MB -> {dst_mb:.3f} MB | ratio={ratio:.3f}")

        if DELETE_UNCOMPRESSED_AFTER:
            src.unlink()

    print()
    print("Building receiver index...")

    receiver_index = build_receiver_index(compressed_paths)

    np.savez_compressed(
        output_archive_dir / "receiver_index.npz",
        receiver_ids=receiver_index["receiver_ids"],
        tile_ids=receiver_index["tile_ids"],
        local_rows=receiver_index["local_rows"],
    )

    elapsed = time.perf_counter() - t0

    write_manifest(
        output_dir=output_archive_dir,
        compressed_tile_paths=compressed_paths,
        receiver_index=receiver_index,
        source_tile_dir=input_tile_dir,
        elapsed_s=elapsed,
    )

    total_in = sum(p.stat().st_size for p in tile_paths if p.exists()) / (1024 * 1024)
    total_out = sum(p.stat().st_size for p in compressed_paths) / (1024 * 1024)

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"input tile size:  {total_in:.3f} MB")
    print(f"output tile size: {total_out:.3f} MB")
    if total_in > 0:
        print(f"ratio:            {total_out / total_in:.3f}")
    print(f"elapsed:          {elapsed:.2f} s")
    print("archive:", output_archive_dir)
    print("=" * 90)


if __name__ == "__main__":
    compress_lobe_tiles()