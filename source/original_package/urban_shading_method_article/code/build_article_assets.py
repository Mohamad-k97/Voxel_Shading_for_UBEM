#!/usr/bin/env python3
"""Build result macros, tables and Python-only PDF figures for the article."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

import benchmark_study as study


ACCENT = "#9B0014"
BLUE = "#355C7D"
TEAL = "#2A7F7F"
GOLD = "#C28E2C"
DARK = "#252525"
MID = "#777777"
LIGHT = "#D9D9D9"
PALE = "#F3F1EE"

CASE_STUDIES = (
    ("tall_buildings", "Compact high-rise core", "HighRise"),
    ("horizontal_city", "Horizontal low-rise district", "LowRise"),
    ("small_town", "Small-town mixed fabric", "SmallTown"),
    ("suburban", "Dispersed suburban fabric", "Suburban"),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        str(path),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


def clean_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        True,
        axis=grid_axis,
        color="#E5E5E5",
        linewidth=0.6,
        zorder=0,
    )


def panel_label(axis: plt.Axes, text: str) -> None:
    axis.text(
        -0.10,
        1.04,
        text,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def bootstrap_cluster_mae(
    rows: Sequence[Dict[str, str]],
    reference_key: str,
    estimate_key: str,
    n_bootstrap: int = 10000,
) -> Tuple[float, float, float]:
    by_receiver = {}
    for row in rows:
        reference = float(row[reference_key])
        if reference < study.MIN_ACTIVE_SHADE:
            continue
        key = (
            row["terrain"],
            int(row["seed"]),
            row["receiver_label"],
        )
        by_receiver.setdefault(key, []).append(
            abs(float(row[estimate_key]) - reference)
        )
    receiver_values = [
        np.asarray(values, dtype=float)
        for values in by_receiver.values()
        if values
    ]
    if not receiver_values:
        return float("nan"), float("nan"), float("nan")

    pooled = np.concatenate(receiver_values)
    point = float(np.mean(pooled))
    rng = np.random.default_rng(20260724)
    samples = np.empty(n_bootstrap, dtype=float)
    n_receivers = len(receiver_values)
    for index in range(n_bootstrap):
        chosen = rng.integers(0, n_receivers, size=n_receivers)
        sample = np.concatenate([receiver_values[i] for i in chosen])
        samples[index] = float(np.mean(sample))
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return point, float(lower), float(upper)


def load_bundle(path: Path):
    # SceneBundle was written while benchmark_study.py ran as __main__.
    sys.modules["__main__"].SceneBundle = study.SceneBundle
    with path.open("rb") as stream:
        return pickle.load(stream)


def validation_bundle_paths(study_dir: Path) -> List[Path]:
    return sorted(
        study_dir.glob(
            "scenes/validation__*/scene_bundle.pkl"
        )
    )


def stable_lobe_timing(
    study_dir: Path,
    calibration: Dict,
    repeats: int = 500,
) -> Dict:
    kernel = str(calibration["selected_kernel"])
    gamma = float(calibration["selected_gamma"])
    top_k = int(calibration["retained_topk"])
    per_receiver_us = []
    preprocess_s = []

    for bundle_path in validation_bundle_paths(study_dir):
        bundle = load_bundle(bundle_path)
        receiver_root = bundle_path.parent / "receivers"
        for label in sorted(bundle.receivers):
            fit_path = receiver_root / label / "fitted_lobes.npz"
            row_path = receiver_root / label / "validation_rows.csv"
            fit = np.load(str(fit_path), allow_pickle=False)
            fit_dict = {key: fit[key] for key in fit.files}
            rank = study.ranking_indices(fit_dict)[
                : min(top_k, len(fit["surface_ids"]))
            ]
            params = np.asarray(
                fit["params_{}".format(kernel)][rank],
                dtype=float,
            )
            rows = read_csv(row_path)
            directions = [
                (float(row["azimuth_deg"]), float(row["elevation_deg"]))
                for row in rows
            ]
            for azimuth, elevation in directions[: min(20, len(directions))]:
                study.evaluate_lobe_union(
                    kernel,
                    params,
                    azimuth,
                    elevation,
                    gamma,
                )

            started = time.perf_counter_ns()
            for _ in range(repeats):
                for azimuth, elevation in directions:
                    study.evaluate_lobe_union(
                        kernel,
                        params,
                        azimuth,
                        elevation,
                        gamma,
                    )
            elapsed_ns = time.perf_counter_ns() - started
            n_queries = max(1, repeats * len(directions))
            per_receiver_us.append(float(elapsed_ns / n_queries / 1000.0))
            preprocess_s.append(float(np.nansum(fit["fit_elapsed_s"])))

    return {
        "lobe_query_median_us": float(np.median(per_receiver_us)),
        "lobe_query_mean_us": float(np.mean(per_receiver_us)),
        "lobe_query_min_us": float(np.min(per_receiver_us)),
        "lobe_query_max_us": float(np.max(per_receiver_us)),
        "fit_preprocess_median_s_per_receiver": float(
            np.median(preprocess_s)
        ),
        "fit_preprocess_mean_s_per_receiver": float(np.mean(preprocess_s)),
        "n_timed_receivers": int(len(per_receiver_us)),
        "timing_repeats_per_direction": int(repeats),
    }


def make_workflow_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.05, 3.05))
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 4.65)
    axis.axis("off")

    boxes = [
        (0.15, 3.00, 1.55, 1.15, "Clean geometry\nand build\nproxy"),
        (2.05, 3.00, 1.55, 1.15, "Tile scene\nand apply\nradius"),
        (3.95, 3.00, 1.55, 1.15, "Fit directional\ncaster lobes"),
        (5.85, 3.00, 1.55, 1.15, "Rank, retain\nand quantise"),
        (7.75, 3.00, 2.05, 1.15, "Evaluate repeated\nsolar directions"),
    ]
    for index, (x, y, width, height, label) in enumerate(boxes):
        face = PALE if index < 4 else "#F6E9EB"
        edge = DARK if index < 4 else ACCENT
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.06",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
        )
        axis.add_patch(patch)
        axis.text(
            x + 0.5 * width,
            y + 0.5 * height,
            label,
            ha="center",
            va="center",
            fontsize=8.0,
            linespacing=1.25,
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            axis.add_patch(
                FancyArrowPatch(
                    (x + width + 0.05, y + 0.5 * height),
                    (next_x - 0.05, y + 0.5 * height),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.9,
                    color=MID,
                )
            )

    reference = FancyBboxPatch(
        (2.10, 0.45),
        5.80,
        1.20,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=BLUE,
        facecolor="#EAF0F4",
    )
    axis.add_patch(reference)
    axis.text(
        4.995,
        1.05,
        "Oblique projection + Sutherland-Hodgman clipping\n"
        "for fitting, convergence and held-out validation",
        ha="center",
        va="center",
        color=DARK,
        fontsize=8.2,
        linespacing=1.25,
    )
    axis.add_patch(
        FancyArrowPatch(
            (4.90, 1.70),
            (4.90, 2.90),
            arrowstyle="<|-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=BLUE,
        )
    )
    save_figure(figure, path)


def make_proxy_figure(study_dir: Path, path: Path) -> None:
    bundle_path = (
        study_dir
        / "scenes"
        / study.scene_key(
            "validation",
            "small_town",
            int(study.VALIDATION_SEEDS["small_town"][0]),
        )
        / "scene_bundle.pkl"
    )
    bundle = load_bundle(bundle_path)
    receiver = bundle.receivers.get("wall", list(bundle.receivers.values())[0])
    centre = np.asarray(receiver.center[:2], dtype=float)
    half_width = 78.0

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for building in bundle.scene.buildings:
        footprint = np.asarray(building.footprint_xy, dtype=float)
        if (
            np.max(footprint[:, 0]) < centre[0] - half_width
            or np.min(footprint[:, 0]) > centre[0] + half_width
            or np.max(footprint[:, 1]) < centre[1] - half_width
            or np.min(footprint[:, 1]) > centre[1] + half_width
        ):
            continue
        axis.add_patch(
            Polygon(
                footprint,
                closed=True,
                facecolor="#E3E3E3",
                edgecolor="#A8A8A8",
                linewidth=0.6,
                zorder=1,
            )
        )

    for polygon in bundle.proxy_profile.polygons:
        if polygon.surface_kind != "barrier_wall_proxy":
            continue
        vertices = np.asarray(polygon.vertices, dtype=float)
        xy = vertices[:2, :2]
        if (
            np.max(xy[:, 0]) < centre[0] - half_width
            or np.min(xy[:, 0]) > centre[0] + half_width
            or np.max(xy[:, 1]) < centre[1] - half_width
            or np.min(xy[:, 1]) > centre[1] + half_width
        ):
            continue
        axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=ACCENT,
            linewidth=1.25,
            alpha=0.9,
            zorder=3,
        )

    receiver_xy = np.asarray(receiver.vertices, dtype=float)[:, :2]
    axis.plot(
        receiver_xy[:, 0],
        receiver_xy[:, 1],
        color=BLUE,
        linewidth=2.5,
        zorder=5,
        label="Selected receiver",
    )
    axis.scatter(
        [centre[0]],
        [centre[1]],
        s=18,
        color=BLUE,
        zorder=6,
    )
    axis.set_xlim(centre[0] - half_width, centre[0] + half_width)
    axis.set_ylim(centre[1] - half_width, centre[1] + half_width)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local east coordinate [m]")
    axis.set_ylabel("Local north coordinate [m]")
    clean_axes(axis, grid_axis="both")

    scale_x = centre[0] + half_width - 35.0
    scale_y = centre[1] - half_width + 8.0
    axis.plot([scale_x, scale_x + 25.0], [scale_y, scale_y], color=DARK, lw=2.0)
    axis.text(scale_x + 12.5, scale_y + 3.0, "25 m", ha="center", va="bottom")

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color="#A8A8A8",
            linewidth=5,
            label="Source footprints",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=ACCENT,
            linewidth=1.5,
            label="Merged-eave barriers",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=BLUE,
            linewidth=2.5,
            label="Selected receiver",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )
    save_figure(figure, path)


def make_case_study_figure(study_dir: Path, path: Path) -> None:
    """Draw one held-out realisation of each fictional morphology."""
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 6.25))
    for index, (axis, case) in enumerate(zip(axes.flat, CASE_STUDIES)):
        terrain, title, _ = case
        seed = int(study.VALIDATION_SEEDS[terrain][0])
        bundle_path = (
            study_dir
            / "scenes"
            / study.scene_key("validation", terrain, seed)
            / "scene_bundle.pkl"
        )
        bundle = load_bundle(bundle_path)
        receiver_object_ids = {
            int(receiver.object_id)
            for receiver in bundle.receivers.values()
        }
        for building in bundle.scene.buildings:
            is_receiver = int(building.object_id) in receiver_object_ids
            axis.add_patch(
                Polygon(
                    np.asarray(building.footprint_xy, dtype=float),
                    closed=True,
                    facecolor="#E9C9CE" if is_receiver else "#D8D8D8",
                    edgecolor=ACCENT if is_receiver else "#8C8C8C",
                    linewidth=0.9 if is_receiver else 0.35,
                    zorder=2 if is_receiver else 1,
                )
            )

        for receiver_label, marker, color in (
            ("wall", "s", BLUE),
            ("roof", "^", ACCENT),
        ):
            receiver = bundle.receivers[receiver_label]
            axis.scatter(
                [float(receiver.center[0])],
                [float(receiver.center[1])],
                marker=marker,
                s=20,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
                label=receiver_label.capitalize(),
            )

        area_size = float(bundle.scene.area_size)
        axis.set_xlim(0.0, area_size)
        axis.set_ylim(0.0, area_size)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks((0.0, 0.5 * area_size, area_size))
        axis.set_yticks((0.0, 0.5 * area_size, area_size))
        axis.tick_params(labelsize=7.2, length=2.5)
        axis.set_title(
            "{}\nseed {}, {} buildings".format(
                title,
                seed,
                len(bundle.scene.buildings),
            ),
            fontsize=8.6,
            pad=4.0,
        )
        panel_label(axis, "({})".format(chr(ord("a") + index)))
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#777777")

    axes[0, 0].set_ylabel("Northing [m]")
    axes[1, 0].set_ylabel("Northing [m]")
    axes[1, 0].set_xlabel("Easting [m]")
    axes[1, 1].set_xlabel("Easting [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.94,
        hspace=0.30,
        wspace=0.20,
    )
    save_figure(figure, path)


def case_study_summaries(
    rows: Sequence[Dict[str, str]],
    study_dir: Path,
) -> Dict[str, Dict]:
    summaries = {}
    for terrain, title, macro_prefix in CASE_STUDIES:
        case_rows = [row for row in rows if row["terrain"] == terrain]
        active_rows = [
            row
            for row in case_rows
            if float(row["raw_full"]) >= study.MIN_ACTIVE_SHADE
        ]
        errors = np.asarray(
            [
                float(row["lobe_quantized"]) - float(row["raw_full"])
                for row in active_rows
            ],
            dtype=float,
        )
        scene_values = []
        for seed in study.VALIDATION_SEEDS[terrain]:
            metadata_path = (
                study_dir
                / "scenes"
                / study.scene_key("validation", terrain, int(seed))
                / "scene_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            scene_values.append(
                {
                    "seed": int(seed),
                    "n_buildings": int(metadata["n_buildings"]),
                    "raw_n_casters": int(metadata["raw_n_casters"]),
                    "proxy_n_casters": int(metadata["proxy_n_casters"]),
                }
            )

        summaries[terrain] = {
            "title": title,
            "macro_prefix": macro_prefix,
            "direction_count": int(len(case_rows)),
            "active_count": int(len(active_rows)),
            "active_mae": float(np.mean(np.abs(errors))),
            "active_bias": float(np.mean(errors)),
            "active_max_abs": float(np.max(np.abs(errors))),
            "scenes": scene_values,
        }
    return summaries


def make_convergence_figure(calibration: Dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    radius_rows = calibration["radius_summary"]
    radius = np.asarray([float(row["radius_m"]) for row in radius_rows])
    radius_mae = np.asarray([float(row["active"]["mae"]) for row in radius_rows])
    radius_p95 = np.asarray(
        [float(row["active"]["p95_abs"]) for row in radius_rows]
    )
    radius_mae_plot = np.maximum(radius_mae, 1.0e-5)
    radius_p95_plot = np.maximum(radius_p95, 1.0e-5)
    axes[0].plot(
        radius,
        radius_mae_plot,
        marker="o",
        color=ACCENT,
        linewidth=1.5,
        markersize=4,
        label="MAE",
    )
    axes[0].plot(
        radius,
        radius_p95_plot,
        marker="s",
        color=BLUE,
        linewidth=1.3,
        markersize=3.5,
        label="95th percentile",
    )
    axes[0].axhline(
        study.RADIUS_MAE_LIMIT,
        color=ACCENT,
        linestyle=":",
        linewidth=0.9,
    )
    axes[0].axhline(
        study.RADIUS_P95_LIMIT,
        color=BLUE,
        linestyle=":",
        linewidth=0.9,
    )
    axes[0].axvline(
        float(calibration["retained_radius_m"]),
        color=MID,
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_xlabel("Neighbour radius [m]")
    axes[0].set_ylabel("Absolute error [log scale]")
    axes[0].set_yscale("log")
    axes[0].set_ylim(8.0e-6, 1.25)
    axes[0].legend(loc="upper right", frameon=False)
    clean_axes(axes[0])
    panel_label(axes[0], "(a)")

    topk_rows = calibration["topk_summary"]
    topk = np.asarray([int(round(float(row["top_k"]))) for row in topk_rows])
    topk_mae = np.asarray([float(row["active"]["mae"]) for row in topk_rows])
    topk_p95 = np.asarray(
        [float(row["active"]["p95_abs"]) for row in topk_rows]
    )
    topk_mae_plot = np.maximum(topk_mae, 1.0e-5)
    topk_p95_plot = np.maximum(topk_p95, 1.0e-5)
    axes[1].plot(
        topk,
        topk_mae_plot,
        marker="o",
        color=ACCENT,
        linewidth=1.5,
        markersize=4,
        label="MAE",
    )
    axes[1].plot(
        topk,
        topk_p95_plot,
        marker="s",
        color=BLUE,
        linewidth=1.3,
        markersize=3.5,
        label="95th percentile",
    )
    axes[1].axhline(
        study.TOPK_MAE_LIMIT,
        color=ACCENT,
        linestyle=":",
        linewidth=0.9,
    )
    axes[1].axhline(
        study.TOPK_P95_LIMIT,
        color=BLUE,
        linestyle=":",
        linewidth=0.9,
    )
    axes[1].axvline(
        float(calibration["retained_topk"]),
        color=MID,
        linestyle="--",
        linewidth=0.9,
    )
    axes[1].set_xlabel("Retained caster lobes, $K$")
    axes[1].set_ylabel("Absolute error [log scale]")
    axes[1].set_yscale("log")
    axes[1].set_ylim(8.0e-6, 0.6)
    clean_axes(axes[1])
    panel_label(axes[1], "(b)")

    figure.subplots_adjust(wspace=0.32)
    save_figure(figure, path)


def make_kernel_figure(calibration: Dict, path: Path) -> None:
    best_by_kernel = {}
    for row in calibration["kernel_summary"]:
        kernel = str(row["kernel"])
        if int(row["active"]["n"]) <= 0:
            continue
        current = best_by_kernel.get(kernel)
        if current is None or float(row["active"]["mae"]) < float(
            current["active"]["mae"]
        ):
            best_by_kernel[kernel] = row

    ordered = sorted(
        best_by_kernel.values(),
        key=lambda row: float(row["active"]["mae"]),
        reverse=True,
    )
    labels = [
        {
            "supergaussian": "super-Gaussian (4th)",
            "supergaussian6": "super-Gaussian (6th)",
            "supergaussian8": "super-Gaussian (8th)",
        }.get(str(row["kernel"]), str(row["kernel"]).capitalize())
        for row in ordered
    ]
    values = [float(row["active"]["mae"]) for row in ordered]
    gammas = [float(row["gamma"]) for row in ordered]
    colors = [
        ACCENT
        if str(row["kernel"]) == str(calibration["selected_kernel"])
        else "#A8A8A8"
        for row in ordered
    ]

    figure, axis = plt.subplots(figsize=(6.1, 3.4))
    y = np.arange(len(ordered))
    axis.barh(y, values, color=colors, height=0.68, zorder=3)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Active-case MAE")
    axis.set_xlim(0.0, max(values) * 1.23)
    for index, (value, gamma) in enumerate(zip(values, gammas)):
        axis.text(
            value + 0.002,
            index,
            r"$\gamma={:.1f}$".format(gamma),
            va="center",
            ha="left",
            fontsize=8,
            color=DARK,
        )
    clean_axes(axis, grid_axis="x")
    save_figure(figure, path)


def heatmap_matrix(
    rows: Sequence[Dict[str, str]],
    key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuths = np.asarray(sorted(set(float(row["azimuth_deg"]) for row in rows)))
    elevations = np.asarray(
        sorted(set(float(row["elevation_deg"]) for row in rows))
    )
    matrix = np.full((len(elevations), len(azimuths)), np.nan, dtype=float)
    az_index = {value: index for index, value in enumerate(azimuths)}
    el_index = {value: index for index, value in enumerate(elevations)}
    for row in rows:
        matrix[
            el_index[float(row["elevation_deg"])],
            az_index[float(row["azimuth_deg"])],
        ] = float(row[key])
    return azimuths, elevations, matrix


def make_heatmap_figure(study_dir: Path, path: Path) -> None:
    rows = read_csv(study_dir / "validation" / "heatmap_example.csv")
    azimuths, elevations, reference = heatmap_matrix(rows, "reference")
    _, _, lobe = heatmap_matrix(rows, "lobe")
    _, _, error = heatmap_matrix(rows, "error")

    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.65), sharex=True, sharey=True)
    cmap_fraction = mpl.colormaps["viridis"].copy()
    cmap_fraction.set_bad("#EFEFEF")
    cmap_error = mpl.colormaps["RdBu_r"].copy()
    cmap_error.set_bad("#EFEFEF")
    mesh0 = axes[0].pcolormesh(
        azimuths,
        elevations,
        reference,
        shading="nearest",
        cmap=cmap_fraction,
        vmin=0.0,
        vmax=1.0,
        rasterized=False,
    )
    axes[1].pcolormesh(
        azimuths,
        elevations,
        lobe,
        shading="nearest",
        cmap=cmap_fraction,
        vmin=0.0,
        vmax=1.0,
        rasterized=False,
    )
    limit = max(0.15, float(np.nanquantile(np.abs(error), 0.98)))
    mesh2 = axes[2].pcolormesh(
        azimuths,
        elevations,
        error,
        shading="nearest",
        cmap=cmap_error,
        vmin=-limit,
        vmax=limit,
        rasterized=False,
    )
    labels = ("Clipping reference", "Fitted lobes", "Lobe - reference")
    for index, axis in enumerate(axes):
        axis.text(
            0.5,
            1.04,
            labels[index],
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        axis.set_xlabel("Solar azimuth [deg]")
        axis.set_xticks((0, 90, 180, 270, 360))
        axis.set_ylim(float(np.min(elevations)), float(np.max(elevations)))
        panel_label(axis, "({})".format(chr(ord("a") + index)))
    axes[0].set_ylabel("Solar elevation [deg]")
    colorbar0 = figure.colorbar(mesh0, ax=axes[:2], fraction=0.035, pad=0.02)
    colorbar0.set_label("Shaded fraction")
    colorbar2 = figure.colorbar(mesh2, ax=axes[2], fraction=0.07, pad=0.03)
    colorbar2.set_label("Error")
    figure.subplots_adjust(wspace=0.13)
    save_figure(figure, path)


def make_validation_figure(
    rows: Sequence[Dict[str, str]],
    validation: Dict,
    path: Path,
) -> None:
    raw = np.asarray([float(row["raw_full"]) for row in rows])
    estimate = np.asarray([float(row["lobe_quantized"]) for row in rows])
    active = raw >= study.MIN_ACTIVE_SHADE

    figure, axes = plt.subplots(1, 2, figsize=(7.20, 3.05))
    axes[0].scatter(
        raw[~active],
        estimate[~active],
        s=10,
        color="#B7B7B7",
        alpha=0.45,
        edgecolors="none",
        label="Unshaded / negligible",
    )
    axes[0].scatter(
        raw[active],
        estimate[active],
        s=14,
        color=ACCENT,
        alpha=0.58,
        edgecolors="none",
        label="Active shadow",
    )
    axes[0].plot((0, 1), (0, 1), color=DARK, linewidth=0.9, linestyle="--")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Raw clipping shaded fraction")
    axes[0].set_ylabel("Quantised-lobe shaded fraction")
    axes[0].legend(loc="upper left", frameon=False)
    clean_axes(axes[0], grid_axis="both")
    panel_label(axes[0], "(a)")

    stages = [
        ("Proxy", "proxy_geometry"),
        ("Radius", "neighbour_radius"),
        ("Top-$K$", "topk_selection"),
        ("Lobe", "lobe_representation"),
        ("Quant.", "quantisation"),
        ("Combined", "combined"),
    ]
    values = [
        float(validation[key]["active"]["mae"])
        for _, key in stages
    ]
    colors = [BLUE, TEAL, GOLD, ACCENT, "#777777", DARK]
    x = np.arange(len(stages))
    axes[1].bar(x, values, color=colors, width=0.72, zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label for label, _ in stages], rotation=0)
    axes[1].tick_params(axis="x", labelsize=7.2)
    axes[1].set_ylabel("Active-case MAE")
    axes[1].set_ylim(0.0, max(values) * 1.20)
    clean_axes(axes[1])
    panel_label(axes[1], "(b)")
    figure.subplots_adjust(wspace=0.33)
    save_figure(figure, path)


def make_per_terrain_figure(
    rows: Sequence[Dict[str, str]],
    path: Path,
) -> None:
    terrains = list(study.TERRAINS)
    labels = ("Tall\nbuildings", "Horizontal\ncity", "Small\ntown", "Suburban")
    wall_values = []
    roof_values = []
    for terrain in terrains:
        wall_values.append(
            [
                abs(float(row["lobe_quantized"]) - float(row["raw_full"]))
                for row in rows
                if row["terrain"] == terrain
                and row["receiver_label"] == "wall"
                and float(row["raw_full"]) >= study.MIN_ACTIVE_SHADE
            ]
        )
        roof_values.append(
            [
                abs(float(row["lobe_quantized"]) - float(row["raw_full"]))
                for row in rows
                if row["terrain"] == terrain
                and row["receiver_label"] == "roof"
                and float(row["raw_full"]) >= study.MIN_ACTIVE_SHADE
            ]
        )

    figure, axis = plt.subplots(figsize=(6.5, 3.25))
    positions = np.arange(len(terrains), dtype=float)
    for values, offset, color, label in (
        (wall_values, -0.18, BLUE, "Walls"),
        (roof_values, 0.18, ACCENT, "Roofs"),
    ):
        plot = axis.boxplot(
            values,
            positions=positions + offset,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": DARK, "linewidth": 1.0},
            whiskerprops={"color": color, "linewidth": 0.9},
            capprops={"color": color, "linewidth": 0.9},
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.55},
        )
        _ = plot
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Absolute combined error")
    clean_axes(axis)
    axis.legend(
        handles=[
            mpl.patches.Patch(facecolor=BLUE, alpha=0.55, label="Walls"),
            mpl.patches.Patch(facecolor=ACCENT, alpha=0.55, label="Roofs"),
        ],
        loc="upper right",
        frameon=False,
    )
    save_figure(figure, path)


def make_performance_figure(
    validation: Dict,
    stable_timing: Dict,
    path: Path,
) -> None:
    raw_ms = float(validation["timing"]["raw_exact_median_ms"])
    proxy_ms = float(validation["timing"]["proxy_exact_median_ms"])
    lobe_ms = float(stable_timing["lobe_query_median_us"]) / 1000.0
    storage = validation["storage"]

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.9))
    methods = ("Raw\nclipping", "Proxy\nclipping", "Packed\nlobes")
    times = (raw_ms, proxy_ms, lobe_ms)
    axes[0].bar(
        np.arange(3),
        times,
        color=("#9E9E9E", BLUE, ACCENT),
        width=0.68,
        zorder=3,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median query time [ms, log scale]")
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(methods)
    clean_axes(axes[0])
    panel_label(axes[0], "(a)")

    byte_labels = ("Packed\npayload", "Compressed\narchive")
    byte_values = (
        float(storage["packed_bytes_per_receiver"]),
        float(storage["compressed_bytes_per_receiver"]),
    )
    axes[1].bar(
        np.arange(2),
        np.asarray(byte_values) / 1024.0,
        color=(BLUE, ACCENT),
        width=0.62,
        zorder=3,
    )
    axes[1].set_ylabel("Storage per receiver [KiB]")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(byte_labels)
    clean_axes(axes[1])
    panel_label(axes[1], "(b)")
    figure.subplots_adjust(wspace=0.35)
    save_figure(figure, path)


def result_macros(
    calibration: Dict,
    validation: Dict,
    stable_timing: Dict,
    confidence: Tuple[float, float, float],
    case_summaries: Dict[str, Dict],
) -> Dict[str, str]:
    combined = validation["combined"]["active"]
    lobe = validation["lobe_representation"]["active"]
    quant = validation["quantisation"]["active"]
    proxy = validation["proxy_geometry"]["active"]
    radius = validation["neighbour_radius"]["active"]
    topk = validation["topk_selection"]["active"]
    storage = validation["storage"]
    profile = validation["profile_reduction"]

    raw_ms = float(validation["timing"]["raw_exact_median_ms"])
    lobe_ms = float(stable_timing["lobe_query_median_us"]) / 1000.0
    stable_speedup = raw_ms / max(1.0e-12, lobe_ms)
    rounded_speedup = int(round(stable_speedup / 100.0) * 100)
    selected_kernel_row = min(
        (
            row
            for row in calibration["kernel_summary"]
            if row["kernel"] == calibration["selected_kernel"]
            and abs(
                float(row["gamma"]) - float(calibration["selected_gamma"])
            )
            < 1.0e-12
        ),
        key=lambda row: float(row["active"]["mae"]),
    )
    radius_knee_row = next(
        row
        for row in calibration["radius_summary"]
        if abs(
            float(row["radius_m"]) - float(calibration["radius_knee_m"])
        )
        < 1.0e-12
    )
    topk_knee_row = next(
        row
        for row in calibration["topk_summary"]
        if int(round(float(row["top_k"]))) == int(calibration["topk_knee"])
    )

    macros = {
        "CalibrationSceneCount": str(calibration["n_calibration_scenes"]),
        "CalibrationReceiverCount": str(calibration["n_calibration_receivers"]),
        "ValidationSceneCount": str(validation["counts"]["n_scenes"]),
        "ValidationReceiverCount": str(validation["counts"]["n_receivers"]),
        "ValidationDirectionCount": str(validation["counts"]["n_rows"]),
        "ValidationActiveCount": str(validation["counts"]["n_active_raw"]),
        "RadiusKneeM": "{:.0f}".format(calibration["radius_knee_m"]),
        "RetainedRadiusM": "{:.0f}".format(calibration["retained_radius_m"]),
        "TopKKnee": str(calibration["topk_knee"]),
        "RetainedTopK": str(calibration["retained_topk"]),
        "SelectedKernel": "fourth-order super-Gaussian",
        "SelectedGamma": "{:.2f}".format(calibration["selected_gamma"]),
        "CalibrationActiveCount": str(selected_kernel_row["active"]["n"]),
        "CalibrationKernelActiveMAE": "{:.4f}".format(
            selected_kernel_row["active"]["mae"]
        ),
        "CalibrationKernelActiveP": "{:.2f}".format(
            100.0 * selected_kernel_row["active"]["mae"]
        ),
        "CalibrationKernelActivePNinetyFive": "{:.4f}".format(
            selected_kernel_row["active"]["p95_abs"]
        ),
        "RadiusKneeActiveMAE": "{:.5f}".format(
            radius_knee_row["active"]["mae"]
        ),
        "RadiusKneeActivePNinetyFive": "{:.5f}".format(
            radius_knee_row["active"]["p95_abs"]
        ),
        "TopKKneeActiveMAE": "{:.5f}".format(
            topk_knee_row["active"]["mae"]
        ),
        "TopKKneeActivePNinetyFive": "{:.5f}".format(
            topk_knee_row["active"]["p95_abs"]
        ),
        "ProxyActiveMAE": "{:.4f}".format(proxy["mae"]),
        "ProxyActiveCount": str(proxy["n"]),
        "ProxyActiveP": "{:.3f}".format(100.0 * proxy["mae"]),
        "ProxyActivePNinetyFive": "{:.4f}".format(proxy["p95_abs"]),
        "ProxyActiveMax": "{:.4f}".format(proxy["max_abs"]),
        "RadiusActiveMAE": "{:.5f}".format(radius["mae"]),
        "RadiusActiveCount": str(radius["n"]),
        "TopKActiveMAE": "{:.5f}".format(topk["mae"]),
        "TopKActiveCount": str(topk["n"]),
        "LobeActiveMAE": "{:.4f}".format(lobe["mae"]),
        "LobeActiveCount": str(lobe["n"]),
        "LobeActiveP": "{:.2f}".format(100.0 * lobe["mae"]),
        "LobeActivePNinetyFive": "{:.4f}".format(lobe["p95_abs"]),
        "LobeActiveMax": "{:.4f}".format(lobe["max_abs"]),
        "QuantActiveMAE": "{:.6f}".format(quant["mae"]),
        "QuantActiveCount": str(quant["n"]),
        "QuantActivePNinetyFive": "{:.6f}".format(quant["p95_abs"]),
        "QuantActiveMax": "{:.6f}".format(quant["max_abs"]),
        "CombinedActiveMAE": "{:.4f}".format(combined["mae"]),
        "CombinedActiveCount": str(combined["n"]),
        "CombinedActiveP": "{:.2f}".format(100.0 * combined["mae"]),
        "CombinedAllMAE": "{:.4f}".format(
            validation["combined"]["all"]["mae"]
        ),
        "CombinedActiveBias": "{:.4f}".format(combined["bias"]),
        "CombinedActiveRMSE": "{:.4f}".format(combined["rmse"]),
        "CombinedActivePNinetyFive": "{:.4f}".format(combined["p95_abs"]),
        "CombinedActiveMax": "{:.4f}".format(combined["max_abs"]),
        "CombinedMAECILower": "{:.4f}".format(confidence[1]),
        "CombinedMAECIUpper": "{:.4f}".format(confidence[2]),
        "BootstrapReplicates": "10000",
        "MeanRawCasterCount": "{:.0f}".format(profile["mean_raw_casters"]),
        "MeanProxyCasterCount": "{:.0f}".format(profile["mean_proxy_casters"]),
        "MeanCasterReductionP": "{:.1f}".format(
            100.0 * profile["mean_reduction_fraction"]
        ),
        "RawQueryMedianMS": "{:.2f}".format(raw_ms),
        "ProxyQueryMedianMS": "{:.2f}".format(
            validation["timing"]["proxy_exact_median_ms"]
        ),
        "LobeQueryMedianUS": "{:.1f}".format(
            stable_timing["lobe_query_median_us"]
        ),
        "StableSpeedup": str(rounded_speedup),
        "PackedBytesPerReceiver": "{:.0f}".format(
            storage["packed_bytes_per_receiver"]
        ),
        "CompressedBytesPerReceiver": "{:.0f}".format(
            storage["compressed_bytes_per_receiver"]
        ),
        "CompressedKiBPerReceiver": "{:.2f}".format(
            storage["compressed_bytes_per_receiver"] / 1024.0
        ),
        "FitPreprocessMedianS": "{:.2f}".format(
            stable_timing["fit_preprocess_median_s_per_receiver"]
        ),
    }
    for terrain, _, macro_prefix in CASE_STUDIES:
        case = case_summaries[terrain]
        first, second = case["scenes"]
        macros.update(
            {
                "Case{}DirectionCount".format(macro_prefix): str(
                    case["direction_count"]
                ),
                "Case{}ActiveCount".format(macro_prefix): str(
                    case["active_count"]
                ),
                "Case{}ActiveMAE".format(macro_prefix): "{:.4f}".format(
                    case["active_mae"]
                ),
                "Case{}ActiveBias".format(macro_prefix): "{:.4f}".format(
                    case["active_bias"]
                ),
                "Case{}ActiveMax".format(macro_prefix): "{:.4f}".format(
                    case["active_max_abs"]
                ),
                "Case{}BuildingsA".format(macro_prefix): str(
                    first["n_buildings"]
                ),
                "Case{}BuildingsB".format(macro_prefix): str(
                    second["n_buildings"]
                ),
                "Case{}RawCastersA".format(macro_prefix): str(
                    first["raw_n_casters"]
                ),
                "Case{}RawCastersB".format(macro_prefix): str(
                    second["raw_n_casters"]
                ),
                "Case{}ProxyCastersA".format(macro_prefix): str(
                    first["proxy_n_casters"]
                ),
                "Case{}ProxyCastersB".format(macro_prefix): str(
                    second["proxy_n_casters"]
                ),
            }
        )
    return macros


def write_results_tex(path: Path, macros: Dict[str, str]) -> None:
    lines = [
        "% Automatically generated by build_article_assets.py.",
        "% Do not edit numerical values here by hand.",
    ]
    for name in sorted(macros):
        lines.append(
            "\\newcommand{{\\{}}}{{{}}}".format(name, macros[name])
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(str(temporary), str(path))


def build_assets(study_dir: Path, article_dir: Path) -> Dict:
    study_dir = Path(study_dir).resolve()
    article_dir = Path(article_dir).resolve()
    figures_dir = article_dir / "figures"
    data_dir = article_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    calibration = json.loads(
        (study_dir / "calibration" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    validation = json.loads(
        (study_dir / "validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_csv(study_dir / "validation" / "validation_all_rows.csv")

    stable_timing = stable_lobe_timing(study_dir, calibration)
    confidence = bootstrap_cluster_mae(
        rows,
        "raw_full",
        "lobe_quantized",
    )
    case_summaries = case_study_summaries(rows, study_dir)
    macros = result_macros(
        calibration,
        validation,
        stable_timing,
        confidence,
        case_summaries,
    )

    make_workflow_figure(figures_dir / "fig01_workflow.pdf")
    make_proxy_figure(study_dir, figures_dir / "fig02_proxy_geometry.pdf")
    make_convergence_figure(calibration, figures_dir / "fig03_convergence.pdf")
    make_kernel_figure(calibration, figures_dir / "fig04_kernel_screen.pdf")
    make_heatmap_figure(study_dir, figures_dir / "fig05_heatmaps.pdf")
    make_validation_figure(
        rows,
        validation,
        figures_dir / "fig06_validation.pdf",
    )
    make_per_terrain_figure(
        rows,
        figures_dir / "fig07_error_by_terrain.pdf",
    )
    make_performance_figure(
        validation,
        stable_timing,
        figures_dir / "fig08_performance.pdf",
    )
    make_case_study_figure(
        study_dir,
        figures_dir / "fig09_case_studies.pdf",
    )

    write_results_tex(article_dir / "results.tex", macros)
    write_json(
        data_dir / "results_values.json",
        {
            "macros": macros,
            "calibration": calibration,
            "validation": validation,
            "case_studies": case_summaries,
            "stable_timing": stable_timing,
            "combined_active_mae_cluster_bootstrap_95ci": {
                "point": confidence[0],
                "lower": confidence[1],
                "upper": confidence[2],
                "clusters": "receiver",
                "replicates": 10000,
                "seed": 20260724,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
        },
    )
    write_csv(data_dir / "validation_all_rows.csv", rows)
    write_csv(
        data_dir / "case_study_summary.csv",
        [
            {
                "terrain": terrain,
                "case_study": case["title"],
                "direction_count": case["direction_count"],
                "active_count": case["active_count"],
                "active_mae": case["active_mae"],
                "active_bias": case["active_bias"],
                "active_max_abs": case["active_max_abs"],
                "validation_seed_a": case["scenes"][0]["seed"],
                "validation_seed_b": case["scenes"][1]["seed"],
                "buildings_a": case["scenes"][0]["n_buildings"],
                "buildings_b": case["scenes"][1]["n_buildings"],
                "raw_casters_a": case["scenes"][0]["raw_n_casters"],
                "raw_casters_b": case["scenes"][1]["raw_n_casters"],
                "proxy_casters_a": case["scenes"][0]["proxy_n_casters"],
                "proxy_casters_b": case["scenes"][1]["proxy_n_casters"],
            }
            for terrain, case in case_summaries.items()
        ],
    )
    write_csv(
        data_dir / "radius_convergence.csv",
        [
            {
                "radius_m": row["radius_m"],
                "active_n": row["active"]["n"],
                "active_bias": row["active"]["bias"],
                "active_mae": row["active"]["mae"],
                "active_rmse": row["active"]["rmse"],
                "active_p95_abs": row["active"]["p95_abs"],
                "active_max_abs": row["active"]["max_abs"],
            }
            for row in calibration["radius_summary"]
        ],
    )
    write_csv(
        data_dir / "topk_convergence.csv",
        [
            {
                "top_k": row["top_k"],
                "active_n": row["active"]["n"],
                "active_bias": row["active"]["bias"],
                "active_mae": row["active"]["mae"],
                "active_rmse": row["active"]["rmse"],
                "active_p95_abs": row["active"]["p95_abs"],
                "active_max_abs": row["active"]["max_abs"],
            }
            for row in calibration["topk_summary"]
        ],
    )
    write_csv(
        data_dir / "kernel_screen.csv",
        [
            {
                "kernel": row["kernel"],
                "gamma": row["gamma"],
                "active_n": row["active"]["n"],
                "active_bias": row["active"]["bias"],
                "active_mae": row["active"]["mae"],
                "active_rmse": row["active"]["rmse"],
                "active_p95_abs": row["active"]["p95_abs"],
            }
            for row in calibration["kernel_summary"]
        ],
    )
    return {
        "article_dir": str(article_dir),
        "figure_count": 9,
        "macros": macros,
        "case_studies": case_summaries,
        "stable_timing": stable_timing,
        "confidence": confidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("study_output"),
    )
    parser.add_argument(
        "--article-dir",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    result = build_assets(args.study_dir, args.article_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
