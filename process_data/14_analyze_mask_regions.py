"""
Purpose: Select regional rectangular boxes from KE-masked regions.
Main inputs: KE-masked region mask NetCDF file on the velocity grid.
Main outputs: Printed region bounds and a diagnostic figure.
Notes: Update placeholder paths before running.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


DDIR = "/path/to"
MASK_FILE = f"{DDIR}/mask/KEmasked_regions_deg025.nc"
OUTPUT_FIGURE = f"{DDIR}/Spectrum/mask_regions_with_bbox.png"
REGION_NAMES = ["ACC", "Kuroshio", "GulfStream", "Agulhas", "EastAustralian", "Malvinas"]
ALPHA = 1.0
PAD_ROWS = 80
PAD_COLS = 80


def kadane_1d(arr):
    n = len(arr)
    best_sum = -np.inf
    best_start = 0
    best_end = 0
    curr_sum = 0.0
    curr_start = 0

    for idx in range(n):
        curr_sum += arr[idx]
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_start = curr_start
            best_end = idx
        if curr_sum < 0:
            curr_sum = 0.0
            curr_start = idx + 1

    return best_sum, best_start, best_end


def max_weight_subrect(mask_2d, alpha=1.0):
    nrows, ncols = mask_2d.shape
    weight = np.where(mask_2d, 1.0, -alpha)

    best_score = -np.inf
    best_rect = None

    for top in range(nrows):
        col_sum = np.zeros(ncols)
        for bottom in range(top, nrows):
            col_sum += weight[bottom, :]
            score, left, right = kadane_1d(col_sum)
            if score > best_score:
                best_score = score
                best_rect = (top, bottom, left, right)

    return best_score, best_rect


def find_optimal_bbox(mask_2d, lat_arr, lon_arr, alpha=1.0, pad_rows=80, pad_cols=80):
    true_locs = np.argwhere(mask_2d)
    if len(true_locs) == 0:
        return None

    row_min, row_max = true_locs[:, 0].min(), true_locs[:, 0].max()
    col_min, col_max = true_locs[:, 1].min(), true_locs[:, 1].max()

    row_min_pad = max(0, row_min - pad_rows)
    row_max_pad = min(mask_2d.shape[0] - 1, row_max + pad_rows)
    col_min_pad = max(0, col_min - pad_cols)
    col_max_pad = min(mask_2d.shape[1] - 1, col_max + pad_cols)

    sub_mask = mask_2d[row_min_pad:row_max_pad + 1, col_min_pad:col_max_pad + 1]
    print(f"  search size: {sub_mask.shape[0]} x {sub_mask.shape[1]}")

    _, (top, bottom, left, right) = max_weight_subrect(sub_mask, alpha=alpha)

    global_top = row_min_pad + top
    global_bottom = row_min_pad + bottom
    global_left = col_min_pad + left
    global_right = col_min_pad + right

    return (
        lon_arr[global_left],
        lon_arr[global_right],
        lat_arr[global_top],
        lat_arr[global_bottom],
    )


def compute_bbox_stats(mask_2d, lat_arr, lon_arr, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_mask = (lat_arr >= lat_min) & (lat_arr <= lat_max)
    lon_mask = (lon_arr >= lon_min) & (lon_arr <= lon_max)
    sub = mask_2d[np.ix_(lat_mask, lon_mask)]
    n_true = int(sub.sum())
    n_false = int(sub.size - n_true)
    total_true = int(mask_2d.sum())
    coverage = n_true / total_true * 100 if total_true > 0 else 0
    return n_true, n_false, coverage


def lon360_to_180(value):
    return ((value + 180) % 360) - 180


def fmt_lon(value):
    value_180 = lon360_to_180(value)
    suffix = "E" if value_180 >= 0 else "W"
    return f"{abs(value_180):.1f}{suffix}"


def fmt_lat(value):
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.1f}{suffix}"


def plot_region_boxes(ds, bboxes, output_figure):
    proj_crs = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9),
        subplot_kw={"projection": proj_crs},
        constrained_layout=True,
    )
    axes = axes.flatten()

    for idx, name in enumerate(REGION_NAMES):
        ax = axes[idx]
        key = f"mask_{name}"
        mask = ds[key].values.astype(float)
        lon_min, lon_max, lat_min, lat_max = bboxes[name]
        margin = 5.0

        ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
        ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="black", linewidth=0.3, zorder=5)
        ax.spines["geo"].set_linewidth(0.5)
        ax.set_extent(
            [
                max(0, lon_min - margin),
                min(360, lon_max + margin),
                lat_min - margin,
                lat_max + margin,
            ],
            crs=data_crs,
        )

        ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            np.where(mask, 1, np.nan),
            cmap="Reds",
            vmin=0,
            vmax=1,
            shading="auto",
            transform=data_crs,
            alpha=0.7,
            zorder=3,
        )
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=2.5,
            edgecolor="red",
            facecolor="none",
            transform=data_crs,
            linestyle="--",
            zorder=10,
        )
        ax.add_patch(rect)
        ax.gridlines(linewidth=0.3, alpha=0.5, xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))

        n_true, n_false, coverage = compute_bbox_stats(ds[key].values, ds["lat"].values, ds["lon"].values, bboxes[name])
        total_true = int(ds[key].values.sum())
        true_ratio = n_true / (n_true + n_false) * 100
        ax.set_title(
            f"{name}\n"
            f"[{fmt_lon(lon_min)}-{fmt_lon(lon_max)}, {fmt_lat(lat_min)}-{fmt_lat(lat_max)}]\n"
            f"True={n_true}/{total_true} ({coverage:.0f}%), False={n_false}, True%={true_ratio:.0f}%",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(f"Optimized Bounding Boxes (alpha={ALPHA})", fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    fig.savefig(output_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ds = xr.open_dataset(MASK_FILE)
    lat = ds["lat"].values
    lon = ds["lon"].values

    bboxes = {}
    for name in REGION_NAMES:
        key = f"mask_{name}"
        mask = ds[key].values
        print(f"\n{name}: searching optimal box (alpha={ALPHA})")
        bbox = find_optimal_bbox(mask, lat, lon, alpha=ALPHA, pad_rows=PAD_ROWS, pad_cols=PAD_COLS)
        bboxes[name] = bbox
        n_true, n_false, coverage = compute_bbox_stats(mask, lat, lon, bbox)
        total_true = int(mask.sum())
        true_ratio = n_true / (n_true + n_false) * 100
        print(f"  bbox = ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
        print(f"  True={n_true}/{total_true} ({coverage:.1f}%), False={n_false}, True ratio={true_ratio:.1f}%")

    plot_region_boxes(ds, bboxes, OUTPUT_FIGURE)
    print(f"\nFigure saved: {OUTPUT_FIGURE}")
    print("\n" + "=" * 90)
    print(f'{"Region":18s} {"lon_min":>8s} {"lon_max":>8s} {"lat_min":>8s} {"lat_max":>8s} {"True":>6s} {"False":>6s} {"Cvg%":>5s} {"True%":>5s}')
    print("=" * 90)
    for name in REGION_NAMES:
        bbox = bboxes[name]
        n_true, n_false, coverage = compute_bbox_stats(ds[f"mask_{name}"].values, lat, lon, bbox)
        true_ratio = n_true / (n_true + n_false) * 100
        print(f"{name:18s} {bbox[0]:9.3f} {bbox[1]:9.3f} {bbox[2]:9.3f} {bbox[3]:9.3f} {n_true:6d} {n_false:6d} {coverage:5.1f} {true_ratio:5.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
