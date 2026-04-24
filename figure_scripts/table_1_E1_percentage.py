"""
Table 1 | Summary of trend characteristics for large-scale (KL) and mesoscale (KM) kinetic energy in different ocean regions derived from the C3S product.
Extended Data Table 1 | Summary of trend characteristics for large-scale (KL) and mesoscale (KM) kinetic energy in different ocean regions derived from the multi-mission product.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class Region:
    name: str
    label: str


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("/path/to")
    dversion: str = "twosat_i8192"
    ftversion: str = "mwt_d184"
    target_time: str = "2024-10-01"
    confidence: float = 1.645


REGIONS: tuple[Region, ...] = (
    Region("mask_Kuroshio", "Kuroshio"),
    Region("mask_GulfStream", "Gulf Stream"),
    Region("mask_Agulhas", "Agulhas Current"),
    Region("mask_EastAustralian", "EAC"),
    Region("mask_Malvinas", "BMC"),
    Region("mask_ACC", "ACC"),
    Region("mask_global_EQ10", "global ocean"),
)


def build_gridcell_area(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """Grid-cell area on sphere for regular lat/lon grid."""
    lon_vals = lon.values
    lat_vals = lat.values

    dlon = float(np.diff(lon_vals).mean())
    dlat = float(np.diff(lat_vals).mean())

    lon_b = np.concatenate(
        ([lon_vals[0] - dlon / 2], 0.5 * (lon_vals[:-1] + lon_vals[1:]), [lon_vals[-1] + dlon / 2])
    )
    lat_b = np.concatenate(
        ([lat_vals[0] - dlat / 2], 0.5 * (lat_vals[:-1] + lat_vals[1:]), [lat_vals[-1] + dlat / 2])
    )

    r_earth = 6371e3
    dmu = np.sin(np.deg2rad(lat_b[1:])) - np.sin(np.deg2rad(lat_b[:-1]))
    dl = np.deg2rad(lon_b[1:] - lon_b[:-1])
    area_np = (r_earth**2) * dmu[:, None] * dl[None, :]

    return xr.DataArray(area_np, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))


def load_significance_and_valid(settings: Settings, varname: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load significance mask and valid mask for one variable at target time."""
    imf_path = (
        settings.data_dir
        / settings.dversion
        / settings.ftversion
        / "global_monthly"
        / f"deg1_{varname}_monthly_eemd.nc"
    )
    ar1_path = (
        settings.data_dir
        / settings.dversion
        / settings.ftversion
        / "global_monthly"
        / f"deg1_{varname}_monthly_AR1.nc"
    )

    with xr.open_dataset(imf_path) as ds_imf:
        imf = ds_imf["IMF"]
        trend = imf.isel(mode=-1).sel(time=settings.target_time) - imf.isel(mode=-1, time=0)

    with xr.open_dataset(ar1_path) as ds_ar1:
        ar1_t = ds_ar1.sel(time=settings.target_time)
        signif = (trend > ar1_t["trends_mean"] + settings.confidence * ar1_t["trends_std"]) | (
            trend < ar1_t["trends_mean"] - settings.confidence * ar1_t["trends_std"]
        )

    area = build_gridcell_area(trend["lon"], trend["lat"])
    valid = trend.notnull()
    return signif, valid, area


def area_weighted_signif_fraction(
    signif: xr.DataArray,
    region_mask: xr.DataArray,
    valid: xr.DataArray,
    area: xr.DataArray,
) -> float:
    region_valid = region_mask.astype(bool) & valid
    signif_region = region_valid & signif.astype(bool)

    area_total = area.where(region_valid).sum()
    area_signif = area.where(signif_region).sum()

    if float(area_total.values) == 0.0:
        return np.nan
    return float((area_signif / area_total * 100.0).values)


def compute_increase(settings: Settings, region_name: str, varname: str) -> float:
    """Regional trend increase (%) from initial time to target time."""
    ts_path = (
        settings.data_dir
        / settings.dversion
        / settings.ftversion
        / "space_mean_monthly"
        / f"{region_name}_{varname}_monthly_eemd.nc"
    )
    with xr.open_dataset(ts_path) as ds:
        imf = ds["IMF"]
        trend0 = imf.isel(mode=-1, time=0)
        trend_t = imf.isel(mode=-1).sel(time=settings.target_time)
        return float(((trend_t - trend0) / trend0 * 100.0).values)


def build_results(settings: Settings) -> pd.DataFrame:
    mask_path = settings.data_dir / "mask" / "KEmasked_regions_deg1.nc"
    with xr.open_dataset(mask_path) as ds_mask:
        region_masks = {region.name: ds_mask[region.name].load() for region in REGIONS}

    signif_l, valid_l, area = load_significance_and_valid(settings, "KE_L")
    signif_m, valid_m, _ = load_significance_and_valid(settings, "KE_M")

    rows = []
    for region in REGIONS:
        mask = region_masks[region.name]

        rows.append(
            {
                "Region": region.label,
                "K_L sig. area": area_weighted_signif_fraction(signif_l, mask, valid_l, area),
                "K_M sig. area": area_weighted_signif_fraction(signif_m, mask, valid_m, area),
                "K_L increase": compute_increase(settings, region.name, "KE_L"),
                "K_M increase": compute_increase(settings, region.name, "KE_M"),
            }
        )

    return pd.DataFrame(rows)


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.1f}")
    return out


def draw_table_only(df_display: pd.DataFrame, jpg_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 7,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(3.5, 2.3))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")

    col_labels = [
        "Region",
        "$K_L$\nsig. area (%)",
        "$K_M$\nsig. area (%)",
        "$K_L$\nincrease (%)",
        "$K_M$\nincrease (%)",
    ]

    table = ax.table(
        cellText=df_display.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.26, 0.185, 0.185, 0.185, 0.185],
        bbox=[0.0, 0.01, 1.0, 0.98],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.10)

    n_rows = len(df_display)
    n_cols = len(col_labels)

    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.set_facecolor("white")


        if c == 0:
            cell.set_text_props(ha="left")


    for c in range(n_cols):
        header_cell = table[(0, c)]
        header_cell.visible_edges = "TB"
        header_cell.set_linewidth(0.8)
        header_cell.set_edgecolor("black")

        bottom_cell = table[(n_rows, c)]
        bottom_cell.visible_edges = "B"
        bottom_cell.set_linewidth(0.8)
        bottom_cell.set_edgecolor("black")


    for c in range(n_cols):
        table[(0, c)].set_height(table[(0, c)].get_height() * 1.35)


    fig.savefig(pdf_path, dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


    plt.close(fig)


def main() -> None:

    settings_twosat = Settings(dversion="twosat_i8192")
    out_csv = Path("table_1_percentage_twosat.csv")
    out_jpg = Path("table_1_percentage_twosat.jpg")
    out_pdf = Path("table_1_percentage_twosat.pdf")

    df = build_results(settings_twosat)
    df.to_csv(out_csv, index=False)
    df_display = format_for_display(df)
    draw_table_only(df_display, out_jpg, out_pdf)


    settings_allsat = Settings(dversion="allsat_i8192")
    out_csv_e1 = Path("table_E1_percentage_allsat.csv")
    out_jpg_e1 = Path("table_E1_percentage_allsat.jpg")
    out_pdf_e1 = Path("table_E1_percentage_allsat.pdf")

    df_e1 = build_results(settings_allsat)
    df_e1.to_csv(out_csv_e1, index=False)
    df_display_e1 = format_for_display(df_e1)
    draw_table_only(df_display_e1, out_jpg_e1, out_pdf_e1)


if __name__ == "__main__":
    main()
