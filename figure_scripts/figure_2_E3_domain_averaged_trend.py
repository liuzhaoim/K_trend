"""
Fig. 2 | Instantaneous trends and trend rates of domain-averaged large-scale (KL) and mesoscale (KM) kinetic energy for selected ocean regions derived from the C3S product.
Extended Data Fig. 3 | Instantaneous trends and trend rates of domain-averaged large-scale (KL) and mesoscale (KM) kinetic energy for selected ocean regions derived from the multi-mission product.
"""


import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import cartopy
import cartopy.crs as ccrs

plt.rcdefaults()
plt.rcParams.update({'font.size': 7})
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['mathtext.tt'] = 'Arial'
plt.rcParams['mathtext.cal'] = 'Symbol'
plt.rcParams['mathtext.fallback'] = 'stix'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'axes.linewidth': 0.5,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
})


rgnames = [
    "mask_Kuroshio",
    "mask_GulfStream",
    "mask_Agulhas",
    "mask_EastAustralian",
    "mask_Malvinas",
    "mask_ACC",
    "mask_global_EQ10",
]
print_texts = [
    "Kuroshio",
    "Gulf Stream",
    "Agulhas Current",
    "East Australian Current",
    "Brazil-Malvinas Confluence",
    "Antarctic Circumpolar Current",
    "global ocean",
]

Paired = list(plt.cm.Paired.colors)
colors = [Paired[1], Paired[7], Paired[3], Paired[11], Paired[9], Paired[5], "black"]

ddir = "/path/to"
dversions = ["twosat_i8192", "allsat_i8192"]
varnames = ["KE_L", "KE_M"]

ftversion = "mwt_d184"
tmversion = "space_mean_monthly"
dsuffix = "monthly_eemd"

ntimeA = "1993-07-01"
ntimeB = "2024-11-01"
ntimeC = "2024-10-31"
PANEL_LABEL_X = -0.24
PANEL_LABEL_Y = 1.02
AB_LABEL_Y_OFFSET = -0.025
D_LABEL_X = 0.02
TICK_LABEL_SIZE = 7.0

proj_crs = ccrs.Robinson(central_longitude=180)
data_crs = ccrs.PlateCarree()


def load_imf(dversion, varname, rgname):
    path = f"{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_{dsuffix}.nc"
    return xr.open_dataset(path)["IMF"]


def get_plot_time(imf, force_month_end=False):
    time = pd.to_datetime(imf["time"].values) + pd.Timedelta(days=15)
    if force_month_end:
        time = time[:-1].append(pd.DatetimeIndex([ntimeC]))
    return time


def load_deimf1_signal(imf, mode_index=1):
    rrest = (
        imf.isel(mode=slice(1, None))
        .isel(mode=slice(None, None, -1))
        .cumsum("mode")
        .isel(mode=slice(None, None, -1))
    )
    return rrest.isel(mode=mode_index)


def compute_signif_mask(imf, dversion, varname, rgname, confidence=1.645):
    trend = imf.isel(mode=-1) - imf.isel(mode=-1, time=0)
    path_m = (
        f"{ddir}/{dversion}/{ftversion}/{tmversion}/"
        f"{rgname}_{varname}_monthly_deIMF1_AR1.nc"
    )
    imfm = xr.open_dataset(path_m)["IMFm"]
    trends = imfm - imfm.isel(time=0)
    trends_mean = trends.mean(dim="samples")
    trends_std = trends.std(dim="samples")
    signif = (trend > trends_mean + confidence * trends_std) | (
        trend < trends_mean - confidence * trends_std
    )
    return signif


def compute_trend_rate(imf):
    trend = imf.isel(mode=-1) - imf.isel(mode=-1, time=0)
    trend = trend.assign_coords(dt=("time", np.arange(len(imf["time"]))))
    trend_rate = trend.differentiate("dt") * 12 * 10
    return trend_rate


def configure_time_axis(ax, show_xtick_labels):
    ax.set_xlim(pd.Timestamp(ntimeA), pd.Timestamp(ntimeB))
    xticks = pd.date_range(start=ntimeA, end=ntimeB, freq="6YS")
    ax.set_xticks(xticks)
    if show_xtick_labels:
        ax.set_xticklabels(xticks.year)
    else:
        ax.set_xticklabels([])


def set_lane_ticks_fixed4(ax, y_low, y_high, is_global):
    step = 0.05 if is_global else 0.5
    interval = np.ceil((y_high - y_low) / (3 * step)) * step
    interval = max(interval, step)
    tick0 = np.floor(y_low / step) * step
    ticks = tick0 + interval * np.arange(4)
    ticks = np.where(np.isclose(ticks, 0.0, atol=step / 100), 0.0, ticks)
    labels = [f"{t:.2f}" if is_global else f"{t:.1f}" for t in ticks]
    ax.set_ylim(float(ticks[0]), float(ticks[-1]))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def plot_stacked_panel(fig, panel_ax, dversion, varname, panel_label):
    if varname == "KE_L":
        var_label = r"$K_L$"
    else:
        var_label = r"$K_M$"
    unit_label = r"(J m$^{-3}$)"

    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)
    panel_ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y + AB_LABEL_Y_OFFSET,
        panel_label,
        fontsize=8, fontweight="bold", fontstyle="normal",
        ha="left",
        va="bottom",
        transform=panel_ax.transAxes,
    )
    n_series = len(rgnames)
    gap = 0.0
    lane_h = (1 - gap * (n_series - 1)) / n_series

    for i, (rgname, rgtext, color) in enumerate(zip(rgnames, print_texts, colors)):
        y0 = 1 - (i + 1) * lane_h - i * gap
        lane_ax = panel_ax.inset_axes([0.0, y0, 1.0, lane_h], transform=panel_ax.transAxes)
        lane_ax.patch.set_alpha(0)

        imf = load_imf(dversion, varname, rgname)
        time = get_plot_time(imf, force_month_end=True)
        monthly = load_deimf1_signal(imf, mode_index=1).values
        trend = imf.isel(mode=-1).values
        signif = compute_signif_mask(imf, dversion, varname, rgname).values

        y_all = np.concatenate([monthly, trend])
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)
        span = y_max - y_min
        pad = 0.10 * span if span > 0 else 0.10 * max(abs(y_max), 1.0)
        y_low = y_min - pad
        y_high = y_max + pad

        lane_ax.plot(time, monthly, color=color, linewidth=0.5, alpha=0.9, zorder=2)
        lane_ax.plot(time, trend, color=color, linewidth=1.5, alpha=0.95, zorder=3)
        configure_time_axis(lane_ax, show_xtick_labels=(i == n_series - 1))
        set_lane_ticks_fixed4(lane_ax, y_low, y_high, is_global=(i == 6))
        ymin, ymax = lane_ax.get_ylim()
        lane_ax.fill_between(time, ymin, ymax, where=signif, color="0.78", zorder=0)
        lane_ax.grid(True, which="major", axis="both", color="0.85", linewidth=0.35, zorder=1)

        is_left = i % 2 == 0
        lane_ax.yaxis.set_ticks_position("left" if is_left else "right")
        lane_ax.tick_params(
            axis="y",
            labelleft=is_left,
            left=is_left,
            labelright=not is_left,
            right=not is_left,
            colors=color,
            pad=2.0,
            labelsize=TICK_LABEL_SIZE,
        )
        lane_ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)

        lane_ax.spines["left"].set_visible(True)
        lane_ax.spines["right"].set_visible(True)
        lane_ax.spines["left"].set_linewidth(0.5)
        lane_ax.spines["right"].set_linewidth(0.5)
        lane_ax.spines["top"].set_visible(i == 0)
        lane_ax.spines["bottom"].set_visible(i == n_series - 1)
        lane_ax.spines["top"].set_linewidth(0.5)
        lane_ax.spines["bottom"].set_linewidth(0.5)

        if i < n_series - 1:
            lane_ax.tick_params(axis="x", bottom=False)

        lane_label = f"{rgtext} {var_label}\n{unit_label}"
        lane_ax.set_ylabel(lane_label, color=color, fontsize=TICK_LABEL_SIZE, rotation=90)
        lane_ax.yaxis.set_label_position("left" if is_left else "right")


def plot_trend_rate_panel(ax, dversion, varname, panel_label):
    if varname == "KE_L":
        var_line = r"$K_L$ trend rate"
    else:
        var_line = r"$K_M$ trend rate"
    unit_line = r"(J m$^{-3}$ decade$^{-1}$)"

    ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        panel_label,
        fontsize=8, fontweight="bold", fontstyle="normal",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.axhline(y=0, linewidth=0.5, color="gray", linestyle=(0, (12, 6)))

    for rgname, rgtext, color in zip(rgnames, print_texts, colors):
        imf = load_imf(dversion, varname, rgname)
        time = get_plot_time(imf)
        trend_rate = compute_trend_rate(imf)
        ax.plot(time, trend_rate.values, color=color, linewidth=1.0, alpha=0.9)

    configure_time_axis(ax, show_xtick_labels=True)
    ax.set_xlabel("")
    ax.set_ylabel(f"{var_line}\n{unit_line}", fontsize=TICK_LABEL_SIZE, rotation=90)
    ax.yaxis.set_label_position("left")
    ax.grid(True, which="major", axis="both", color="0.85", linewidth=0.35, zorder=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    ylim_map = {
        ("KE_L", "twosat_i8192"): (-3.2, 3.2, 1, None),
        ("KE_M", "twosat_i8192"): (-3.2, 5.51, 2, 1),
        ("KE_L", "allsat_i8192"): (-2.2, 4.51, 2, 1),
        ("KE_M", "allsat_i8192"): (-0.2, 8.51, 2, 1),
    }
    if (varname, dversion) in ylim_map:
        ymin, ymax, major, minor = ylim_map[(varname, dversion)]
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(major))
        if minor is not None:
            ax.yaxis.set_minor_locator(MultipleLocator(minor))


def plot_region_definition_panel(ax, panel_label):
    ax.text(
        D_LABEL_X,
        PANEL_LABEL_Y,
        panel_label,
        fontsize=8, fontweight="bold", fontstyle="normal",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )


    ax.add_feature(cartopy.feature.OCEAN, color="white", zorder=0)
    ax.add_feature(
        cartopy.feature.LAND,
        edgecolor="black",
        facecolor="black",
        linewidth=0.3,
        zorder=4,
    )
    ax.spines["geo"].set_linewidth(0.5)
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_zorder(20)
    ax.set_global()


    ds_mask = xr.open_dataset(f"{ddir}/mask/KEmasked_regions_deg1.nc")
    for r in range(6):
        mask_all = ds_mask[rgnames[r]]
        ax.pcolormesh(
            ds_mask["lon"],
            ds_mask["lat"],
            np.where(mask_all, 1, np.nan),
            cmap=mcolors.ListedColormap([colors[r]]),
            transform=data_crs,
            alpha=1.0,
            zorder=3,
        )


    ax.set_xticks([])
    ax.set_yticks([])


def plot_merged_figure(dversion):
    fig = plt.figure(figsize=(7.086, 6.692))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, height_ratios=[6.2, 1.0], hspace=0.12, left=0.095, right=0.905, bottom=0.025, top=0.975)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.6)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[35, 30, 35], wspace=0.05)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1], projection=proj_crs)
    ax_e = fig.add_subplot(bottom[0, 2])


    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    digit_w = 0.010
    ax_c.set_position(
        [pos_c.x0 - 1 * digit_w, pos_c.y0, pos_c.width, pos_c.height]
    )
    ax_e.set_position(
        [pos_e.x0 + 7 * digit_w, pos_e.y0, pos_e.width, pos_e.height]
    )
    ax_d.set_position(
        [pos_d.x0 - 6 * digit_w, pos_d.y0, pos_d.width + 12 * digit_w, pos_d.height]
    )

    plot_stacked_panel(fig, ax_a, dversion, "KE_L", "a")
    plot_stacked_panel(fig, ax_b, dversion, "KE_M", "c")
    plot_trend_rate_panel(ax_c, dversion, "KE_L", "b")
    plot_region_definition_panel(ax_d, "e")
    plot_trend_rate_panel(ax_e, dversion, "KE_M", "d")

    if dversion == dversions[0]:
        figure_name = f"fig2 domain averaged trend"
    else:
        figure_name = f"figE3 domain averaged trend {dversion[:6]}"

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


plot_merged_figure(dversions[0])
plot_merged_figure(dversions[1])
