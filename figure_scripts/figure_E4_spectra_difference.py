"""
Extended Data Fig. 4 | Differences in kinetic energy spectra between the multi-mission and C3S products across major jet regions.
"""


import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

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


proj_crs = ccrs.Robinson(central_longitude=180)
data_crs = ccrs.PlateCarree()


def plot_crs(ax):

    ax.set_extent([0.01, 359.99, -70, 70], crs=data_crs)
    ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
    ax.add_feature(
        cartopy.feature.LAND,
        edgecolor='black',
        facecolor='black',
        linewidth=0.3,
        zorder=4,
    )

    ax.spines['geo'].set_linewidth(0.5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_zorder(20)


def load_psd_data(dversion, rgname):

    ds_twosat = xr.open_dataset(
        f'{ddir}/Spectrum/{dversion}/psd_twosat_{rgname}_{period_label}.nc'
    )
    ds_allsat = xr.open_dataset(
        f'{ddir}/Spectrum/{dversion}/psd_allsat_{rgname}_{period_label}.nc'
    )

    wavenumbers = ds_twosat['wn'].values
    frequencies = ds_twosat['freq'].values

    K, F = np.meshgrid(wavenumbers * 1e3, frequencies * 86400.0)

    ke_twosat = (ds_twosat['u_psd'].values + ds_twosat['v_psd'].values) / 2
    ke_allsat = (ds_allsat['u_psd'].values + ds_allsat['v_psd'].values) / 2

    jacobian = 1e-3 / 86400.0
    ke_twosat = ke_twosat * jacobian * 1.0e4
    ke_allsat = ke_allsat * jacobian * 1.0e4

    dk = wavenumbers[1] - wavenumbers[0]
    ef_twosat = np.sum((ds_twosat['u_psd'].values + ds_twosat['v_psd'].values) / 2, axis=1)
    ef_allsat = np.sum((ds_allsat['u_psd'].values + ds_allsat['v_psd'].values) / 2, axis=1)
    ef_twosat = ef_twosat * dk / 86400.0 * 1.0e4
    ef_allsat = ef_allsat * dk / 86400.0 * 1.0e4

    freq_cpd = frequencies * 86400.0
    pos = freq_cpd > 0

    ds_twosat.close()
    ds_allsat.close()

    return {
        'K': K,
        'F': F,
        'freq_cpd': freq_cpd[pos],
        'vp_diff': K * F * (ke_allsat - ke_twosat),
        'vp_ef_c3s': freq_cpd[pos] * ef_twosat[pos],
        'vp_ef_multimission': freq_cpd[pos] * ef_allsat[pos],
    }


def load_km_difference():

    ds_twosat = xr.open_dataset(
        f'{ddir}/{trend_dversions[0]}/{ftversion}/{tmversion}/'
        f'{degree}_{trend_varname}_{dsuffix}.nc'
    )
    ds_allsat = xr.open_dataset(
        f'{ddir}/{trend_dversions[1]}/{ftversion}/{tmversion}/'
        f'{degree}_{trend_varname}_{dsuffix}.nc'
    )

    imf_twosat = ds_twosat['IMF']
    imf_allsat = ds_allsat['IMF']

    trend_twosat = imf_twosat.isel(mode=-1).sel(time=ntime) - imf_twosat.isel(mode=-1, time=0)
    trend_allsat = imf_allsat.isel(mode=-1).sel(time=ntime) - imf_allsat.isel(mode=-1, time=0)

    mask_twosat = imf_twosat.isel(mode=0, time=0).isnull()
    mask_allsat = imf_allsat.isel(mode=0, time=0).isnull()
    mask_diff = (mask_twosat | mask_allsat).load()
    trend_diff = (trend_allsat - trend_twosat).where(~mask_diff).load()

    ds_twosat.close()
    ds_allsat.close()

    return trend_diff, mask_diff


def compute_fk_limits(data_list):

    all_k = np.concatenate([data['K'].ravel() for data in data_list])
    all_f = np.concatenate([data['F'].ravel() for data in data_list])

    k_pos = all_k[all_k > 0]
    f_pos = all_f[all_f > 0]

    return (f_pos.min(), f_pos.max()), (k_pos.min(), k_pos.max())


def compute_fk_diff_clim(data_list):

    all_diff = np.concatenate([np.abs(data['vp_diff'].ravel()) for data in data_list])
    all_diff = all_diff[np.isfinite(all_diff)]
    vmax = np.percentile(all_diff, 100) * 0.2
    vmax = 50
    return -vmax, vmax


def compute_freq_ylim(data_list):

    all_values = np.concatenate([
        np.concatenate([data['vp_ef_c3s'], data['vp_ef_multimission']])
        for data in data_list
    ])
    all_values = all_values[np.isfinite(all_values) & (all_values > 0)]

    return 0.0, all_values.max() * 1.05


def add_fk_ref_lines(ax):

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    for f_val, label in f_refs:
        ax.axvline(f_val, color='k', ls='--', lw=LINE_WIDTH, alpha=0.8)
        ax.text(
            f_val * 1.1,
            ylim[0] * 1.15,
            label,
            rotation=90,
            ha='left',
            va='bottom',
            color='k',
            fontsize=TICK_LABEL_SIZE,
        )

    for k_val, label in k_refs:
        ax.axhline(k_val, color='k', ls='--', lw=LINE_WIDTH, alpha=0.7)
        ax.text(
            xlim[0] * 1.1,
            k_val,
            label,
            ha='left',
            va='bottom',
            color='k',
            fontsize=TICK_LABEL_SIZE,
        )


def add_freq_ref_lines(ax):

    for f_val, _ in f_refs:
        ax.axvline(f_val, color='k', ls='--', lw=LINE_WIDTH, alpha=0.7)


def add_region_boxes(ax):

    for rgname in rgnames:
        lon_min, lon_max, lat_min, lat_max = region_bounds[rgname]
        ax.add_patch(
            Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                fill=False,
                edgecolor=colors_hex[0],
                linewidth=0.8,
                transform=data_crs,
                zorder=30,
            )
        )


def style_log_tick_lengths(ax, style_x=False, style_y=False, major_length=2.5, minor_length=1.5):

    if style_x:
        ax.tick_params(axis='x', which='major', length=major_length, width=0.5)
        ax.tick_params(axis='x', which='minor', length=minor_length, width=0.5)

    if style_y:
        ax.tick_params(axis='y', which='major', length=major_length, width=0.5)
        ax.tick_params(axis='y', which='minor', length=minor_length, width=0.5)


def plot_ax_fk(ax, data, rgtext, freq_xlim, diff_ylim, diff_clim, show_ylabel=False):

    p = ax.pcolormesh(
        data['F'][:-1, :],
        data['K'][:-1, :],
        data['vp_diff'][:-1, :],
        cmap='RdBu_r',
        vmin=diff_clim[0],
        vmax=diff_clim[1],
        shading='auto',
        rasterized=True,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(freq_xlim)
    ax.set_ylim(4e-4, diff_ylim[1])
    style_log_tick_lengths(ax, style_x=True, style_y=True)
    ax.text(
        0.5,
        1.015,
        rgtext,
        color='black',
        fontsize=TICK_LABEL_SIZE,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE, labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r'Wavenumber $k_h$ (cpkm)', fontsize=TICK_LABEL_SIZE)
    else:
        ax.tick_params(labelleft=False)

    add_fk_ref_lines(ax)

    return p


def plot_ax_freq(ax, data, rgtext, freq_xlim, freq_ylim, show_xlabel=False, show_ylabel=False):

    ax.plot(data['freq_cpd'], data['vp_ef_c3s'], color=colors_hex[0], linewidth=0.65, label='C3S', alpha=0.8)
    ax.plot(data['freq_cpd'], data['vp_ef_multimission'], color=colors_hex[4], linewidth=0.65, label='Multi-mission', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlim(freq_xlim)
    ax.set_ylim(freq_ylim)
    style_log_tick_lengths(ax, style_x=True)
    ax.text(
        0.5,
        1.015,
        rgtext,
        color='black',
        fontsize=TICK_LABEL_SIZE,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=TICK_LABEL_SIZE,
        handlelength=1.0,
        borderaxespad=0.15,
        labelspacing=0.3,
    )

    if show_xlabel:
        ax.set_xlabel(r'Frequency $\omega$ (cpd)', fontsize=TICK_LABEL_SIZE)
    else:
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r'$\omega \cdot K$ (cm$^2$/s$^2$)', fontsize=TICK_LABEL_SIZE)
    else:
        ax.tick_params(labelleft=False)

    add_freq_ref_lines(ax)


def plot_ax_map(ax):

    trend_diff, mask_diff = load_km_difference()

    plot_crs(ax)

    p = ax.pcolormesh(
        trend_diff['lon'],
        trend_diff['lat'],
        trend_diff,
        cmap=cmap,
        vmin=-80,
        vmax=80,
        transform=data_crs,
        zorder=3,
        rasterized=True,
    )
    ax.contourf(
        mask_diff['lon'],
        mask_diff['lat'],
        mask_diff,
        levels=[0.5, 1.5],
        colors='k',
        alpha=0.6,
        transform=data_crs,
        zorder=1,
    )

    add_region_boxes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        1.015,
        r'$K_M$ difference (Multi-mission minus C3S)',
        color='black',
        fontsize=TICK_LABEL_SIZE,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )

    return p


def plot_axs(dversion):

    all_data = [(rgname, load_psd_data(dversion, rgname)) for rgname in rgnames]
    data_list = [data for _, data in all_data]

    freq_xlim, diff_ylim = compute_fk_limits(data_list)
    diff_clim = compute_fk_diff_clim(data_list)
    freq_ylim = compute_freq_ylim(data_list)

    fig = plt.figure(figsize=(7.086, 6.692))
    fig.patch.set_facecolor('white')

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[5.0, 1.0],
        hspace=0.15,
        left=0.07,
        right=0.925,
        bottom=0.0,
        top=0.97,
    )
    top = outer[0].subgridspec(
        4, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.2,
        hspace=0.25,
    )
    bottom = outer[1].subgridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.0,
    )

    ax00 = fig.add_subplot(top[0, 0])
    ax01 = fig.add_subplot(top[0, 1])
    ax02 = fig.add_subplot(top[0, 2])
    ax10 = fig.add_subplot(top[1, 0])
    ax11 = fig.add_subplot(top[1, 1])
    ax12 = fig.add_subplot(top[1, 2])
    ax20 = fig.add_subplot(top[2, 0])
    ax21 = fig.add_subplot(top[2, 1])
    ax22 = fig.add_subplot(top[2, 2])
    ax30 = fig.add_subplot(top[3, 0])
    ax31 = fig.add_subplot(top[3, 1])
    ax32 = fig.add_subplot(top[3, 2])
    ax40 = fig.add_subplot(bottom[0, 0])
    ax41 = fig.add_subplot(bottom[0, 1], projection=proj_crs)
    ax42 = fig.add_subplot(bottom[0, 2])

    ax40.set_axis_off()
    ax42.set_axis_off()

    l_x, t_y = -0.1, 1.02
    ax00.text(l_x, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax00.transAxes)
    ax10.text(l_x, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax10.transAxes)
    ax20.text(l_x, t_y, 'c', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax20.transAxes)
    ax30.text(l_x, t_y, 'd', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax30.transAxes)
    ax01.text(l_x, t_y, 'e', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax01.transAxes)
    ax11.text(l_x, t_y, 'f', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax11.transAxes)
    ax21.text(l_x, t_y, 'g', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax21.transAxes)
    ax31.text(l_x, t_y, 'h', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax31.transAxes)
    ax02.text(l_x, t_y, 'i', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax02.transAxes)
    ax12.text(l_x, t_y, 'j', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax12.transAxes)
    ax22.text(l_x, t_y, 'k', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax22.transAxes)
    ax32.text(l_x, t_y, 'l', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax32.transAxes)
    ax41.text(l_x, 1.05, 'm', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax41.transAxes)

    p_top = plot_ax_fk(ax00, all_data[0][1], print_texts[0], freq_xlim, diff_ylim, diff_clim, show_ylabel=True)
    plot_ax_fk(ax01, all_data[2][1], print_texts[2], freq_xlim, diff_ylim, diff_clim)
    plot_ax_fk(ax02, all_data[4][1], print_texts[4], freq_xlim, diff_ylim, diff_clim)

    plot_ax_freq(ax10, all_data[0][1], print_texts[0], freq_xlim, freq_ylim, show_ylabel=True)
    plot_ax_freq(ax11, all_data[2][1], print_texts[2], freq_xlim, freq_ylim)
    plot_ax_freq(ax12, all_data[4][1], print_texts[4], freq_xlim, freq_ylim)

    p_bottom = plot_ax_fk(ax20, all_data[1][1], print_texts[1], freq_xlim, diff_ylim, diff_clim, show_ylabel=True)
    plot_ax_fk(ax21, all_data[3][1], print_texts[3], freq_xlim, diff_ylim, diff_clim)
    plot_ax_fk(ax22, all_data[5][1], print_texts[5], freq_xlim, diff_ylim, diff_clim)

    plot_ax_freq(ax30, all_data[1][1], print_texts[1], freq_xlim, freq_ylim, show_xlabel=True, show_ylabel=True)
    plot_ax_freq(ax31, all_data[3][1], print_texts[3], freq_xlim, freq_ylim, show_xlabel=True)
    plot_ax_freq(ax32, all_data[5][1], print_texts[5], freq_xlim, freq_ylim, show_xlabel=True)

    p_map = plot_ax_map(ax41)

    pos02 = ax02.get_position()
    pos22 = ax22.get_position()
    pos41 = ax41.get_position()
    cax_top = fig.add_axes([pos02.x1 + 0.01, pos02.y0, 0.008, pos02.height])
    cax_bottom = fig.add_axes([pos22.x1 + 0.01, pos22.y0, 0.008, pos22.height])
    cax_map = fig.add_axes([pos41.x1 + 0.01, pos41.y0, 0.008, pos41.height])

    cb_top = fig.colorbar(
        p_top,
        cax=cax_top,
        orientation='vertical',
        extend='both',
        label=r'$k_h \cdot \omega \cdot K$ (cm$^2$/s$^2$)',
    )
    cb_bottom = fig.colorbar(
        p_bottom,
        cax=cax_bottom,
        orientation='vertical',
        extend='both',
        label=r'$k_h \cdot \omega \cdot K$ (cm$^2$/s$^2$)',
    )
    cb_map = fig.colorbar(
        p_map,
        cax=cax_map,
        orientation='vertical',
        extend='both',
        label=r'$K_M$ increase (J m$^{-3}$)',
    )

    cb_top.outline.set_linewidth(0.5)
    cb_bottom.outline.set_linewidth(0.5)
    cb_map.outline.set_linewidth(0.5)

    cb_top.set_ticks(np.linspace(diff_clim[0], diff_clim[1], 5))
    cb_bottom.set_ticks(np.linspace(diff_clim[0], diff_clim[1], 5))
    cb_map.set_ticks(np.linspace(-80, 80, 5))

    cb_top.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    cb_bottom.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    cb_map.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    cb_top.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    cb_bottom.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    cb_map.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    cb_top.ax.yaxis.labelpad = 1
    cb_bottom.ax.yaxis.labelpad = 1
    cb_map.ax.yaxis.labelpad = 1

    figure_name = f'figE4 spectra difference'

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor='white', transparent=False)


ddir = '/path/to'

dversions = ['v2']
rgnames = [
    'Kuroshio',
    'GulfStream',
    'Agulhas',
    'EastAustralian',
    'Malvinas',
    'ACC',
]
print_texts = [
    'Kuroshio',
    'Gulf Stream',
    'Agulhas Current',
    'East Australian Current',
    'Brazil-Malvinas Confluence',
    'Antarctic Circumpolar Current',
]

region_bounds = {
    'Kuroshio': (140.625, 183.625, 30.375, 39.875),
    'GulfStream': (286.375, 321.625, 32.875, 42.375),
    'Agulhas': (6.125, 49.875, -42.875, -29.875),
    'EastAustralian': (153.375, 166.375, -37.125, -22.125),
    'Malvinas': (304.125, 330.875, -50.875, -36.125),
    'ACC': (78.125, 223.625, -54.625, -48.375),
}

period_label = '1993_2024'
trend_dversions = ['twosat_i8192', 'allsat_i8192']
trend_varname = 'KE_M'

ftversion = 'mwt_d184'
degree = 'deg1'
tmversion = 'global_monthly'
dsuffix = 'monthly_eemd'
ntime = '2024-10-01'

TICK_LABEL_SIZE = 7.0
LINE_WIDTH = 0.5


k_refs = [
    (1.0 / 500.0, '500 km'),
    (1.0 / 80.0, '80 km'),
]
f_refs = [
    (1.0 / 184.0, '184 day'),
    (1.0 / 10.0, '10 day'),
]

colors_hex = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
nodes = [0.0, 0.45, 0.5, 0.55, 1.0]
rgb_colors = [mcolors.to_rgb(c) for c in colors_hex]
cdict = {'red': [], 'green': [], 'blue': []}
for i, node in enumerate(nodes):
    cdict['red'].append((node, rgb_colors[i][0], rgb_colors[i][0]))
    cdict['green'].append((node, rgb_colors[i][1], rgb_colors[i][1]))
    cdict['blue'].append((node, rgb_colors[i][2], rgb_colors[i][2]))
cmap = mcolors.LinearSegmentedColormap('custom_bwr_c4', cdict)
cmap1 = mcolors.LinearSegmentedColormap('custom_bwr_c4_1', cdict)


for dversion in dversions:
    print(f'Plotting {dversion} for {period_label} ...', flush=True)
    plot_axs(dversion)

print('All done!')
