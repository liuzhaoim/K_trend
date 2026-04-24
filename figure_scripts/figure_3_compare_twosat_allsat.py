"""
Fig. 3 | Comparison of results derived from the C3S product and the multi-mission product.
"""


import csv
import numpy as np
import xarray as xr
from datetime import datetime

import matplotlib.dates as mdates
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

def plot_crs(ax, gl_l=True, gl_b=True):


    ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor='black', linewidth=0.3, zorder=4)


    ax.spines['geo'].set_linewidth(0.5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_zorder(20)


def add_region_boxes(ax):
    for rgname in ['GulfStream']:
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


def plot_ax_trend(ax, dversion, varname, ntime, vmin_plot, vmax_plot):
    '''
    Plot EEMD trend ending at ntime for KE, with specified vmin/vmax.
    '''

    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_{dsuffix}.nc')['IMF']

    trend = IMF.isel(mode=-1).sel(time=ntime) - IMF.isel(mode=-1, time=0)

    p = ax.pcolormesh(trend['lon'], trend['lat'], trend, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, transform=data_crs, zorder=3)

    mask_invalid = IMF.isel(mode=0, time=0).isnull()
    ax.contourf(mask_invalid['lon'], mask_invalid['lat'], mask_invalid,
               levels=[0.5, 1.5], colors='k', alpha=0.6,
               transform=data_crs, zorder=1)

    return p, trend, mask_invalid


def plot_ax_diff(ax, trend_a, trend_b, mask_a, mask_b, vmin_plot, vmax_plot):
    '''
    Plot trend difference (allsat - twosat) with combined invalid mask.
    '''
    mask_diff = mask_a | mask_b
    trend_diff = (trend_a - trend_b).where(~mask_diff)

    p = ax.pcolormesh(trend_diff['lon'], trend_diff['lat'], trend_diff, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, transform=data_crs, zorder=3)

    ax.contourf(mask_diff['lon'], mask_diff['lat'], mask_diff,
               levels=[0.5, 1.5], colors='k', alpha=0.6,
               transform=data_crs, zorder=1)

    return p


def load_trend_mask(dversion, varname, ntime):
    ds = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_{dsuffix}.nc')
    IMF = ds['IMF']

    trend = (IMF.isel(mode=-1).sel(time=ntime) - IMF.isel(mode=-1, time=0)).load()
    mask_invalid = IMF.isel(mode=0, time=0).isnull().load()

    ds.close()

    return trend, mask_invalid


def load_satnum_data(csv_path):
    dates = []
    counts = []

    with open(csv_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dates.append(datetime.strptime(row['monthly'], '%Y-%m-%d'))
            counts.append(int(row['count']))

    return dates, counts


def plot_ax_satnum(ax):
    dates, counts = load_satnum_data(satnum_csv)

    ax.plot(
        dates,
        counts,
        color='black',
        linewidth=0.8,
        alpha=0.95,
    )

    ax.text(0.5, 1.015, 'Satellite number', color='black', fontsize=7, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_xlabel('Year', fontsize=7)

    ax.set_ylim(0, 8)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, which='major', axis='both', color='0.85', linewidth=0.35, zorder=0)
    ax.tick_params(axis='both', labelsize=7)


def load_if_data(ke_var, sig_type, tp_label):
    ds_allsat = xr.open_dataset(f'{ddir}/SatNum/IF_space_mean/IF_SatNum_{ke_var}_{sig_type}_allsat_{tp_label}.nc')
    ds_twosat = xr.open_dataset(f'{ddir}/SatNum/IF_space_mean/IF_SatNum_{ke_var}_{sig_type}_twosat_{tp_label}.nc')

    data = {
        'T21_allsat': ds_allsat['T21'].values,
        'T21_twosat': ds_twosat['T21'].values,
        'E95_21_allsat': ds_allsat['E95_21'].values,
        'E95_21_twosat': ds_twosat['E95_21'].values,
    }

    ds_allsat.close()
    ds_twosat.close()

    return data


def plot_ax_if(ax, ke_var, show_ylabel=False, show_legend=False):
    data = load_if_data(ke_var, 'deimf1', 'full')

    x = np.arange(len(if_region_names))
    width = 0.38


    colors = [colors_hex[0], colors_hex[1], colors_hex[3], colors_hex[4]]

    ax.bar(
        x - width / 2,
        data['T21_twosat'],
        width=width,
        yerr=data['E95_21_twosat'],
        capsize=2,
        color=colors[1],
        alpha=0.9,
        label='C3S',
        edgecolor=colors[0],
        linewidth=0.5,
        error_kw={
            'elinewidth': 0.8,
            'capthick': 0.8,
            'ecolor': colors[0]
        }
    )
    ax.bar(
        x + width / 2,
        data['T21_allsat'],
        width=width,
        yerr=data['E95_21_allsat'],
        capsize=2,
        color=colors[2],
        alpha=0.9,
        label='Multi-mission',
        edgecolor=colors[3],
        linewidth=0.5,
        error_kw={
            'elinewidth': 0.8,
            'capthick': 0.8,
            'ecolor': colors[3]
        }
    )

    ax.axhline(0, linewidth=0.5, color='k')
    ax.text(0.5, 1.015, rf'$T_{{\mathrm{{SN}}\to K_{{{ke_var[-1]}}}}}$', color='black', fontsize=7, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_xticks(x)
    ax.set_xticklabels(if_region_names, rotation=25, ha='right')
    ax.tick_params(axis='x', labelsize=7, pad=0.2)
    ax.tick_params(axis='y', labelsize=7)

    if show_ylabel:
        ax.set_ylabel(r'(nats month$^{-1}$)', fontsize=7)

    if show_legend:
        ax.legend(loc='upper left', frameon=False, fontsize=7, handlelength=1.0, borderaxespad=0.15, labelspacing=0.3)


def load_psd_data(dversion, rgname):
    ds_twosat = xr.open_dataset(f'{ddir}/Spectrum/{dversion}/psd_twosat_{rgname}_{period_label}.nc')
    ds_allsat = xr.open_dataset(f'{ddir}/Spectrum/{dversion}/psd_allsat_{rgname}_{period_label}.nc')

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


def compute_fk_limits(data_list):
    all_k = np.concatenate([data['K'].ravel() for data in data_list])
    all_f = np.concatenate([data['F'].ravel() for data in data_list])

    k_pos = all_k[all_k > 0]
    f_pos = all_f[all_f > 0]

    return (f_pos.min(), f_pos.max()), (k_pos.min(), k_pos.max())


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
        ax.axvline(f_val, color='k', ls='--', lw=0.5, alpha=0.8)
        ax.text(
            f_val * 1.1,
            ylim[0] * 1.15,
            label,
            rotation=90,
            ha='left',
            va='bottom',
            color='k',
            fontsize=7,
        )

    for k_val, label in k_refs:
        ax.axhline(k_val, color='k', ls='--', lw=0.5, alpha=0.7)
        ax.text(
            xlim[0] * 1.1,
            k_val,
            label,
            ha='left',
            va='bottom',
            color='k',
            fontsize=7,
        )


def add_freq_ref_lines(ax):
    for f_val, _ in f_refs:
        ax.axvline(f_val, color='k', ls='--', lw=0.5, alpha=0.7)


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
    ax.text(0.5, 1.015, rgtext, color='black', fontsize=7, ha='center', va='bottom', transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=7, labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r'Wavenumber $k_h$ (cpkm)', fontsize=7)
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

    ax.tick_params(axis='both', labelsize=7)
    ax.legend(loc='upper left', frameon=False, fontsize=7, handlelength=1.0, borderaxespad=0.15, labelspacing=0.3)

    if show_xlabel:
        ax.set_xlabel(r'Frequency $\omega$ (cpd)', fontsize=7, labelpad=1)
    else:
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r'$\omega \cdot K$ (cm$^2$/s$^2$)', fontsize=7)
    else:
        ax.tick_params(labelleft=False)

    add_freq_ref_lines(ax)


def plot_axs():

    all_spectra = [(rgname, load_psd_data(spectra_dversion, rgname)) for rgname in spectra_rgnames]
    spectra_list = [data for _, data in all_spectra]
    freq_xlim, diff_ylim = compute_fk_limits(spectra_list)
    diff_clim = (-50, 50)
    freq_ylim = compute_freq_ylim(spectra_list)


    fig = plt.figure(figsize=(7.086, 6.692))
    fig.patch.set_facecolor('white')
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[2, 1],
        hspace=0.15,
        left=0.0,
        right=0.925,
        bottom=0.0,
        top=0.97,
    )
    top = outer[0].subgridspec(
        2, 3,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.6,
        hspace=0.12,
    )
    bottom = outer[1].subgridspec(
        1, 3,
        width_ratios=[1, 1, 1],
        wspace=0.25,
    )

    ax00 = fig.add_subplot(top[0, 0:2], projection=proj_crs)
    ax10 = fig.add_subplot(top[1, 0:2], projection=proj_crs)
    ax01 = fig.add_subplot(top[0, 2])
    ax11 = fig.add_subplot(top[1, 2])
    ax20 = fig.add_subplot(bottom[0, 0])
    ax21 = fig.add_subplot(bottom[0, 1])
    ax22 = fig.add_subplot(bottom[0, 2])

    bottom_dx = ax20.get_position().width * 0.2
    bottom_dy = ax20.get_position().height * 0.25
    for ax in [ax20, ax21, ax22]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + bottom_dx, pos.y0 + bottom_dy, pos.width, pos.height * 0.7])

    plot_crs(ax00, gl_b=False)
    plot_crs(ax10, gl_b=False)

    l_x, t_y = -0.13, 1.03
    ax00.text(0.03, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax00.transAxes)
    ax10.text(0.03, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax10.transAxes)
    ax01.text(l_x, t_y, 'c', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax01.transAxes)
    ax11.text(l_x, t_y, 'd', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax11.transAxes)
    ax20.text(l_x, t_y, 'e', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax20.transAxes)
    ax21.text(l_x, t_y, 'f', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax21.transAxes)
    ax22.text(l_x, t_y, 'g', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax22.transAxes)

    ax00.text(0.5, 1.015, r'$K_L$ difference (Multi-mission minus C3S)', ha='center', va='bottom', transform=ax00.transAxes)
    ax10.text(0.5, 1.015, r'$K_M$ difference (Multi-mission minus C3S)', ha='center', va='bottom', transform=ax10.transAxes)

    trend_allsat_L, mask_allsat_L = load_trend_mask(dversions[1], 'KE_L', ntime)
    trend_twosat_L, mask_twosat_L = load_trend_mask(dversions[0], 'KE_L', ntime)
    trend_allsat_M, mask_allsat_M = load_trend_mask(dversions[1], 'KE_M', ntime)
    trend_twosat_M, mask_twosat_M = load_trend_mask(dversions[0], 'KE_M', ntime)

    p_L2 = plot_ax_diff(ax00, trend_allsat_L, trend_twosat_L, mask_allsat_L, mask_twosat_L, vmin_diff_L, vmax_diff_L)
    p_M2 = plot_ax_diff(ax10, trend_allsat_M, trend_twosat_M, mask_allsat_M, mask_twosat_M, vmin_diff_M, vmax_diff_M)
    add_region_boxes(ax10)
    p_spec = plot_ax_fk(ax01, all_spectra[1][1], spectra_print_texts[1], freq_xlim, diff_ylim, diff_clim, show_ylabel=True)
    plot_ax_freq(ax11, all_spectra[1][1], spectra_print_texts[1], freq_xlim, freq_ylim, show_xlabel=True, show_ylabel=True)
    style_log_tick_lengths(ax01, style_x=True, style_y=True)
    style_log_tick_lengths(ax11, style_x=True)
    plot_ax_satnum(ax20)
    plot_ax_if(ax21, 'KE_L', show_ylabel=False, show_legend=True)
    plot_ax_if(ax22, 'KE_M', show_ylabel=False, show_legend=False)

    pos01 = ax01.get_position()
    cax_spec = fig.add_axes([pos01.x1 + 0.01, pos01.y0, 0.008, pos01.height])
    cb_spec = fig.colorbar(
        p_spec,
        cax=cax_spec,
        orientation='vertical',
        extend='both',
        label=r'$k_h \cdot \omega \cdot K$ (cm$^2$/s$^2$)',
    )
    cb_spec.outline.set_linewidth(0.5)
    cb_spec.set_ticks(np.linspace(diff_clim[0], diff_clim[1], 5))
    cb_spec.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    cb_spec.ax.tick_params(labelsize=7)
    cb_spec.ax.yaxis.labelpad = 1


    cbar_fraction = 0.035 * (2.0 / 3.0)

    cbs = [
        fig.colorbar(p_L2, ax=ax00, orientation='vertical', pad=0.02, aspect=28, shrink=0.9, fraction=cbar_fraction, extend='both', label=r'$K_L$ increase (J m$^{-3}$)'),
        fig.colorbar(p_M2, ax=ax10, orientation='vertical', pad=0.02, aspect=28, shrink=0.9, fraction=cbar_fraction, extend='both', label=r'$K_M$ increase (J m$^{-3}$)'),
    ]

    for cb in cbs:
        cb.outline.set_linewidth(0.5)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        cb.ax.yaxis.labelpad = 1

    cbs[0].set_ticks(np.linspace(vmin_diff_L, vmax_diff_L, 5))
    cbs[1].set_ticks(np.linspace(vmin_diff_M, vmax_diff_M, 5))

    pos10 = ax10.get_position()
    poscb10 = cbs[1].ax.get_position()
    ax10_dy = pos10.height * 0.08
    ax10.set_position([pos10.x0, pos10.y0 - ax10_dy, pos10.width, pos10.height])
    cbs[1].ax.set_position([poscb10.x0, poscb10.y0 - ax10_dy, poscb10.width, poscb10.height])

    figure_name = 'fig3 compare twosat allsat'

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


ddir = '/path/to'

dversions = ['twosat_i8192', 'allsat_i8192']
varnames = ['KE_L', 'KE_M']

ftversion = 'mwt_d184'


degree = 'deg1'


tmversion = 'global_monthly'

dsuffix = 'monthly_eemd'

ntime = '2024-10-01'
period_label = '1993_2024'
satnum_csv = f'{ddir}/SatNum/SatNum_monthly_count.csv'

if_region_names = ["Kuroshio", "Gulf Stream", "Agulhas Current", "EAC", "BMC", "ACC", "global ocean"]

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

vmin_L, vmax_L = -80, 80
vmin_M, vmax_M = -80, 80
vmin_diff_L, vmax_diff_L = -80, 80
vmin_diff_M, vmax_diff_M = -80, 80

spectra_dversion = 'v2'
spectra_rgnames = [
    'Kuroshio',
    'GulfStream',
    'Agulhas',
    'EastAustralian',
    'Malvinas',
    'ACC',
]
spectra_print_texts = [
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
k_refs = [
    (1.0 / 500.0, '500 km'),
    (1.0 / 80.0, '80 km'),
]
f_refs = [
    (1.0 / 184.0, '184 day'),
    (1.0 / 10.0, '10 day'),
]

plot_axs()
