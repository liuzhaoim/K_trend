"""
Fig. 1 | Global distributions of time-varying trends in large-scale and mesoscale kinetic energy derived from the C3S product.
Extended Data Fig. 2 | Global distributions of time-varying trends in large-scale and mesoscale kinetic energy derived from the multi-mission product.
"""


import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

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

plt.rcParams['hatch.linewidth'] = 0.08

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


def calc_signif_area_fraction(da, signif):
    '''
    input:
        da: xr.DataArray (lat, lon)
        signif : xr.DataArray (lat, lon), bool
    output:
        frac_pct : float, signif_area / total_valid_area * 100 (%)
    '''

    lat = da['lat'].values
    lon = da['lon'].values

    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())

    lat_b = np.concatenate(([lat[0] - dlat/2], 0.5*(lat[:-1] + lat[1:]), [lat[-1] + dlat/2]))
    lon_b = np.concatenate(([lon[0] - dlon/2], 0.5*(lon[:-1] + lon[1:]), [lon[-1] + dlon/2]))

    R = 6371e3
    dmu = np.sin(np.deg2rad(lat_b[1:])) - np.sin(np.deg2rad(lat_b[:-1]))
    dl  = np.deg2rad(lon_b[1:] - lon_b[:-1])

    area_np = (R**2) * dmu[:, None] * dl[None, :]
    area = xr.DataArray(area_np, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))

    valid = da.notnull()

    area_total  = area.where(valid).sum()
    area_signif = area.where(valid & signif).sum()

    frac_pct = float((area_signif / area_total * 100.0).values)
    return frac_pct


def plot_ax_trend(ax, dversion, varname, ntime, vmin_plot, vmax_plot, cmap_plot):
    '''
    Plot EEMD trend ending in different years for KE.
    '''

    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_{dsuffix}.nc')['IMF']

    trend = IMF.isel(mode=-1).sel(time=ntime) - IMF.isel(mode=-1, time=0)

    p = ax.pcolormesh(trend['lon'], trend['lat'], trend, cmap=cmap_plot, vmin=vmin_plot, vmax=vmax_plot, transform=data_crs, zorder=3, rasterized=False)

    ax.text(0.5, 1.015, f'{ntime[:4]}', color='black', ha='center', va='bottom', transform=ax.transAxes)

    mask_invalid = IMF.isel(mode=0, time=0).isnull()

    ax.contourf(mask_invalid['lon'], mask_invalid['lat'], mask_invalid,
               levels=[0.5, 1.5], colors='k', alpha=0.6,
               transform=data_crs, zorder=1)

    return p

def plot_ax_trend_signif(ax, dversion, varname, ntime):
    '''
    Plot statistically significant EEMD trend ending in final years for KE.

    Overlay statistically significant trend areas on an existing ax
    WITHOUT changing the base plot.
    '''

    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_{dsuffix}.nc')['IMF']

    trend = IMF.isel(mode=-1).sel(time=ntime) - IMF.isel(mode=-1, time=0)

    mask_invalid = IMF.isel(mode=0, time=0).isnull()

    ds = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_monthly_AR1.nc').sel(time=ntime)
    trends_mean = ds['trends_mean']
    trends_std = ds['trends_std']
    confid = 1.645

    signif = (trend > trends_mean + confid * trends_std) | (trend < trends_mean - confid * trends_std)


    lons, lats = np.meshgrid(signif['lon'].values, signif['lat'].values)
    sig_mask = signif.values
    ax.scatter(lons[sig_mask], lats[sig_mask], s=0.13, c='k', edgecolors='none',
               transform=data_crs, zorder=3)


def plot_axs(dversion):

    fig = plt.figure(figsize=(7.086, 4.35))
    fig.patch.set_facecolor('white')
    nrows, ncols = 2, 2
    gs = fig.add_gridspec(nrows, ncols, width_ratios=np.ones(ncols), height_ratios=np.ones(nrows), wspace=0.02, hspace=0.0, left=0.01, right=0.99, bottom=0.08, top=0.99)

    ax00 = fig.add_subplot(gs[0, 0], projection=proj_crs)
    ax01 = fig.add_subplot(gs[0, 1], projection=proj_crs)
    ax10 = fig.add_subplot(gs[1, 0], projection=proj_crs)
    ax11 = fig.add_subplot(gs[1, 1], projection=proj_crs)

    plot_crs(ax00, gl_b=False)
    plot_crs(ax10)
    plot_crs(ax01, gl_b=False, gl_l=False)
    plot_crs(ax11, gl_l=False)

    l_x, r_x, t_y = 0.03, 0.03, 1.015
    ax00.text(l_x, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax00.transAxes)
    ax10.text(l_x, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax10.transAxes)
    ax01.text(r_x, t_y, 'c', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax01.transAxes)
    ax11.text(r_x, t_y, 'd', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax11.transAxes)


    p_L = plot_ax_trend(ax00, dversion, varnames[0], ntimes[0], vmin_L, vmax_L, cmap)
    plot_ax_trend(ax10, dversion, varnames[0], ntimes[1], vmin_L, vmax_L, cmap)


    p_M = plot_ax_trend(ax01, dversion, varnames[1], ntimes[0], vmin_M, vmax_M, cmap)
    plot_ax_trend(ax11, dversion, varnames[1], ntimes[1], vmin_M, vmax_M, cmap)


    plot_ax_trend_signif(ax00, dversion, varnames[0], ntimes[0])
    plot_ax_trend_signif(ax10, dversion, varnames[0], ntimes[1])
    plot_ax_trend_signif(ax01, dversion, varnames[1], ntimes[0])
    plot_ax_trend_signif(ax11, dversion, varnames[1], ntimes[1])

    cax = fig.add_axes([0.102, 0.04, 0.3, 0.01])
    cax1 = fig.add_axes([0.598, 0.04, 0.3, 0.01])

    cb_label = r'$K_L$ increase (J m$^{-3}$)'
    cb_label1 = r'$K_M$ increase (J m$^{-3}$)'

    if dversion == dversions[0]:
        figure_name = f'fig1 global increase'
    else:
        figure_name = f'figE2 global increase {dversion[:6]}'

    cb = fig.colorbar(p_L, cax=cax, orientation='horizontal', extend='both', label=cb_label)
    cb1 = fig.colorbar(p_M, cax=cax1, orientation='horizontal', extend='both', label=cb_label1)

    cb.outline.set_linewidth(0.5)
    cb1.outline.set_linewidth(0.5)
    cax.xaxis.set_label_position('top')
    cax1.xaxis.set_label_position('top')

    cb.set_ticks(np.linspace(vmin_L, vmax_L, 9))
    cb1.set_ticks(np.linspace(vmin_M, vmax_M, 9))
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    cb1.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


ddir = '/path/to'

dversions = ['twosat_i8192', 'allsat_i8192']
varnames = ['KE_L', 'KE_M']

ftversion = 'mwt_d184'


degree = 'deg1'


tmversion = 'global_monthly'

dsuffix = 'monthly_eemd'


ntimes = ['2000-10-01', '2024-10-01']

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
plot_axs(dversions[0])
plot_axs(dversions[1])
