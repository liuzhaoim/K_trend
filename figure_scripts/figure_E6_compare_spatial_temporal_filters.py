"""
Extended Data Fig. 6 | Comparison of scale decomposition using spatial and temporal filters.
"""


import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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


proj_crs = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree()

def plot_crs(ax, gl_l=True, gl_b=True):

    ax.set_extent(lonlat, crs=data_crs)
    ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor=[0.7, 0.7, 0.7], linewidth=0.3, zorder=4)

    if gl_b:
        ax.set_xticks(np.linspace(lonlat[0]+2, lonlat[1]-2, 4), crs=data_crs)
        ax.xaxis.set_major_formatter(LongitudeFormatter())


    if gl_l:
        ax.set_yticks(np.linspace(lonlat[2], lonlat[3], 3), crs=data_crs)
        ax.yaxis.set_major_formatter(LatitudeFormatter())


    ax.spines['geo'].set_linewidth(0.5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_zorder(20)


def plot_axs():

    fig = plt.figure(figsize=(6.0, 6.692))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2], wspace=0.05, hspace=0.08, left=0.08, right=0.99, bottom=0.0, top=0.98)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0:2], projection=proj_crs))
    ax.append(fig.add_subplot(gs[0, 2:4], projection=proj_crs))
    ax.append(fig.add_subplot(gs[1, 0:2], projection=proj_crs))
    ax.append(fig.add_subplot(gs[1, 2:4], projection=proj_crs))
    ax.append(fig.add_subplot(gs[2, 0:2], projection=proj_crs))
    ax.append(fig.add_subplot(gs[2, 2:4], projection=proj_crs))
    ax.append(fig.add_subplot(gs[3, 1:3], projection=proj_crs))

    plot_crs(ax[0], gl_b=False)
    plot_crs(ax[1], gl_l=False, gl_b=False)
    plot_crs(ax[2], gl_b=False)
    plot_crs(ax[3], gl_l=False, gl_b=False)
    plot_crs(ax[4])
    plot_crs(ax[5], gl_l=False)
    plot_crs(ax[6])

    l_x, r_x, t_y = -0.2, -0.02, 1.04
    ax[0].text(l_x, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[0].transAxes)
    ax[1].text(r_x, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[1].transAxes)
    ax[2].text(l_x, t_y, 'c', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[2].transAxes)
    ax[3].text(r_x, t_y, 'd', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[3].transAxes)
    ax[4].text(l_x, t_y, 'e', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[4].transAxes)
    ax[5].text(r_x, t_y, 'f', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[5].transAxes)
    ax[6].text(l_x, t_y, 'g', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[6].transAxes)

    [ax[i].text(1.05, 1.01, '(m)', color='black', ha='center', va='bottom', transform=ax[i].transAxes) for i in range(7)]


    ddir = '/path/to'

    shrk = 0.7
    pad = 0.03

    mask_dailynan_twosat = xr.open_dataset(f'{ddir}/mask/mask_dailynan_twosat_deg025.nc')['mask_dailynan'].sel(lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1])).values


    ftversion = '200km'

    adt = xr.open_dataset(f'{ddir}/twosat_daily/coarse_graining/adt_demo.nc')['adt']
    adt = adt.isel(depth=0).assign_coords(time=pd.to_datetime(adt['time'].values, origin='1993-01-01', unit='D')).rename({'latitude': 'lat', 'longitude': 'lon'})
    adt_L = xr.open_dataset(f'{ddir}/twosat_daily/coarse_graining/adt_filter_{ftversion}.nc')['adt']
    adt_L = adt_L.isel(depth=0).assign_coords(time=pd.to_datetime(adt_L['time'].values, origin='1993-01-01', unit='D')).rename({'latitude': 'lat', 'longitude': 'lon'})

    adt = adt.sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1])).where(~mask_dailynan_twosat)
    adt_L = adt_L.sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1])).where(~mask_dailynan_twosat)
    adt_M = adt - adt_L


    adt_L.plot(ax=ax[0], vmin=vmin_L, vmax=vmax_L, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_L, vmax_L, 3)}, transform=data_crs)
    adt_M.plot(ax=ax[1], vmin=vmin_M, vmax=vmax_M, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_M, vmax_M, 3)}, transform=data_crs)


    adt_L.plot.contour(ax=ax[0], levels=np.arange(vmin_L, vmax_L, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)
    adt_M.plot.contour(ax=ax[1], levels=np.arange(vmin_M, vmax_M, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)

    ax[0].text(0.5, 1.01, f'Large-scale SSH (> {ftversion[:3]} km)', ha='center', va='bottom', transform=ax[0].transAxes)
    ax[1].text(0.5, 1.01, f'Mesoscale SSH (< {ftversion[:3]} km)', ha='center', va='bottom', transform=ax[1].transAxes)


    ftversion = '500km'

    adt = xr.open_dataset(f'{ddir}/twosat_daily/coarse_graining/adt_demo.nc')['adt']
    adt = adt.isel(depth=0).assign_coords(time=pd.to_datetime(adt['time'].values, origin='1993-01-01', unit='D')).rename({'latitude': 'lat', 'longitude': 'lon'})
    adt_L = xr.open_dataset(f'{ddir}/twosat_daily/coarse_graining/adt_filter_{ftversion}.nc')['adt']
    adt_L = adt_L.isel(depth=0).assign_coords(time=pd.to_datetime(adt_L['time'].values, origin='1993-01-01', unit='D')).rename({'latitude': 'lat', 'longitude': 'lon'})

    adt = adt.sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1])).where(~mask_dailynan_twosat)
    adt_L = adt_L.sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1])).where(~mask_dailynan_twosat)
    adt_M = adt - adt_L


    adt_L.plot(ax=ax[2], vmin=vmin_L, vmax=vmax_L, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_L, vmax_L, 3)}, transform=data_crs)
    adt_M.plot(ax=ax[3], vmin=vmin_M, vmax=vmax_M, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_M, vmax_M, 3)}, transform=data_crs)


    adt_L.plot.contour(ax=ax[2], levels=np.arange(vmin_L, vmax_L, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)
    adt_M.plot.contour(ax=ax[3], levels=np.arange(vmin_M, vmax_M, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)

    ax[2].text(0.5, 1.01, f'Large-scale SSH (> {ftversion[:3]} km)', ha='center', va='bottom', transform=ax[2].transAxes)
    ax[3].text(0.5, 1.01, f'Mesoscale SSH (< {ftversion[:3]} km)', ha='center', va='bottom', transform=ax[3].transAxes)


    ftversion = 'mwt_d184'


    adt_L = xr.open_dataset(f'{ddir}/twosat_i8192/{ftversion}/ssh_L_daily.nc')['ssh_L'].sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1]))
    adt_M = xr.open_dataset(f'{ddir}/twosat_i8192/{ftversion}/ssh_M_daily.nc')['ssh_M'].sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1]))


    adt_L.plot(ax=ax[4], vmin=vmin_L, vmax=vmax_L, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_L, vmax_L, 3)}, transform=data_crs)
    adt_M.plot(ax=ax[5], vmin=vmin_M, vmax=vmax_M, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk, 'pad': pad, 'ticks': np.linspace(vmin_M, vmax_M, 3)}, transform=data_crs)


    adt_L.plot.contour(ax=ax[4], levels=np.arange(vmin_L, vmax_L, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)
    adt_M.plot.contour(ax=ax[5], levels=np.arange(vmin_M, vmax_M, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)

    ax[4].text(0.5, 1.01, f'Large-scale SSH (> {ftversion[-3:]} day)', ha='center', va='bottom', transform=ax[4].transAxes)
    ax[5].text(0.5, 1.01, f'Mesoscale SSH (< {ftversion[-3:]} day)', ha='center', va='bottom', transform=ax[5].transAxes)


    adt = xr.open_dataset(f'{ddir}/twosat_daily/input/adt.nc')['adt'].sel(time=ntime, lat=slice(lonlat[2], lonlat[3]), lon=slice(lonlat[0], lonlat[1]))
    adt.plot(ax=ax[6], vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'label': f'', 'shrink':shrk/1.2, 'pad': pad, 'ticks': np.linspace(vmin, vmax, 3)}, transform=data_crs)
    adt.plot.contour(ax=ax[6], levels=np.arange(vmin, vmax, 0.1), colors='black', linewidths=0.5, transform=data_crs, zorder=1)
    ax[6].text(0.5, 1.01, 'Original SSH', ha='center', va='bottom', transform=ax[6].transAxes)


    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [axi.set_title('') for axi in ax]

    figure_name = f'figE6 compare spatial temporal filters'

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


ntime = '2020-06-01'

lonlat = [278, 312, 25, 45]
vmin, vmax = -0.6, 1.4
vmin_L, vmax_L = -0.6, 1.4
vmin_M, vmax_M = -0.6, 0.6
step, qscale = 2, 15


cmap = 'RdBu_r'


plot_axs()
