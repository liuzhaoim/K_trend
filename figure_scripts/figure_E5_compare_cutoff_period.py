"""
Extended Data Fig. 5 | Sensitivity of the scale decomposition to the choice of cutoff period.
"""


import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


def _set_four_halfstep_yticks(ax, first_panel=False):
    """Keep ylim, use 4 evenly spaced ticks with minimal padding on a fixed grid."""
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or np.isclose(ymin, ymax):
        return

    span = ymax - ymin
    unit = 0.5 if first_panel else 0.05


    step_mul = int(np.floor((span / 3.0) / unit + 1e-12))
    if step_mul < 1:
        ticks = np.linspace(ymin, ymax, 4)
        ticks = np.round(ticks / unit) * unit
    else:
        step = step_mul * unit
        start_min = np.ceil(ymin / unit) * unit
        start_max = np.floor((ymax - 3.0 * step) / unit) * unit

        if start_min > start_max:
            ticks = np.linspace(ymin, ymax, 4)
            ticks = np.round(ticks / unit) * unit
        else:
            ideal_start = (ymin + ymax - 3.0 * step) / 2.0
            start = np.round(ideal_start / unit) * unit
            start = min(max(start, start_min), start_max)
            ticks = start + step * np.arange(4)

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f' if first_panel else '%.2f'))


def plot_ax(ax, varname, ftversion, print_text, color, color2):


    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_{dsuffix}.nc')['IMF']
    time = pd.to_datetime(IMF['time'].values) + pd.Timedelta(days=15)


    def load_Rrest_mode1(IMF, m):
        Rrest = (IMF.isel(mode=slice(1, None)).isel(mode=slice(None, None, -1)).cumsum('mode').isel(mode=slice(None, None, -1)))
        return Rrest.isel(mode=m).values
    ax.plot(time, load_Rrest_mode1(IMF, 1), color=color, linewidth=0.8, alpha=0.9, zorder=2)

    ax.plot(time, IMF.isel(mode=-1), color=color, linewidth=2, alpha=0.9, zorder=4)

    ax.text(0.5, 1.01, f'{print_text}', color='black', ha='center', va='bottom', transform=ax.transAxes)


    ax.set_xlim(pd.Timestamp(ntimeA), pd.Timestamp(ntimeB))
    ax.set_xticks(pd.date_range(start=ntimeA, end=ntimeB, freq='6YS'))
    ax.set_xticklabels([])


def plot_significant_time(ax, varname, ftversion):
    '''
    Plot significant EEMD trend for KE.
    '''

    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_{dsuffix}.nc')['IMF']
    trend = IMF.isel(mode=-1) - IMF.isel(mode=-1, time=0)


    IMFm = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_monthly_deIMF1_AR1.nc')['IMFm']
    autocorr = IMFm.attrs['autocorr']

    trends = IMFm - IMFm.isel(time=0)
    trends_mean = trends.mean(dim='samples').values
    trends_std = trends.std(dim='samples').values
    confid = 1.645

    signif = (trend > trends_mean + confid * trends_std) | (trend < trends_mean - confid * trends_std)


    time = pd.to_datetime(IMF['time'].values) + pd.Timedelta(days=15)
    time = time[:-1].append(pd.DatetimeIndex([ntimeC]))


    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.fill_between(time, ymin, ymax, where=signif, color='gray', alpha=0.5, zorder=0)


def plot_axs():

    fig = plt.figure(figsize=(7.086, 4.3))
    fig.patch.set_facecolor('white')
    nrows, ncols = 3, 2
    gs = fig.add_gridspec(nrows, ncols, width_ratios=np.ones(ncols), height_ratios=np.ones(nrows), wspace=0.22, hspace=0.25, left=0.08, right=0.98, bottom=0.08, top=0.95)

    ax = []
    for i in range(nrows * ncols):
        ax.append(fig.add_subplot(gs[i]))

    for i in range(6):
        plot_ax(ax[i], varnames[i], ftversions[i], print_texts[i], colors[0], colors[1])
        plot_significant_time(ax[i], varnames[i], ftversions[i])
        _set_four_halfstep_yticks(ax[i], first_panel=(i == 0))


    y_label0 = 'Kinetic energy (J m$^{-3}$)'
    y_label1 = 'Kinetic energy (J m$^{-3}$)'

    ax[0].set_ylabel(y_label0)
    ax[1].set_ylabel(y_label1)
    ax[2].set_ylabel(y_label0)
    ax[3].set_ylabel(y_label1)
    ax[4].set_ylabel(y_label0)
    ax[5].set_ylabel(y_label1)
    ax[4].set_xlabel('Year')
    ax[5].set_xlabel('Year')
    ax[4].set_xticklabels(pd.date_range(start=ntimeA, end=ntimeB, freq='6YS').year)
    ax[5].set_xticklabels(pd.date_range(start=ntimeA, end=ntimeB, freq='6YS').year)

    l_x, r_x, t_y = -0.15, -0.15, 1.01
    ax[0].text(l_x, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[0].transAxes)
    ax[2].text(l_x, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[2].transAxes)
    ax[4].text(l_x, t_y, 'c', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[4].transAxes)
    ax[1].text(r_x, t_y, 'd', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[1].transAxes)
    ax[3].text(r_x, t_y, 'e', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[3].transAxes)
    ax[5].text(r_x, t_y, 'f', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[5].transAxes)

    figure_name = f'figE5 compare cutoff period'


    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


rgnames = ['mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', 'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', 'mask_global_EQ10']

colors = ['black', '#bebebe']

ddir = '/path/to'

dversion = 'twosat_i8192'

varnames = ['KE_L', 'KE_M', 'KE_L', 'KE_M', 'KE_L', 'KE_M']

ftversions = ['mwt_d92', 'mwt_d92', 'mwt_d184', 'mwt_d184', 'mwt_d369', 'mwt_d369']
print_texts = [r'$K_L$ (> 92 day)', r'$K_M$ (< 92 day)', r'$K_L$ (> 184 day)', r'$K_M$ (< 184 day)', r'$K_L$ (> 369 day)', r'$K_M$ (< 369 day)']

tmversion = 'space_mean_monthly'

dsuffix = 'monthly_eemd'

ntimeA = '1993-07-01'
ntimeB = '2024-11-01'
ntimeC = '2024-10-31'

rgname = 'mask_global_EQ10'
rgtext = 'global ocean'

plot_axs()
