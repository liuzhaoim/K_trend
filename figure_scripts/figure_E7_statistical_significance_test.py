"""
Extended Data Fig. 6 | Comparison of scale decomposition using spatial and temporal filters.
"""


import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_trends(ax, dversion, varname, rgname, print_text, color, color2):
    '''
    Plot EEMD trend for KE.
    '''

    IMF = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_{dsuffix}.nc')['IMF']
    trend = IMF.isel(mode=-1) - IMF.isel(mode=-1, time=0)


    IMFm = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_monthly_deIMF1_AR1.nc')['IMFm']
    autocorr = IMFm.attrs['autocorr']

    trends = IMFm - IMFm.isel(time=0)
    trends_mean = trends.mean(dim='samples').values
    trends_std = trends.std(dim='samples').values
    confid = 1.645

    trends_2std_max = trends_mean + confid * trends_std
    trends_2std_min = trends_mean - confid * trends_std

    signif = (trend > trends_2std_max) | (trend < trends_2std_min)


    ax.plot(trend['time'], trend, color=colors[4], linewidth=1, zorder=3)

    for m in range(trends.shape[1]):
        ax.plot(trends['time'], trends.isel(samples=m), color=color2, linewidth=0.2, alpha=0.8, zorder=1)

    ax.plot(trends['time'], trends_2std_max, color=color, linewidth=0.8, zorder=2)
    ax.plot(trends['time'], trends_2std_min, color=color, linewidth=0.8, zorder=2)


    def get_inp():
        with xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_monthly_eemd.nc') as ds:
            Rrest = (ds['IMF'].isel(mode=slice(1, None)).isel(mode=slice(None, None, -1)).cumsum('mode').isel(mode=slice(None, None, -1)))
            return Rrest.isel(mode=1).values
    Var_x = np.var(get_inp())

    ax.text(0.0, 1.002, f'{print_text}', color='black', ha='left', va='bottom', transform=ax.transAxes)
    ax.text(1.0, 1.002, r'$\sigma^2$' + f' = {Var_x:.2f}, ' + r'$\alpha$' + f' = {autocorr:.4f}', color='black', ha='right', va='bottom', transform=ax.transAxes)

    ax.set_xlim(pd.Timestamp(ntimeA), pd.Timestamp(ntimeB))
    ax.set_xticks(pd.date_range(start=ntimeA, end=ntimeB, freq='6YS'))
    ax.set_xticklabels([])


    tmp = trends_2std_max[-1] * 1.8
    ylim = np.ceil(tmp*10)/10 if tmp < 1 else np.ceil(tmp) + (np.ceil(tmp) % 2)


    return ylim


def plot_axs(varname):

    fig = plt.figure(figsize=(3.503, 2.5))
    fig.patch.set_facecolor('white')
    nrows, ncols = 2, 1
    gs = fig.add_gridspec(nrows, ncols, width_ratios=np.ones(ncols), height_ratios=np.ones(nrows), wspace=0.0, hspace=0.3, left=0.16, right=0.98, bottom=0.13, top=0.93)

    ax = []
    for i in range(nrows * ncols):
        ax.append(fig.add_subplot(gs[i]))

    ylim = plot_trends(ax[0], dversions[0], varname, rgnames[0], print_texts[0], colors[0], colors[2])
    plot_trends(ax[1], dversions[1], varname, rgnames[0], print_texts[1], colors[1], colors[3])
    ax[0].set_ylim(-ylim, ylim)
    ax[0].set_yticks(np.arange(-ylim, ylim*1.1, ylim/2))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_yticks(np.arange(-ylim, ylim*1.1, ylim/2))

    if varname == 'KE_L':
        y_label = r'Trend of $K_L$' + '\n' + r'(J m$^{-3}$)'
        figure_name = f'skip'

        ylim = 1
        ax[0].set_ylim(-ylim, ylim)
        ax[0].set_yticks(np.arange(-ylim, ylim*1.1, ylim/2))
        ax[1].set_ylim(-ylim, ylim)
        ax[1].set_yticks(np.arange(-ylim, ylim*1.1, ylim/2))
    else:
        y_label = r'Trend of $K_M$' + '\n' + r'(J m$^{-3}$)'
        figure_name = f'figE7 statistical significance test KM'

    ax[0].set_ylabel(y_label, labelpad=0)
    ax[1].set_ylabel(y_label, labelpad=0)
    ax[1].set_xlabel('Year')
    ax[1].set_xticklabels(pd.date_range(start=ntimeA, end=ntimeB, freq='6YS').year)

    l_x, r_x, t_y = -0.17, -0.08, 1.002
    ax[0].text(l_x, t_y, 'a', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[0].transAxes)
    ax[1].text(l_x, t_y, 'b', fontsize=8, fontweight='bold', fontstyle='normal', ha='left', va='bottom', transform=ax[1].transAxes)

    fig.savefig(f'./{figure_name.replace(" ", "_")}.pdf', dpi=600, bbox_inches=None, pad_inches=0, facecolor="white", transparent=False)


rgnames = ['mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', 'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', 'mask_global_EQ10']

Paired = sns.color_palette('Paired')
colors = ['black', 'black', '#bebebe', '#bebebe', Paired[5]]

ddir = '/path/to'

print_texts = ['Multi-mission', 'C3S']
dversions = ['allsat_i8192', 'twosat_i8192']
varnames = ['KE_L', 'KE_M']

ftversion = 'mwt_d184'


tmversion = 'space_mean_monthly'

dsuffix = 'monthly_eemd'

ntimeA = '1993-07-01'
ntimeB = '2024-11-01'

plot_axs('KE_M')
