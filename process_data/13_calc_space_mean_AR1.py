"""
Purpose: Generate AR(1) red-noise trend envelopes for regional mean time series.
Main inputs: Regional mean monthly KE time series.
Main outputs: Regional AR(1) trend-envelope NetCDF files.
Notes: EEMD trend estimation is implemented with the pyeemd package; update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

import pyeemd

from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore', message='Sending large graph of size', category=UserWarning,)


def cal_eemd_trend(inp):
    IMFs = pyeemd.eemd(inp, ensemble_size=400, noise_strength=0.2, S_number=0, num_siftings=10, rng_seed=0)
    return IMFs[-1, :]


def cal_AR1_eemd(inp, time, ntime, n_samples):

    autocorr = pd.Series(inp).autocorr(lag=1)

    sigma_z = np.sqrt(np.var(inp) * (1 - autocorr**2))

    np.random.seed(0)
    w = np.random.randn(ntime, n_samples)

    AR1_samples = np.zeros((ntime, n_samples))

    for t in range(1, ntime):
        AR1_samples[t, :] = autocorr * AR1_samples[t - 1, :] + sigma_z * w[t, :]

    ds_AR1 = xr.Dataset(
        data_vars=dict(AR1=(('time', 'samples'), AR1_samples)),
        coords=dict(samples=np.arange(n_samples), time=time),
    )

    # AR1_samples_MS = ds_AR1['AR1'].resample(time='MS').mean(dim=['time'], skipna=True).chunk({'time': -1, 'samples': 80})  # for daily input
    AR1_samples_MS = ds_AR1['AR1'].chunk({'time': -1, 'samples': 80})  # for monthly input

    IMFm_MS = xr.apply_ufunc(
        cal_eemd_trend,
        AR1_samples_MS,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        output_dtypes=[np.float64],
        vectorize=True,
        dask='parallelized',
    )

    IMFm_MS = IMFm_MS.transpose('time', 'samples')

    IMFm_MS = IMFm_MS.compute()

    return IMFm_MS, autocorr


def main():
    client = Client(n_workers=64, threads_per_worker=1)

    dversion = 'twosat_i8192'
    # dversion = 'allsat_i8192'

    ftversion = 'mwt_d184'
    # ftversion = 'mwt_d369'
    # ftversion = 'mwt_d92'

    # dversion = 'twosat_daily'
    # ftversion = 'bw6_d184'

    # tmversion = 'space_mean_daily'
    tmversion = 'space_mean_monthly'
    # tmversion = 'space_mean_monthly_deg025'

    varnames = ['KE_L', 'KE_M']

    ddir = f'/path/to/{dversion}/{ftversion}/{tmversion}'

    rgnames = ['mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', 'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', 'mask_global_EQ10']

    varname, rgname = varnames[0], rgnames[0]
    da = xr.open_dataset(f'{ddir}/{rgname}_{varname}.nc')[varname]
    time = da['time'].values
    ntime = len(time)

    # time_MS = da['time'].resample(time='MS').mean(dim=['time'], skipna=True).values  # for daily input
    time_MS = time  # for monthly input

    n_samples = 5000

    tasks = [(varname, rgname) for rgname in rgnames for varname in varnames]

    # inp_list = [xr.open_dataset(f'{ddir}/{rgname}_{varname}.nc')[varname].values for (varname, rgname) in tasks]

    # for deIMF1
    def load_Rrest_mode1(ddir, rgname, varname):
        with xr.open_dataset(f'{ddir}/{rgname}_{varname}_monthly_eemd.nc') as ds:
            Rrest = (ds['IMF'].isel(mode=slice(1, None)).isel(mode=slice(None, None, -1)).cumsum('mode').isel(mode=slice(None, None, -1)))
            return Rrest.isel(mode=1).values
    inp_list = [load_Rrest_mode1(ddir, rgname, varname) for (varname, rgname) in tasks]

    for idx, inp in tqdm(enumerate(inp_list)):
        IMFm_MS, autocorr = cal_AR1_eemd(inp, time, ntime, n_samples)
        varname, rgname = tasks[idx]

        ds_out = xr.Dataset(
            data_vars=dict(IMFm=(('time', 'samples'), IMFm_MS.values)),
            coords=dict(samples=np.arange(n_samples), time=time_MS),
        )
        ds_out['IMFm'].attrs['autocorr'] = autocorr
        ds_out['time'].attrs['long_name'] = 'Time'
        encoding = {
            'IMFm': {'dtype': 'float64'},
            'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
        }
        # ds_out.to_netcdf(f'{ddir}/{rgname}_{varname}_monthly_AR1.nc', encoding=encoding)
        ds_out.to_netcdf(f'{ddir}/{rgname}_{varname}_monthly_deIMF1_AR1.nc', encoding=encoding)

    client.close()

if __name__ == '__main__':
    main()
