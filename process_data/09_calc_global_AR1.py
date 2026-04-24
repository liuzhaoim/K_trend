"""
Purpose: Generate AR(1) red-noise ensembles for gridded time series.
Main inputs: Monthly gridded KE fields and EEMD settings.
Main outputs: Latitude-wise AR(1) trend-envelope NetCDF files.
Notes: EEMD trend estimation is implemented with the pyeemd package; update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm

import pyeemd

# from multiprocessing import Pool
from dask.distributed import Client


def get_mask_select(lon, lat):
    Lon, Lat = np.meshgrid(lon, lat)
    lonlat = [130, 180, 25, 45]
    mask_Kuroshio = (Lon > lonlat[0]) & (Lon < lonlat[1]) & (Lat > lonlat[2]) & (Lat < lonlat[3])
    lonlat = [278, 327, 27, 53]
    mask_GulfStream = (Lon > lonlat[0]) & (Lon < lonlat[1]) & (Lat > lonlat[2]) & (Lat < lonlat[3])
    mask_select = mask_Kuroshio | mask_GulfStream
    return mask_select


def cal_eemd_trend(inp):
    IMFs = pyeemd.eemd(inp, ensemble_size=400, noise_strength=0.2, S_number=0, num_siftings=10, rng_seed=0)
    return IMFs[-1, :]


def cal_AR1_trends_std(inp):

    autocorr = pd.Series(inp.values).autocorr(lag=1)

    # if np.isnan(autocorr):
    #     return np.full(ntime, np.nan), np.full(ntime, np.nan)

    sigma_z = np.sqrt(np.var(inp.values) * (1 - autocorr**2))

    np.random.seed(0)
    w = np.random.randn(ntime, n_samples)

    AR1_samples = np.zeros((ntime, n_samples))

    for t in range(1, ntime):
        AR1_samples[t, :] = autocorr * AR1_samples[t - 1, :] + sigma_z * w[t, :]

    ds_AR1 = xr.Dataset(
        data_vars=dict(AR1=(('time', 'samples'), AR1_samples)),
        coords=dict(samples=np.arange(n_samples), time=time),
    )

    ## ----------- apply_ufunc ----------------

    # AR1_samples_MS = ds_AR1['AR1'].resample(time='MS').mean(dim=['time'], skipna=True).chunk({'time': -1, 'samples': 9})  # for input global_daily
    AR1_samples_MS = ds_AR1['AR1'].chunk({'time': -1, 'samples': 8})  # for input global_monthly

    IMFm_MS = xr.apply_ufunc(
        cal_eemd_trend,
        AR1_samples_MS,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        output_dtypes=[np.float64],
        # output_dtypes=[AR1_samples_MS.dtype],
        vectorize=True,
        dask='parallelized',
    )

    IMFm_MS = IMFm_MS.transpose('time', 'samples')

    IMFm_MS = IMFm_MS.compute()

    trends = IMFm_MS - IMFm_MS.isel(time=0)

    trends_mean0 = trends.mean(dim='samples').values
    trends_std0 = trends.std(dim='samples').values

    ## ----------- pool ----------------

    # AR1_samples_MS = ds_AR1['AR1'].resample(time='MS').mean(dim=['time'], skipna=True)

    # IMFm_MS = np.zeros((ntime, n_samples))

    # # for k in range(n_samples):
    # #     IMFm_MS[:, k] = cal_eemd_trend(AR1_samples_MS.isel(samples=k).values)

    # if __name__ == '__main__':
    #     with Pool(processes=64) as pool:
    #         results = list(pool.imap(cal_eemd_trend, [AR1_samples_MS.isel(samples=k).values for k in range(n_samples)]))
    #     for k, result in enumerate(results):
    #         IMFm_MS[:, k] = result

    # trends = IMFm_MS - IMFm_MS[0, :]
    # trends_mean0 = np.mean(trends, axis=1)
    # trends_std0 = np.std(trends, axis=1)

    return trends_mean0, trends_std0


def main():

    client = Client(n_workers=64, threads_per_worker=1)

    dversion = 'twosat_i8192'
    # dversion = 'allsat_i8192'

    ftversion = 'mwt_d184'
    # ftversion = 'mwt_d184_win3'

    varname = 'KE_L'
    # varname = 'KE_M'
    # varname = 'KE_S'

    degree = 'deg1'
    # degree = 'deg025'
    # degree = 'deg1Rdeg5'

    # tmversion = 'global_daily'
    tmversion = 'global_monthly'

    ddir = f'/path/to/{dversion}/{ftversion}/{tmversion}'
    # ddir = f'.'  # scnet

    da = xr.open_dataset(f'{ddir}/{degree}_{varname}.nc')[varname]

    global lon, lat, time, nlon, nlat, ntime, n_samples

    lon = da['lon'].values
    lat = da['lat'].values
    time = da['time'].values
    nlon = len(lon)
    nlat = len(lat)
    ntime = len(time)

    # # for input global_daily
    # time = da.isel(lon=0, lat=0).resample(time='MS').mean(dim=['time'], skipna=True)['time'].values
    # ntime = len(time)

    n_samples = 500

    trends_mean = np.full((ntime, nlat, nlon), np.nan)
    trends_std = np.full((ntime, nlat, nlon), np.nan)

    valid2d = da.isel(time=0).notnull().values
    # valid2d = da.isel(time=0).where(get_mask_select(lon, lat)).notnull().values

    prev_file = None

    # for j in range(nlat):  # All latitudes (-89.5, 89.5) nlat=180
    # for j in range(455, 580):  # deg025 nlat=720

    # for j in range(81):  # South: lat = (-89.5, -9.5, 1.0)
    # for j in range(99, nlat):  # North: lat = (9.5, 89.5, 1.0)
    # for j in list(range(81)) + list(range(99, nlat)):

    # for j in range(68):  # South: lat = (-89.5, -23.5, 1.0)
    # for j in range(68, nlat):  # North: lat = (-23.5, 89.5, 1.0)
    for j in list(range(68)) + list(range(68, nlat)):

        ii = np.where(valid2d[j, :])[0]
        for i in tqdm(ii, desc=f'lat {j:03d}', total=ii.size):
        # for i in tqdm(range(nlon), desc=f'lat {j:03d}-{nlat-1:03d}'):  # All longitudes (0.5, 359.5) nlon=360
            inp = da.isel(lon=i, lat=j)
            tm, ts = cal_AR1_trends_std(inp)
            trends_mean[:, j, i], trends_std[:, j, i] = tm, ts

    # # tasks = [(j, i) for j in range(nlat) for i in range(nlon)]  # All latitudes (-89.5, 89.5) nlat=180
    # # tasks = [(j, i) for j in range(81) for i in range(nlon)]  # South: lat = (-89.5, -9.5, 1.0)
    # tasks = [(j, i) for j in range(99, nlat) for i in range(nlon)]  # North: lat = (9.5, 89.5, 1.0)
    # inp_list = [da.isel(lon=i, lat=j) for (j, i) in tasks]

    # for (j, i), inp in tqdm(zip(tasks, inp_list), total=len(tasks)):
    #     tm, ts = cal_AR1_trends_std(inp)
    #     trends_mean[:, j, i], trends_std[:, j, i] = tm, ts

    # # if __name__ == '__main__':
    # #     with Pool(processes=64) as pool:
    # #         results = list(pool.imap(cal_AR1_trends_std, inp_list))
    # #     for (j, i), (tm, ts) in zip(tasks, results):
    # #         trends_mean[:, j, i], trends_std[:, j, i] = tm, ts

        ds_out = xr.Dataset(
                data_vars=dict(
                    trends_mean=(['time', 'lat', 'lon'], trends_mean),
                    trends_std=(['time', 'lat', 'lon'], trends_std),
                ),
                coords=dict(
                    lon=(['lon'], lon),
                    lat=(['lat'], lat),
                    time=(['time'], time),
                ),
        )
        ds_out['lon'].attrs['units'] = 'degrees_east'
        ds_out['lon'].attrs['long_name'] = 'Longitude'
        ds_out['lat'].attrs['units'] = 'degrees_north'
        ds_out['lat'].attrs['long_name'] = 'Latitude'
        ds_out['time'].attrs['long_name'] = 'Time'
        encoding = {
            'trends_mean': {'dtype': 'float64'},
            'trends_std': {'dtype': 'float64'},
            'lon': {'dtype': 'float64'},
            'lat': {'dtype': 'float64'},
            'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
        }

        # fname = f'{ddir}/AR1/{degree}_{varname}_monthly_AR1.nc'
        fname = f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_lat{j:03d}.nc'
        # fname = f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_Kuroshio_lat{j:03d}.nc'
        ds_out.to_netcdf(fname, encoding=encoding)

        if prev_file:
            os.remove(prev_file)
        prev_file = fname

    client.close()

if __name__ == '__main__':
    main()
