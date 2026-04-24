"""
Purpose: Calculate grid-point standard deviation and lag-1 autocorrelation.
Main inputs: Monthly gridded KE fields.
Main outputs: Autocorrelation and standard-deviation NetCDF files.
Notes: Update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
import pandas as pd

def calc_global_autocorr():

    da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}.nc')[varname]

    # def compute_autocorr(time_series, lag=1):
    #     x = time_series - np.mean(time_series)
    #     N = len(x)
    #     C0 = np.sum(x * x) / N
    #     Ck = np.sum(x[:-lag] * x[lag:]) / (N - lag)
    #     return Ck / C0

    def compute_autocorr(time_series):
        return pd.Series(time_series).autocorr(lag=1)

    autocorr = xr.apply_ufunc(
        compute_autocorr,
        da,
        vectorize=True,
        input_core_dims=[['time']],
        output_core_dims=[[]],
    )

    var_x = np.var(da.values, axis=0)

    sigma_z = np.sqrt(var_x * (1 - autocorr ** 2))

    ds_out = xr.Dataset(
        data_vars=dict(
            autocorr=(('lat', 'lon'), autocorr.values),
            sigma_x=(('lat', 'lon'), np.sqrt(var_x)),
            sigma_z=(('lat', 'lon'), sigma_z.values),
        ),
        coords=dict(
            lon=(['lon'], da['lon'].values),
            lat=(['lat'], da['lat'].values),
        ),
    )
    ds_out['lon'].attrs['units'] = 'degrees_east'
    ds_out['lon'].attrs['long_name'] = 'Longitude'
    ds_out['lat'].attrs['units'] = 'degrees_north'
    ds_out['lat'].attrs['long_name'] = 'Latitude'

    encoding = {
        'autocorr': {'dtype': 'float64'},
        'sigma_x': {'dtype': 'float64'},
        'sigma_z': {'dtype': 'float64'},
        'lon': {'dtype': 'float64'},
        'lat': {'dtype': 'float64'},
    }

    # ds_out.to_netcdf(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_autocorr_formula.nc', encoding=encoding)
    ds_out.to_netcdf(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_autocorr.nc', encoding=encoding)


ddir = '/path/to'
dversion = 'twosat_i8192'
ftversion = 'mwt_d184'
degree = 'deg1'
# degree = 'deg1Rdeg5'

for tmversion in ['global_daily', 'global_monthly']:
    for varname in ['KE_L', 'KE_M']:
        calc_global_autocorr()
