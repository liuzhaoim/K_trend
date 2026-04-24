"""
Purpose: Apply a Butterworth low-pass filter for sensitivity tests.
Main inputs: Daily input fields for the selected product and variable.
Main outputs: Butterworth-filtered daily fields.
Notes: Update placeholder paths and product/version switches before running.
"""

def write_nc_TLL(fname, varname, v, lon, lat, time):
    ds_out = xr.Dataset(
            data_vars={varname: (['time', 'lat', 'lon'], v)},
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
        varname: {'dtype': 'float32', '_FillValue': 9.96921e+36, 'missing_value': 9.96921e+36},
        'lon': {'dtype': 'float64'},
        'lat': {'dtype': 'float64'},
        'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
    }
    ds_out.to_netcdf(f'{fname}', encoding=encoding)


import numpy as np
import xarray as xr
from scipy import signal


def calc_lowpass_filter(da):
    sos = signal.butter(N=order, Wn=fc, btype='lowpass', fs=fs, output='sos')
    da_low = signal.sosfiltfilt(sos, da, axis=0)
    return da_low

ftname = 'bw6'
order = 6
fs = 1.0             # the sampling frequency

# # the critical frequency
# dc = 'd300'
# fc = 1/300.0
dc = 'd184'
fc = 1/184.5


def calc_filter(varname):

    da = xr.open_dataset(f'{ddir}/{dversion}/input/{varname}.nc')[varname]

    time = da['time'].values
    lat = da['lat'].values
    lon = da['lon'].values

    da_low = calc_lowpass_filter(da)

    fname = f'{ddir}/{dversion}/{ftname}_{dc}/{varname}.nc'

    write_nc_TLL(fname, varname, da_low, lon, lat, time)


ddir = '/path/to'

dversion = 'twosat_daily'

varnames = ['ugos', 'vgos', 'adt']

for varname in varnames:
    calc_filter(varname)
