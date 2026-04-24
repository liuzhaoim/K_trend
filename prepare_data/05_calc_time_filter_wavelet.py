"""
Purpose: Apply the wavelet low-pass time filter.
Main inputs: Input fields for the selected product and variable.
Main outputs: Filtered fields under the selected filter directory.
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
import pywt

def fix_length(arr_lp, N):
    '''
    arr_lp : wavelet low-pass numpy array with dimensions (time, lat, lon)
    N      : original time length
    return : array trimmed or padded to the original length
    '''
    nt = arr_lp.shape[0]
    if nt == N:
        return arr_lp
    elif nt < N:
        pad = arr_lp[-1:,...].repeat(N - nt, axis=0)
        return np.concatenate([arr_lp, pad], axis=0)
    else:
        return arr_lp[:N,...]


def calc_lowpass_filter(da):
    coeffs = pywt.wavedec(da, wavelet=w, mode=m, level=l, axis=0)
    coeffs_lp = [coeffs[0]] + [np.zeros_like(cD) for cD in coeffs[1:]]
    da_lp = pywt.waverec(coeffs_lp, wavelet=w, mode=m, axis=0)
    return da_lp


m = 'symmetric'

l = 6                # 2^(6+1) * 1 = 128 d
dc = 'd128'
# l = 7                # 2^(7+1) * 1 = 256 d
# dc = 'd256'

ftname = 'db8'
w = pywt.Wavelet(ftname)


def calc_filter(varname):

    da = xr.open_dataset(f'{ddir}/{dversion}/input/{varname}.nc')[varname]

    time = da['time'].values
    lat = da['lat'].values
    lon = da['lon'].values

    arr = da.values   # shape (N, lat, lon)
    N = arr.shape[0]

    da_low = calc_lowpass_filter(arr)
    da_low = fix_length(da_low, N)

    fname = f'{ddir}/{dversion}/{ftname}_{dc}/{varname}.nc'

    write_nc_TLL(fname, varname, da_low, lon, lat, time)


ddir = '/path/to'

dversion = 'twosat_daily'

# varnames = ['ugos', 'vgos', 'adt']
varnames = ['ugos', 'vgos']

for varname in varnames:
    calc_filter(varname)
