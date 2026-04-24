"""
Purpose: Calculate large-scale, mesoscale, and residual kinetic energy.
Main inputs: Filtered and unfiltered geostrophic velocity components.
Main outputs: KE_L, KE_M, and KE_R NetCDF files.
Notes: Update placeholder paths and product/version switches before running.
"""

import xarray as xr
import gc
from tqdm import tqdm


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


def calc_KE(ftversion):

    ddir = '/path/to'

    dversion = 'twosat_daily'

    varname0 = 'ugos'
    varname1 = 'vgos'

    u_L = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{varname0}.nc')[varname0]
    v_L = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{varname1}.nc')[varname1]

    ugos = xr.open_dataset(f'{ddir}/{dversion}/input/{varname0}.nc')[varname0]
    vgos = xr.open_dataset(f'{ddir}/{dversion}/input/{varname1}.nc')[varname1]

    time = ugos['time'].values
    lat = ugos['lat'].values
    lon = ugos['lon'].values

    u_M = ugos - u_L
    v_M = vgos - v_L


    KE_L = 0.5 * (u_L**2 + v_L**2)
    write_nc_TLL(f'{ddir}/{dversion}/{ftversion}/KE_L.nc', 'KE_L', KE_L.values, lon, lat, time)
    print('KE_L done')
    del KE_L
    gc.collect()

    KE_M = 0.5 * (u_M**2 + v_M**2)
    write_nc_TLL(f'{ddir}/{dversion}/{ftversion}/KE_M.nc', 'KE_M', KE_M.values, lon, lat, time)
    print('KE_M done')
    del KE_M
    gc.collect()

    KE_R = u_L * u_M + v_L * v_M
    write_nc_TLL(f'{ddir}/{dversion}/{ftversion}/KE_R.nc', 'KE_R', KE_R.values, lon, lat, time)
    print('KE_R done')
    del KE_R
    gc.collect()


# ftversions = ['bw6_d128', 'bw6_d256', 'bw6_d300', 'bw6_d360', 'bw6_d512', 'db8_d128', 'db8_d256', 'db8_d512']
ftversions = ['bw6_d184']

for ftversion in tqdm(ftversions):
    calc_KE(ftversion)
