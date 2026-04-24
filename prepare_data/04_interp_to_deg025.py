"""
Purpose: Interpolate the all-satellite product to the 0.25-degree grid.
Main inputs: All-satellite 0.125-degree daily input files.
Main outputs: All-satellite 0.25-degree daily input files.
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


import xarray as xr

ddir = '/path/to'

# varname = 'ugos'
# varname = 'vgos'
varname = 'adt'

da = xr.open_dataset(f'{ddir}/allsat_daily/input/{varname}_deg0125.nc')[varname]
da2 = xr.open_dataset(f'{ddir}/twosat_daily/input/{varname}.nc')[varname]

da = da.chunk({'time': 1000})

da_interp = da.interp(lon=da2['lon'].values, lat=da2['lat'].values)

data = da_interp.compute()

write_nc_TLL(f'{ddir}/allsat_daily/input/{varname}.nc', varname, data.values, data['lon'].values, data['lat'].values, data['time'].values)
