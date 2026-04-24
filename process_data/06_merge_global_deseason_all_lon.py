"""
Purpose: Merge longitude-chunk deseasoned files into global fields.
Main inputs: Chunked deseasoned NetCDF files.
Main outputs: Global daily deseasoned NetCDF files.
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

def calc_merge_deseason(dversion, ftversion, varname):
    da = xr.open_mfdataset(f'{ddir}/{dversion}/{ftversion}/{varname}/{varname}_daily_deseason_lon*of16.nc', combine='by_coords')[varname]

    v = da.compute()

    fname = f'{ddir}/{dversion}/{ftversion}/{varname}_daily_deseason.nc'
    write_nc_TLL(fname, varname, v.values, v['lon'].values, v['lat'].values, v['time'].values)


ddir = '/path/to'


# %%

# ftversion = 'mwt_d184'

# dversion = 'twosat_i8192'
# for varname in ['KE_L', 'KE_M']:
#     calc_merge_deseason(dversion, ftversion, varname)

# dversion = 'allsat_i8192'
# for varname in ['KE_L', 'KE_M']:
#     calc_merge_deseason(dversion, ftversion, varname)


# %%

# ftversion = 'mwt_d369'
ftversion = 'mwt_d92'
# ftversion = 'bw6_d184'

dversion = 'twosat_i8192'
# dversion = 'twosat_daily'

varname = 'KE_L'
# varname = 'KE_M'

calc_merge_deseason(dversion, ftversion, varname)
