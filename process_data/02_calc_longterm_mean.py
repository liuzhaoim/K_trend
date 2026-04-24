"""
Purpose: Calculate long-term mean KE and SSH fields.
Main inputs: Daily KE or SSH fields for the selected product/filter.
Main outputs: Long-term mean NetCDF files.
Notes: Update placeholder paths and product/version switches before running.
"""

import xarray as xr

from dask.distributed import Client

# %%

def write_nc_vLL(fname, varname, v, lon, lat):
    ds_out = xr.Dataset(
            data_vars={varname: (['lat', 'lon'], v)},
            coords=dict(
                lon=(['lon'], lon),
                lat=(['lat'], lat),
            ),
    )
    ds_out['lon'].attrs['units'] = 'degrees_east'
    ds_out['lon'].attrs['long_name'] = 'Longitude'
    ds_out['lat'].attrs['units'] = 'degrees_north'
    ds_out['lat'].attrs['long_name'] = 'Latitude'
    encoding = {
        varname: {'dtype': 'float32', '_FillValue': 9.96921e+36, 'missing_value': 9.96921e+36},
        'lon': {'dtype': 'float64'},
        'lat': {'dtype': 'float64'},
    }
    ds_out.to_netcdf(f'{fname}', encoding=encoding)


def calc_longterm_mean(ddir, dversion, ftversion, varname):
    da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{varname}_daily.nc', chunks={'time': -1, 'lat': -1, 'lon': 90})[varname]
    longterm_mean = da.mean(dim='time', skipna=True)
    v = longterm_mean.compute()
    write_nc_vLL(f'{ddir}/{dversion}/{ftversion}/longterm_mean/{varname}_deg025.nc', varname, v.values, v['lon'].values, v['lat'].values)


def calc_longterm_mean_input(ddir, dversion0, ftversion, varname0, varname1):
    da = xr.open_dataset(f'{ddir}/{dversion0}_daily/input/{varname0}.nc', chunks={'time': -1, 'lat': -1, 'lon': 90})[varname0]

    mask_dailynan_allsat = xr.open_dataset(f'{ddir}/mask/mask_dailynan_allsat_deg025.nc')['mask_dailynan']
    mask_dailynan_twosat = xr.open_dataset(f'{ddir}/mask/mask_dailynan_twosat_deg025.nc')['mask_dailynan']
    mask_EQ10 = xr.open_dataset(f'{ddir}/mask/mask_EQ10_deg025.nc')['mask_EQ10']
    da = da.where(~mask_dailynan_allsat & ~mask_dailynan_twosat & ~mask_EQ10)

    da = da.isel(time=slice(181, -182))

    longterm_mean = da.mean(dim='time', skipna=True)
    v = longterm_mean.compute()
    write_nc_vLL(f'{ddir}/{dversion0}_i8192/{ftversion}/longterm_mean/{varname1}_deg025.nc', varname1, v.values, v['lon'].values, v['lat'].values)


def main():

    client = Client()

    ddir = '/path/to'
    ftversion = 'mwt_d184'

    dversion = 'twosat_i8192'
    for varname in ['KE_L', 'KE_M']:
        calc_longterm_mean(ddir, dversion, ftversion, varname)

    dversion = 'allsat_i8192'
    for varname in ['KE_L', 'KE_M']:
        calc_longterm_mean(ddir, dversion, ftversion, varname)

    dversion = 'twosat_i8192'
    for varname in ['ssh_L', 'ssh_M']:
        calc_longterm_mean(ddir, dversion, ftversion, varname)

    dversion = 'allsat_i8192'
    for varname in ['ssh_L', 'ssh_M']:
        calc_longterm_mean(ddir, dversion, ftversion, varname)


    ddir = '/path/to'
    ftversion = 'mwt_d184'
    varname0 = 'adt'
    varname1 = 'ssh'
    for dversion0 in ['twosat', 'allsat']:
        calc_longterm_mean_input(ddir, dversion0, ftversion, varname0, varname1)

    client.close()

if __name__ == '__main__':
    main()
