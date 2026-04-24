"""
Purpose: Area-weight and coarsen fields from 0.25 degrees to 1 degree.
Main inputs: Daily, monthly, or long-term fields and valid-region masks.
Main outputs: Coarsened NetCDF files for global analysis.
Notes: Update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr


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


def calc_area_weighted_mean(da, mask_valid):

    lat = da['lat'].values  # (Nlat,)
    lon = da['lon'].values  # (Nlon,)

    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())

    lat_b = np.concatenate(([lat[0] - dlat/2], 0.5*(lat[:-1] + lat[1:]), [lat[-1] + dlat/2]))  # (Nlat+1,)
    lon_b = np.concatenate(([lon[0] - dlon/2], 0.5*(lon[:-1] + lon[1:]), [lon[-1] + dlon/2]))  # (Nlon+1,)

    R = 6371e3
    dmu = np.sin(np.deg2rad(lat_b[1:])) - np.sin(np.deg2rad(lat_b[:-1]))  # (Nlat,)
    dl  = np.deg2rad(lon_b[1:] - lon_b[:-1])  # (Nlon,)

    area_np = (R**2) * dmu[:, None] * dl[None, :]  # (Nlat, Nlon)
    area = xr.DataArray(area_np, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))

    area_valid = area.where(mask_valid)
    weighted_sum = (da * area_valid).coarsen(lat=4, lon=4, boundary='trim').sum(skipna=True)  # from 0.25 degrees to 1 degree
    area_sum = (area_valid).coarsen(lat=4, lon=4, boundary='trim').sum(skipna=True)  # from 0.25 degrees to 1 degree

    return weighted_sum / area_sum


def calc_downsampling(dversion, ftversion, varname):

    da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{varname}_daily_deseason.nc', chunks={'time': 256})[varname]
    # da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{varname}_daily.nc', chunks={'time': 256})[varname]  # only for win3_KE_L (mean window)
    # da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/longterm_mean/{varname}_deg025.nc')[varname]  # only for longterm mean

    # da = da.isel(time=slice(184, -184))  # only for cut1year
    # print(da['time'].values[0], da['time'].values[-1])

    da = da.resample(time='MS').mean(dim=['time'], skipna=True)  # global_monthly

    mask_valid = xr.open_dataset(f'{ddir}/mask/mask_valid_twoall_deg025.nc')['mask_valid']
    da = da.where(mask_valid)

    # da_deg1 = da.coarsen(lat=4, lon=4, boundary='trim').mean(skipna=True)  # from 0.25 degrees to 1 degree
    da_deg1 = calc_area_weighted_mean(da, mask_valid)

    v = da_deg1.compute()

    # fname = f'{ddir}/{dversion}/{ftversion}/global_daily/deg1_{varname}.nc'
    fname = f'{ddir}/{dversion}/{ftversion}/global_monthly/deg1_{varname}.nc'  # global_monthly
    write_nc_TLL(fname, varname, v.values, v['lon'].values, v['lat'].values, v['time'].values)

    # write_nc_vLL(f'{ddir}/{dversion}/{ftversion}/longterm_mean/{varname}.nc', varname, v.values, v['lon'].values, v['lat'].values)  # only for longterm mean


ddir = '/path/to'


# %%

ftversion = 'mwt_d184'
# ftversion = 'mwt_d369'
# ftversion = 'mwt_d92'
# ftversion = 'bw6_d184'

dversion = 'twosat_i8192'
for varname in ['KE_L', 'KE_M']:
    calc_downsampling(dversion, ftversion, varname)

dversion = 'allsat_i8192'
for varname in ['KE_L', 'KE_M']:
    calc_downsampling(dversion, ftversion, varname)
