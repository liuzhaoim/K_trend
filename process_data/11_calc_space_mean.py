"""
Purpose: Calculate area-weighted regional mean KE time series.
Main inputs: Coarsened gridded fields and region masks.
Main outputs: Regional mean NetCDF time series.
Notes: Update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
from tqdm import tqdm


# %%
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
    weighted_sum = (da * area_valid).sum(dim=['lat', 'lon'], skipna=True)
    area_sum = (area_valid).sum(dim=['lat', 'lon'], skipna=True)

    return weighted_sum / area_sum

# %%

def calc_time_series(args):

    ddir, dversion, ftversion, tmversion, tmversion2, varname, rgname = args

    da0 = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}.nc', chunks={'time': 12})[varname]

    mask_region = xr.open_dataset(f'{ddir}/mask/KEmasked_regions_{degree}.nc')[rgname]
    da0 = da0.where(mask_region)

    # da = da0.where(mask_region).mean(dim=['lon', 'lat'], skipna=True)
    da = calc_area_weighted_mean(da0, mask_region)

    v = da.compute()

    ds_out = xr.Dataset(
        data_vars={varname: (['time'], v.values)},
        coords=dict(
            time=(['time'], v['time'].values),
        ),
        attrs=dict(description=f'time series of regional averaged ({rgname}) {varname}'),
    )
    ds_out[varname].attrs['units'] = 'J/m^3'
    ds_out[varname].attrs['long_name'] = f'regional averaged ({rgname}) {varname}'
    ds_out['time'].attrs['long_name'] = 'Time'
    encoding = {
        varname: {'dtype': 'float64', '_FillValue': 9.96921e+36, 'missing_value': 9.96921e+36},
        'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
    }
    ds_out.to_netcdf(f'{ddir}/{dversion}/{ftversion}/{tmversion2}/{rgname}_{varname}.nc', encoding=encoding)


ddir = '/path/to'

dversions = ['allsat_i8192', 'twosat_i8192']
ftversion = 'mwt_d184'
# ftversion = 'mwt_d369'
# ftversion = 'mwt_d92'

# dversions = ['twosat_daily']
# ftversion = 'bw6_d184'

degree = 'deg1'
# degree = 'deg025'

# tmversion = 'global_daily'
# tmversion2 = 'space_mean_daily'

tmversion = 'global_monthly'
tmversion2 = 'space_mean_monthly'
# tmversion2 = 'space_mean_monthly_deg025'

varnames = ['KE_L', 'KE_M']

rgnames = ['mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', 'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', 'mask_global_EQ10']

tasks = [(ddir, dversion, ftversion, tmversion, tmversion2, varname, rgname) for rgname in rgnames for varname in varnames for dversion in dversions]

for task in tqdm(tasks):
    calc_time_series(task)
