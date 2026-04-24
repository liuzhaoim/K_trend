"""
Purpose: Apply masks, convert longitude coordinates to 0-360 degrees, and merge yearly fields.
Main inputs: Original yearly fields and the combined original-data mask.
Main outputs: Masked daily input files under {dversion}_daily/input/.
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


def lonflip(varin, lon, option):
    '''
    Convert longitude and rearrange variable accordingly.

    Parameters:
    varin (np.ndarray): Input data array with dimensions [time, lat, lon].
    lon (array-like): Input longitude array.
    option (int): If 1, convert to -180 to 180; else, convert to 0 to 360.

    Returns:
    varout (np.ndarray): Rearranged variable based on longitude.
    lon_flip (np.ndarray): Flipped longitude array.

    Updated by Zhao Liu at 20241004
    '''
    # Remove any singleton dimensions
    lon = np.squeeze(lon)

    if option == 1:
        # Convert to -180 to 180
        ind1 = lon > 180
        ind2 = lon <= 180
        varout = np.concatenate((varin[:, :, ind1], varin[:, :, ind2]), axis=2)
        lon_flip = np.concatenate((lon[ind1] - 360, lon[ind2]))
    else:
        # Convert to 0 to 360
        ind1 = lon >= 0
        ind2 = lon < 0
        varout = np.concatenate((varin[:, :, ind1], varin[:, :, ind2]), axis=2)
        lon_flip = np.concatenate((lon[ind1], lon[ind2] + 360))

    return varout, lon_flip


import numpy as np
import xarray as xr
from tqdm import tqdm

# dversion = 'allsat'
dversion = 'twosat'
ddir = '/path/to'

varname = 'ugos'
# varname = 'vgos'
# varname = 'adt'

mask = xr.open_dataset(f'{ddir}/mask/mask_ori/mask_{dversion}.nc')['mask']

for year in tqdm(range(1993, 2026)):
    da = xr.open_dataset(f'{ddir}/{dversion}_ori/{varname}_{year}.nc')[varname].where(~mask)

    varout, lon_flip = lonflip(da, da['lon'], 0)

    write_nc_TLL(f'{ddir}/{dversion}_daily/input/{varname}_{year}.nc', varname, varout, lon_flip, da['lat'].values, da['time'].values)


# %% merge_all_years

import xarray as xr

# dversion = 'allsat'
dversion = 'twosat'
ddir = '/path/to'

varname = 'ugos'
# varname = 'vgos'
# varname = 'adt'

da = xr.open_mfdataset(f'{ddir}/{dversion}_daily/input/{varname}_*.nc')[varname].compute()

write_nc_TLL(f'{ddir}/{dversion}_daily/input/{varname}.nc', varname, da.values, da['lon'].values, da['lat'].values, da['time'].values)
