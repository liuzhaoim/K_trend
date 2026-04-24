"""
Purpose: Create masks for grid cells with missing geostrophic velocity data.
Main inputs: Yearly ugos and vgos files in {dversion}_ori/.
Main outputs: Yearly masks and a combined mask under mask/mask_ori/.
Notes: Update placeholder paths and product/version switches before running.
"""

import xarray as xr

from tqdm import tqdm
from multiprocessing import Pool


def calc_mask(args):
    year = args

    # dversion = 'allsat'
    dversion = 'twosat'

    varname0 = 'ugos'
    varname1 = 'vgos'

    ddir = '/path/to'

    mask_u = xr.open_dataset(f'{ddir}/{dversion}_ori/{varname0}_{year}.nc')[varname0].isnull().any(dim='time').compute()
    mask_v = xr.open_dataset(f'{ddir}/{dversion}_ori/{varname1}_{year}.nc')[varname1].isnull().any(dim='time').compute()
    mask = mask_u | mask_v

    print(year, mask_u.sum().item(), mask_v.sum().item(), mask.sum().item())

    ds_out = xr.Dataset(
            data_vars=dict(
                mask=(['lat', 'lon'], mask.values),
            ),
            coords=dict(
                lon=(['lon'], mask['lon'].values),
                lat=(['lat'], mask['lat'].values),
            ),
            attrs=dict(description='Presence of NaN values along the time dimension')
    )

    ds_out['lon'].attrs['units'] = 'degrees_east'
    ds_out['lon'].attrs['long_name'] = 'Longitude'
    ds_out['lat'].attrs['units'] = 'degrees_north'
    ds_out['lat'].attrs['long_name'] = 'Latitude'

    ds_out.to_netcdf(f'{ddir}/mask/mask_ori/mask_{dversion}/mask_{dversion}_{year}.nc')

tasks = [(year) for year in range(1993, 2026)]

if __name__ == '__main__':
    with Pool(processes=5) as pool:
        list(tqdm(pool.imap(calc_mask, tasks), total=len(tasks)))


# %% merge_mask_gos_years

import xarray as xr

# dversion = 'allsat'
dversion = 'twosat'
ddir = '/path/to'

mask = None
for year in range(1993, 2026):
    ds_mask = xr.open_dataset(f'{ddir}/mask/mask_ori/mask_{dversion}/mask_{dversion}_{year}.nc')

    if mask is None:
        mask = ds_mask['mask']
    else:
        mask = mask | ds_mask['mask']

    print(year, mask.sum().item())

ds_out = xr.Dataset(
        data_vars=dict(
            mask=(['lat', 'lon'], mask.values),
        ),
        coords=dict(
            lon=(['lon'], mask['lon'].values),
            lat=(['lat'], mask['lat'].values),
        ),
        attrs=dict(description='Presence of NaN values along the time dimension')
)

ds_out['lon'].attrs['units'] = 'degrees_east'
ds_out['lon'].attrs['long_name'] = 'Longitude'
ds_out['lat'].attrs['units'] = 'degrees_north'
ds_out['lat'].attrs['long_name'] = 'Latitude'

ds_out.to_netcdf(f'{ddir}/mask/mask_ori/mask_{dversion}.nc')
