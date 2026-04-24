"""
Purpose: Extract yearly ugos, vgos, and adt fields from daily satellite altimetry products.
Main inputs: Raw DUACS multi-mission or C3S altimeter files from CMEMS and the selected dversion.
Main outputs: Yearly NetCDF files in {dversion}_ori/.
Notes: Update placeholder paths and product/version switches before running.
"""

def write_nc_TLL64(fname, varname, v, lon, lat, time):
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
        varname: {'dtype': 'float64'},
        'lon': {'dtype': 'float32'},
        'lat': {'dtype': 'float32'},
        'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
    }
    ds_out.to_netcdf(f'{fname}', encoding=encoding)


import xarray as xr
from tqdm import tqdm

# dversion = 'allsat'
dversion = 'twosat'

# ddir_in = '/path/to/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D'
ddir_in = '/path/to/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D'

ddir_out = f'/path/to/{dversion}_ori'

varname0 = 'ugos'
varname1 = 'vgos'
varname2 = 'adt'

for year in tqdm(range(1993, 2026)):

    ds = xr.open_mfdataset(f'{ddir_in}/{year}/dt_global_{dversion}_phy_l4_*.nc')

    data0 = ds[varname0].compute()
    data1 = ds[varname1].compute()
    data2 = ds[varname2].compute()

    write_nc_TLL64(f'{ddir_out}/{varname0}_{year}.nc', varname0, data0.values, data0['longitude'].values, data0['latitude'].values, data0['time'].values)
    write_nc_TLL64(f'{ddir_out}/{varname1}_{year}.nc', varname1, data1.values, data1['longitude'].values, data1['latitude'].values, data1['time'].values)
    write_nc_TLL64(f'{ddir_out}/{varname2}_{year}.nc', varname2, data2.values, data2['longitude'].values, data2['latitude'].values, data2['time'].values)
