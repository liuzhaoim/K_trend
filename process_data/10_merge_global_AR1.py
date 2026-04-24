"""
Purpose: Merge latitude-wise AR(1) files into one global file.
Main inputs: AR(1) NetCDF files split by latitude.
Main outputs: Merged monthly AR(1) NetCDF file.
Notes: Update placeholder paths and product/version switches before running.
"""

import xarray as xr

dversion = 'twosat_i8192'
# dversion = 'allsat_i8192'

ftversion = 'mwt_d184'
# ftversion = 'mwt_d184_win3'

varname = 'KE_L'
# varname = 'KE_M'
# varname = 'KE_S'

degree = 'deg1'
# degree = 'deg025'
# degree = 'deg1Rdeg5'

# tmversion = 'global_daily'
tmversion = 'global_monthly'

ddir = f'/path/to/{dversion}/{ftversion}/{tmversion}'

nlat_S = 67
nlat_N = 179

ds0 = xr.open_dataset(f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_lat{nlat_S:03d}.nc')
ds1 = xr.open_dataset(f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_lat{nlat_N:03d}.nc')
trends_mean = ds0['trends_mean'].combine_first(ds1['trends_mean'])
trends_std = ds0['trends_std'].combine_first(ds1['trends_std'])

ds_out = xr.Dataset(
        data_vars=dict(
            trends_mean=(['time', 'lat', 'lon'], trends_mean.values),
            trends_std=(['time', 'lat', 'lon'], trends_std.values),
        ),
        coords=dict(
            lon=(['lon'], trends_std['lon'].values),
            lat=(['lat'], trends_std['lat'].values),
            time=(['time'], trends_std['time'].values),
        ),
)
ds_out['lon'].attrs['units'] = 'degrees_east'
ds_out['lon'].attrs['long_name'] = 'Longitude'
ds_out['lat'].attrs['units'] = 'degrees_north'
ds_out['lat'].attrs['long_name'] = 'Latitude'
ds_out['time'].attrs['long_name'] = 'Time'
encoding = {
    'trends_mean': {'dtype': 'float64'},
    'trends_std': {'dtype': 'float64'},
    'lon': {'dtype': 'float64'},
    'lat': {'dtype': 'float64'},
    'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
}
ds_out.to_netcdf(f'{ddir}/{degree}_{varname}_monthly_AR1.nc', encoding=encoding)


# # # test merge

# import xarray as xr
# import matplotlib.pyplot as plt

# # dversion = 'twosat_i8192'
# dversion = 'allsat_i8192'

# ftversion = 'mwt_d184'
# # ftversion = 'mwt_d184_win3'

# # varname = 'KE_L'
# varname = 'KE_M'
# # varname = 'KE_S'

# degree = 'deg1'
# # degree = 'deg1Rdeg5'

# # tmversion = 'global_daily'
# tmversion = 'global_monthly'

# ddir = f'/path/to/{dversion}/{ftversion}/{tmversion}'

# nlat_S = 67
# nlat_N = 179

# da0 = xr.open_dataset(f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_lat{nlat_S:03d}.nc')
# da1 = xr.open_dataset(f'{ddir}/AR1/{degree}_{varname}_monthly_AR1_lat{nlat_N:03d}.nc')
# da2 = xr.open_dataset(f'{ddir}/AR1/{degree}_{varname}_monthly_AR1.nc')

# print('max mean diff S', (da2['trends_mean']-da0['trends_mean']).max().values)
# print('max mean diff N', (da2['trends_mean']-da1['trends_mean']).max().values)
# print('max std diff S', (da2['trends_std']-da0['trends_std']).max().values)
# print('max std diff N', (da2['trends_std']-da1['trends_std']).max().values)

# fig, ax = plt.subplots(1, 3, figsize=(12, 2), constrained_layout=True)
# da0['trends_std'].isel(time=-1).plot(ax=ax[0])
# da1['trends_std'].isel(time=-1).plot(ax=ax[1])
# da2['trends_std'].isel(time=-1).plot(ax=ax[2])

# # figure_name = 'demo'
# # fig.savefig(f'./{figure_name.replace(' ', '_')}.pdf', dpi=300, bbox_inches='tight')
# # fig.savefig(f'./{figure_name.replace(' ', '_')}.png', dpi=500, bbox_inches='tight')
# plt.show()
