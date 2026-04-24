"""
Purpose: Run EEMD at each global grid point.
Main inputs: Monthly gridded KE fields.
Main outputs: Monthly EEMD mode and trend NetCDF files.
Notes: EEMD is implemented with the pyeemd package; update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool

import pyeemd

def calc_eemd(inp):
    IMFm = pyeemd.eemd(inp, ensemble_size=400, noise_strength=0.2, S_number=0, num_siftings=10, rng_seed=0)
    return IMFm


ddir = '/path/to'

dversion = 'twosat_i8192'
# dversion = 'allsat_i8192'
# dversion = 'twosat_daily'

ftversion = 'mwt_d184'
# ftversion = 'mwt_d369'
# ftversion = 'mwt_d92'
# ftversion = 'bw6_d184'

varname = 'KE_L'
# varname = 'KE_M'

degree = 'deg1'
# degree = 'deg025'
# degree = 'deg1Rdeg5'

# tmversion = 'global_daily'
tmversion = 'global_monthly'

da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}.nc')[varname]

# varin = da.resample(time='MS').mean(dim=['time'], skipna=True)  # for daily data
varin = da

inp = varin.values

time = varin['time'].values
lat = varin['lat'].values
lon = varin['lon'].values
ntime = len(time)
nlat = len(lat)
nlon = len(lon)


inp = inp.reshape((ntime, nlat * nlon))

nIMFs = int(np.fix(np.log2(ntime)))  # Number of IMFs

IMF = np.full((nIMFs+1, ntime, nlat * nlon), np.nan)

valid_idx = np.where(~np.isnan(inp).any(axis=0))[0]
valid_inp = inp[:, valid_idx]
valid_IMF = np.full((nIMFs+1, ntime, len(valid_idx)), np.nan)
valid_IMF[0, :, :] = valid_inp

def compute_eemd(k):
    return calc_eemd(valid_inp[:, k])

# for k in tqdm(range(len(valid_idx))):
#     valid_IMF[1:, :, k] = compute_eemd(k)

if __name__ == '__main__':
    with Pool(processes=64) as pool:
        results = list(tqdm(pool.imap(compute_eemd, range(len(valid_idx))), total=len(valid_idx)))
    for k, result in enumerate(results):
        valid_IMF[1:, :, k] = result

IMF[:, :, valid_idx] = valid_IMF

IMF = IMF.reshape(nIMFs+1, ntime, nlat, nlon)

ds_out = xr.Dataset(
    data_vars=dict(
        IMF=(['mode', 'time', 'lat', 'lon'], IMF),
    ),
    coords=dict(
        lon=(['lon'], lon),
        lat=(['lat'], lat),
        time=(['time'], time),
        mode=(['mode'], np.arange(nIMFs+1)),
    ),
)
ds_out['lon'].attrs['units'] = 'degrees_east'
ds_out['lon'].attrs['long_name'] = 'Longitude'
ds_out['lat'].attrs['units'] = 'degrees_north'
ds_out['lat'].attrs['long_name'] = 'Latitude'
ds_out['time'].attrs['long_name'] = 'Time'

encoding = {
    'IMF': {'dtype': 'float32', '_FillValue': 9.96921e+36, 'missing_value': 9.96921e+36},
    'lon': {'dtype': 'float64'},
    'lat': {'dtype': 'float64'},
    'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
}

ds_out.to_netcdf(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{degree}_{varname}_monthly_eemd.nc', encoding=encoding)
