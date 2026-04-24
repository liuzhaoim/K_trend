"""
Purpose: Run EEMD on regional mean monthly time series.
Main inputs: Regional mean monthly KE time series.
Main outputs: Regional EEMD mode and trend NetCDF files.
Notes: EEMD is implemented with the pyeemd package; update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr
from tqdm import tqdm

import pyeemd

def calc_eemd(inp):
    IMFm = pyeemd.eemd(inp, ensemble_size=400, noise_strength=0.2, S_number=0, num_siftings=10, rng_seed=0)
    return IMFm

def process_eemd(args):

    ddir, dversion, ftversion, tmversion, varname, rgname = args

    da = xr.open_dataset(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}.nc')[varname]

    monthly = da  # if the input is monthly data
    # monthly = da.resample(time='MS').mean(dim=['time'], skipna=True)  # if the input is daily data

    inp = monthly.values
    time = monthly['time'].values

    ntime = len(inp)
    nIMFs = int(np.fix(np.log2(ntime)))  # Number of IMFs

    v = np.full((nIMFs+1, ntime), np.nan)
    v[0, :]  = inp
    v[1:, :] = calc_eemd(inp)

    ds_out = xr.Dataset(
        data_vars=dict(
            IMF=(['mode', 'time'], v),
        ),
        coords=dict(
            time=(['time'], time),
            mode=(['mode'], np.arange(nIMFs+1)),
        ),
    )
    ds_out['time'].attrs['long_name'] = 'Time'
    encoding = {
        'IMF': {'dtype': 'float64', '_FillValue': 9.96921e+36, 'missing_value': 9.96921e+36},
        'time': {'dtype': 'float64', 'units': 'days since 1993-01-01 00:00:00', 'calendar': 'gregorian'},
    }
    ds_out.to_netcdf(f'{ddir}/{dversion}/{ftversion}/{tmversion}/{rgname}_{varname}_monthly_eemd.nc', encoding=encoding)


ddir = '/path/to'

dversions = ['allsat_i8192', 'twosat_i8192']
ftversion = 'mwt_d184'
# ftversion = 'mwt_d369'
# ftversion = 'mwt_d92'

# dversions = ['twosat_daily']
# ftversion = 'bw6_d184'

# tmversion = 'space_mean_daily'
tmversion = 'space_mean_monthly'
# tmversion = 'space_mean_monthly_deg025'

varnames = ['KE_L', 'KE_M']

rgnames = ['mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', 'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', 'mask_global_EQ10']

tasks = [(ddir, dversion, ftversion, tmversion, varname, rgname) for rgname in rgnames for varname in varnames for dversion in dversions]

for task in tqdm(tasks):
    process_eemd(task)
