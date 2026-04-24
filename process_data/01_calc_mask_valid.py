"""
Purpose: Calculate bathymetry, missing-value, equatorial, and combined valid-region masks.
Main inputs: Topography files and daily input velocity fields.
Main outputs: Mask NetCDF files under mask/.
Notes: Update placeholder paths and product/version switches before running.
"""

import numpy as np
import xarray as xr


DDIR = "/path/to"
DEGREE = "deg025"
DVERSION_LIST = ["allsat_i8192", "twosat_i8192"]
U_VAR = "uvel"
V_VAR = "vvel"


def write_nc_bool_ll(fname, varname, da):
    ds_out = xr.Dataset(
        data_vars={varname: (["lat", "lon"], da.values)},
        coords=dict(
            lon=(["lon"], da["lon"].values),
            lat=(["lat"], da["lat"].values),
        ),
    )
    ds_out["lon"].attrs["units"] = "degrees_east"
    ds_out["lon"].attrs["long_name"] = "Longitude"
    ds_out["lat"].attrs["units"] = "degrees_north"
    ds_out["lat"].attrs["long_name"] = "Latitude"
    encoding = {
        varname: {"dtype": "bool"},
        "lon": {"dtype": "float64"},
        "lat": {"dtype": "float64"},
    }
    ds_out.to_netcdf(fname, encoding=encoding)


def calc_mask_100m():
    depth = xr.open_dataset(f"{DDIR}/topo/ETOPO1_ECCO2grid.nc")["z"]
    mask_100m = depth >= -100
    write_nc_bool_ll(f"{DDIR}/mask/mask_100m_{DEGREE}.nc", "mask_100m", mask_100m)


def calc_mask_dailynan(dversion):
    mask_u = xr.open_dataset(f"{DDIR}/{dversion}/input/{U_VAR}_lev1.nc")[U_VAR].isnull().any(dim="time").compute()
    mask_v = xr.open_dataset(f"{DDIR}/{dversion}/input/{V_VAR}_lev1.nc")[V_VAR].isnull().any(dim="time").compute()
    mask_dailynan = mask_u | mask_v
    write_nc_bool_ll(
        f"{DDIR}/mask/mask_dailynan_{dversion[:6]}_{DEGREE}.nc",
        "mask_dailynan",
        mask_dailynan,
    )


def calc_mask_EQ10():
    lon = xr.open_dataset(f"{DDIR}/twosat_i8192/input/{U_VAR}_lev1.nc")["lon"].values
    lat = xr.open_dataset(f"{DDIR}/twosat_i8192/input/{U_VAR}_lev1.nc")["lat"].values
    _, lat_grid = np.meshgrid(lon, lat)
    mask_EQ10 = xr.DataArray(
        (lat_grid >= -10) & (lat_grid <= 10),
        coords=dict(lat=lat, lon=lon),
        dims=("lat", "lon"),
    )
    write_nc_bool_ll(f"{DDIR}/mask/mask_EQ10_{DEGREE}.nc", "mask_EQ10", mask_EQ10)


def calc_mask_valid():
    mask_dir = f"{DDIR}/mask"
    mask_100m = xr.open_dataset(f"{mask_dir}/mask_100m_{DEGREE}.nc")["mask_100m"]
    mask_dailynan_allsat = xr.open_dataset(f"{mask_dir}/mask_dailynan_allsat_{DEGREE}.nc")["mask_dailynan"]
    mask_dailynan_twosat = xr.open_dataset(f"{mask_dir}/mask_dailynan_twosat_{DEGREE}.nc")["mask_dailynan"]
    mask_EQ10 = xr.open_dataset(f"{mask_dir}/mask_EQ10_{DEGREE}.nc")["mask_EQ10"]

    mask_valid = ~mask_100m & ~mask_dailynan_allsat & ~mask_dailynan_twosat & ~mask_EQ10
    write_nc_bool_ll(f"{mask_dir}/mask_valid_twoall_{DEGREE}.nc", "mask_valid", mask_valid)


if __name__ == "__main__":
    calc_mask_100m()
    for dversion in DVERSION_LIST:
        calc_mask_dailynan(dversion)
    calc_mask_EQ10()
    calc_mask_valid()
