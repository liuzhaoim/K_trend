"""
Purpose: Calculate regional masks from long-term mean large-scale and mesoscale KE.
Main inputs: Deg025 long-term mean KE_L and KE_M fields.
Main outputs: KEmasked_regions_deg025.nc and KEmasked_regions_deg1.nc under mask/.
Notes: Update placeholder paths and product/filter switches before running.
"""

import numpy as np
import xarray as xr


DDIR = "/path/to"
DVERSION = "twosat_i8192"
FTVERSION = "mwt_d184"
KE_VARNAMES = ["KE_L", "KE_M"]
THRESHOLD = 20


def write_region_masks(fname, masks, lon, lat):
    ds_out = xr.Dataset(
        data_vars={name: (["lat", "lon"], value) for name, value in masks.items()},
        coords=dict(
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description=f"KE-masked regions (KE_sum > {THRESHOLD})"),
    )
    ds_out["lon"].attrs["units"] = "degrees_east"
    ds_out["lon"].attrs["long_name"] = "Longitude"
    ds_out["lat"].attrs["units"] = "degrees_north"
    ds_out["lat"].attrs["long_name"] = "Latitude"
    encoding = {
        **{varname: {"dtype": "bool"} for varname in ds_out.data_vars},
        "lon": {"dtype": "float64"},
        "lat": {"dtype": "float64"},
    }
    ds_out.to_netcdf(fname, encoding=encoding)


def calc_area(da):
    lat = da["lat"].values
    lon = da["lon"].values
    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())

    lat_b = np.concatenate(([lat[0] - dlat / 2], 0.5 * (lat[:-1] + lat[1:]), [lat[-1] + dlat / 2]))
    lon_b = np.concatenate(([lon[0] - dlon / 2], 0.5 * (lon[:-1] + lon[1:]), [lon[-1] + dlon / 2]))

    radius = 6371e3
    dmu = np.sin(np.deg2rad(lat_b[1:])) - np.sin(np.deg2rad(lat_b[:-1]))
    dlambda = np.deg2rad(lon_b[1:] - lon_b[:-1])
    area = (radius ** 2) * dmu[:, None] * dlambda[None, :]
    return xr.DataArray(area, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def area_weighted_coarsen_to_deg1(da, mask_valid):
    valid = np.isfinite(da)
    if mask_valid is not None:
        valid = valid & mask_valid

    area = calc_area(da)
    weighted_sum = (da.where(valid) * area).coarsen(lat=4, lon=4, boundary="trim").sum(skipna=True)
    area_sum = area.where(valid).coarsen(lat=4, lon=4, boundary="trim").sum(skipna=True)
    return weighted_sum / area_sum


def calc_region_masks(ke_l, ke_m):
    mask_lon = ke_l["lon"].values
    mask_lat = ke_l["lat"].values
    lon, lat = np.meshgrid(mask_lon, mask_lat)

    mask_global = (lon > 0) & (lon < 360) & (lat > -90) & (lat < 90)
    mask_global_EQ10 = (lon > 0) & (lon < 360) & ((lat > 10) | (lat < -10))

    mask_Kuroshio = (
        (lon > 120)
        & (lon < 190)
        & (lat > 17)
        & (lat < 45)
        & (lat <= (3 / 4) * lon - 65)
        & ~((lon >= 144) & (lat < 28))
        & ~((lon <= 144) & (lat < (2 / 5) * lon - 30))
    )
    mask_GulfStream = (lon > 280) & (lon < 330) & (np.abs(lat - (2 / 5) * lon + 84) <= 10)
    mask_Agulhas = (lon > 0) & (lon < 50) & (lat > -44) & (lat < -18)
    mask_EastAustralian = (lon > 147) & (lon < 167) & (lat > -46) & (lat < -18)
    mask_Malvinas = (lon > 298) & (lon < 331) & (lat > -51) & (lat < -34)

    mask_ACC = (lat > -70) & (lat < -34) & ~((lon > 20) & (lon < 288) & (lat > -(3 / 40) * lon - 63 / 2))
    mask_ACC = mask_ACC & ~mask_Agulhas & ~mask_EastAustralian & ~mask_Malvinas

    KE_sum = ke_l.values + ke_m.values
    masks = {
        "mask_global": mask_global,
        "mask_global_EQ10": mask_global_EQ10,
        "mask_ACC": mask_ACC & (KE_sum > THRESHOLD),
        "mask_Kuroshio": mask_Kuroshio & (KE_sum > THRESHOLD),
        "mask_GulfStream": mask_GulfStream & (KE_sum > THRESHOLD),
        "mask_Agulhas": mask_Agulhas & (KE_sum > THRESHOLD),
        "mask_EastAustralian": mask_EastAustralian & (KE_sum > THRESHOLD),
        "mask_Malvinas": mask_Malvinas & (KE_sum > THRESHOLD),
    }
    return masks, mask_lon, mask_lat


def calc_KEmasked_regions():
    base = f"{DDIR}/{DVERSION}/{FTVERSION}/longterm_mean"
    ke_l_deg025 = xr.open_dataset(f"{base}/{KE_VARNAMES[0]}_deg025.nc")[KE_VARNAMES[0]]
    ke_m_deg025 = xr.open_dataset(f"{base}/{KE_VARNAMES[1]}_deg025.nc")[KE_VARNAMES[1]]

    masks_deg025, lon_deg025, lat_deg025 = calc_region_masks(ke_l_deg025, ke_m_deg025)
    write_region_masks(f"{DDIR}/mask/KEmasked_regions_deg025.nc", masks_deg025, lon_deg025, lat_deg025)

    mask_valid = xr.open_dataset(f"{DDIR}/mask/mask_valid_twoall_deg025.nc")["mask_valid"]
    ke_l_deg1 = area_weighted_coarsen_to_deg1(ke_l_deg025, mask_valid)
    ke_m_deg1 = area_weighted_coarsen_to_deg1(ke_m_deg025, mask_valid)

    masks_deg1, lon_deg1, lat_deg1 = calc_region_masks(ke_l_deg1, ke_m_deg1)
    write_region_masks(f"{DDIR}/mask/KEmasked_regions_deg1.nc", masks_deg1, lon_deg1, lat_deg1)


if __name__ == "__main__":
    calc_KEmasked_regions()
