"""
Purpose: Calculate regional frequency-wavenumber KE spectra.
Main inputs: Daily ugos/vgos fields and regional boxes selected by the previous script.
Main outputs: Regional PSD NetCDF files under Spectrum/.
Notes: Update placeholder paths and product/version switches before running.
"""

# %% Calculate KE spectra for multiple regions and datasets.

import xarray as xr
import numpy as np
from scipy.fft import fftn, fftfreq
from scipy.signal.windows import tukey
import os

# ========================================================================
# Region definitions  (lon_min, lon_max, lat_min, lat_max)
# ========================================================================
REGIONS = {
    'ACC':            ( 78.125, 223.625, -54.625, -48.375),
    'Kuroshio':       (140.625, 183.625,  30.375,  39.875),
    'GulfStream':     (286.375, 321.625,  32.875,  42.375),
    'Agulhas':        (  6.125,  49.875, -42.875, -29.875),
    'EastAustralian': (153.375, 166.375, -37.125, -22.125),
    'Malvinas':       (304.125, 330.875, -50.875, -36.125),
}

DVERSIONS = ['twosat', 'allsat']

timea, timeb = '1993-07-01', '2024-10-31'

ddir = '/path/to'

# ========================================================================
# PSD function (unchanged from calc_KE_spectrum.py)
# ========================================================================

def calculate_vel_psd(dx, dy, dt, vel):
    """
    Calculate velocity frequency-wavenumber power spectral density (PSD)
    with 3D sequential demean and Tukey windowing (r=1/6, Arbic et al. 2014).

    Parameters:
        dx (float): Grid spacing in x-direction [m]
        dy (float): Grid spacing in y-direction [m]
        dt (float): Time step [s]
        vel (np.ndarray): Zonal or Meridional velocity [m/s] (dimensions: [time, y, x])

    Returns:
    dict: Contains PSD results and spectral axes
        psd : PSD(f, k), units [(m/s)^2] / [Hz * cycles/m]
    """

    nt, ny, nx = vel.shape
    Lx = dx * nx
    Ly = dy * ny

    vel = vel.copy().astype(np.float64)
    vel -= np.mean(vel, axis=0, keepdims=True)
    vel -= np.mean(vel, axis=(1, 2), keepdims=True)

    wt = tukey(nt, alpha=1.0/6.0)
    wy = tukey(ny, alpha=1.0/6.0)
    wx = tukey(nx, alpha=1.0/6.0)
    window = wt[:, None, None] * wy[None, :, None] * wx[None, None, :]
    window_power = float(np.sum(window**2))

    energy_before = float(np.sum((vel * window)**2)) / window_power

    vel_hat = fftn(vel * window, axes=(0,1,2))

    psd_vel = (np.abs(vel_hat)**2) * dx*dy*dt / window_power

    Lt = dt * nt
    energy_after = float(np.sum(psd_vel)) / (Lx * Ly * Lt)
    rel_err = abs(energy_after - energy_before) / max(energy_before, 1.0e-30)
    print(f'  Parseval check: before_fft={energy_before:.6e}, '
          f'after_fft={energy_after:.6e}, rel_err={rel_err:.2e}')
    if rel_err > 0.05:
        import warnings
        warnings.warn(
            f'Parseval relative error {rel_err:.2e} exceeds 5%. '
            'The windowed PSD may not conserve energy well - '
            'this is expected when the window attenuates a large fraction of the data.',
            stacklevel=2,
        )

    kx = fftfreq(nx, d=dx)  # [1/m]
    ky = fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    wavenumbers = np.sqrt(kx_grid**2 + ky_grid**2)
    freq = fftfreq(nt, d=dt)  # [Hz]

    dkx = 1.0 / Lx
    dky = 1.0 / Ly
    dk = min(dkx, dky)

    k_edges = np.arange(0, np.max(wavenumbers)+dk, dk)
    if k_edges[-1] <= np.max(wavenumbers):
        k_edges = np.append(k_edges, k_edges[-1] + dk)
    k_centers = (k_edges[:-1] + k_edges[1:])/2

    shell_index = np.digitize(wavenumbers.ravel(), bins=k_edges) - 1
    valid = (shell_index >= 0) & (shell_index < k_centers.size)
    shell_index = shell_index[valid]

    area_factor = dkx * dky / dk
    psd_fk = np.zeros((len(freq), len(k_centers)))
    for ifreq in range(len(freq)):
        shell_sum = np.bincount(
            shell_index,
            weights=psd_vel[ifreq].ravel()[valid],
            minlength=k_centers.size,
        )
        psd_fk[ifreq] = shell_sum * area_factor

    positive_freq = freq >= 0.0
    freq_pos = freq[positive_freq]
    psd_2d = psd_fk[positive_freq].copy()

    positive_interior = freq_pos > 0.0
    if nt % 2 == 0:
        nyquist = 0.5 / dt
        positive_interior &= ~np.isclose(freq_pos, nyquist)
    psd_2d[positive_interior] *= 2.0

    return {
        'psd': psd_2d,
        'wavenumbers': k_centers,
        'frequencies': freq_pos,
        'kx': kx,
        'ky': ky,
        'freq': freq
    }

# ========================================================================
# Main loop
# ========================================================================

outdir = f'{ddir}/Spectrum'
os.makedirs(outdir, exist_ok=True)

for dversion in DVERSIONS:
    print(f'\n{"="*72}')
    print(f'Dataset: {dversion}')
    print(f'{"="*72}')

    dsu_full = xr.open_dataset(f'{ddir}/{dversion}_daily/input/ugos.nc').sel(time=slice(timea, timeb))
    dsv_full = xr.open_dataset(f'{ddir}/{dversion}_daily/input/vgos.nc').sel(time=slice(timea, timeb))

    for region_name, (lon_min, lon_max, lat_min, lat_max) in REGIONS.items():
        print(f'\n--- {region_name} ({dversion}) ---')
        print(f'    lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]')

        dsu = dsu_full.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        dsv = dsv_full.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        u = dsu['ugos'].values[:]
        v = dsv['vgos'].values[:]

        u[np.isnan(u)] = 0.0
        v[np.isnan(v)] = 0.0

        print(f'    shape = {u.shape}  (nt, nlat, nlon)')

        dt = 86400.0
        dlat = np.diff(dsu['lat'].values[:])[0]
        dlon = np.diff(dsu['lon'].values[:])[0]
        R = 6371e3
        lats = dsu['lat'].values
        dy = np.radians(dlat) * R                                              # [m]
        dx = float(np.mean(np.radians(dlon) * R * np.cos(np.radians(lats))))   # [m]

        print(f'    dx = {dx:.1f} m, dy = {dy:.1f} m')
        print(f'    lat range = [{lats[0]:.2f}, {lats[-1]:.2f}]')

        print('  Computing u PSD ...')
        u_result = calculate_vel_psd(dx, dy, dt, u)
        print('  Computing v PSD ...')
        v_result = calculate_vel_psd(dx, dy, dt, v)

        wavenumbers = u_result['wavenumbers']  # [1/m] = cycles per meter
        frequencies = u_result['frequencies']  # [Hz]  = cycles per second

        outpath = f'{outdir}/psd_{dversion}_{region_name}_{timea[:4]}_{timeb[:4]}.nc'
        ds = xr.Dataset(
            data_vars={
                'u_psd': (['freq', 'wn'], u_result['psd'],
                          {'units': '(m/s)^2 / (Hz * cycles/m)',
                           'long_name': 'Zonal velocity PSD'}),
                'v_psd': (['freq', 'wn'], v_result['psd'],
                          {'units': '(m/s)^2 / (Hz * cycles/m)',
                           'long_name': 'Meridional velocity PSD'}),
            },
            coords={
                'freq': ('freq', frequencies,
                         {'units': 'Hz', 'long_name': 'Frequency'}),
                'wn':   ('wn', wavenumbers,
                         {'units': 'cycles/m', 'long_name': 'Isotropic wavenumber'}),
            },
            attrs={
                'description': 'One-sided frequency-wavenumber PSD',
                'dversion': dversion,
                'region': region_name,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'time_start': timea,
                'time_end': timeb,
            },
        )
        ds.to_netcdf(outpath)
        print(f'  Saved: {outpath}')

    dsu_full.close()
    dsv_full.close()

print(f'\nAll done. Output directory: {outdir}/')
