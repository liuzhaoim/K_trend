% Purpose: Remove the daily seasonal cycle in longitude chunks.
% Main inputs: Daily fields for the selected variable.
% Main outputs: Longitude-chunk deseasoned NetCDF files.
% Notes: Update placeholder paths and product/version switches before running. The deseason function is from the Climate Data Toolbox.

clear; close all; clc;

addpath('/path/to/matlab_tools/');
add_matlab_path;

ddir = '/path/to';

dversion = 'twosat_i8192';
% dversion = 'allsat_i8192';
% dversion = 'twosat_daily';

ftversion = 'mwt_d184';
% ftversion = 'mwt_d369';
% ftversion = 'mwt_d92';
% ftversion = 'mwt_d184_win3'
% ftversion = 'bw6_d184';

varname = 'KE_L';
% varname = 'KE_M';
% varname = 'KE_S';

filein = fullfile(ddir, dversion, ftversion, [varname, '_daily.nc']);

lon = ncread(filein, 'lon');
lat = ncread(filein, 'lat');
time0 = ncread(filein, 'time');

time_units = ncreadatt(filein, 'time', 'units');

nlon = length(lon);
nlat = length(lat);
ntime = length(time0);

T0 = time0(1);
dt = datenum(1993,1,1) - T0;
[year, month, day] = datevec(time0 + dt);
time = datenum(year, month, day);


fillvalue = single(9.96921e+36);

nchunks = 16;
lon_chunk = nlon / nchunks;

outdir = fullfile(ddir, dversion, ftversion, varname);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

for k = 1:nchunks
    i0 = (k - 1) * lon_chunk + 1;
    i1 = k * lon_chunk;

    start = [i0, 1, 1];
    count = [lon_chunk, nlat, ntime];

    varin = ncread(filein, varname, start, count);
    varout = deseason(varin, time, 'daily');

    varout = single(varout);
    varout(isnan(varout)) = fillvalue;

    lon_sub = lon(i0:i1);

    fname = fullfile(outdir, sprintf('%s_daily_deseason_lon%02dof%02d.nc', varname, k, nchunks));

    nccreate(fname, 'lat', 'Dimensions', {'lat', nlat}, 'Datatype', 'double', 'Format', '64bit');
    ncwrite(fname, 'lat', lat);
    nccreate(fname, 'lon', 'Dimensions', {'lon', lon_chunk}, 'Datatype', 'double', 'Format', '64bit');
    ncwrite(fname, 'lon', lon_sub);
    nccreate(fname, 'time', 'Dimensions', {'time', ntime}, 'Datatype', 'double', 'Format', '64bit');
    ncwrite(fname, 'time', time0);
    nccreate(fname, varname, 'Dimensions', {'lon', lon_chunk, 'lat', nlat, 'time', ntime}, 'Datatype', 'single', 'Format', '64bit');
    ncwrite(fname, varname, varout);

    ncwriteatt(fname, varname, '_FillValue', fillvalue);
    ncwriteatt(fname, varname, 'missing_value', fillvalue);
    ncwriteatt(fname, 'lat', 'units', 'degrees_north');
    ncwriteatt(fname, 'lat', 'long_name', 'Latitude');
    ncwriteatt(fname, 'lon', 'units', 'degrees_east');
    ncwriteatt(fname, 'lon', 'long_name', 'Longitude');
    ncwriteatt(fname, 'time','units', time_units);
    ncwriteatt(fname, 'time','long_name', 'Time');

    fprintf('     wrote %s\n', fname);
end

fprintf('*** Finished variable %s (16 chunks)\n', varname);
