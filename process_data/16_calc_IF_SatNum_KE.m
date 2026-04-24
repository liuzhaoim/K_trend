% Purpose: Calculate regional information flow between satellite number and KE.
% Main inputs: Regional EEMD KE files and a monthly satellite-number table.
% Main outputs: Regional information-flow NetCDF files.
% Notes: Update placeholder paths and product/version switches before running.

clc;
clear all;

ke_label = 'twosat';
% ke_label = 'allsat';
ke_var = 'KE_M';
% ke_var = 'KE_L';
% sig_type = 'monthly';    % original monthly series (mode 1)
sig_type = 'deimf1';   % signal after removing the highest-frequency IMF1 = sum(mode 3:N)
% sig_type = 'trend';    % trend component (mode N)
% ---- Select time period {start date, end date, label} ----
time_period = {'1993-07-01', '2024-10-31', 'full'};
% time_period = {'1993-07-01', '2014-10-31', 'early'};
% time_period = {'2014-11-01', '2024-10-31', 'late'};
% ---- End of switches ----
tp_start = datetime(time_period{1});
tp_end   = datetime(time_period{2});
tp_label = time_period{3};

var_short = {'T21', 'tau21', 'E90_21', 'E95_21', 'E99_21', ...
             'T12', 'tau12', 'E90_12', 'E95_12', 'E99_12'};
nvar = length(var_short);

regions = {'mask_Kuroshio', 'mask_GulfStream', 'mask_Agulhas', ...
           'mask_EastAustralian', 'mask_Malvinas', 'mask_ACC', ...
           'mask_global_EQ10'};
region_names = {'Kuroshio', 'Gulf Stream', 'Agulhas', ...
                'EAC', 'BMC', 'ACC', 'Global Ocean'};
nreg = length(regions);

ddir = ['/path/to/', ke_label, '_i8192/mwt_d184/space_mean_monthly/'];
sat_file = '/path/to/SatNum/SatNum_monthly_count.csv';

fprintf('== Configuration ==\n');
fprintf('  Satellite product: %s\n', ke_label);
fprintf('  KE variable:  %s\n', ke_var);
fprintf('  Signal type: %s\n', sig_type);
fprintf('  Time period:   %s (%s ~ %s)\n', tp_label, time_period{1}, time_period{2});
fprintf('  Data directory: %s\n', ddir);

sat_data = readtable(sat_file);
sat_dates = datetime(sat_data.monthly, 'InputFormat', 'yyyy-MM-dd');
sat_count = sat_data.count;

results = NaN(nreg, nvar);

for ir = 1:nreg

    rgname = regions{ir};
    imf_file = [ddir, rgname, '_', ke_var, '_monthly_eemd.nc'];

    fprintf('\n[%d/%d] %s (%s)\n', ir, nreg, region_names{ir}, rgname);

    imf_time  = ncread(imf_file, 'time');
    mode_vals = ncread(imf_file, 'mode');
    IMF_all   = ncread(imf_file, 'IMF');

    nmode = length(mode_vals);
    imf_dates = datetime(1993, 1, 1) + days(imf_time);

    switch sig_type
        case 'monthly', ke_sig = double(IMF_all(:, 1));
        case 'deimf1',  ke_sig = double(sum(IMF_all(:, 3:nmode), 2));
        case 'trend',   ke_sig = double(IMF_all(:, nmode));
        otherwise,      error('Unknown sig_type: %s', sig_type);
    end

    fprintf('  IMF: %d modes, time=%d\n', nmode, length(imf_time));

    sat_ym = sat_dates.Year * 12 + sat_dates.Month;
    imf_ym = imf_dates.Year * 12 + imf_dates.Month;

    [~, sat_idx, imf_idx] = intersect(sat_ym, imf_ym);

    aligned_dates = sat_dates(sat_idx);
    tmask = aligned_dates >= tp_start & aligned_dates <= tp_end;

    sat_idx = sat_idx(tmask);
    imf_idx = imf_idx(tmask);

    sat_aligned = double(sat_count(sat_idx));
    ntime = length(sat_aligned);

    fprintf('  Time period: %s ~ %s, total %d months\n', ...
        datestr(sat_dates(sat_idx(1))), datestr(sat_dates(sat_idx(end))), ntime);

    ke_ts = ke_sig(imf_idx);

    if any(isnan(ke_ts))
        fprintf('  NaN values found; skipping this region.\n');
        continue;
    end

    warning off;

    % causality_relative(X1, X2, np): T21 = X2 -> X1
    % X1=KE, X2=SatNum -> T21 = SatNum -> KE
    [T21, tau21, E90_21, E95_21, E99_21] = causality_relative(ke_ts, sat_aligned, 1);
    [T12, tau12, E90_12, E95_12, E99_12] = causality_relative(sat_aligned, ke_ts, 1);

    results(ir, :) = [T21, tau21, E90_21, E95_21, E99_21, ...
                      T12, tau12, E90_12, E95_12, E99_12];

    fprintf('  T21=%.4e, tau21=%.4f, T12=%.4e, tau12=%.4f\n', T21, tau21, T12, tau12);

    warning on;
end

fprintf('\n\n========================================\n');
fprintf('  Result summary: %s %s %s (%s)\n', ke_var, sig_type, ke_label, tp_label);
fprintf('========================================\n');

fprintf('%-16s  %12s  %8s  %12s %12s %12s  | %12s  %8s  %12s %12s %12s\n', ...
    'Region', 'T21(Sat->KE)', 'tau21', 'E90_21', 'E95_21', 'E99_21', ...
    'T12(KE->Sat)', 'tau12', 'E90_12', 'E95_12', 'E99_12');
fprintf('%s\n', repmat('-', 1, 150));
for ir = 1:nreg
    r = results(ir, :);
    sig21 = ''; sig12 = '';
    if abs(r(1)) > abs(r(4)); sig21 = ' *'; end   % |T21| > E95_21
    if abs(r(6)) > abs(r(9)); sig12 = ' *'; end   % |T12| > E95_12
    fprintf('%-16s  %12.4e  %8.4f  %12.4e %12.4e %12.4e%s | %12.4e  %8.4f  %12.4e %12.4e %12.4e%s\n', ...
        region_names{ir}, r(1), r(2), r(3), r(4), r(5), sig21, ...
        r(6), r(7), r(8), r(9), r(10), sig12);
end
fprintf('\n(* indicates significance at the 95%% confidence level)\n');

%% ========== 4. Save results to NetCDF ==========
fillvalue = single(9.96921e+36);
outdir = '/path/to/SatNum/IF_space_mean/';
if ~exist(outdir, 'dir'); mkdir(outdir); end
fname = fullfile(outdir, ['IF_SatNum_', ke_var, '_', sig_type, '_', ke_label, '_', tp_label, '.nc']);
fprintf('\nSaving results to %s ...\n', fname);

if exist(fname, 'file'); delete(fname); end

nccreate(fname, 'region', 'Dimensions', {'region', nreg}, ...
    'Datatype', 'int32', 'Format', '64bit');
ncwrite(fname, 'region', int32(1:nreg));
ncwriteatt(fname, 'region', 'long_name', 'Region index');
ncwriteatt(fname, 'region', 'region_names', strjoin(region_names, ', '));

for iv = 1:nvar
    vn = var_short{iv};
    v = results(:, iv);
    v(isnan(v)) = fillvalue;

    nccreate(fname, vn, 'Dimensions', {'region', nreg}, ...
        'Datatype', 'single', 'Format', '64bit');
    ncwrite(fname, vn, single(v));
    ncwriteatt(fname, vn, '_FillValue', fillvalue);
    ncwriteatt(fname, vn, 'missing_value', fillvalue);
end

ncwriteatt(fname, '/', 'description', ...
    ['Regional info flow: SatNum and ', ke_var, ' (', sig_type, ', ', ke_label, ', ', tp_label, ')']);
ncwriteatt(fname, '/', 'regions', strjoin(regions, ', '));
ncwriteatt(fname, '/', 'region_names', strjoin(region_names, ', '));
ncwriteatt(fname, '/', 'signal_type', sig_type);
ncwriteatt(fname, '/', 'SatNum_source', sat_file);
ncwriteatt(fname, '/', 'KE_source_dir', ddir);
ncwriteatt(fname, '/', 'method', 'Liang-Kleeman information flow (causality_relative.m)');
ncwriteatt(fname, '/', 'creation_date', datestr(now));

fprintf('Finished! Results were saved to %s\n', fname);
