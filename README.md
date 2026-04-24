# Limited intensification of oceanic large-scale and mesoscale circulations since the 1990s

This repository contains the analysis code for *"Limited intensification of oceanic large-scale and mesoscale circulations since the 1990s"* by **Yang Yang et al.**.

The code processes satellite altimetry fields, calculates large-scale and mesoscale geostrophic kinetic energy, estimates trends and red-noise uncertainty, evaluates satellite-number effects, and generates figures and tables.

## Repository Structure

- `prepare_data/`: Scripts for extracting input fields, building masks, converting longitude coordinates, interpolating grids, applying temporal filters, and calculating kinetic energy.
- `process_data/`: Scripts for valid-mask construction, long-term means, grid coarsening, deseasoning, EEMD, AR(1) uncertainty, regional means, spectra, and information-flow diagnostics.
- `figure_scripts/`: Scripts for generating figures and summary tables.

## Getting Started

Run the numbered scripts in `prepare_data/` and `process_data/` in sequence, adjusting product, variable, filter, region, and time-period switches as needed.

Before running, replace the placeholder paths in the scripts:

- `/path/to`: root directory for the workflow data files and outputs; it should contain subfolders such as `mask/`, `topo/`, `SatNum/`, `twosat_i8192/`, and `allsat_i8192/`.
- `/path/to/SEALEVEL_GLO_PHY_L4_MY_008_047`: replace this full string with the local root for the DUACS multi-mission altimeter data from CMEMS.
- `/path/to/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057`: replace this full string with the local root for the C3S altimeter data from CMEMS.
- `/path/to/matlab_tools/`: local MATLAB path setup; it should include the Climate Data Toolbox if the deseasoning script is used.

Large intermediate NetCDF files, satellite-number tables, and generated figures are not included.

## Data Availability

The DUACS multi-mission altimeter data are publicly available from CMEMS at https://doi.org/10.48670/MOI-00148.

The C3S altimeter data are available from CMEMS at https://doi.org/10.48670/MOI-00145.

The scripts expect daily gridded fields including `ugos`, `vgos`, and `adt`.

## Software Requirements

Python dependencies are listed in `requirements.txt`. The EEMD analysis was implemented using the `pyeemd` package, available at https://pyeemd.readthedocs.io. Some processing steps use MATLAB. The `deseason` function used by `process_data/05_calc_global_deseason_chunks.m` is from the Climate Data Toolbox, and the information-flow routine `causality_relative.m` is included.

## Code Availability Note

This repository provides code only. Large input products and derived NetCDF outputs should be downloaded, prepared, or regenerated separately.
