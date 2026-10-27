from pathlib import Path
import nbformat
path = Path('SST_highest_SSP.ipynb')
nb = nbformat.read(path, as_version=4)
nb.cells[0].source = '''# CMIP SST: highest available SSP scenario

Inspect the SST spatial patterns for all models in the same year. Input data are in `/Users/pz/CMIP_AMOC_SST`. Select the highest-forcing SSP **with an available SST file** for each model; the default map shows **annual-mean SST in 2100**.

- Five models provide `ssp585` SST. **GISS-E2-1-G provides only MOC under `ssp585`; its highest available SST scenario is `ssp126`.** Its forcing therefore differs from that of the other five models.
- The files already contain a regular 1° × 1° grid and an annual `year` coordinate. Select one year directly, without interpolation, smoothing, or multi-year averaging. Preserve missing values.
- Convert K to °C using the file's `units` attribute and use a shared color scale. The variable is named `SST`; its metadata describe sea-water potential temperature at scalar `lev = 0`.
- SSP means *Shared Socioeconomic Pathway*. In SSP5-8.5, 8.5 denotes the nominal 2100 radiative-forcing level (W m⁻²), not a CO₂ concentration.
- The SST files contain neither atmospheric CO₂ concentrations nor member identifiers. **The approximately 1135 / 446 ppm labels are standard scenario reference concentrations for 2100, not concentrations read from or verified against these model files.** The reference labels still refer to 2100 if the map year is changed.

Scenario definitions: [DKRZ](https://www.dkrz.de/en/communication/climate-simulations/cmip6-en/the-ssp-scenarios/). Concentrations: [Meinshausen et al. (2020)](https://doi.org/10.5194/gmd-13-3571-2020); rounded values are tabulated in [Jiang et al. (2023), Table 1](https://doi.org/10.1029/2022MS003563).

Plotting code is adapted from cell 6 of `Last_deglaciation/N_minors_S_V3.ipynb` and cell 7 of `Figure_1_Ens.ipynb` (counting all cells from 1). Reused lines and comments include the imports, `ax.set_global()`, grid-label settings, `# Plotting`, `# Using pcolormesh`, `# Add a colorbar`, `# add panel labels`, and `# adjust vertical space between subplots`. Data selection, variable names, layout, and color limits are adapted for SST.

In VS Code, select this project's `.venv/bin/python` and **Run All** from the project root. Dependencies: numpy, pandas, xarray, netCDF4, matplotlib, cartopy, and ipykernel. Coastlines are cached in `data/cartopy/`. Figures are saved as **PNG and PDF** in `figures/SST_highest_SSP/`.'''
nb.cells[2].source = '''## Select SSP files

Rank scenarios using SST files that actually exist. A scenario containing only MOC cannot be selected for SST. The table lists scenario directories, available SST scenarios, and the selected scenario.'''
nb.cells[4].source = '''## Inspect coordinates and select a common year

Read coordinates and metadata before loading a temperature field, then find the intersection of available years. The table reports `units` and `lev` directly from each file.'''
nb.cells[6].source = '''## Load SST and convert K to °C

Keep the original values in `sst_raw` and the two-dimensional plotting fields in `sst_celsius`. Load only the selected year and close each NetCDF file afterwards. Check finite grid-cell counts and temperature ranges.'''
nb.cells[8].source = '''## Plot SST

Use the Robinson global projection and the actual longitude and latitude coordinates. Grey represents land or missing data in the input file; no values are filled. Each panel shows the model, selected SSP, and **2100 reference CO₂ concentration**. The missing `ssp585` SST file for GISS is explicitly marked. Export the same figure to PNG and PDF; the PDF retains vector text and coastlines with a rasterized SST mesh.'''
nb.cells[9].source = nb.cells[9].source.replace("fig.savefig(figure_path, dpi=300, bbox_inches='tight')", "fig.savefig(figure_path, dpi=300, bbox_inches='tight')\nfig.savefig(figure_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')")
nb.cells[9].source = nb.cells[9].source.replace('print(figure_path)', "print(figure_path)\nprint(figure_path.with_suffix('.pdf'))")
for c in nb.cells:
 if c.cell_type == 'code':
  c.outputs = []
  c.execution_count = None
nbformat.write(nb,path)
