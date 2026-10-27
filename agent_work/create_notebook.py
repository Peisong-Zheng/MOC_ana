from pathlib import Path
import nbformat as nbf

root = Path('/Users/pz/VScode/MOC_ana')
nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name': 'Python (MOC_ana .venv)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.5'},
}
cells = []
def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))
def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))

markdown('''
# CMIP SST: highest available SSP scenario

粗看各模型在同一年的 SST 空间分布。数据来自 `/Users/pz/CMIP_AMOC_SST`，每个模型只选**有 SST 文件的最高强迫 SSP 情景**，默认绘制 **2100 年的年平均场**。

- 五个模型有 `ssp585` SST；**GISS-E2-1-G 的 `ssp585` 只有 MOC，SST 最高可用情景为 `ssp126`**。它不能当作与其他五个模型相同强迫的比较。
- 文件已提供 1° × 1° 经纬网格和 `year` 坐标；直接读取选定年份，不重新插值、平滑或计算多年平均。缺测值保持不变。
- 根据文件的 `units` 将 K 转成 °C，六张图共用色标。文件变量为 `SST`，元数据描述为表层海水位温，标量 `lev = 0`。
- SSP 是 *Shared Socioeconomic Pathway*。`ssp585` 对应 SSP5-8.5，8.5 是 2100 年名义辐射强迫等级（W m⁻²），不是 CO₂ 浓度。
- SST 文件没有 CO₂ 浓度或成员编号。图中 **2100 年 CO₂ 约 1135 / 446 ppm 是标准情景参考值，不是从这些模型文件中读取或验证的实际浓度**；修改绘图年份时，该标注仍明确指 2100 年。

情景说明：[DKRZ](https://www.dkrz.de/en/communication/climate-simulations/cmip6-en/the-ssp-scenarios/)。浓度参考：[Meinshausen et al. (2020)](https://doi.org/10.5194/gmd-13-3571-2020)，取整值见 [Jiang et al. (2023), Table 1](https://doi.org/10.1029/2022MS003563)。

代码参考 `Last_deglaciation/N_minors_S_V3.ipynb` 的第 6 个单元格和 `Figure_1_Ens.ipynb` 的第 7 个单元格（按所有单元格从 1 计数）。保留可直接复用的 imports、`ax.set_global()`、网格标签设置和 `# Plotting`、`# Using pcolormesh`、`# Add a colorbar`、`# add panel labels`、`# adjust vertical space between subplots` 等原注释；数据选择、变量名、布局及色标按本次任务调整。

在 VS Code 选择当前目录的 `.venv/bin/python`，然后 **Run All**。依赖：numpy、pandas、xarray、netCDF4、matplotlib、cartopy、ipykernel。海岸线已缓存到 `data/cartopy/`。
''')
code('''
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from IPython.display import display

project_dir = Path.cwd()
data_dir = Path('/Users/pz/CMIP_AMOC_SST')
figure_dir = project_dir / 'figures' / 'SST_highest_SSP'
figure_dir.mkdir(parents=True, exist_ok=True)
cartopy.config['data_dir'] = project_dir / 'data' / 'cartopy'

plot_year = 2100  # Set to None to use the latest year shared by all models.
vmin, vmax = -2, 38  # SST (°C); one scale for all panels.

# Nominal 2100 forcing, used to rank the SSP scenarios available locally.
ssp_forcing = {'ssp119': 1.9, 'ssp126': 2.6, 'ssp245': 4.5,
               'ssp370': 7.0, 'ssp434': 3.4, 'ssp460': 6.0, 'ssp585': 8.5}
# Rounded global atmospheric CO2 in 2100; scenario references, not file metadata.
co2_2100 = {'ssp126': 446, 'ssp585': 1135}
''')
markdown('''
## Select SSP files

按本地实际存在的 SST 文件筛选，避免把仅有 MOC 的情景当作可绘制的 SST。下表同时列出情景目录、SST 可用情景和最终选择。
''')
code('''
selected_files = {}
selected_scenarios = {}
selection_rows = []

model_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
if not model_dirs:
    raise FileNotFoundError(f'No model directories found in {data_dir}')

for model_dir in model_dirs:
    scenario_dirs = sorted(model_dir.glob('ssp*'))
    sst_files = {}
    for scenario_dir in scenario_dirs:
        files = sorted((scenario_dir / 'SST').glob('*.nc'))
        if files:
            if scenario_dir.name not in ssp_forcing:
                raise ValueError(f'Check SSP forcing before ranking: {scenario_dir.name}')
            if len(files) != 1:
                raise ValueError(f'Check members or time chunks in {scenario_dir / "SST"}')
            sst_files[scenario_dir.name] = files[0]
    if not sst_files:
        raise FileNotFoundError(f'No future SST files for {model_dir.name}')

    scenario = max(sst_files, key=ssp_forcing.get)
    model = model_dir.name
    selected_files[model] = sst_files[scenario]
    selected_scenarios[model] = scenario
    missing_sst = [path.name for path in scenario_dirs if path.name not in sst_files]
    selection_rows.append({
        'model': model,
        'SSP directories': ', '.join(path.name for path in scenario_dirs),
        'available SST': ', '.join(sst_files),
        'selected SSP': scenario,
        '2100 forcing (W m-2)': ssp_forcing[scenario],
        '2100 reference CO2 (ppm)': co2_2100.get(scenario, np.nan),
        'note': f'No SST: {", ".join(missing_sst)}' if missing_sst else '',
    })

selection = pd.DataFrame(selection_rows).set_index('model')
display(selection)
''')
markdown('''
## Inspect coordinates and select a common year

先只读取坐标及元数据，再取年份交集。表中的 `units` 与 `lev` 直接来自文件。
''')
code('''
available_years = {}
metadata_rows = []

for model, path in selected_files.items():
    with xr.open_dataset(path) as ds_sst:
        if set(ds_sst['SST'].dims) != {'year', 'lat', 'lon'}:
            raise ValueError(f'{model}: check SST dimensions {ds_sst.SST.dims}')
        years = ds_sst.year.values
        if not np.all(np.diff(years) > 0):
            raise ValueError(f'{model}: years must be unique and increasing')
        for coord in ['lat', 'lon']:
            if ds_sst[coord].ndim != 1 or not np.all(np.diff(ds_sst[coord]) > 0):
                raise ValueError(f'{model}: check {coord} coordinates')
        available_years[model] = set(years.tolist())
        metadata_rows.append({
            'model': model,
            'first year': int(years.min()),
            'last year': int(years.max()),
            'grid (lat, lon)': (ds_sst.sizes['lat'], ds_sst.sizes['lon']),
            'units': ds_sst.SST.attrs.get('units', ds_sst.attrs.get('units')),
            'lev': float(ds_sst.lev),
            'file': str(path.relative_to(data_dir)),
        })

common_years = sorted(set.intersection(*available_years.values()))
if not common_years:
    raise ValueError('The selected SST files have no common year')
if plot_year is None:
    plot_year = common_years[-1]
if plot_year not in common_years:
    raise ValueError(f'{plot_year} is not shared by all models. Available: {common_years}')

display(pd.DataFrame(metadata_rows).set_index('model'))
print(f'Common years: {common_years[0]}–{common_years[-1]} ({len(common_years)} years)')
print(f'Selected annual-mean SST: {plot_year}')
''')
markdown('''
## Load SST and convert K to °C

`sst_raw` 保留文件中的原值；`sst_celsius` 保存用于绘图的二维场。只加载一个年份，随后关闭 NetCDF 文件。检查每个场的有效格点数和温度范围。
''')
code('''
sst_raw = {}
sst_celsius = {}
diagnostic_rows = []

for model, path in selected_files.items():
    with xr.open_dataset(path) as ds_sst:
        sst_raw[model] = ds_sst.SST.sel(year=plot_year).transpose('lat', 'lon').load()
        units = ds_sst.SST.attrs.get('units', ds_sst.attrs.get('units', ''))

    if units.strip().lower() in ['k', 'kelvin']:
        # Convert Kelvin to Celsius; preserve the raw field and missing values.
        sst_celsius[model] = sst_raw[model] - 273.15
    elif units.strip().lower() in ['degc', 'degrees_celsius', 'degree_celsius', '°c']:
        sst_celsius[model] = sst_raw[model].copy()
    else:
        raise ValueError(f'{model}: unrecognized SST units {units!r}')
    sst_celsius[model].attrs = dict(sst_raw[model].attrs, units='degC')

    sst = sst_celsius[model]
    finite = np.isfinite(sst)
    if not bool(finite.any()) or bool(np.isinf(sst).any()):
        raise ValueError(f'{model}: SST is empty or contains infinite values')
    diagnostic_rows.append({
        'model': model,
        'year': plot_year,
        'min (°C)': float(sst.min()),
        'max (°C)': float(sst.max()),
        'valid grid cells': int(finite.sum()),
        'missing (%)': float(sst.isnull().mean() * 100),
        'outside color scale': int(((sst < vmin) | (sst > vmax)).sum()),
    })

diagnostics = pd.DataFrame(diagnostic_rows).set_index('model')
display(diagnostics.round(2))
''')
markdown('''
## Plot SST

Robinson 全球投影；使用文件中真实的经纬度格点。灰色表示陆地或原文件缺测区域，不补值。每个面板标注模型、实际选中的 SSP，以及该情景 **2100 年参考 CO₂**。GISS 单独标注 `ssp585` SST 缺失。
''')
code('''
# Plotting
ncols = 3
nrows = int(np.ceil(len(sst_celsius) / ncols))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows),
                        subplot_kw={'projection': ccrs.Robinson()}, dpi=150,
                        squeeze=False)

for i, (model, sst) in enumerate(sst_celsius.items()):
    ax = axs.flat[i]
    scenario = selected_scenarios[model]
    ax.set_global()
    ax.set_facecolor('0.88')

    # Using pcolormesh
    map_plot = ax.pcolormesh(sst['lon'], sst['lat'], sst,
                            cmap='coolwarm', vmin=vmin, vmax=vmax,
                            transform=ccrs.PlateCarree(), shading='auto',
                            rasterized=True)
    ax.coastlines(resolution='110m', linewidth=0.6)
    gl = ax.gridlines(draw_labels=False, linewidth=0, color='grey', linestyle='--')
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    ssp_label = f'SSP{scenario[3]}-{ssp_forcing[scenario]:.1f}'
    co2_label = (f'{co2_2100[scenario]:.0f} ppm' if scenario in co2_2100 else 'not supplied')
    ax.set_title(f'{model} | {ssp_label}\\n2100 reference CO$_2$: {co2_label}',
                 fontsize=10, pad=25)
    if selection.loc[model, 'note']:
        ax.text(0.5, -0.09, selection.loc[model, 'note'], transform=ax.transAxes,
                fontsize=9, ha='center', color='firebrick')

    # add panel labels
    ax.text(-0.06, 1.18, chr(97 + i), transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

for ax in axs.flat[len(sst_celsius):]:
    ax.set_visible(False)

# adjust vertical space between subplots
plt.subplots_adjust(hspace=0.35)
fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.19, wspace=0.18)

# Add a colorbar
cbar_ax = fig.add_axes([0.28, 0.12, 0.44, 0.025])
cbar = fig.colorbar(map_plot, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('SST (°C)')
cbar.set_ticks([0, 5, 10, 15, 20, 25, 30, 35])

fig.suptitle(f'Annual-mean SST | {plot_year} | highest available SSP per model',
             fontsize=15, y=0.98)
fig.text(0.5, 0.035,
         'CO$_2$ labels: standard scenario references for 2100 (Meinshausen et al., 2020; Jiang et al., 2023), not file metadata.\\n'
         'GISS-E2-1-G uses SSP1-2.6 because its SSP5-8.5 SST is absent. Grey: land / missing data.',
         ha='center', fontsize=9)

figure_path = figure_dir / f'SST_highest_SSP_{plot_year}.png'
fig.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.show()
print(figure_path)
''')
nb.cells = cells
nbf.validate(nb)
nbf.write(nb, root / 'SST_highest_SSP.ipynb')
print(root / 'SST_highest_SSP.ipynb')
