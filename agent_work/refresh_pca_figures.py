from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xarray as xr
import SST_GMM as analysis

files = sorted(analysis.output_dir.glob('*/*_PCA_GMM.nc'))
assert len(files) == 6
for path in files:
    with xr.open_dataset(path) as source:
        ds = source.load()
    title = f"{ds.attrs['model']} | {ds.attrs['scenario'].upper()} | {ds.attrs['year_start']}–{ds.attrs['year_end']}"
    analysis.sf.plot_pca(ds, title, analysis.figure_dir / ds.attrs['model'])
    print(ds.attrs['model'], 'PC/EOF figures refreshed with individual EOF color scales.', flush=True)
