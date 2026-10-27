from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import cartopy
import xarray as xr
from toolbox.SST_k_figures import plot_k_maps
root = Path(__file__).resolve().parents[1]
cartopy.config['data_dir'] = root / 'data/cartopy'
files = sorted((root / 'data/processed/SST_GMM_K_comparison').glob('*/*.nc'))
assert len(files) == 6
for path in files:
    with xr.open_dataset(path) as source:
        ds = source.load()
    plot_k_maps(ds, root / 'figures/SST_GMM_K_comparison' / ds.attrs['model'])
    print(ds.attrs['model'], 'map layout refreshed', flush=True)
