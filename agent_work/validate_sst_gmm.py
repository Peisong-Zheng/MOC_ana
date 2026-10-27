"""Temporary scientific checks against the original PCA implementation."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolbox import PCA_functions as pf
from toolbox import GMM_functions as gf

spec = importlib.util.spec_from_file_location('legacy_pca', '/Users/pz/VScode/Last_deglaciation/toolbox/PCA_functions.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
rng = np.random.default_rng(1)
years = np.arange(2015, 2035)
lat = [-60, -20, 20, 60]
lon = [-150, -90, -30, 30, 90, 150]
values = rng.normal(size=(20, 4, 6)) + np.arange(20)[:, None, None] * rng.normal(size=(4, 6))[None, :, :]
ds = xr.Dataset({'SST': (('year', 'lat', 'lon'), values)}, coords={'year': years, 'lat': lat, 'lon': lon})
new = pf.apply_weighting(pf.center_data(ds))
old = legacy.apply_weighting(legacy.center_data(ds.rename({'SST': 'sat', 'year': 'age'})))
np.testing.assert_allclose(new.sst_centered_weighted, old.sat_centered_weighted)
new_variance, new_eof, new_pc, valid = pf.sat_PCA(new, n_modes=6)
old_variance, old_eof, old_pc = legacy.sat_PCA(old)
np.testing.assert_allclose(new_variance, old_variance, atol=1e-12)
np.testing.assert_allclose(new_pc[:, :2] @ new_eof[:, :2].T, old_pc @ old_eof.T, atol=1e-12)
np.testing.assert_allclose(new_eof.T @ new_eof, np.eye(6), atol=1e-12)
assert valid.all()

masked = ds.copy(deep=True)
masked.SST.values[:, 0, 0] = np.nan
masked.SST.values[3, 1, 1] = np.nan
masked = pf.apply_weighting(pf.center_data(masked))
variance, eof, pc, valid = pf.sat_PCA(masked, n_modes=6)
assert valid.sum() == 22 and not valid[0, 0] and not valid[1, 1]
assert np.isnan(eof[~valid.ravel()]).all()
features = eof[valid.ravel(), :2] * np.sqrt(valid.sum())
aic, delta, fits = gf.gmm_aic_curve(features, max_components=4, n_init=2)
result, probabilities = gf.GMM4EOFS(features, masked, valid, fits[1])
np.testing.assert_array_equal(result.lon, masked.lon)
np.testing.assert_allclose(probabilities.sum(axis=1), 1)
assert np.isnan(result.class_label_raw.values[~valid]).all()
np.testing.assert_array_equal(result.class_label_raw.values[valid], fits[1].predict(features))
print('PASS: legacy weighting and PCA reconstruction preserved; masked-cell/grid/probability checks passed.')
