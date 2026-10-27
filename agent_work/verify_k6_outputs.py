"""Check saved scientific outputs independently of the plotting pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import pymupdf
import nbformat

root = Path(__file__).resolve().parents[1]
summary = pd.read_csv(root / 'data/processed/SST_GMM_K6/model_summary.csv')
assert len(summary) == 6
for row in summary.itertuples():
    folder = root / 'data/processed/SST_GMM_K6' / row.model
    path = folder / f'{row.model}_{row.scenario}_PCA_GMM.nc'
    with xr.open_dataset(path) as saved, xr.open_dataset(saved.attrs['source_file']) as source:
        ds = saved.load()
        raw = source.SST.sel(year=ds.year).load().astype(float)
    assert ds.attrs['selected_k'] == ds.sizes['class_id'] == 6
    assert ds.attrs['selection_mode'] == 'fixed'
    valid = ds.valid_mask.values.astype(bool)
    np.testing.assert_array_equal(ds.lon, raw.lon)
    np.testing.assert_array_equal(ds.lat, raw.lat)
    np.testing.assert_array_equal(valid, np.isfinite(raw).all(axis=0))
    np.testing.assert_allclose(ds.sst_mean, raw.mean('year') - 273.15, atol=1e-10)
    np.testing.assert_allclose(ds.posterior_probability.values[valid].sum(axis=1), 1, atol=1e-12)
    assert np.isnan(ds.posterior_probability.values[~valid]).all()
    np.testing.assert_array_equal(ds.class_label.values[valid], ds.posterior_probability.values[valid].argmax(axis=1) + 1)
    np.testing.assert_allclose(ds.class_area_fraction.sum(), 1, atol=1e-12)
    assert np.all(np.diff(ds.class_warming.values) >= 0)
    np.testing.assert_allclose(ds.explained_variance.sum(), 1, atol=1e-12)
    eofs = ds.eof.values[valid]
    np.testing.assert_allclose(eofs.T @ eofs, np.eye(6), atol=1e-10)
    weights = ds.pca_weight.broadcast_like(ds.valid_mask)
    residual_fraction = float(((ds.reconstruction_rmse * weights)**2).sum() / ((ds.sst_std.where(ds.valid_mask == 1) * weights)**2).sum())
    np.testing.assert_allclose(residual_fraction, 1 - ds.explained_variance.values[:2].sum(), atol=1e-9)
    features = eofs[::41, :2] * ds.attrs['feature_scale']
    log_components = np.column_stack([
        multivariate_normal.logpdf(features, mean=ds.gmm_mean.values[j], cov=ds.gmm_covariance.values[j]) + np.log(ds.gmm_mixture_weight.values[j])
        for j in range(ds.sizes['class_id'])])
    expected = np.exp(log_components - logsumexp(log_components, axis=1, keepdims=True))
    np.testing.assert_allclose(expected, ds.posterior_probability.values[valid][::41], atol=1e-10)
    curve = pd.read_csv(folder / 'AIC_dAIC.csv')
    np.testing.assert_allclose(curve.delta_aic.values[1:], np.diff(curve.aic), atol=1e-9)
    assert curve.converged.all()
    assert ds.attrs['k_aic_min'] == int(curve.loc[curve.aic.idxmin(), 'k'])
    print(row.model, 'PASS: coordinates, mask, units, EOFs, reconstruction, probabilities, classes, AIC')

nb = nbformat.read(root / 'SST_highest_SSP.ipynb', as_version=4)
nbformat.validate(nb)
assert not any('\u4e00' <= ch <= '\u9fff' for cell in nb.cells for ch in cell.source)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', []))
pngs = sorted((root / 'figures/SST_GMM_K6').rglob('*.png'))
pdfs = sorted((root / 'figures/SST_GMM_K6').rglob('*.pdf'))
assert len(pngs) == len(pdfs) == 43
for path in pdfs:
    assert path.with_suffix('.png').exists()
    with pymupdf.open(path) as doc:
        assert len(doc) == 1 and len(doc[0].get_text()) > 40
print('PASS: English executed notebook; 43 fixed-k PNG/PDF pairs, all PDFs readable.')

comparison = pd.read_csv(root / 'data/processed/SST_GMM_K6/comparison_with_automatic_k.csv')
assert len(comparison) == 6 and (comparison.fixed_k == 6).all()
for row in comparison.itertuples():
    old_dir = root / 'data/processed/SST_GMM' / row.model
    new_dir = root / 'data/processed/SST_GMM_K6' / row.model
    old_curve = pd.read_csv(old_dir / 'AIC_dAIC.csv')
    new_curve = pd.read_csv(new_dir / 'AIC_dAIC.csv')
    np.testing.assert_allclose(old_curve.aic, new_curve.aic, atol=1e-8, rtol=1e-12)
    overlap = pd.read_csv(new_dir / 'baseline_overlap_area_fraction.csv', index_col=0)
    np.testing.assert_allclose(overlap.to_numpy().sum(), 1, atol=1e-12)
    if row.model == 'GISS-E2-1-G':
        assert row.adjusted_rand_index_vs_baseline == 1
        assert abs(row.AIC_fixed_minus_baseline) < 1e-8
print('PASS: fixed k=6, identical AIC sweeps, six comparisons, and identical GISS partition.')
