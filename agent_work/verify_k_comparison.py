from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pymupdf
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

root = Path(__file__).resolve().parents[1]
folder = root / 'data/processed/SST_GMM_K_comparison'
summary = pd.read_csv(folder / 'all_models_K4_K9_summary.csv')
assert len(summary) == 36 and summary.model.nunique() == 6
for model, rows in summary.groupby('model'):
    files = list((folder / model).glob('*.nc'))
    assert len(files) == 1
    with xr.open_dataset(files[0]) as source:
        ds = source.load()
    valid = ds.valid_mask.values.astype(bool)
    weights = ds.weight.broadcast_like(ds.valid_mask).values[valid]
    features = ds.eof.values[valid] * ds.attrs['feature_scale']
    fits = pd.read_csv(folder / model / 'initialization_checks.csv')
    with xr.open_dataset(root / 'data/processed/SST_GMM' / model / f'{model}_{ds.attrs["scenario"]}_PCA_GMM.nc') as old:
        np.testing.assert_array_equal(ds.valid_mask, old.valid_mask)
        np.testing.assert_allclose(ds.eof, old.eof.isel(mode=slice(0, 2)), atol=1e-10)
        np.testing.assert_allclose(ds.pc, old.pc.isel(mode=slice(0, 2)), atol=1e-10)
    previous = None
    for k in range(4, 10):
        result = ds.sel(k=k)
        labels = result.class_label.values[valid].astype(int)
        raw = result.raw_component.values[valid].astype(int)
        probabilities = result.posterior_probability.values[valid, :k]
        mapping = result.display_id_for_raw_component.values[:k].astype(int)
        assert sorted(np.unique(labels)) == list(range(1, k + 1))
        assert sorted(mapping) == list(range(1, k + 1))
        np.testing.assert_array_equal(labels, mapping[raw])
        assert adjusted_rand_score(labels, raw) == 1
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(probabilities.argmax(axis=1) + 1, labels)
        assert np.isnan(result.posterior_probability.values[~valid]).all()
        assert np.isnan(result.posterior_probability.values[:, :, k:]).all()
        np.testing.assert_allclose(result.class_area_fraction.values[:k].sum(), 1, atol=1e-12)
        row = rows.loc[rows.k == k].iloc[0]
        batches = fits.loc[fits.k == k]
        assert batches.converged.all() and batches.selected.sum() == 1
        np.testing.assert_allclose(row.aic, batches.aic.min(), atol=1e-8)
        logs = np.column_stack([
            multivariate_normal.logpdf(features[::101], mean=result.gmm_mean.values[j], cov=result.gmm_covariance.values[j])
            + np.log(result.gmm_mixture_weight.values[j]) for j in range(k)])
        expected = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
        np.testing.assert_allclose(expected, probabilities[::101], atol=1e-10)
        if previous is not None:
            overlap = np.zeros((k - 1, k))
            np.add.at(overlap, (previous - 1, labels - 1), weights)
            saved = pd.read_csv(folder / model / f'overlap_K{k-1}_K{k}.csv', index_col=0).values
            np.testing.assert_allclose(saved, overlap / weights.sum(), atol=1e-12)
            old_ids, new_ids = linear_sum_assignment(-overlap)
            matched = weights[labels == previous].sum() / weights.sum()
            np.testing.assert_allclose(matched, overlap[old_ids, new_ids].sum() / weights.sum(), atol=1e-12)
            np.testing.assert_allclose(matched, row.matched_area_fraction, atol=1e-12)
        previous = labels
    print(model, 'PASS: shared PCA, six K values, best fits, posterior/parameter consistency, maximum-area label matching.')
pdfs = sorted((root / 'figures/SST_GMM_K_comparison').rglob('*.pdf'))
assert len(pdfs) == 12
for path in pdfs:
    assert path.with_suffix('.png').exists()
    with pymupdf.open(path) as pdf:
        assert len(pdf) == 1 and len(pdf[0].get_text()) > 40
print('PASS: 36 classifications and 12 PNG/PDF figure pairs.')
