"""Compare K=4..9 spatial SST classifications, one six-panel map per model.

Run: .venv/bin/python SST_GMM_K_comparison.py
Reuses the existing PCA/GMM algorithms. No previous output files are needed.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import cartopy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from toolbox import PCA_functions as pf
from toolbox import GMM_functions as gf
from toolbox import SST_k_figures as sf

project_dir = Path(__file__).resolve().parent
input_dir = Path('/Users/pz/CMIP_AMOC_SST')
output_dir = project_dir / 'data' / 'processed' / 'SST_GMM_K_comparison'
figure_dir = project_dir / 'figures' / 'SST_GMM_K_comparison'
cartopy.config['data_dir'] = project_dir / 'data' / 'cartopy'

k_values = np.arange(4, 10)
first_year, last_year = 2015, 2100
n_gmm_eofs = 2
pca_weight_power = 1.0          # Preserve the original cos(lat) weighting.
seed_starts = {42: 5, 43: 3, 44: 3}  # Keep the best fit across all three seed batches.
max_iter, tol, reg_covar = 500, 1e-4, 1e-6
mean_window = 20               # Used only to order the colors in the first K panel.
confidence_threshold = 0.8
ssp_forcing = {'ssp119': 1.9, 'ssp126': 2.6, 'ssp245': 4.5,
               'ssp370': 7.0, 'ssp434': 3.4, 'ssp460': 6.0, 'ssp585': 8.5}
co2_2100 = {'ssp126': 446.0, 'ssp585': 1135.0}  # Jiang et al. (2023), Table 1.


def select_inputs():
    selected = {}
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        scenarios = sorted(path for path in model_dir.glob('ssp*') if path.is_dir())
        if not scenarios:
            continue
        candidates = {}
        for scenario in scenarios:
            files = sorted((scenario / 'SST').glob('*.nc'))
            if files:
                if len(files) != 1 or scenario.name not in ssp_forcing:
                    raise ValueError(f'Check scenario/member/time-chunk selection: {scenario}')
                candidates[scenario.name] = files[0]
        if not candidates:
            raise FileNotFoundError(f'No future SST for {model_dir.name}')
        scenario = max(candidates, key=ssp_forcing.get)
        selected[model_dir.name] = (scenario, candidates[scenario])
    if not selected:
        raise FileNotFoundError(f'No model SST files under {input_dir}')
    return selected


def align_colors(previous, labels, weights, warming, k):
    """Return raw-component -> display-ID mapping and area overlap matrix.

    Only IDs change: fitted parameters, probabilities, and partitions do not.
    A rectangular Hungarian assignment maximizes preserved area. The unmatched
    component gets the new ID. This does not impose a nested hierarchy.
    """
    if previous is None:
        area = np.bincount(labels, weights=weights, minlength=k)
        change = np.bincount(labels, weights=weights * warming, minlength=k) / area
        mapping = np.empty(k, dtype=int)
        mapping[np.argsort(change, kind='stable')] = np.arange(1, k + 1)
        return mapping, None
    previous_k = k - 1
    overlap = np.zeros((previous_k, k))
    np.add.at(overlap, (previous - 1, labels), weights)
    old_ids, raw_ids = linear_sum_assignment(-overlap)
    mapping = np.zeros(k, dtype=int)
    mapping[raw_ids] = old_ids + 1
    mapping[mapping == 0] = k
    return mapping, overlap[:, np.argsort(mapping)]


def analyze_model(model, scenario, path, years):
    print(f'{model} | {scenario} | {years[0]}–{years[-1]}', flush=True)
    model_output = output_dir / model
    model_figures = figure_dir / model
    model_output.mkdir(parents=True, exist_ok=True)
    model_figures.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(path) as source:
        ds_sst = source[['SST']].sel(year=years).load()
        units = source.SST.attrs.get('units', source.attrs.get('units', ''))
    if set(ds_sst.SST.dims) != {'year', 'lat', 'lon'}:
        raise ValueError(f'{model}: unexpected SST dimensions')
    ds_sst = ds_sst.transpose('year', 'lat', 'lon')
    for coord in ['lat', 'lon']:
        if ds_sst[coord].ndim != 1 or not np.all(np.diff(ds_sst[coord]) > 0):
            raise ValueError(f'{model}: invalid {coord} coordinates')
    ds_sst['SST'] = ds_sst.SST.astype(float)
    if units.lower() in ['k', 'kelvin']:
        ds_sst['SST'] = ds_sst.SST - 273.15
    elif units.lower() not in ['degc', 'degrees_celsius', 'degree_celsius', '°c']:
        raise ValueError(f'Unknown SST units: {units}')
    if bool(np.isinf(ds_sst.SST).any()):
        raise ValueError(f'{model}: infinite SST values')
    ds_sst.SST.attrs['units'] = 'degC'

    # 1. Center the Data
    ds_sst = pf.center_data(ds_sst)
    # 2. Weight the Data
    ds_sst = pf.apply_weighting(ds_sst, weight_power=pca_weight_power)
    # 4. Perform PCA, once per model; all K values use the same two EOFs.
    variance, eofs, pcs, valid = pf.sat_PCA(ds_sst, n_modes=n_gmm_eofs)
    feature_scale = np.sqrt(valid.sum())
    features = eofs[valid.ravel()] * feature_scale
    area = ds_sst.weight.broadcast_like(ds_sst.SST.isel(year=0)).values[valid]
    warming = (ds_sst.SST.isel(year=slice(-mean_window, None)).mean('year')
               - ds_sst.SST.isel(year=slice(0, mean_window)).mean('year')).values[valid]
    shape = valid.shape
    nk, max_k = len(k_values), int(k_values.max())
    labels_all = np.full((nk, *shape), np.nan)
    raw_all = np.full_like(labels_all, np.nan)
    probability_all = np.full((nk, *shape, max_k), np.nan)
    maximum_all = np.full_like(labels_all, np.nan)
    mapping_all = np.full((nk, max_k), np.nan)
    area_all = np.full((nk, max_k), np.nan)
    mean_all = np.full((nk, max_k, n_gmm_eofs), np.nan)
    covariance_all = np.full((nk, max_k, n_gmm_eofs, n_gmm_eofs), np.nan)
    mixture_all = np.full((nk, max_k), np.nan)
    summary_rows, fit_rows = [], []
    previous = None

    for index, k in enumerate(k_values):
        k = int(k)
        candidates = []
        seeds = list(seed_starts)
        for seed in seeds:
            aic, _, fits = gf.gmm_aic_curve(
                features, min_components=k, max_components=k,
                random_state=seed, n_init=seed_starts[seed],
                max_iter=max_iter, reg_covar=reg_covar, tol=tol)
            candidates.append((float(aic[0]), fits[0]))
        best = int(np.argmin([value for value, fit in candidates]))
        aic, gmm = candidates[best]
        mapped, probabilities = gf.GMM4EOFS(features, ds_sst, valid, gmm)
        raw = mapped.class_label_raw.values[valid].astype(int)
        if len(np.unique(raw)) != k:
            raise ValueError(f'{model}, K={k}: a component has no hard-assigned cells')
        mapping, overlap = align_colors(previous, raw, area, warming, k)
        labels = mapping[raw]
        order = np.argsort(mapping)
        labels_all[index][valid] = labels
        raw_all[index][valid] = raw
        maximum_all[index][valid] = probabilities.max(axis=1)
        probability_all[index, :, :, :k][valid] = probabilities[:, order]
        mapping_all[index, :k] = mapping
        area_all[index, :k] = np.bincount(labels - 1, weights=area, minlength=k) / area.sum()
        mean_all[index, :k] = gmm.means_[order]
        covariance_all[index, :k] = gmm.covariances_[order]
        mixture_all[index, :k] = gmm.weights_[order]
        retained = np.nan
        if previous is not None:
            retained = float(area[labels == previous].sum() / area.sum())
            pd.DataFrame(overlap / area.sum(), index=np.arange(1, k),
                         columns=np.arange(1, k + 1)).rename_axis('previous_matched_id').to_csv(
                             model_output / f'overlap_K{k - 1}_K{k}.csv')
        ari_values = []
        for seed, (fit_aic, fit) in zip(seeds, candidates):
            ari = adjusted_rand_score(raw, fit.predict(features))
            ari_values.append(ari)
            fit_rows.append({'model': model, 'k': k, 'seed': seed, 'n_init': seed_starts[seed],
                             'aic': fit_aic, 'converged': fit.converged_, 'n_iter': fit.n_iter_,
                             'selected': seed == seeds[best], 'ari_to_selected': ari})
        summary_rows.append({
            'model': model, 'scenario': scenario, 'k': k, 'aic': aic,
            'selected_seed': seeds[best], 'minimum_restart_ARI': min(ari_values),
            'low_confidence_area_fraction': float(area[probabilities.max(axis=1) < confidence_threshold].sum() / area.sum()),
            'smallest_class_area_fraction': float(area_all[index, :k].min()),
            'matched_area_fraction': retained,
            'ARI_vs_previous_k': adjusted_rand_score(previous, labels) if previous is not None else np.nan,
        })
        previous = labels
        print(f'  K={k}: best seed={seeds[best]}, AIC={aic:.1f}, min restart ARI={min(ari_values):.3f}', flush=True)

    summary = pd.DataFrame(summary_rows)
    summary['delta_aic'] = summary.aic.diff()  # AIC(K) - AIC(K-1), undefined at K=4.
    fits = pd.DataFrame(fit_rows)
    summary.to_csv(model_output / 'K4_K9_summary.csv', index=False)
    fits.to_csv(model_output / 'initialization_checks.csv', index=False)
    result = xr.Dataset(
        {'class_label': (('k', 'lat', 'lon'), labels_all),
         'raw_component': (('k', 'lat', 'lon'), raw_all),
         'posterior_probability': (('k', 'lat', 'lon', 'class_id'), probability_all),
         'max_probability': (('k', 'lat', 'lon'), maximum_all),
         'display_id_for_raw_component': (('k', 'component'), mapping_all),
         'class_area_fraction': (('k', 'class_id'), area_all),
         'gmm_mean': (('k', 'class_id', 'feature'), mean_all),
         'gmm_covariance': (('k', 'class_id', 'feature', 'feature_2'), covariance_all),
         'gmm_mixture_weight': (('k', 'class_id'), mixture_all),
         'valid_mask': (('lat', 'lon'), valid.astype('int8')),
         'eof': (('lat', 'lon', 'mode'), eofs.reshape(*shape, n_gmm_eofs)),
         'pc': (('year', 'mode'), pcs),
         'explained_variance': ('spectrum_mode', variance),
         'weight': ds_sst.weight,
         'aic': ('k', summary.aic.values),
         'selected_seed': ('k', summary.selected_seed.values)},
        coords={'k': k_values, 'lat': ds_sst.lat, 'lon': ds_sst.lon, 'year': years,
                'class_id': np.arange(1, max_k + 1), 'component': np.arange(max_k),
                'mode': np.arange(1, n_gmm_eofs + 1), 'spectrum_mode': np.arange(1, len(variance) + 1),
                'feature': np.arange(1, n_gmm_eofs + 1), 'feature_2': np.arange(1, n_gmm_eofs + 1)},
        attrs={'model': model, 'scenario': scenario, 'source_file': str(path),
               'year_start': int(years[0]), 'year_end': int(years[-1]),
               'two_eof_variance_fraction': float(variance[:2].sum()),
               'feature_scale': float(feature_scale), 'pca_weight_power': pca_weight_power,
               'seed_starts': json.dumps(seed_starts), 'max_iter': max_iter, 'tol': tol, 'reg_covar': reg_covar,
               'co2_2100_reference_ppm': co2_2100.get(scenario, np.nan),
               'co2_reference': 'Jiang et al. (2023), Table 1, doi:10.1029/2022MS003563; not file metadata',
               'selection': 'Lowest AIC converged fit among seed batches at each fixed K',
               'class_ids': 'K=4: warming rank; K>4: maximum cosine-area overlap with preceding K, unmatched class gets new ID',
               'caution': 'Colors are model-specific display correspondences; GMM partitions need not be nested',
               'pca_definition': 'Temporal centering, cos(lat) multiplier, first two unit-norm weighted EOFs',
               'likelihood': 'Equal grid-cell sample weights; spatial correlation is not modeled',
               'probability_padding': 'Absent class IDs > K and excluded cells are NaN'})
    result.to_netcdf(model_output / f'{model}_{scenario}_K4_K9.nc',
                     encoding={name: {'zlib': True, 'complevel': 4} for name in result.data_vars})
    sf.plot_k_maps(result, model_figures)
    sf.plot_k_checks(result, summary, fits, model_figures)
    print(f'  saved spatial comparison and diagnostics: {model_figures}', flush=True)
    return summary


def main():
    if not np.array_equal(k_values, np.arange(4, 10)):
        raise ValueError('This six-panel experiment is configured for K=4..9')
    selected = select_inputs()
    common = None
    for model, (scenario, path) in selected.items():
        with xr.open_dataset(path) as source:
            years = source.year.values
            if not np.all(np.diff(years) > 0):
                raise ValueError(f'{model}: years must be unique and increasing')
            common = set(years.tolist()) if common is None else common & set(years.tolist())
    years = sorted(year for year in common if first_year <= year <= last_year)
    if len(years) < 2 * mean_window or not np.all(np.diff(years) == 1):
        raise ValueError('Need consecutive common years spanning both mean windows')
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    settings = {'k_values': k_values.tolist(), 'years': years, 'n_gmm_eofs': n_gmm_eofs,
                'pca_weight_power': pca_weight_power, 'seed_starts': seed_starts,
                'max_iter': max_iter, 'tol': tol, 'reg_covar': reg_covar,
                'mean_window': mean_window, 'confidence_threshold': confidence_threshold}
    (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    summaries = []
    with threadpool_limits(limits=2):
        for model, (scenario, path) in selected.items():
            summaries.append(analyze_model(model, scenario, path, years))
            pd.concat(summaries, ignore_index=True).to_csv(output_dir / 'all_models_K4_K9_summary.csv', index=False)
    print(f'Complete. Figures: {figure_dir}', flush=True)


if __name__ == '__main__':
    main()
