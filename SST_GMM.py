"""Cluster each model's highest available SSP SST using the legacy EOF–GMM workflow.

Run with .venv/bin/python SST_GMM.py from any directory. Settings are below.
The copied/adapted PCA and GMM algorithms live in toolbox/. All figures are
saved as PDF and PNG; labeled numerical outputs are saved as NetCDF and CSV.
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
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from toolbox import PCA_functions as pf
from toolbox import GMM_functions as gf
from toolbox import SST_figures as sf


project_dir = Path(__file__).resolve().parent
input_dir = Path('/Users/pz/CMIP_AMOC_SST')
output_dir = project_dir / 'data' / 'processed' / 'SST_GMM'
figure_dir = project_dir / 'figures' / 'SST_GMM'
cartopy.config['data_dir'] = project_dir / 'data' / 'cartopy'

first_year, last_year = 2015, 2100
n_pca_modes = 6                # Retained for diagnostics; full spectrum is also saved.
n_gmm_eofs = 2                 # Same EOF-loading features as Last_deglaciation.
pca_weight_power = 1.0         # Original cos(lat); 0.5 would use sqrt(cos(lat)).
max_components = 12
marginal_fraction = 0.10        # Elbow check: each of the next two gains <=10% total gain.
random_state = 42
n_init = 5
max_iter = 500
tol = 1e-4
reg_covar = 1e-6               # On EOFs multiplied by one common sqrt(N) factor.
stability_seeds = [43, 44, 45]
stability_n_init = 3
mean_window = 20               # Early/late windows for class curves and label ordering.

ssp_forcing = {'ssp119': 1.9, 'ssp126': 2.6, 'ssp245': 4.5,
               'ssp370': 7.0, 'ssp434': 3.4, 'ssp460': 6.0, 'ssp585': 8.5}
# Rounded standard 2100 pathway references, not concentrations in the SST files.
# Jiang et al. (2023), Table 1: https://doi.org/10.1029/2022MS003563
co2_2100 = {'ssp126': 446.0, 'ssp585': 1135.0}


def select_sst_files():
    selected = {}
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        candidates = {}
        for scenario_dir in sorted(model_dir.glob('ssp*')):
            files = sorted((scenario_dir / 'SST').glob('*.nc'))
            if files:
                if len(files) != 1 or scenario_dir.name not in ssp_forcing:
                    raise ValueError(f'Check scenario or file/member selection in {scenario_dir}')
                candidates[scenario_dir.name] = files[0]
        if not candidates:
            raise FileNotFoundError(f'No future SST for {model_dir.name}')
        scenario = max(candidates, key=ssp_forcing.get)
        selected[model_dir.name] = (scenario, candidates[scenario])
    if not selected:
        raise FileNotFoundError(f'No model directories in {input_dir}')
    return selected


def class_statistics(ds_sst, result, probabilities, valid):
    """Rank classes by warming and calculate means with cos(lat) area weights."""
    k = probabilities.shape[1]
    curves = []
    areas = []
    for class_label in range(k):
        class_mask = result.class_label_raw == class_label
        # Weighted Average SST Curve
        ds_subset = ds_sst.where(class_mask)
        weights_broadcasted = ds_sst['weight'].broadcast_like(ds_subset['SST'])
        weights_broadcasted = weights_broadcasted.where(ds_subset.SST.notnull(), 0)
        sum_weighted_sst = (ds_subset['SST'] * weights_broadcasted).sum(dim=['lat', 'lon'])
        sum_weight_sst = weights_broadcasted.sum(dim=['lat', 'lon'])
        weighted_avg_sst = sum_weighted_sst / sum_weight_sst
        if not bool(np.isfinite(weighted_avg_sst).all()):
            raise ValueError(f'Empty or invalid hard-assigned class {class_label}')
        curves.append(weighted_avg_sst)
        areas.append(float(ds_sst.weight.broadcast_like(class_mask).where(class_mask, 0).sum()))
    curves = xr.concat(curves, dim='class_id')
    warming = curves.isel(year=slice(-mean_window, None)).mean('year') - curves.isel(year=slice(0, mean_window)).mean('year')
    order = np.argsort(warming.values, kind='stable')
    curves = curves.isel(class_id=order).assign_coords(class_id=np.arange(1, k + 1))
    mapping = np.empty(k, dtype=int)
    mapping[order] = np.arange(1, k + 1)
    labels = np.full(valid.shape, np.nan)
    labels[valid] = mapping[result.class_label_raw.values[valid].astype(int)]
    posterior = np.full((*valid.shape, k), np.nan)
    posterior[valid] = probabilities[:, order]
    result['class_label'] = (('lat', 'lon'), labels)
    result['posterior_probability'] = (('lat', 'lon', 'class_id'), posterior)
    result['class_sst'] = curves
    result['class_anomaly'] = curves - curves.isel(year=slice(0, mean_window)).mean('year')
    result['class_warming'] = ('class_id', warming.values[order])
    result['class_area_fraction'] = ('class_id', np.asarray(areas)[order] / np.sum(areas))
    result['raw_component_for_class'] = ('class_id', order)
    return result, order


def check_initializations(features, reference, selected_k):
    rows = []
    reference_labels = reference.predict(features)
    for seed in stability_seeds:
        gmm = GaussianMixture(n_components=selected_k, covariance_type='full',
                              random_state=seed, n_init=stability_n_init,
                              max_iter=max_iter, tol=tol, reg_covar=reg_covar)
        gmm.fit(features)
        if not gmm.converged_:
            raise RuntimeError(f'Stability fit at seed={seed} did not converge')
        labels = gmm.predict(features)
        overlap = confusion_matrix(reference_labels, labels, labels=np.arange(selected_k))
        reference_index, trial_index = linear_sum_assignment(-overlap)
        rows.append({'seed': seed, 'aic': gmm.aic(features),
                     'adjusted_rand_index': adjusted_rand_score(reference_labels, labels),
                     'aligned_agreement': overlap[reference_index, trial_index].sum() / len(labels),
                     'converged': gmm.converged_, 'n_iter': gmm.n_iter_})
    return pd.DataFrame(rows)


def analyze_model(model, scenario, path, years):
    print(f'{model} | {scenario} | loading {years[0]}–{years[-1]}', flush=True)
    model_output = output_dir / model
    model_figures = figure_dir / model
    model_output.mkdir(parents=True, exist_ok=True)
    model_figures.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(path) as ds:
        ds_sst = ds[['SST']].sel(year=years).load()
        units = ds.SST.attrs.get('units', ds.attrs.get('units', ''))
    if set(ds_sst.SST.dims) != {'year', 'lat', 'lon'}:
        raise ValueError(f'{model}: unexpected SST dimensions')
    ds_sst = ds_sst.transpose('year', 'lat', 'lon')
    for coord in ['lat', 'lon']:
        if ds_sst[coord].ndim != 1 or not np.all(np.diff(ds_sst[coord]) > 0):
            raise ValueError(f'{model}: check {coord} coordinates')
    if units.strip().lower() in ['k', 'kelvin']:
        ds_sst['SST'] = ds_sst.SST.astype('float64') - 273.15
    elif units.strip().lower() not in ['degc', 'degrees_celsius', 'degree_celsius', '°c']:
        raise ValueError(f'{model}: unrecognized SST units {units!r}')
    ds_sst.SST.attrs['units'] = 'degC'
    if bool(np.isinf(ds_sst.SST).any()):
        raise ValueError(f'{model}: infinite SST values')

    # 1. Center the Data
    ds_sst = pf.center_data(ds_sst)
    # 2. Weight the Data
    ds_sst = pf.apply_weighting(ds_sst, weight_power=pca_weight_power)
    # 4. Perform PCA
    exp_variance, eofs, pcs, valid = pf.sat_PCA(ds_sst, n_modes=n_pca_modes)
    # All modes get the SAME numerical scale; no relative feature rescaling/whitening.
    feature_scale = np.sqrt(valid.sum())
    features = eofs[valid.ravel(), :n_gmm_eofs] * feature_scale
    print(f'  {valid.sum()} complete cells; EOF1+2 explain {exp_variance[:2].sum():.1%}; fitting k=1..{max_components}', flush=True)
    aic, delta_aic, models = gf.gmm_aic_curve(
        features, max_components=max_components, random_state=random_state,
        n_init=n_init, max_iter=max_iter, reg_covar=reg_covar, tol=tol)
    selected_k, k_aic, k_elbow, selection_reason = gf.select_k(aic, marginal_fraction)
    gmm = models[selected_k - 1]
    print(f'  selected k={selected_k}; AIC minimum k={k_aic}; {selection_reason}', flush=True)
    table = pd.DataFrame({'k': np.arange(1, len(aic) + 1), 'aic': aic,
                          'delta_aic': np.r_[np.nan, delta_aic],
                          'aic_above_min': aic - aic.min(),
                          'converged': [m.converged_ for m in models],
                          'n_iter': [m.n_iter_ for m in models]})
    table.to_csv(model_output / 'AIC_dAIC.csv', index=False)
    result, probabilities = gf.GMM4EOFS(features, ds_sst, valid, gmm)
    result, order = class_statistics(ds_sst, result, probabilities, valid)
    result = result.assign_coords(mode=np.arange(1, n_pca_modes + 1),
                                   spectrum_mode=np.arange(1, len(exp_variance) + 1),
                                   feature=np.arange(1, n_gmm_eofs + 1),
                                   feature_2=np.arange(1, n_gmm_eofs + 1))
    result['eof'] = (('lat', 'lon', 'mode'), eofs.reshape(*valid.shape, n_pca_modes))
    result['pc'] = (('year', 'mode'), pcs)
    result['explained_variance'] = ('spectrum_mode', exp_variance)
    result['valid_mask'] = (('lat', 'lon'), valid.astype('int8'))
    result['sst_mean'] = ds_sst.SST.mean('year')
    result['sst_std'] = ds_sst.SST.std('year')
    result['finite_year_count'] = np.isfinite(ds_sst.SST).sum('year')
    result['pca_weight'] = ds_sst.pca_weight
    result['gmm_mean'] = (('class_id', 'feature'), gmm.means_[order])
    result['gmm_covariance'] = (('class_id', 'feature', 'feature_2'), gmm.covariances_[order])
    result['gmm_mixture_weight'] = ('class_id', gmm.weights_[order])

    # Reconstruct the unweighted anomaly using exactly the two clustering EOFs.
    reconstructed = pcs[:, :n_gmm_eofs] @ eofs[valid.ravel(), :n_gmm_eofs].T
    spatial_weight = ds_sst.pca_weight.broadcast_like(ds_sst.SST.isel(year=0)).values[valid]
    reconstructed = reconstructed / spatial_weight
    centered = ds_sst.sst_centered.values[:, valid]
    mse = np.mean((centered - reconstructed)**2, axis=0)
    variance = np.mean(centered**2, axis=0)
    rmse = np.full(valid.shape, np.nan)
    represented = np.full(valid.shape, np.nan)
    rmse[valid] = np.sqrt(mse)
    represented[valid] = np.divide(variance - mse, variance, out=np.full_like(variance, np.nan), where=variance > 1e-12)
    result['reconstruction_rmse'] = (('lat', 'lon'), rmse)
    result['retained_variance_fraction'] = (('lat', 'lon'), represented)
    lon, lat = np.meshgrid(ds_sst.lon, ds_sst.lat)
    distance = (lat[valid] - 30)**2 + (((lon[valid] + 45 + 180) % 360) - 180)**2
    point = int(np.argmin(distance))
    result['example_anomaly'] = ('year', centered[:, point])
    result['example_reconstruction'] = ('year', reconstructed[:, point])
    result.attrs = {
        'model': model, 'scenario': scenario, 'source_file': str(path),
        'year_start': int(years[0]), 'year_end': int(years[-1]),
        'selected_k': selected_k, 'k_aic_min': k_aic, 'k_elbow': k_elbow,
        'selection_reason': selection_reason,
        'aic_min_at_boundary': int(k_aic == max_components),
        'max_components': max_components, 'marginal_fraction': marginal_fraction,
        'selected_aic_above_min': float(aic[selected_k - 1] - aic.min()),
        'n_gmm_eofs': n_gmm_eofs, 'pca_weight_power': pca_weight_power,
        'feature_scale': float(feature_scale), 'random_state': random_state,
        'n_init': n_init, 'max_iter': max_iter, 'tol': tol, 'reg_covar': reg_covar,
        'reference_co2_2100_ppm': co2_2100.get(scenario, np.nan),
        'co2_source': 'Jiang et al. (2023), Table 1; doi:10.1029/2022MS003563',
        'co2_status': 'Standard 2100 scenario reference, not model-file metadata',
        'input_units': units, 'example_lat': float(lat[valid][point]),
        'example_lon': float(lon[valid][point]), 'class_mean_window_years': mean_window,
        'method': 'Temporal centering; legacy cos(lat) weighting; GMM on first two weighted EOF loadings',
        'gmm_sampling': 'Each complete grid cell has equal likelihood weight; spatial samples are correlated',
        'label_order': 'Ascending area-weighted last-20-year minus first-20-year SST; within-model only',
        'eof_sign': 'Largest absolute loading positive; PC sign flipped together; modes not aligned across models',
    }
    for name in ['class_sst', 'class_anomaly', 'class_warming', 'sst_mean', 'sst_std', 'reconstruction_rmse', 'example_anomaly', 'example_reconstruction']:
        result[name].attrs['units'] = 'degC'
    result.eof.attrs['description'] = 'Unit-norm loading of the weighted SST anomaly field'
    result.pc.attrs['description'] = 'Temporal coefficient paired with weighted EOF; no display normalization applied'
    result.class_label.attrs['description'] = '1-based within-model warming rank; NaN outside complete-cell mask'
    stability = check_initializations(features, gmm, selected_k)
    stability.to_csv(model_output / 'initialization_stability.csv', index=False)
    class_table = result[['class_warming', 'class_area_fraction', 'raw_component_for_class']].to_dataframe().reset_index()
    class_table.to_csv(model_output / 'class_summary.csv', index=False)
    encoding = {name: {'zlib': True, 'complevel': 4} for name in result.data_vars}
    result.to_netcdf(model_output / f'{model}_{scenario}_PCA_GMM.nc', encoding=encoding)
    title = f'{model} | {scenario.upper()} | {years[0]}–{years[-1]}'
    sf.plot_pca(result, title, model_figures)
    sf.plot_selection(result, table, features, gmm, title, model_figures)
    sf.plot_classes(result, title, model_figures)
    sf.plot_checks(result, stability, title, model_figures)
    summary = {'model': model, 'scenario': scenario, 'selected_k': selected_k,
               'k_aic_min': k_aic, 'aic_min_at_upper_bound': k_aic == max_components,
               'selection_reason': selection_reason,
               'aic_gain_captured': float((aic[0] - aic[selected_k - 1]) / (aic[0] - aic.min())),
               'aic_above_min': float(aic[selected_k - 1] - aic.min()),
               'EOF1_EOF2_variance_fraction': float(exp_variance[:2].sum()),
               'valid_cells': int(valid.sum()),
               'minimum_initialization_ARI': float(stability.adjusted_rand_index.min()),
               'fraction_probability_below_0.8': float(np.mean(probabilities.max(axis=1) < 0.8))}
    print(f'  saved maps and diagnostics; minimum restart ARI={summary["minimum_initialization_ARI"]:.3f}', flush=True)
    return result, summary


def main():
    if n_gmm_eofs != 2 or n_pca_modes != 6:
        raise ValueError('The current diagnostic layout expects 2 GMM EOFs and 6 retained modes')
    selected = select_sst_files()
    common = None
    for model, (scenario, path) in selected.items():
        with xr.open_dataset(path) as ds:
            if not np.all(np.diff(ds.year.values) > 0):
                raise ValueError(f'{model}: duplicate or unordered years')
            years = set(ds.year.values.tolist())
        common = years if common is None else common & years
    years = sorted(year for year in common if first_year <= year <= last_year)
    if len(years) < 2 * mean_window or not np.all(np.diff(years) == 1):
        raise ValueError('Need consecutive common years spanning both averaging windows')
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    settings = {name: globals()[name] for name in [
        'first_year', 'last_year', 'n_pca_modes', 'n_gmm_eofs', 'pca_weight_power',
        'max_components', 'marginal_fraction', 'random_state', 'n_init', 'max_iter',
        'tol', 'reg_covar', 'stability_seeds', 'stability_n_init', 'mean_window']}
    settings['actual_common_years'] = years
    (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    results, summaries = {}, []
    # Keep BLAS from oversubscribing the many small covariance calculations.
    with threadpool_limits(limits=2):
        for model, (scenario, path) in selected.items():
            result, summary = analyze_model(model, scenario, path, years)
            results[model] = result
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(output_dir / 'model_summary.csv', index=False)
    sf.plot_overview(results, figure_dir)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'Figures: {figure_dir}\nNumerical outputs: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
