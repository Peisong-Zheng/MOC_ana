"""Adapted from Last_deglaciation/toolbox/GMM_functions.py.

Reuses the original gmm_aic_curve and GMM4EOFS fit/predict logic and comments.
Adds reproducible restarts, convergence checks, fitted-model reuse, an explicit
AIC/Delta-AIC elbow rule, and restoration of the masked SST grid.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def gmm_aic_curve(data, max_components=12, covariance_type='full',
                  init_params='kmeans', random_state=42, n_init=5,
                  max_iter=500, reg_covar=1e-6, tol=1e-4):
    """Fit k=1..max_components; return AIC, successive differences, and fits."""
    aic = []
    models = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k,
                              covariance_type=covariance_type,
                              init_params=init_params,
                              random_state=random_state,
                              n_init=n_init, max_iter=max_iter,
                              reg_covar=reg_covar, tol=tol)
        gmm.fit(data)
        if not gmm.converged_:
            raise RuntimeError(f'GMM k={k} did not converge; increase max_iter')
        aic.append(gmm.aic(data))
        models.append(gmm)

    aic = np.array(aic)
    delta_aic = np.diff(aic)      # consecutive differences
    return aic, delta_aic, models


def select_k(aic, marginal_fraction=0.10):
    """Choose an exploratory AIC elbow, checked against two following gains.

    Normalize the best-so-far AIC gain and k to [0, 1]. The elbow maximizes
    gain minus relative k (distance from the endpoint chord). Starting at
    that elbow, accept the first k whose next TWO raw AIC improvements are
    each <= marginal_fraction of the entire tested AIC improvement.
    If no such k exists, use the tested AIC minimum. This is explicitly a
    parsimony heuristic, not the formal AIC-minimizing selection rule.
    """
    aic = np.asarray(aic, dtype=float)
    if len(aic) < 4 or not np.isfinite(aic).all():
        raise ValueError('At least four finite AIC values are required')
    k_aic = int(np.argmin(aic) + 1)
    gain_total = aic[0] - np.min(aic)
    if gain_total <= 2:
        return 1, k_aic, 1, 'No material AIC gain over k=1'
    gain = (aic[0] - np.minimum.accumulate(aic)) / gain_total
    relative_k = np.linspace(0, 1, len(aic))
    k_elbow = int(np.argmax(gain - relative_k) + 1)
    improvements = np.maximum(-np.diff(aic), 0)
    for k in range(max(2, k_elbow), min(k_aic, len(aic) - 2) + 1):
        if np.all(improvements[k - 1:k + 1] <= marginal_fraction * gain_total):
            return k, k_aic, k_elbow, 'AIC elbow with two small subsequent gains'
    return k_aic, k_aic, k_elbow, 'No confirmed elbow; tested AIC minimum'


def GMM4EOFS(data, ds_sst, valid_mask, gmm_model):
    """Map predictions from the already-fitted selected GMM back to the grid."""
    if data.shape[0] != np.count_nonzero(valid_mask):
        raise ValueError('EOF samples do not match the valid-cell mask')

    # get the predicted class labels for each data point
    class_labels = gmm_model.predict(data)
    probabilities = gmm_model.predict_proba(data)
    max_prob = np.amax(probabilities, axis=1)

    # add the class labels to the xarray dataset
    ds = ds_sst[['weight']].copy().assign_coords(lon=ds_sst.lon)
    labels_full = np.full(valid_mask.shape, np.nan)
    labels_full[valid_mask] = class_labels
    reshaped_probs = np.full(valid_mask.shape, np.nan)
    reshaped_probs[valid_mask] = max_prob
    ds['class_label_raw'] = (('lat', 'lon'), labels_full)
    ds['max_probability'] = (('lat', 'lon'), reshaped_probs)
    return ds, probabilities
