"""SST adaptation of Last_deglaciation/toolbox/PCA_functions.py.

Retains center_data, apply_weighting, and sat_PCA's SVD/PC definitions.
Changes: year/SST names, explicit weighting power, complete-cell mask,
economy SVD, configurable retained modes, deterministic paired EOF/PC signs.
"""
import numpy as np
import xarray as xr


def center_data(ds):
    if 'SST' not in ds:
        raise ValueError("The input xarray dataset does not contain an 'SST' variable.")

    # Calculate the mean across the 'year' dimension
    mean_data = ds['SST'].mean(dim='year')

    # Subtract the mean from the data
    centered_data = ds['SST'] - mean_data

    # Add the centered data back to the dataset
    ds = ds.copy()
    ds['sst_centered'] = centered_data
    return ds


def apply_weighting(ds, weight_power=1.0):
    # Check for necessary variables
    for var in ['lat', 'sst_centered']:
        if var not in ds:
            raise ValueError(f"The input xarray dataset does not contain a '{var}' variable.")

    # Calculate the weight
    weight = np.cos(np.deg2rad(ds['lat']))

    # Set weight to 0 if it is smaller than 0
    weight = xr.where(weight < 0, 0, weight)

    # Add the weight to the dataset
    ds = ds.copy()
    ds['weight'] = weight

    # Legacy weighting uses cos(lat); power=0.5 gives conventional area-metric EOFs.
    # The default power=1 preserves the original analysis (cos² variance metric).
    ds['pca_weight'] = weight ** weight_power
    ds['sst_centered_weighted'] = ds['sst_centered'] * ds['pca_weight']
    return ds


def sat_PCA(ds, n_modes=6):
    """Return explained variance, masked spatial EOFs, PCs, and valid-cell mask.

    EOFs are unit-norm vectors of the weighted field; PCs reconstruct that
    field, not the unweighted SST. A cell must be finite in every input year.
    """
    # Reshape the sat_centered_weighted data
    sst_centered_weighted = ds['sst_centered_weighted'].transpose('year', 'lat', 'lon').values
    original_shape = sst_centered_weighted.shape
    sst_centered_weighted = sst_centered_weighted.reshape(original_shape[0], -1).T
    valid = np.isfinite(sst_centered_weighted).all(axis=1)
    if valid.sum() < n_modes or original_shape[0] <= n_modes:
        raise ValueError('Insufficient complete cells or years for PCA')
    sst_centered_weighted = sst_centered_weighted[valid].astype(np.float64)

    # Perform Singular Value Decomposition
    u, s, vh = np.linalg.svd(sst_centered_weighted, full_matrices=False)
    if s[0] <= 0:
        raise ValueError('The weighted SST field has no temporal variance')

    # Extract the first EOFs
    eofs = u[:, :n_modes]

    # Calculate the Principal Components
    pcs = sst_centered_weighted.T.dot(eofs)

    # Calculate the variance explained by the PCs
    exp_variance = s**2 / np.sum(s**2)

    # Flip EOF and PC together so the largest-magnitude loading is positive.
    signs = np.sign(eofs[np.argmax(np.abs(eofs), axis=0), np.arange(n_modes)])
    eofs = eofs * signs
    pcs = pcs * signs
    eofs_full = np.full((valid.size, n_modes), np.nan)
    eofs_full[valid] = eofs
    return exp_variance, eofs_full, pcs, valid.reshape(original_shape[1:])
