"""PCA/GMM diagnostics, adapted from Last_deglaciation plotting routines."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs


CLASS_COLORS = ['#7FC97F', '#FDC086', '#F0027F', '#4682B4', '#666666',
                '#A6761D', '#7570B3', '#E6AB02', '#1B9E77', '#E7298A',
                '#66A61E', '#A6CEE3']


def save_figure(fig, path):
    for suffix in ['.png', '.pdf']:
        fig.savefig(path.with_suffix(suffix), dpi=250, bbox_inches='tight')
    plt.close(fig)


def map_field(ax, ds, field, **kwargs):
    ax.set_global()
    ax.set_facecolor('0.88')
    # Using pcolormesh
    image = ax.pcolormesh(ds.lon, ds.lat, field, transform=ccrs.PlateCarree(),
                         shading='auto', rasterized=True, **kwargs)
    ax.coastlines(resolution='110m', linewidth=0.6)
    return image


def class_colors(k):
    colors = CLASS_COLORS[:k] if k <= len(CLASS_COLORS) else plt.get_cmap('turbo')(np.linspace(0, 1, k))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, k + 1.5), k)
    return cmap, norm


def plot_pca(ds, title, output_dir):
    variance = ds.explained_variance.values
    n_modes = ds.sizes['mode']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    mode_numbers = np.arange(1, len(variance) + 1)
    axes[0, 0].bar(mode_numbers[:15], variance[:15] * 100, color='steelblue')
    axes[0, 0].set(xlabel='EOF mode', ylabel='Explained variance (%)', title='Weighted-field variance spectrum')
    axes[0, 1].plot(mode_numbers, np.cumsum(variance) * 100, 'k.-')
    axes[0, 1].axvline(ds.attrs['n_gmm_eofs'], color='firebrick', ls='--', label='GMM feature count')
    axes[0, 1].set(xlabel='Number of EOF modes', ylabel='Cumulative variance (%)', ylim=(0, 101))
    axes[0, 1].legend()
    for j in range(n_modes):
        ax = axes[1, j // 3]
        # plot the PCs against the year
        pc = ds.pc.isel(mode=j)
        ax.plot(ds.year, pc / pc.std('year'), label=f'PC{j + 1} ({variance[j] * 100:.1f}%)')
    for ax in axes[1]:
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set(xlabel='Year CE', ylabel='PC / temporal standard deviation')
        ax.legend(fontsize=8)
    fig.suptitle(title + '\nPCA diagnostics; GMM uses EOF1 and EOF2', fontsize=12)
    save_figure(fig, output_dir / 'PCs_variance')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), subplot_kw={'projection': ccrs.Robinson()})
    for j, ax in enumerate(axes.flat):
        if j >= n_modes:
            ax.set_visible(False)
            continue
        eof = ds.eof.isel(mode=j)
        limit = float(abs(eof).max())
        image = map_field(ax, ds, eof, cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set_title(f'EOF{j + 1} | {variance[j] * 100:.2f}%', fontsize=10)
        cbar = fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.75, pad=0.06)
        cbar.set_label('Weighted loading (unit norm)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(top=0.87, bottom=0.08, wspace=0.08, hspace=0.12)
    fig.suptitle(title + '\nEOFs of temporally centered, latitude-weighted SST | individual color scales', fontsize=12)
    save_figure(fig, output_dir / 'EOF_maps')


def plot_selection(ds, table, features, gmm, title, output_dir):
    k = ds.attrs['selected_k']
    k_aic = ds.attrs['k_aic_min']
    cmap, norm = class_colors(k)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    # AIC
    axes[0, 0].plot(table.k, table.aic, marker='o')
    axes[0, 0].axvline(k, color='firebrick', label=f'Selected k={k}')
    axes[0, 0].axvline(k_aic, color='0.3', ls='--', label=f'Tested AIC minimum: {k_aic}')
    axes[0, 0].set(xlabel='Number of components (k)', ylabel='AIC')
    axes[0, 0].legend(fontsize=8)
    # ΔAIC = AIC(k) − AIC(k−1)
    axes[0, 1].plot(table.k.iloc[1:], table.delta_aic.iloc[1:], marker='o')
    axes[0, 1].axhline(0, color='k', lw=0.8)
    axes[0, 1].axvline(k, color='firebrick')
    threshold = ds.attrs['marginal_fraction'] * (table.aic.iloc[0] - table.aic.min())
    axes[0, 1].axhline(-threshold, color='0.5', ls=':', label='Small-gain threshold')
    axes[0, 1].set(xlabel='Number of components (k)', ylabel=r'$\Delta$AIC = AIC$_k$ - AIC$_{k-1}$')
    axes[0, 1].legend(fontsize=8)
    for ax in axes[0]:
        ax.grid(alpha=0.25)
    labels = ds.class_label.values[ds.valid_mask.values.astype(bool)]
    axes[1, 0].scatter(features[:, 0], features[:, 1], c=labels,
                       cmap=cmap, norm=norm, s=1, alpha=0.5, rasterized=True)
    # plot the contour plot of the fitted GMM
    x, y = np.meshgrid(np.linspace(np.min(features[:, 0]), np.max(features[:, 0]), 150),
                      np.linspace(np.min(features[:, 1]), np.max(features[:, 1]), 150))
    XX = np.array([x.ravel(), y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(x.shape)
    axes[1, 1].contour(x, y, Z, levels=20, cmap='coolwarm_r')
    for ax in axes[1]:
        ax.set(xlabel=r'EOF1 loading $\times\sqrt{N}$', ylabel=r'EOF2 loading $\times\sqrt{N}$')
    axes[1, 0].set_title('GMM classes in EOF space')
    axes[1, 1].set_title('Fitted negative log density')
    boundary = ' | AIC minimum at upper search bound' if ds.attrs['aic_min_at_boundary'] else ''
    fig.suptitle(title + f'\nSelected k={k} (AIC/Delta-AIC selection){boundary}', fontsize=11)
    save_figure(fig, output_dir / 'AIC_dAIC_EOF_scatter')


def plot_classes(ds, title, output_dir):
    k = ds.attrs['selected_k']
    window = ds.attrs['class_mean_window_years']
    cmap, norm = class_colors(k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), subplot_kw={'projection': ccrs.Robinson()})
    image = map_field(axes[0], ds, ds.class_label, cmap=cmap, norm=norm)
    axes[0].set_title(f'Spatial classes | k={k}')
    fig.colorbar(image, ax=axes[0], orientation='horizontal', shrink=0.8,
                 pad=0.07, ticks=np.arange(1, k + 1), label='Class (within-model warming rank)')
    image = map_field(axes[1], ds, ds.max_probability, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Maximum posterior probability')
    fig.colorbar(image, ax=axes[1], orientation='horizontal', shrink=0.8,
                 pad=0.07, label='GMM assignment confidence')
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.80, bottom=0.12)
    save_figure(fig, output_dir / 'classes_probability')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for j in range(k):
        color = cmap(j)
        label = f'Class {j + 1} ({float(ds.class_area_fraction.isel(class_id=j)) * 100:.1f}%)'
        axes[0, 0].plot(ds.year, ds.class_sst.isel(class_id=j), color=color, label=label)
        axes[0, 1].plot(ds.year, ds.class_anomaly.isel(class_id=j), color=color)
    axes[0, 0].set(xlabel='Year CE', ylabel='Area-weighted SST (°C)', title='Class mean temperature')
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].set(xlabel='Year CE', ylabel='SST anomaly (°C)', title=f'Relative to the first {window} years')
    axes[0, 1].axhline(0, color='0.6', lw=0.7)
    axes[1, 0].bar(ds.class_id, ds.class_warming, color=cmap(np.arange(k)))
    axes[1, 0].set(xlabel='Class (within-model rank)', ylabel='Late minus early SST (°C)',
                   title=f'Last {window} years minus first {window} years', xticks=ds.class_id.values)
    axes[1, 1].hist(ds.max_probability.values[np.isfinite(ds.max_probability)], bins=30, range=(0, 1), color='steelblue')
    axes[1, 1].axvline(0.8, color='firebrick', ls='--', label='0.8 diagnostic threshold')
    axes[1, 1].set(xlabel='Maximum posterior probability', ylabel='Grid-cell count', title='Assignment confidence distribution')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(title + '\nClass numbers are model-specific; equal numbers do not imply matching regions.', fontsize=11)
    save_figure(fig, output_dir / 'class_curves_confidence')


def plot_checks(ds, stability, title, output_dir):
    fig = plt.figure(figsize=(12, 7), layout='constrained')
    spec = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(spec[0, 0], projection=ccrs.Robinson())
    image = map_field(ax, ds, ds.retained_variance_fraction, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Local variance represented by the two GMM EOFs')
    fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05, label='Explained fraction')
    ax = fig.add_subplot(spec[0, 1], projection=ccrs.Robinson())
    image = map_field(ax, ds, ds.reconstruction_rmse, cmap='magma', vmin=0)
    ax.set_title('Two-mode reconstruction error (unweighted SST)')
    fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05, label='RMSE (°C)')
    ax = fig.add_subplot(spec[1, 0])
    ax.plot(ds.year, ds.example_anomaly, 'k-', label='Centered SST')
    ax.plot(ds.year, ds.example_reconstruction, 'r-', label='Two-mode reconstruction')
    ax.set(xlabel='Year CE', ylabel='SST anomaly (°C)',
           title=f"Nearest complete cell to 30°N, 45°W: {ds.attrs['example_lat']:.1f}°, {ds.attrs['example_lon']:.1f}°")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(spec[1, 1])
    ax.plot(stability.seed, stability.adjusted_rand_index, 'o-', label='Adjusted Rand index')
    ax.plot(stability.seed, stability.aligned_agreement, 's-', label='Label-aligned agreement')
    ax.set(xlabel='Independent GMM seed', ylabel='Agreement with selected fit', ylim=(-0.05, 1.05),
           title='Initialization sensitivity (fixed EOFs and k)', xticks=stability.seed)
    ax.legend(fontsize=8)
    fig.suptitle(title + '\nReconstruction and initialization diagnostics', fontsize=12)
    save_figure(fig, output_dir / 'reconstruction_stability')


def plot_overview(results, output_dir):
    first_result = next(iter(results.values()))
    period = f"{first_result.attrs['year_start']}–{first_result.attrs['year_end']}"
    nrows = int(np.ceil(len(results) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, nrows * 4),
                              subplot_kw={'projection': ccrs.Robinson()}, squeeze=False)
    for ax, (model, ds) in zip(axes.flat, results.items()):
        k = ds.attrs['selected_k']
        cmap, norm = class_colors(k)
        image = map_field(ax, ds, ds.class_label, cmap=cmap, norm=norm)
        co2 = ds.attrs['reference_co2_2100_ppm']
        note = ' | SSP585 SST absent' if ds.attrs['scenario'] == 'ssp126' else ''
        ax.set_title(f"{model} | {ds.attrs['scenario'].upper()} | k={k}{note}\n2100 reference CO$_2$: {co2:g} ppm", fontsize=9)
        fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.75,
                     pad=0.05, ticks=np.arange(1, k + 1))
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.subplots_adjust(top=0.87, bottom=0.12, hspace=0.25, wspace=0.08)
    fig.suptitle(f'SST GMM spatial classes | {period} | highest available SSP per model', fontsize=14)
    fig.text(0.5, 0.04, 'Classes are ranked within each model by late-minus-early warming; colors do not establish cross-model correspondence.\n'
             'CO$_2$: standard 2100 scenario references (Meinshausen et al., 2020; Jiang et al., 2023), not file metadata.', ha='center', fontsize=9)
    save_figure(fig, output_dir / 'all_models_classes')
