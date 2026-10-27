"""Compact figures for spatial changes across a sequence of GMM class counts."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from toolbox.SST_figures import CLASS_COLORS, map_field, save_figure


def plot_k_maps(ds, output_dir):
    """Use one fixed categorical palette across the six area-matched maps."""
    ks = ds.k.values
    cmap = ListedColormap(CLASS_COLORS[:int(ks.max())])
    norm = BoundaryNorm(np.arange(0.5, ks.max() + 1.5), len(cmap.colors))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6),
                              subplot_kw={'projection': ccrs.Robinson()})
    for ax, k in zip(axes.flat, ks):
        map_field(ax, ds, ds.class_label.sel(k=k), cmap=cmap, norm=norm)
        ax.set_title(f'K = {k}', fontsize=13)
    fig.subplots_adjust(left=0.025, right=0.98, bottom=0.22, top=0.85,
                        wspace=0.07, hspace=0.12)
    cax = fig.add_axes([0.27, 0.14, 0.46, 0.026])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation='horizontal', ticks=np.arange(1, ks.max() + 1))
    cbar.set_label('Area-overlap-matched class ID (within this model only)')
    attrs = ds.attrs
    note = ' | SSP585 SST unavailable' if attrs['scenario'] == 'ssp126' else ''
    fig.suptitle(f"{attrs['model']} | {attrs['scenario'].upper()}{note}\n"
                 f"Annual SST, {attrs['year_start']}–{attrs['year_end']} | EOF1+2: {attrs['two_eof_variance_fraction']:.1%} | "
                 f"2100 reference CO$_2$: {attrs['co2_2100_reference_ppm']:g} ppm", fontsize=13, y=0.97)
    fig.text(0.5, 0.025,
             'Colors follow maximum area overlap with the preceding K; splits and boundary rearrangements are allowed.\n'
             'K=4 IDs start in warming order; later IDs track overlap, not warming rank. Grey background: land / missing data.\n'
             'CO$_2$: scenario references (Meinshausen et al., 2020; Jiang et al., 2023), not model-file concentrations.',
             ha='center', fontsize=8)
    save_figure(fig, output_dir / 'spatial_classes_K4_K9')


def plot_k_checks(ds, summary, fits, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout='constrained')
    minimum = summary.aic.min()
    for seed, group in fits.groupby('seed'):
        axes[0, 0].plot(group.k, group.aic - minimum, '.--', alpha=0.55, label=f'Seed {seed}')
        axes[1, 0].plot(group.k, group.ari_to_selected, '.-', label=f'Seed {seed}')
    axes[0, 0].plot(summary.k, summary.aic - minimum, 'ko-', label='Best of seed batches')
    axes[0, 0].set(ylabel='AIC above minimum over K=4–9', title='Fit quality and local optima')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(summary.k, summary.low_confidence_area_fraction * 100, 'o-', label='Max posterior < 0.8')
    axes[0, 1].plot(summary.k, summary.smallest_class_area_fraction * 100, 's-', label='Smallest class')
    axes[0, 1].set(ylabel='Fraction of analyzed ocean area (%)', title='Ambiguity and small regions')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(ylabel='Adjusted Rand index vs selected fit', ylim=(-0.05, 1.05),
                   title='Initialization sensitivity at each K')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(summary.k, summary.matched_area_fraction * 100, 'o-')
    axes[1, 1].set(ylabel='Area retaining its matched color (%)', ylim=(0, 105),
                   title='Spatial persistence from previous K')
    for ax in axes.flat:
        ax.set(xlabel='Number of classes (K)', xticks=summary.k)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{ds.attrs['model']} | {ds.attrs['scenario'].upper()} | essential K-comparison checks", fontsize=13)
    save_figure(fig, output_dir / 'K4_K9_diagnostics')
