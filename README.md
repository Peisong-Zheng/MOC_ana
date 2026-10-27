# CMIP SST inspection and GMM analysis

## Run

Use the local Python environment:

```bash
.venv/bin/python SST_GMM.py
```

For `SST_highest_SSP.ipynb`, select `.venv/bin/python` in VS Code and run all cells from this directory. The notebook is entirely in English and saves its 2100 annual-mean SST figure as PNG and PDF.

`SST_GMM.py` runs independently of the notebook and can be launched from any working directory. Editable scientific settings are at the top of the script. Dependencies are NumPy, pandas, xarray, netCDF4, SciPy, scikit-learn, threadpoolctl, Matplotlib, and Cartopy; the notebook additionally uses ipykernel. The project environment inherits the existing Anaconda packages and contains Cartopy. Natural Earth 110 m coastlines are cached under `data/cartopy/`.

## Inputs and comparison

Inputs are read directly from `/Users/pz/CMIP_AMOC_SST`. Each model uses the highest-forcing SSP with an actual SST file. ACCESS-CM2, CESM2, MIROC6, MPI-ESM1-2-HR, and MRI-ESM2-0 use SSP5-8.5. GISS-E2-1-G uses SSP1-2.6 because its SSP5-8.5 folder contains MOC but no SST. The files lack member identifiers and atmospheric CO2 concentrations.

The analysis uses the common annual sequence, 2015–2100. K is explicitly converted to degrees Celsius. The supplied regular 1-degree grid is retained. Land and any cell missing even one annual value are excluded from PCA and GMM; they are never filled with zero or interpolated. There is no temporal smoothing, detrending, or per-cell standardization. Thus the clusters describe spatial variation in the time-dependent SST signal, including long-term warming, rather than mean-temperature zones or clusters of individual years.

The 2100 CO2 labels are rounded **scenario references**, 1135 ppm for SSP5-8.5 and 446 ppm for SSP1-2.6. They are not verified model-file concentrations. Sources: [Meinshausen et al. (2020)](https://doi.org/10.5194/gmd-13-3571-2020) and [Jiang et al. (2023), Table 1](https://doi.org/10.1029/2022MS003563).

## Reused algorithms and adaptations

`toolbox/PCA_functions.py` and `toolbox/GMM_functions.py` are adapted copies of the corresponding functions in `/Users/pz/VScode/Last_deglaciation/toolbox/`. The originals are unchanged, and running this analysis does not require importing the old project. Original calculation lines and comments are retained where applicable, including the time centering, cosine weights, SVD/PC calculation, GMM fit/predict/probability calculation, AIC loop, and `np.diff(aic)` definition. Plotting in `toolbox/SST_figures.py` follows the original Robinson maps, EOF scatter/density diagnostics, area-weighted class curves, palette, and comments.

The analysis follows `TraCE_GMM_v2.ipynb`:

1. Subtract each grid cell's mean over the full common interval.
2. Multiply anomalies by `cos(latitude)`, preserving the original implementation. This produces a **cos²(latitude) variance metric**, not the conventional area metric. `pca_weight_power = 0.5` is available for a scientifically different, square-root-cosine analysis; the default is 1.0. Class-area fractions and class temperature means always use `cos(latitude)` area weights.
3. Compute economy SVD on complete cells. This preserves the original nonzero EOF/PC decomposition without allocating a full grid-by-grid matrix. Retain six modes for diagnostics and the complete explained-variance spectrum. Flip each EOF and its paired PC together so its largest absolute loading is positive. Modes are not matched across models.
4. Fit full-covariance GMMs to the first **two spatial EOF loading columns**, as in the original work. Both columns receive the same multiplier, `sqrt(number of complete cells)`, to keep covariance regularization small relative to feature variance. There is no relative reweighting or whitening of the two columns. The map EOFs themselves remain unit norm. Every complete grid cell has equal likelihood weight in GMM; this is not an area-weighted GMM likelihood.
5. For every k from 1 to 12, use five initializations, seed 42, `max_iter=500`, `tol=1e-4`, and `reg_covar=1e-6` in the scaled feature space. Stop with an error if a returned fit is unconverged. Reuse the selected fitted model for map predictions rather than silently fitting a different solution.

The changes from the old functions are explicit year/SST names, a persistent land/missing-data mask, memory-efficient SVD, configurable retained modes, consistent EOF/PC signs, reproducible GMM restarts, saved fitted parameters/probabilities, and forward-running calendar years instead of reversed years BP. The original default regularization acted on very small unscaled loadings; the shared feature multiplier changes that numerical regularization scale. No legacy zero filling or paleoclimate event annotations are carried over.

## Selection of k

When `fixed_k = None`, the selected k is an **exploratory AIC/Delta-AIC elbow**, not a claim to be the formal AIC minimum or the true number of physical ocean regions. This distinction is preserved in the plots, CSV summaries, and NetCDF attributes.

- AIC is computed on the same complete-cell EOF sample for every k within a model. Delta-AIC is `AIC(k) - AIC(k-1)`; a negative value means improvement.
- Normalize the running-best AIC improvement and the tested k interval to [0, 1]. The initial elbow maximizes normalized AIC improvement minus normalized k, i.e. departure from the endpoint chord.
- Starting at that elbow, choose the first k for which **each of the next two AIC improvements** is at most 10% of the full tested AIC improvement. This threshold is exposed as `marginal_fraction`.
- If there is no confirmed elbow, choose the tested AIC minimum. If the total improvement is no more than 2 AIC units, choose k=1.
- Also report the tested AIC minimum, the selected model's AIC distance from it, the fraction of the total improvement captured, and whether the minimum lies at k=12. A minimum at the upper bound is not an interior optimum. Changing the search interval or threshold may change the elbow.

This rule favors interpretable broad regions and may select a model with substantially higher AIC than the minimum. It is not an AIC equivalence test. Spatial correlation and the equal-grid-cell likelihood also mean these curves should be treated as descriptive selection diagnostics rather than independent-sample evidence for a physical class count. AIC values cannot be compared between models with different samples/EOF spaces. The scikit-learn API defines [AIC and GMM initialization/convergence behavior](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

## Labels and checks

Classes are reordered by increasing area-weighted warming between 2015–2034 and 2081–2100. The original GMM component index and its mapping to the displayed class are saved. Equal class numbers/colors across different models do **not** establish geographical or physical correspondence.

Three additional fits (seeds 43, 44, 45; three starts each) test initialization sensitivity at the selected k. Report adjusted Rand index and agreement after optimal label permutation; this is optimization sensitivity, not a sampling uncertainty interval. Posterior probabilities are GMM assignment confidence, not calibrated probabilities that a physical region is correct.

PCA diagnostics include six EOF maps, PCs normalized only for display, the full variance spectrum, local variance represented by the two GMM EOFs, unweighted SST reconstruction RMSE, and a North Atlantic point comparison. The reconstruction divides out the latitude multiplier. These checks reveal where the two-feature legacy representation misses variability. The raw PC coefficients and unit-norm EOFs are saved for reconstruction.

## Outputs

- `figures/SST_highest_SSP/`: English notebook snapshot in PNG and PDF.
- `figures/SST_GMM/all_models_classes.{png,pdf}`: six-model class-map overview.
- `figures/SST_GMM/<model>/`: six diagnostic figures, each as PNG and PDF: `PCs_variance`, `EOF_maps`, `AIC_dAIC_EOF_scatter`, `classes_probability`, `class_curves_confidence`, and `reconstruction_stability`.
- `data/processed/SST_GMM/<model>/`: NetCDF with EOFs, PCs, spectra, masks, class labels, posterior probabilities, fitted GMM parameters, class curves, reconstruction diagnostics, and method/source metadata; CSVs with AIC/Delta-AIC, class summaries, and initialization checks.
- `data/processed/SST_GMM/model_summary.csv`: selected k, AIC minimum, explained variance, and stability across models.
- `data/processed/SST_GMM/settings.json`: settings and the actual common years used.

PNG and PDF maps use rasterized meshes with vector coastlines/text in PDF. Temporary development checks and logs are isolated under `agent_work/`; the research script does not depend on them.

## Fixed-k experiment

`SST_GMM.py` currently sets `fixed_k = 6` near the top. Run the same command to reproduce the experiment. Set `fixed_k = None` to restore the automatic AIC/Delta-AIC elbow selection, or choose another integer within `1..max_components` for a different fixed-k experiment.

A fixed-k run derives its output names from that parameter: `figures/SST_GMM_K6/` and `data/processed/SST_GMM_K6/`. The previous automatic-run outputs remain in `SST_GMM/`. The full k=1..12 sweep is retained for diagnostic AIC plots, but it does not determine the class count in a fixed-k run. SST selection, common years, masks, latitude weighting, EOF features, seeds, and optimization settings remain the same as in the automatic run.

When the previous automatic results exist, the script checks matching years, coordinates, masks, EOFs, and PCs and creates `comparison_with_automatic_k.{png,pdf}` for each model. The overlap heatmap reports, for each previous class, the percentage of its ocean area assigned to each fixed-k class. It uses cosine-latitude weights and does not assume that equal class numbers correspond across fits. `baseline_overlap_area_fraction.csv` stores the same intersections as fractions of total analyzed ocean area. `comparison_with_automatic_k.csv` summarizes label-invariant adjusted Rand index, AIC change, low-confidence fractions, initialization sensitivity, and smallest-class area. A higher class count need not be a nested subdivision of the previous partition.

## Spatial comparison across K=4–9

Run `.venv/bin/python SST_GMM_K_comparison.py`. This new script reads the raw SST files directly, reuses the existing PCA/GMM modules, and does not require previous analysis outputs. Settings are at the top. The shared GMM function now accepts `min_components` (default 1), so existing scripts retain their original behavior.

Each model has one **2 × 3 spatial figure** covering K=4, 5, 6, 7, 8, 9, plus one compact diagnostic figure. Both figures are saved as PDF and PNG under `figures/SST_GMM_K_comparison/<model>/`. PCA is computed once per model; every K uses the same 2015–2100 interval, complete-cell mask, legacy cosine-latitude multiplier, and two EOF features. PCs and EOF maps are not plotted again because they do not vary with K.

For each K, seed 42 has five initializations and seeds 43/44 have three each. The lowest-AIC fit among these three batches is retained. This is a deliberate improvement over retaining seed 42 alone: the earlier experiments found occasional better local optima under additional seeds. Thus a K=6 panel here may differ from the earlier fixed-K=6 panel. The diagnostic figure shows all batch AICs and their adjusted Rand indices against the chosen partition; optimization sensitivity is not a sampling confidence interval.

Colors are matched **within each model** across successive K values. K=4 starts in increasing early-to-late warming order. Thereafter a rectangular Hungarian assignment maximizes cosine-area overlap between the preceding displayed classes and the new raw GMM components. Matched components inherit the old IDs/colors and the unmatched component receives the new ID. This changes labels only, not the fitted partition or probabilities. It does not force nested subdivisions, and later IDs are not warming ranks. Colors do not establish physical correspondence between models.

The diagnostic figure shows AIC/local-optimum differences, ocean-area fraction with maximum posterior below 0.8, smallest-class ocean-area fraction, restart agreement, and the area retaining its matched color from the preceding K. Area fractions use cosine-latitude weights, whereas adjusted Rand indices and the GMM likelihood treat complete grid cells equally. Lower color persistence can reflect a split as well as a broader reorganization; it is not itself a measure of fit quality. Delta-AIC is saved in the summary CSV, with K=4 undefined because K=3 is not fitted in this experiment.

`data/processed/SST_GMM_K_comparison/<model>/` contains a single NetCDF with all six partitions, raw/display-label mappings, posterior probabilities, fitted parameters, area fractions, and shared EOF/PC data, plus fit/initialization summary CSVs and five adjacent-K overlap tables. In each overlap table, entries are fractions of total analyzed ocean area, rows are previous displayed IDs, and columns are new displayed IDs. Unused component slots above the current K are NaN. `all_models_K4_K9_summary.csv` combines all 36 solutions. The GISS SST scenario remains SSP1-2.6 because SSP5-8.5 SST is absent.
