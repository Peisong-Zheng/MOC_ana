# CMIP SST inspection and GMM analysis

Explore sea surface temperature (SST) in the highest-forcing SSP scenario available for each climate model, and compare spatial classifications obtained with principal component analysis (PCA) and Gaussian mixture models (GMMs).

The project includes an SST snapshot notebook, an automatic/fixed-K GMM workflow, and a dedicated comparison of K = 4–9. All figures are saved in **both PNG and PDF**. Historical and MOC data are present in the source archive but are not used by these SST workflows.

## Quick start

Run the following commands from the project root using the existing local environment:

```bash
# Compare K = 4, 5, 6, 7, 8, 9 for every model.
.venv/bin/python SST_GMM_K_comparison.py

# Run the main analysis with detailed diagnostics; currently defaults to K = 6.
.venv/bin/python SST_GMM.py
```

For a quick temperature-field inspection, open [SST_highest_SSP.ipynb](SST_highest_SSP.ipynb) in VS Code, select `.venv/bin/python` as the kernel, and run all cells with the project root as the working directory. The default snapshot year is 2100; set `plot_year = None` to use the latest year shared by all models.

Scientific settings are edited near the top of each script; there is no command-line argument interface. The Python scripts resolve output paths relative to their own location and can also be launched using absolute paths from another directory. Neither script requires running the notebook first.

| Analysis | Entry point / setting | Figure directory | Numerical output directory |
| --- | --- | --- | --- |
| Annual SST snapshot | `SST_highest_SSP.ipynb` | `figures/SST_highest_SSP/` | Selection and metadata displayed in the notebook |
| Automatic K selection | `SST_GMM.py`, `fixed_k = None` | `figures/SST_GMM/` | `data/processed/SST_GMM/` |
| Fixed K = 6 | `SST_GMM.py`, `fixed_k = 6` **(current default)** | `figures/SST_GMM_K6/` | `data/processed/SST_GMM_K6/` |
| Spatial comparison, K = 4–9 | `SST_GMM_K_comparison.py` | `figures/SST_GMM_K_comparison/` | `data/processed/SST_GMM_K_comparison/` |

Different workflows use separate output directories. Rerunning a workflow writes to its existing output paths; change the output directory before running if an earlier version must be preserved.

## Environment

The existing `.venv` inherits scientific packages from the local Anaconda installation and includes Cartopy. To create an environment on another machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas xarray netCDF4 scipy scikit-learn threadpoolctl matplotlib cartopy ipykernel jupyter
```

This is a dependency installation guide, not a version-pinned environment. Package versions can affect numerical reproducibility. Natural Earth 110 m coastlines are cached in `data/cartopy/`; Cartopy may need internet access to download them if the cache is absent.

## Input data and scenario selection

The default source directory is `/Users/pz/CMIP_AMOC_SST`. Change `input_dir` in the scripts or `data_dir` in the notebook when using another location. Expected input layout:

```text
CMIP_AMOC_SST/
└── <model>/
    ├── historical/
    └── <ssp scenario>/
        ├── SST/
        │   └── sst_<model>_<scenario>_ann_2015_2100.nc
        └── MOC/
```

The supplied SST files contain `SST(year, lat, lon)` in kelvin, with annual data for 2015–2100 on a regular 1-degree grid. Temperatures are converted to degrees Celsius. The analysis retains the supplied grid; it does not perform additional regridding.

Scenarios are ranked by nominal 2100 radiative forcing, and selection requires an actual SST file. A scenario containing only MOC data is not eligible. The current selections are:

| Model | Selected SST scenario | Reference CO2 in 2100 |
| --- | --- | --- |
| ACCESS-CM2 | SSP5-8.5 | approximately 1135 ppm |
| CESM2 | SSP5-8.5 | approximately 1135 ppm |
| GISS-E2-1-G | SSP1-2.6 | approximately 446 ppm |
| MIROC6 | SSP5-8.5 | approximately 1135 ppm |
| MPI-ESM1-2-HR | SSP5-8.5 | approximately 1135 ppm |
| MRI-ESM2-0 | SSP5-8.5 | approximately 1135 ppm |

GISS-E2-1-G uses SSP1-2.6 because its SSP5-8.5 directory contains MOC but no SST. Its forcing therefore differs from the other models. The CO2 labels are rounded **standard scenario references**, not concentrations read or verified from these files. The scenario suffixes 2.6 and 8.5 denote nominal radiative forcing in W m−2, not CO2 concentration in ppm. References: [Meinshausen et al. (2020)](https://doi.org/10.5194/gmd-13-3571-2020) and [Jiang et al. (2023), Table 1](https://doi.org/10.1029/2022MS003563).

Each model/scenario SST directory is expected to contain one NetCDF file. Multiple members or time chunks require explicit selection or preprocessing; the scripts raise an error instead of silently choosing one. They ignore top-level directories without SSP subdirectories. The snapshot notebook still treats every top-level directory as a model, so its `model_dirs` list must exclude unrelated analysis folders when present.

## Shared PCA–GMM method

The GMM workflows use the common annual interval, currently 2015–2100, and the following preprocessing:

1. Subtract each grid cell's temporal mean over the common interval. Retain the long-term trend; apply no temporal smoothing, detrending, or per-cell standardization.
2. Exclude land and cells missing any annual value. Missing values are not filled or interpolated.
3. Multiply anomalies by `cos(latitude)` and compute an economy SVD. This preserves the legacy weighting and produces a **cos²(latitude) variance metric**. Setting `pca_weight_power = 0.5` instead gives square-root-cosine weighting and constitutes a different analysis.
4. Use the first **two spatial EOF loading columns** as the GMM features. Both columns receive the same `sqrt(number of complete cells)` multiplier to control the numerical scale of covariance regularization. There is no relative whitening or reweighting of the two features.
5. Fit full-covariance GMMs with `max_iter = 500`, `tol = 1e-4`, and `reg_covar = 1e-6`. Unconverged returned fits cause an error. Predictions and probabilities come from the retained fitted model.

Grid cells are the GMM samples, so the classification describes spatial variation in SST evolution, including warming. It is not a classification of individual years or mean-temperature zones. Every complete grid cell has equal weight in the GMM likelihood; class-area fractions and class temperature means use cosine-latitude area weights.

EOF signs are oriented consistently within each model by making the largest absolute loading positive and flipping the paired PC with it. EOF modes are not matched between models. Raw PC coefficients, unit-norm EOFs, masks, fitted parameters, and posterior probabilities are saved for inspection and reconstruction.

## Comparing K = 4–9

[SST_GMM_K_comparison.py](SST_GMM_K_comparison.py) is the main entry point for examining how spatial patterns change as the number of classes increases. It reads the source SST directly and computes PCA once per model, so all six K values share the same years, mask, weighting, and EOF features.

For each K, seed 42 uses five initializations and seeds 43 and 44 use three each. The lowest-AIC fit among the three batches is retained. This differs from the original fixed-K workflow, which retains the seed-42 fit and uses other seeds only for checks. Consequently, the K = 6 panel here can differ from the earlier fixed-K = 6 result.

### Class colors

At K = 4, displayed classes are ordered by increasing area-weighted warming between 2015–2034 and 2081–2100. For each subsequent K, a rectangular Hungarian assignment maximizes cosine-area overlap with the preceding partition. Matched components inherit the preceding IDs/colors; the unmatched component receives the new ID.

This operation changes labels only. It does not force a nested hierarchy, and class IDs after K = 4 are not warming ranks. Matching applies within each model; equal colors across models do not imply corresponding physical regions.

### Figures and numerical outputs

Each model has two figure pairs under `figures/SST_GMM_K_comparison/<model>/`:

| Figure stem | Contents |
| --- | --- |
| `spatial_classes_K4_K9` | Six Robinson maps in a 2 × 3 layout, with matched colors, scenario information, and a shared class colorbar |
| `K4_K9_diagnostics` | AIC by initialization batch, low-confidence ocean area, smallest-class area, restart agreement, and area retaining its matched color between adjacent K values |

PC and EOF maps are not repeated because their inputs do not change with K. Low confidence means maximum posterior probability below 0.8. Color persistence can decrease because of a split or broader reorganization; it is not a fit-quality score.

Each model directory under `data/processed/SST_GMM_K_comparison/` contains:

- `<model>_<scenario>_K4_K9.nc`: all six partitions, raw/display-label mappings, posterior probabilities, fitted GMM parameters, class areas, shared EOFs/PCs, and method metadata. Unused class slots above the current K are NaN.
- `K4_K9_summary.csv`: fit and spatial diagnostics for each K, including Delta-AIC. Delta-AIC at K = 4 is undefined because K = 3 is not fitted.
- `initialization_checks.csv`: per-batch AIC and adjusted Rand index (ARI) against the retained partition.
- `overlap_K4_K5.csv` through `overlap_K8_K9.csv`: adjacent-K intersections as fractions of total analyzed ocean area; rows are previous displayed classes and columns are new displayed classes.

The run directory also contains `all_models_K4_K9_summary.csv` for all 36 solutions and `settings.json`.

## Automatic and fixed-K analysis

[SST_GMM.py](SST_GMM.py) provides more extensive PCA and reconstruction diagnostics. Its current setting is `fixed_k = 6`. Set it to `None` for automatic selection, or another integer within `1..max_components` for a fixed-K experiment. Both modes fit K = 1–12 for the AIC plots, using five initializations with seed 42 for each K.

### Automatic selection

Automatic selection uses an exploratory AIC/Delta-AIC elbow rule:

1. Normalize running-best AIC improvement and the tested K interval to [0, 1]. Locate the initial elbow by maximizing normalized improvement minus normalized K.
2. Starting there, choose the first K for which each of the next two AIC improvements is at most `marginal_fraction = 0.10` of the full tested improvement.
3. If no elbow is confirmed, use the tested AIC minimum. If total improvement is at most two AIC units, choose K = 1.

Delta-AIC is `AIC(K) - AIC(K-1)`; negative values indicate improvement. The summaries also report the tested AIC minimum, the selected fit's distance from it, and whether the minimum occurs at the search boundary.

The elbow is a parsimony heuristic, not the formal AIC minimum or an AIC equivalence test. It may select a fit with substantially higher AIC. Changing the tested interval or threshold can change the selected K.

### Diagnostics and outputs

Classes are ordered independently for each fit by increasing area-weighted early-to-late warming. Additional seeds 43, 44, and 45, with three starts each, assess initialization sensitivity at the selected K; they do not replace the seed-42 fit.

Each model has six figure pairs:

- `PCs_variance`: PC evolution and explained variance.
- `EOF_maps`: the six retained diagnostic EOFs.
- `AIC_dAIC_EOF_scatter`: K-selection curves and the two-feature GMM representation.
- `classes_probability`: spatial labels and assignment confidence.
- `class_curves_confidence`: area-weighted class SST evolution and confidence checks.
- `reconstruction_stability`: two-EOF reconstruction quality and initialization sensitivity.

An `all_models_classes` figure summarizes the model maps. Per-model numerical outputs include `<model>_<scenario>_PCA_GMM.nc`, `AIC_dAIC.csv`, `initialization_stability.csv`, and `class_summary.csv`. Run-level files include `model_summary.csv` and `settings.json`.

For fixed-K runs, existing automatic outputs are used for comparison after checking matching years, coordinates, masks, EOFs, and PCs. Additional outputs are `comparison_with_automatic_k` figure pairs, per-model `baseline_overlap_area_fraction.csv`, and run-level `comparison_with_automatic_k.csv`. The overlap figure normalizes each previous class to its own area; the overlap CSV stores fractions of total analyzed ocean area.

## Interpretation and reproducibility

GMM classes are exploratory statistical regions. Spatial dependence is not modeled, and AIC values should be compared only within a model using the same sample and EOF representation. AIC is not comparable across different models' feature spaces. Even multiple initializations do not guarantee a global optimum.

ARI measures label-invariant partition agreement on equally weighted grid cells. Restart agreement describes optimization sensitivity, not sampling uncertainty. Posterior probabilities describe assignment confidence under the fitted GMM, not calibrated probabilities that a physical ocean region is correct. The two-EOF reconstruction diagnostics help identify variability omitted by the clustering features.

PDF maps use rasterized data meshes with vector text and coastlines. All workflows save PNG copies for quick inspection.

## Code organization and provenance

```text
MOC_ana/
├── README.md
├── SST_highest_SSP.ipynb
├── SST_GMM.py
├── SST_GMM_K_comparison.py
├── toolbox/
│   ├── PCA_functions.py
│   ├── GMM_functions.py
│   ├── SST_figures.py
│   └── SST_k_figures.py
├── data/
│   ├── cartopy/
│   └── processed/
├── figures/
└── agent_work/
```

The PCA/GMM modules are adapted from `/Users/pz/VScode/Last_deglaciation/toolbox/`, following the workflow in `TraCE_GMM_v2.ipynb`. Original calculation lines and comments are retained where applicable. Adaptations include SST/year dimension names, persistent missing-data masks, economy SVD, reproducible fitting, saved probabilities and fitted parameters, and forward-running calendar years. The common EOF feature multiplier also changes the numerical regularization scale relative to the legacy unscaled features.

The original project is unchanged and is not required at runtime. Plot helpers reuse its map conventions and diagnostic style. `agent_work/` contains development checks, logs, and previews; analysis scripts do not depend on it.

For the estimator's AIC, initialization, and convergence definitions, see the [scikit-learn GaussianMixture documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
