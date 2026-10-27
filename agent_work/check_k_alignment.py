from pathlib import Path
import itertools
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SST_GMM_K_comparison import align_colors
from toolbox.GMM_functions import gmm_aic_curve

old = np.repeat(np.arange(1, 5), 8)
raw = np.repeat([3, 0, 4, 1], 8)
raw[20:24] = 2  # Split one previous class, then arbitrarily permute raw IDs.
weights = np.arange(1, 33, dtype=float)
mapping, overlap = align_colors(old, raw, weights, np.zeros(32), 5)
assert sorted(mapping.tolist()) == [1, 2, 3, 4, 5]
matched = sum(weights[mapping[raw] == old])
brute = max(sum(weights[np.asarray(order)[raw] == old]) for order in itertools.permutations(range(1, 6)))
assert matched == brute
np.testing.assert_allclose(overlap.sum(), weights.sum())
# A permutation of the new raw component IDs must preserve the displayed map.
permutation = np.array([2, 4, 1, 0, 3])
new_mapping, new_overlap = align_colors(old, permutation[raw], weights, np.zeros(32), 5)
np.testing.assert_array_equal(mapping[raw], new_mapping[permutation[raw]])
np.testing.assert_allclose(overlap, new_overlap)

rng=np.random.default_rng(0)
features=rng.normal(size=(200,2))
aic_full, _, full=gmm_aic_curve(features,max_components=3,n_init=2)
aic_subset, _, subset=gmm_aic_curve(features,min_components=2,max_components=3,n_init=2)
np.testing.assert_allclose(aic_full[1:],aic_subset)
assert [fit.n_components for fit in full]==[1,2,3]
assert [fit.n_components for fit in subset]==[2,3]
print('PASS: maximum-area matching, label-permutation invariance, and backward-compatible GMM K ranges.')
