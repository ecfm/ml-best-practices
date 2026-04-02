# Methodology Deep-Dive: Train/Test Integrity

Trace ALL data flows from raw input to final evaluation metric. At each point,
check if held-out data influences training, feature construction, or labels.

## Files to read

[LIST_ALL_ANALYSIS_SCRIPTS]

## Check at each stage

### a) Scale/normalization parameters
- Computed from all data or train-only?
- Do they affect evaluation labels or metrics?

### b) Dimensionality reduction
- PCA/SVD/embedding fit on all data or train-only?
- For cross-validation: refit per fold or once globally?

### c) Feature engineering
- Any features derived from test-set response distributions?
- Are "pre-available" features truly available before data collection?

### d) Evaluation thresholds
- How are thresholds (midpoints, quantiles, terciles) computed?
- From test data or held-out?

## Known fixes (verify, don't re-report)

[LIST_KNOWN_FIXES_FROM_DECISIONS_MD]

## Output format

For each script:

```
## Script: [name]
### Data Flow
raw → [transform] → features → [split] → train/test → [model] → [evaluate]

### Leakage Points Found
| Stage | What leaks | Direction | Magnitude |

### Verified Clean
[stages checked and found correct]
```
