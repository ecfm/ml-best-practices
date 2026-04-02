# Methodology Deep-Dive: Statistical Claims

Review ALL confidence intervals, p-values, and significance claims. For each
test, assess whether it actually tests what the paper claims.

## Files to read

[LIST_SCRIPTS_WITH_BOOTSTRAP_OR_PVALUES]
[MAIN_TEX_FOR_CLAIMS]

## For each CI or p-value, answer:

### 1. Resampling unit
What is resampled — observations, clusters, studies, or predictions? Is this
the correct unit of analysis?

### 2. Test-statistic alignment
Does the test match the claim? Examples of misalignment:
- Permuting labels post-fit tests "association" not "classifier beats chance"
- Paired bootstrap tests "difference on this sample" not "population superiority"

### 3. Edge cases
- Can p = 0.000 be reported with finite permutations?
- Are CIs symmetric when the distribution is skewed?
- What happens with degenerate cases (zero variance, single class)?

### 4. Multiple comparisons
- How many tests? Any correction?
- Are p-values reported selectively?

## Output format

```
## Test: [description]
- **Script**: [file:line]
- **Resampling unit**: [what's resampled]
- **Claim**: [what the paper says]
- **What's actually tested**: [what the test measures]
- **Aligned?**: [yes/no + explanation]
- **Magnitude of concern**: [negligible / moderate / serious]
```
