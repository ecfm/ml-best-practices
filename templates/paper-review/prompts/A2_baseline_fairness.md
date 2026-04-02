# Adversarial: Is the Comparison Fair?

The paper compares [METHOD_A] against [METHOD_B] and concludes [CONCLUSION].
Your job: check whether the comparison systematically advantages one side.

## Files to read

[LIST_COMPARISON_SCRIPTS_AND_TEX_SECTIONS]

## For each comparison, check:

### a) Information parity
- What features/information does each method get?
- Are they informationally equivalent?

### b) Sample alignment
- Same test set? Same N? Same respondents?
- Same train set or different training regimes?

### c) Evaluation metric
- Same metric and denominator?
- Same aggregation method?
- Same treatment of missing/out-of-range values?

### d) Hyperparameter tuning
- Is one method tuned on test data while the other isn't?

### e) Structural advantages
- Does one method get supervised training while the other doesn't?
- Is the comparison testing what the paper claims?

## Output format

For each comparison:

```
## Comparison: [dataset, method A vs method B]
### Information
| | Method A | Method B |
|---|---|---|
| Features | ... | ... |
| Training data | ... | ... |

### Direction of Unfairness
[advantages A / advantages B / fair / ambiguous]

### Magnitude
[negligible / moderate / large]
```
