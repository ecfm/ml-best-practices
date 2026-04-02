# Methodology Deep-Dive: Metric Validity

For each metric the paper reports, question whether it measures what the paper
claims it measures. Do NOT check whether formulas are coded correctly — assume
the code computes what it says. Instead, question what the metric *means*.

## Files to read

[LIST_SCRIPTS_AND_METHODS_SECTION]

## For EACH metric, answer three questions:

1. **Null baseline**: What would a random/constant/trivial predictor score?
   Compute it if possible.
2. **Sensitivity**: Is the metric sensitive to scale, sample size, or outcome
   composition? Would removing 5% of outcomes change the conclusion?
3. **Conflation**: Does the metric conflate two things the paper treats as
   distinct?

## Metrics to examine

[LIST_EACH_METRIC_WITH_PAPER_DEFINITION]

## What NOT to report

- Code bugs (that's for E-tracks)
- Train/test leakage (that's for M2)
- Statistical test issues (that's for M3)
- Known issues already in DECISIONS.md

## Output format

```
## Metric: [name]
- **Paper's interpretation**: [what the paper says it means]
- **Null baseline**: [trivial predictor score]
- **Sensitivity**: [what changes the metric most]
- **Conflation risk**: [if any]
- **Verdict**: [does it measure what the paper claims?]
```
