# Report Writing — Reasoning Transparency

Source: [Coefficient Giving](https://coefficientgiving.org/research/reasoning-transparency/)

## Core Principle

Help the reader calibrate how much to trust each claim, rather than presenting results as facts.

## Required Practices

1. **Lead with linked summary** — Open with key conclusions, each linking to the detailed section. Reader should get the full picture in 30 seconds

2. **State confidence levels explicitly** — For every major claim:
   - "~90% confident this is real (replicated across 3 seeds)"
   - "Suggestive but single-run — treat with caution"
   - Use probabilistic language, not just "we found X"

3. **Show which evidence matters most** — Don't just list results. Say which experiments were most informative and which were secondary: "The strongest evidence for X comes from experiment Y"

4. **Disclose uncertainty and limitations honestly**:
   - Flag known confounds you didn't control for
   - Note where you ran out of time or compute
   - Distinguish "checked carefully" vs "assume based on intuition"

5. **Make it verifiable** — Include seeds, hyperparameters, exact configs. Point to where every number came from. Include raw results, not just highlights

6. **Document your process** — How many experiments ran, what didn't work, search strategy, time spent, shortcuts taken

## Report Structure Template

```markdown
## Summary of Findings
- Finding 1 (high confidence) — [link to section]
- Finding 2 (moderate confidence) — [link to section]

## Experiment N: [Title]
- **Setup**: exact config, seeds, hyperparams
- **Results**: with error bars or variance across seeds
- **Confidence**: how much to trust this and why
- **Limitations**: what could be confounded

## Discussion
- Which experiments were most vs least informative
- What would you do with more time
- Known systematic issues
```
