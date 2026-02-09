# ML/LLM Experiment Best Practices

Lessons learned from building and evaluating LLM-based prediction systems.
These apply broadly to any ML project that involves model inference, evaluation,
and manual validation of results.

---

## 1. Output Structure: Hierarchical Drill-Down

Design outputs so a human can start from a 1-line summary and drill into any
specific data point. Every level should answer "is this good?" and "where
should I look next?"

```
run_dir/
├── README.txt              ← Purpose, scope, data, model, metrics, navigation
├── config.json             ← Exact reproduction parameters (resolved, not file ref)
├── metrics.json            ← Top-level numbers
├── predictions.parquet     ← All predictions (programmatic access)
├── raw_outputs.parquet     ← All raw LLM outputs + input prompts
│
├── groups/                 ← Per-group breakdown (rename for your domain)
│   ├── _index.csv          ← ENTRY POINT: one row per group, sorted by metric
│   └── {group}/
│       ├── group_metrics.json
│       ├── predictions.csv      ← Human-readable (not parquet)
│       ├── per_entity.csv       ← Per-entity summary (sorted worst-first)
│       └── entities/
│           └── {entity_id}/
│               ├── summary.txt  ← All predictions at a glance
│               └── {item}.txt   ← Individual LLM input/output
│
└── diagnostics/
    ├── parse_failures.csv       ← Raw outputs that failed parsing
    ├── outliers.csv             ← Predictions outside valid range
    ├── value_ranges.json        ← Min/max/scale per group
    └── actual_vs_predicted.csv  ← Full flat table for custom analysis
```

### Why this structure works
- `_index.csv` sorted worst-first tells you immediately which groups need attention
- `per_entity.csv` sorted worst-first within each group pinpoints problem cases
- `summary.txt` shows all predictions for one entity at a glance — no cross-referencing
- Individual `.txt` files give the full LLM prompt + response for any single prediction
- `diagnostics/` separates "what went wrong" from "what are the results"

## 2. README.txt: The Most Important File

Every run must have a README.txt that a stranger (or yourself in 3 months) can
read and fully understand:

- **Purpose**: What hypothesis is this testing? Why does it matter?
- **Scope**: What dataset, how many subjects, what split, what subset
- **Model**: Exact model ID, mode (zero-shot/ICL/SFT), adapter path if any
- **Results**: Key numbers at a glance
- **Metrics definition**: What does each metric actually mean?
- **Navigation**: How to find what you're looking for in the file tree
- **Comparison context**: How does this relate to other methods' results?

A raw accuracy number is meaningless without context. README should state what
metric is being used, what the baselines are, and whether higher/lower is better.

## 3. Dual Format: CSV for Eyes, Parquet for Code

- CSVs in the hierarchy for `cat`, `head`, Excel, quick inspection
- Parquets at the root for pandas/polars programmatic analysis
- Never have parquet-only outputs in inspection directories

## 4. Sort Worst-First

Every summary table should be sorted so the worst results are at the top:
- `_index.csv`: groups sorted by ascending accuracy
- `per_entity.csv`: entities sorted by ascending accuracy within each group

You always want to investigate failures first. If everything looks good at the
top, you can stop reading.

## 5. LLM Input/Output Logging

Every LLM call should produce a plain-text file with clear sections:

```
=== INPUT ===
<full prompt with chat template applied>

=== OUTPUT ===
<raw LLM generation>

=== METADATA ===
entity_id: 42
group: task_A
item: question_1
actual: 5.0
predicted: 3.0
abs_error: 2.0
value_range: [1.0, 6.0]
```

Key lessons:
- **Include the templated prompt**, not just the messages dict. The actual tokens
  sent to the model matter (special tokens, chat format, system prompts).
- **Include ground truth in metadata** so you can evaluate each file standalone.
- **Include value range** so you can see if the prediction is plausible.
- Store `input_text` in the raw_outputs parquet too — if you forget to include it
  initially, you'll have to backfill it from log files later.

## 6. Diagnostics as First-Class Outputs

After every run, the first thing to check is not the accuracy number — it's:

1. **Parse failures**: How many outputs couldn't be parsed? What did they say?
   A small fraction is fine. A large fraction means your prompt or parsing is broken.
2. **Outliers**: How many predictions are outside the valid range?
   Models predicting 2-3 for binary 0/1 scales tells you immediately the model
   has no scale calibration.
3. **Value ranges**: What's the actual min/max per group? Comparing pred_mean
   vs actual_mean instantly reveals systematic biases.

## 7. Config Reproducibility

Save the **resolved** config, not a reference to a config file:

```json
{
  "mode": "zero_shot",
  "base_model": "org/model-name",
  "condition": "demographics",
  "seed": 42,
  "batch_size": 8,
  "max_new_tokens": 16
}
```

If you only save `"config": "configs/experiment.yaml"`, the config file may change
before you look at the results.

## 8. Train/Test Split Hygiene

- Split at the **entity level** (person, not observation). All observations from
  one entity must be in the same split.
- Save the split as a standalone JSON with explicit train/test ID lists.
- Every component (data prep, training, inference, evaluation, comparison) must
  load the same split file. Grep for "split" across the codebase to verify.
- When comparing against external baselines, document whether they used the same
  split. Often they didn't, and you need to note you're evaluating their
  predictions on your test subset.

## 9. Smoke Test Before Full Run

Before launching a multi-hour training or a large-scale inference run:

1. Run on 4 examples to verify the pipeline produces valid output
2. Check that the model loads, generates, and parses correctly
3. Verify the output schema matches expectations
4. Check for obvious issues (all predictions identical, all NaN, etc.)

This catches model loading issues, auth failures, prompt formatting bugs, and
parsing errors before you waste hours on a broken pipeline.

## 10. Post-Run Review Checklist

After every experiment run, systematically check:

- [ ] Parse failure rate (<1% acceptable, >5% investigate prompt/parsing)
- [ ] Prediction distribution vs actual distribution (mean, std)
- [ ] Per-group accuracy spread (which groups are catastrophic?)
- [ ] Outlier count and nature (systematic or random?)
- [ ] Spot-check 3-5 individual LLM input/output files for sanity
- [ ] Compare against baselines — does the ranking make sense?
- [ ] Update experiment notes with findings

---

## Anti-Patterns

- **Flat directory with thousands of files** — use group/entity subdirectories
- **Parquet-only outputs** — humans can't inspect without writing code
- **Discarding raw outputs after parsing** — you will need them for debugging
- **Undocumented runs** — saving metrics without context on what was tested
- **Config-by-reference** — pointing to a mutable config file instead of saving resolved values
- **Silent failures** — parse errors that decrement the count but don't tell you what happened
- **Forgetting input prompts** — saving the output but not the input that produced it
- **Optimistic sorting** — showing best results first; you need to see failures first
