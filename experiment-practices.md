# ML Experiment Practices

Lessons learned from building and evaluating LLM-based prediction systems.
These apply broadly to any ML project that involves model inference, evaluation,
and manual validation of results.

General coding rules (fail-fast, logging, monitoring) are in
[claude/CLAUDE.md](claude/CLAUDE.md). Pipeline architecture patterns are in
[pipeline-architecture.md](pipeline-architecture.md).

---

## 1. Output Structure: Hierarchical Drill-Down

Design outputs so a human can start from a 1-line summary and drill into any
specific data point. Every level should answer "is this good?" and "where
should I look next?"

Example structure (adapt the hierarchy and naming for your domain):

```
run_dir/
├── README.txt              ← Purpose, scope, data, model, metrics, navigation
├── config.json             ← Exact reproduction parameters (resolved, not file ref)
├── metrics.json            ← Top-level numbers
├── predictions.parquet     ← All predictions (programmatic access)
├── raw_outputs.parquet     ← All raw model outputs + input prompts
│
├── groups/                 ← Per-group breakdown (adapt naming for your domain)
│   ├── _index.csv          ← ENTRY POINT: one row per group, sorted worst-first
│   └── {group_name}/
│       ├── metrics.json
│       ├── predictions.csv      ← Human-readable (not parquet)
│       ├── per_sample.csv       ← Per-sample summary (sorted worst-first)
│       └── samples/
│           └── {sample_id}/
│               ├── summary.txt  ← All predictions for this sample at a glance
│               └── {input_id}.txt  ← Individual model input/output
│
└── diagnostics/
    ├── parse_failures.csv       ← Raw outputs that failed parsing
    ├── outliers.csv             ← Predictions outside valid range
    ├── value_ranges.json        ← Min/max/scale per group
    └── actual_vs_predicted.csv  ← Full flat table for custom analysis
```

### Why this structure works
- `_index.csv` sorted worst-first tells you immediately which groups need attention
- `per_sample.csv` sorted worst-first within each group pinpoints problem cases
- `summary.txt` shows all predictions for one sample at a glance — no cross-referencing
- Individual `.txt` files give the full model input + output for any single prediction
- `diagnostics/` separates "what went wrong" from "what are the results"

When comparing multiple methods, this inner structure nests inside an experiment
tracking layout (see section 11).

**Avoid:** Flat directories with thousands of files — use group/entity subdirectories.

## 2. README.txt: The Most Important File

Every run must have a README.txt that a stranger (or yourself in 3 months) can
read and fully understand:

- **Purpose**: What hypothesis is this testing? Why does it matter?
- **Scope**: What dataset, how many samples, what split, what subset
- **Model**: Exact model ID, training method, key hyperparameters
- **Results**: Key numbers at a glance
- **Metrics definition**: What does each metric actually mean?
- **Navigation**: How to find what you're looking for in the file tree
- **Comparison context**: How does this relate to other methods' results?

A raw number is meaningless without context. README should state what metric
is being used, what the baselines are, and whether higher/lower is better.

**Avoid:** Saving metrics without documenting what was tested — undocumented runs
become useless within weeks.

## 3. Dual Format: CSV for Eyes, Parquet for Code

- CSVs in the hierarchy for `cat`, `head`, Excel, quick inspection
- Parquets at the root for pandas/polars programmatic analysis
- Never have parquet-only outputs in inspection directories

**Avoid:** Parquet-only outputs — humans can't inspect without writing code.

## 4. Sort Worst-First

Every summary table should be sorted so the worst results are at the top:
- `_index.csv`: groups sorted worst-first by primary metric
- Per-sample tables: sorted worst-first within each group

You always want to investigate failures first. If everything looks good at the
top, you can stop reading.

**Avoid:** Showing best results first — you need to see failures first.

## 5. LLM Input/Output Logging

Every LLM call should produce a plain-text file with clear sections:

```
=== INPUT ===
<full prompt with chat template applied>

=== OUTPUT ===
<raw LLM generation>

=== METADATA ===
sample_id: 42
group: category_A
ground_truth: 5.0
predicted: 3.0
error: 2.0
valid_range: [1.0, 6.0]
```

Key lessons:
- **Include the templated prompt**, not just the messages dict. The actual tokens
  sent to the model matter (special tokens, chat format, system prompts).
- **Include ground truth in metadata** so you can evaluate each file standalone.
- **Include valid range** so you can see if the prediction is plausible.
- Store `input_text` in the raw_outputs parquet too — if you forget to include it
  initially, you'll have to backfill it from log files later.

**Avoid:** Discarding raw outputs after parsing — you will need them for debugging.
Also avoid saving the output without the input prompt that produced it.

## 6. Diagnostics as First-Class Outputs

After every run, the first thing to check is not the headline metric — it's:

1. **Parse failures**: How many outputs couldn't be parsed? What did they say?
   A small fraction is fine. A large fraction means your prompt or parsing is broken.
2. **Outliers**: How many predictions are outside the valid range?
   Models predicting 2-3 for binary 0/1 scales tells you immediately the model
   has no scale calibration.
3. **Value ranges**: What's the actual min/max per group? Comparing pred_mean
   vs actual_mean instantly reveals systematic biases.

**Avoid:** Silent parse errors that decrement the count but don't tell you what
happened — every failure should be logged with the raw output that caused it.

## 7. Train/Test Split Hygiene

- Split at the **unit of independence** (e.g., patient, document, user — not
  individual observations). All data from one unit must be in the same split.
- Save the split as a standalone JSON with explicit train/test ID lists.
- Every component (data prep, training, inference, evaluation, comparison) must
  load the same split file. Grep for "split" across the codebase to verify.
- When comparing against external baselines, document whether they used the same
  split. Often they didn't, and you need to note you're evaluating their
  predictions on your test subset.

## 8. Study-Specific Decisions: When to Ask vs Decide

Some decisions look like implementation details but can invalidate a study if
wrong. The rule: **make obvious choices and state them; ask when genuinely
ambiguous.**

### Usually determinable from context — decide and state clearly

| Decision | When it's obvious |
|----------|-------------------|
| **Data split strategy** | Time series → temporal; multi-row-per-unit → grouped; i.i.d. → random; imbalanced → stratified |
| **Split unit** | Data has `patient_id`/`user_id` with multiple rows → split by that ID |
| **Train/test ratio** | 80/20 or 80/10/10 is a reasonable default for most datasets |
| **Categorical encoding** | Nominal → one-hot; ordinal → ordinal encoding. Clear from variable type |
| **Missing data** | Trivial amount in non-critical features → drop rows |
| **Normalization** | Standardize numerical features. Rarely controversial |

### Must ask the user — genuinely ambiguous or high-stakes

| Decision | Why you can't guess |
|----------|---------------------|
| **Evaluation metric** | Accuracy vs F1 vs AUC vs MAE tell different stories; must match the research question |
| **Binarization/discretization thresholds** | Median split? Top/bottom quartile? No universal default |
| **Outlier exclusion criteria** | What counts as an outlier and whether to exclude is domain-specific |
| **Comparison baselines** | Which methods to compare against is the research question itself |
| **Feature selection** | When many options exist, which to include changes the study |

### Always ask when:

- Deviating from standard practice for a study-specific reason
- The best practice in this document conflicts with what the study requires
- Multiple reasonable approaches exist and the choice materially affects results

## 9. Smoke Test Before Full Run

Before launching a multi-hour training or a large-scale inference run, smoke test
on a small representative sample. The sample should:

- **Cover key variations**: different groups, edge cases, boundary values — not
  just the first N rows, which may all be from one group
- **Trigger the full lifecycle**: the sample and config must be small enough that
  evaluation, metrics logging, checkpointing, and output saving all fire within
  minutes. If your eval runs every 500 steps, use a config where 500 steps takes
  2 minutes, not 2 hours. Every behavior should be exercised, not just "training
  starts."

What to verify:
1. Model loads, generates, and parses correctly
2. Evaluation callback runs and produces valid metrics
3. Checkpoints save and can be loaded
4. Output files are written with expected schema and count
5. No obvious issues (all predictions identical, all NaN, etc.)

After the smoke test completes, run the full post-run review checklist (section 10)
on the smoke output. This catches model loading issues, auth failures, prompt
formatting bugs, parsing errors, broken eval callbacks, and checkpoint failures
before you waste hours on a broken pipeline.

## 10. Post-Run Review Checklist

After every experiment run, systematically check in this order (see sections
1, 5, 6 for where to find each piece):

**First — verify the job actually produced output:**
- [ ] Output files exist and are non-empty
- [ ] Output count matches expected input count (no silent data loss)
- [ ] Resolved config was logged — verify it matches what you intended

**Then — check for problems before looking at metrics:**
- [ ] Parse failure rate (<1% acceptable, >5% investigate prompt/parsing)
- [ ] Prediction distribution vs actual distribution (mean, std)
- [ ] Per-group metric spread (which groups are catastrophic?)
- [ ] Outlier count and nature (systematic or random?)
- [ ] Spot-check individual LLM input/output files for sanity

**Finally — evaluate and compare:**
- [ ] Compare against baselines — does the ranking make sense?
- [ ] If results are significantly better than expected, investigate for data
      leaks or bugs before trusting them
- [ ] Update experiment notes with findings

---

## 11. Experiment Lifecycle and Deduplication

Every pipeline run follows the same lifecycle via `ExperimentTracker`:

1. **prepare()** — compute experiment identity from config hash + git commit.
   Check registry: if this identity already completed, skip.
2. **start()** — create output dirs, write frozen config snapshot, record git state
3. **complete(metrics)** or **fail(error)** — update registry, save results

```
identity = {config_hash[:8]}_{git_commit_short}
```

Same config + same code = same identity → skip. Changed config or changed code =
new identity → run. This prevents re-running completed experiments and prevents
silently comparing results from different code versions.

A registry file (JSON with file locking) tracks status per identity. Failed
experiments are recorded (not just missing), so you don't waste time re-running
known failures.

The experiment output wraps around the per-run structure from section 1:

```
outputs/{method}/
├── registry.json                ← All experiments for this method
└── experiments/
    └── {identity}/
        ├── config.yaml          ← Frozen config snapshot (from to_dict)
        ├── metadata.json        ← Git commit, timestamp, host
        ├── metrics.json         ← Final evaluation results
        ├── stages/              ← Per-stage intermediate outputs
        ├── llm_logs/            ← LLM I/O for debugging (section 5)
        ├── groups/              ← Per-group drill-down (section 1)
        └── diagnostics/         ← Parse failures, outliers (section 6)
```

**Avoid:** Re-running identical experiments — it wastes compute and creates
confusion about which result is canonical.

### Run Stability Log

Maintain a `runs.yaml` in the project repo to track which runs are stable (safe to
run unmonitored) vs need active watching. This is separate from the experiment
registry — the registry tracks *what was run*, the stability log tracks *whether
it's safe to trust*.

```yaml
# runs.yaml
- id: "abc123_8f3d"
  config: configs/method_a/base.yaml
  status: stable        # stable | unstable | investigating
  git_commit: "a1b2c3d"
  date: "2026-02-09 14:30"
  env:
    server: gpu-server-1
    gpu: A100-80GB
    python: "3.11"
    vllm: "0.4.1"
    torch: "2.2.0"
  notes: "Baseline run, validated on full test set"
  metrics: {mae: 0.82, correlation: 0.45}

- id: "def456_2a1b"
  config: configs/method_b/large.yaml
  status: unstable
  git_commit: "e4f5g6h"
  date: "2026-02-08 09:15"
  env:
    server: gpu-server-2
    gpu: A6000-48GB
    python: "3.11"
    vllm: "0.4.0"
    torch: "2.2.0"
  notes: "Parse failure rate 12%, prompt needs fixing"
```

Key fields for determining stability:
- **git_commit**: Code changes can break previously stable runs
- **date**: With timestamp, so you know recency
- **env**: Server, GPU, Python version, key library versions — a run stable on
  A100 may OOM on A6000, or behave differently with a different vLLM version
- **status**: `stable` means safe to run without active monitoring; `unstable`
  means must watch the first 5 minutes; `investigating` means known issues

Reference this from the project's CLAUDE.md:
```
Check runs.yaml for prior run status before launching jobs.
```

---

## 12. Real-Time Progress Monitoring

Long-running jobs (training, batch inference) MUST produce output within the first
few minutes or you risk wasting hours on a silently broken run.

### The 5-Minute Rule

If no progress output appears within 5 minutes of launch, something is wrong.
Every long-running job must:

1. **Log step 1 immediately.** Don't set `logging_steps=10` and wait 15 minutes
   for the first sign of life. Log every step (or at least step 1) so you can
   compute s/step and ETA from the very first iteration.

2. **Write to a real-time log file**, not just stdout. Stdout may be buffered,
   piped, or lost. Always write to `run_dir/training.log` or similar:

   ```python
   log_file = open(run_dir / "training.log", "w")
   def _log(msg: str) -> None:
       ts = time.strftime("%H:%M:%S")
       line = f"[{ts}] {msg}"
       print(line, flush=True)
       log_file.write(line + "\n")
       log_file.flush()
   ```

   Monitor with `tail -f run_dir/training.log`.

3. **Validate timing on step 1 before walking away.** After launching, wait for
   step 1, compute: `total_steps × s_per_step = ETA`. If ETA is unreasonable,
   kill immediately and fix config. A 3-epoch training run showing 85s/step ×
   7389 steps = 175 hours means your config is wrong — don't let it run overnight
   hoping it speeds up.

4. **Batch long blocking calls.** If a library call like `llm.generate()` blocks
   for the entire dataset with no progress callback, split into batches and log
   between them:

   ```python
   BATCH = 5000
   for start in range(0, len(prompts), BATCH):
       batch = prompts[start:start+BATCH]
       outputs.extend(model.generate(batch))
       _log(f"Generated {start+BATCH}/{len(prompts)} ({elapsed}s, ~{eta}s remaining)")
   ```

### What to Log

Every training step should log:
- Step N / total steps
- Loss (train and/or eval)
- Elapsed time and ETA
- Learning rate (if using schedule)

Every inference batch should log:
- N processed / total
- Elapsed time, prompts/second, ETA
- Parse failure count so far

### Pre-Flight Validation

Before launching multi-hour jobs, validate:
- [ ] Expected s/step matches hardware capability (benchmark 1-2 steps first)
- [ ] Total ETA is acceptable
- [ ] GPU memory usage leaves headroom (check after step 1)
- [ ] Data pipeline produces correct shapes (spot-check first batch)
- [ ] Output directory is writable and has enough disk space

## 13. Common ML Silent Failures

Silent failures are the most expensive bugs in ML. A model that trains for 8 hours
on corrupted data, or inference that runs for 2 hours producing garbage, wastes
both compute and human attention. Watch for these:

- `device_map="auto"` silently uses pipeline parallelism (1 GPU idle during
  training). Use explicit device placement or DDP/FSDP.
- `max_length` set too high → extreme padding → 10x slower training with no error.
- Tokenizer `pad_token` defaulting to `eos_token` → model learns to predict EOS
  everywhere. Usually works but can cause subtle generation issues.
- `gradient_accumulation_steps` silently changes effective batch size. Always log
  the effective batch size at training start.
- Data collator silently truncating sequences longer than model's max position
  embeddings — no error, just wrong results.

## 14. Complete Each Run Fully Before Starting the Next

The fastest way to find bugs is to produce a result. Don't batch all training
runs, then all inference runs, then all evaluations. Instead, for each
configuration: **train → infer → validate**, then move on.

### Why

1. **Earlier bug detection.** A broken prompt, misconfigured model, or data
   pipeline bug shows up in the first complete run's output — not after N
   training jobs have all consumed GPU-hours.

2. **Results inform next steps.** If the smallest config already matches the
   baseline, larger configs may be unnecessary. If it catastrophically fails,
   you know before wasting time on bigger runs.

3. **Partial work has zero value.** A trained model with no inference results
   is useless for the paper. A completed training + inference + validated result
   is a full data point you can report, even if everything else fails.

### Where This Applies

- **Hyperparameter sweeps**: Run the fastest config fully (smallest data, 1
  epoch) before scaling up. Validate output quality, then proceed.
- **Data budget experiments** (e.g. 10% → 25% → 50% → 100% of training data;
  actual splits depend on the study and should be confirmed with the user):
  Each budget is a self-contained experiment. Finish smallest first.
- **Multi-condition experiments**: Complete one condition end-to-end before
  starting others.
- **Pipeline changes**: After modifying any stage, run one example end-to-end
  before scaling up.
- **Multi-model comparisons**: Finish one model's full pipeline before starting
  the next.

### Anti-Pattern: Batch-by-Stage

```
# BAD: batch by stage
Train config_A, config_B, config_C     # 8 hours
Then infer config_A, config_B, config_C  # 2 hours
Then validate all                          # 30 min
→ First result at hour 8.5, bugs found at hour 10.5

# GOOD: batch by config
Train config_A → Infer → Validate      # 1.5 hours → first result + validated
Train config_B → Infer → Validate      # 2 hours
...
→ First result at hour 1.5, bugs found at hour 1.5
```

### Queue Script Pattern

```bash
for config in config_small config_medium config_large; do
    echo "=== ${config}: Training ==="
    train $config

    echo "=== ${config}: Inference ==="
    infer $config

    echo "=== ${config}: Validation ==="
    validate $config
done
```

## 15. Early Stopping for Experiments

Not every experiment needs to run to completion. Know when to stop early — both
within a training run and across an experiment series.

### Within a Training Run

Use early stopping when training loss has clearly plateaued or diverged:

- **Patience-based**: Stop if eval loss doesn't improve for N consecutive
  evaluations. HuggingFace Trainer supports this with
  `EarlyStoppingCallback(early_stopping_patience=3)`.
- **Save checkpoints frequently enough** that early stopping doesn't lose work.
  `save_strategy="steps"` with `save_steps=200` is better than
  `save_strategy="epoch"` for long epochs.
- **Watch for overfitting**: If train loss drops but eval loss rises, stop
  immediately. More training is actively harmful.
- **Track prediction quality, not just loss**: Eval loss can decrease while
  actual prediction quality stagnates. Add eval callbacks that generate
  sample predictions and compute task-specific metrics (MAE, correlation)
  during training.

### Across an Experiment Series

Use the "complete each run fully" principle (section 14) to enable early
stopping of the series:

- If the smallest config produces terrible results (e.g., metric below
  random baseline), investigate before scaling up — the problem may be in
  the pipeline, not the data volume.
- If the smallest config already matches the target baseline, larger configs
  may give diminishing returns — run them for completeness but deprioritize.
- If results degrade with more data or compute, something is wrong. Stop
  and investigate before proceeding.

### What to Monitor for Early Stopping Decisions

| Signal | Action |
|--------|--------|
| Eval loss flat for 3+ evals | Stop training (patience exceeded) |
| Eval loss increasing | Stop immediately (overfitting) |
| Config N below random baseline | Investigate pipeline before config N+1 |
| Config N ≈ Config N-1 (within noise) | Diminishing returns — consider stopping series |
| Config N worse than Config N-1 | Stop and investigate before continuing |

### Implementation

```python
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

Ensure `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"`
so the final saved model is the best checkpoint, not the last one.

---

## 16. vLLM and Prefix Caching for Batch Inference

Use [vLLM](https://github.com/vllm-project/vllm) instead of HuggingFace
`model.generate()` for batch inference. vLLM provides continuous batching,
PagedAttention, and automatic prefix caching — often 10-50x faster.

**Prompt ordering matters for prefix caching.** vLLM caches the KV cache for
shared prompt prefixes. If multiple prompts share a long common prefix, group
them together so the prefix is computed once and reused. The optimal ordering
depends on your prompt structure — identify which part is longest and shared
(system prompt, context, instructions) vs which part varies per call, and sort
so prompts with the same prefix are adjacent.

**Design prompts for prefix compatibility.** The shared content must come before
the variable content in the prompt. The prefix is everything up to the first
token that differs between prompts — anything after that point is recomputed.

Other vLLM tips:
- Use `tensor_parallel_size=N` to shard across multiple GPUs
- Set `max_model_len` to just above your longest prompt (saves KV cache memory)
- `gpu_memory_utilization=0.85` is a safe default; lower if other processes
  share the GPU
