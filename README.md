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

When comparing multiple methods, this inner structure nests inside an experiment
tracking layout (see section 15).

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

## 7. Train/Test Split Hygiene

- Split at the **entity level** (person, not observation). All observations from
  one entity must be in the same split.
- Save the split as a standalone JSON with explicit train/test ID lists.
- Every component (data prep, training, inference, evaluation, comparison) must
  load the same split file. Grep for "split" across the codebase to verify.
- When comparing against external baselines, document whether they used the same
  split. Often they didn't, and you need to note you're evaluating their
  predictions on your test subset.

## 8. Smoke Test Before Full Run

Before launching a multi-hour training or a large-scale inference run:

1. Run on 4 examples to verify the pipeline produces valid output
2. Check that the model loads, generates, and parses correctly
3. Verify the output schema matches expectations
4. Check for obvious issues (all predictions identical, all NaN, etc.)

This catches model loading issues, auth failures, prompt formatting bugs, and
parsing errors before you waste hours on a broken pipeline.

## 9. Post-Run Review Checklist

After every experiment run, systematically check:

- [ ] Parse failure rate (<1% acceptable, >5% investigate prompt/parsing)
- [ ] Prediction distribution vs actual distribution (mean, std)
- [ ] Per-group accuracy spread (which groups are catastrophic?)
- [ ] Outlier count and nature (systematic or random?)
- [ ] Spot-check 3-5 individual LLM input/output files for sanity
- [ ] Compare against baselines — does the ranking make sense?
- [ ] Update experiment notes with findings

## 10. vLLM and Prefix Caching for Batch Inference

Use [vLLM](https://github.com/vllm-project/vllm) instead of HuggingFace
`model.generate()` for batch inference. vLLM provides continuous batching,
PagedAttention, and automatic prefix caching — often 10-50x faster.

**Prompt ordering matters for prefix caching.** When you have N entities × M
items (e.g. 589 people × 72 survey questions), all prompts for the same entity
share a long prefix (system prompt + entity profile). Sort prompts by entity
so vLLM's prefix cache can reuse the KV cache:

```
# BAD: sorted by item (default from data)
# Each prompt gets full prefill — cache is cold
person_1/item_A, person_2/item_A, person_3/item_A, ...

# GOOD: sorted by entity
# Only first prompt per person pays full prefill cost
person_1/item_A, person_1/item_B, ..., person_1/item_N,
person_2/item_A, person_2/item_B, ...
```

**Design prompts for prefix compatibility.** The shared context (entity profile)
must come before the variable part (question) in the prompt. The prefix is
everything up to the first token that differs between prompts:

```
[SHARED PREFIX — cached after first prompt]
<system>You are simulating...</system>
<user>Here is the profile:
feature_1: 3.0
feature_2: 1.0
...
feature_251: 5.0

Question: In study '
[VARIABLE SUFFIX — unique per prompt]
study_name', for item 'variable_name'...
```

With 589 people × 72 items, this turns 42K full prefills (3K tokens each) into
589 full prefills + 41K cache hits (~20 tokens each). In practice: **~3 hours
→ ~20 minutes**.

Other vLLM tips:
- Use `tensor_parallel_size=N` to shard across multiple GPUs
- Set `max_model_len` to just above your longest prompt (saves KV cache memory)
- `gpu_memory_utilization=0.85` is a safe default; lower if other processes
  share the GPU

---

## 11. Shared Core, Isolated Pipelines

Separate universal infrastructure from method-specific logic:

```
src/
├── core/                    ← Shared by ALL pipelines
│   ├── types.py             ← Sample, Prediction (universal currency)
│   ├── data.py              ← DataLoader (one source of truth)
│   ├── evaluation.py        ← evaluate() (unified metrics)
│   ├── experiment.py        ← ExperimentTracker (lifecycle)
│   └── utils.py             ← JSON parsing, retry, formatting
│
├── pipelines/               ← Each method is self-contained
│   ├── method_a/
│   │   ├── __init__.py      ← Exports Config + Pipeline
│   │   ├── config.py        ← Dataclass with to_dict/from_dict/validate
│   │   └── runner.py        ← Pipeline.run(train, val, test)
│   └── method_b/
│       └── ...
│
└── run_experiment.py        ← Dispatcher: --method flag → pipeline
```

Adding a new method means creating a directory and registering it in the
dispatcher. Zero changes to existing pipelines, data loading, or evaluation.

## 12. Universal Type Contracts

All pipelines consume `Sample` and produce `Prediction`. No exceptions.

- **Sample**: stores raw labels (floats), converts to classification/binary/continuous
  on demand via `get_label(trait, mode)`. One storage format, many views.
- **Prediction**: carries `sample_id`, `trait`, `value`, plus a `metadata` dict for
  method-specific info.
- **Evaluation**: a single `evaluate(predictions, ground_truth, trait)` handles any
  prediction type — bins regression outputs into classes, maps categorical outputs to
  ordinals for correlation — so every method gets the same metrics automatically.

This makes methods directly comparable. Swap pipeline A for pipeline B and the rest
of the system doesn't change.

## 13. Config as Dataclass, Not Dict

If you only save `"config": "configs/experiment.yaml"`, the config file may change
before you look at the results. And if configs are plain dicts, typos become silent
bugs. Solve both problems by making every pipeline config a dataclass with three
required methods:

```python
@dataclass
class PipelineConfig:
    learning_rate: float = 1e-3
    batch_size: int = 4
    traits: list[str] = field(default_factory=lambda: ["extraversion"])

    def to_dict(self) -> dict:       # Serializes resolved values for hashing and storage
        ...
    @classmethod
    def from_dict(cls, d) -> Self:   # Deserializes from YAML
        ...
    def validate(self) -> list[str]: # Returns warnings for suspicious values
        ...
```

`to_dict()` produces the resolved config that gets saved with every experiment —
no mutable file references, no missing fields. Dataclasses reject unknown fields
at construction. `validate()` catches "technically valid but probably wrong" configs
(learning rate of 100, negative epochs).

Support config inheritance (`_base: base.yaml`) and presets
(`_presets: [fast_test, binary]`) for composable experiment variants:

```yaml
# configs/method_a/experiment_1.yaml
_base: base.yaml
_presets:
  - fast_test        # small sample sizes for iteration
  - binary           # median-split labels

traits: [extraversion, agreeableness]
learning_rate: 5e-4
```

Presets let you mix-and-match parameter groups without a combinatorial explosion of
config files.

## 14. Multi-Stage Pipelines with Stage Caching

For complex pipelines, define stages as independent units:

```python
class Stage(ABC):
    name: str
    depends_on: list[str]          # Explicit DAG

    def run(self, inputs: dict) -> StageResult
    def load_cached(self) -> dict | None
    def save_outputs(self, outputs: dict, metrics: dict)
```

Each stage reads from previous stages' outputs, saves its own outputs with a manifest,
and can be independently cached or re-run. The runner orchestrates:
check cache → run if missing → pass outputs forward.

A 3-stage pipeline (extract → score → predict) where stage 1 takes 2 hours should not
re-run stage 1 when you're iterating on stage 3. Stage caching makes this automatic.

## 15. Experiment Lifecycle and Deduplication

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
        ├── stages/              ← Per-stage intermediate outputs (section 14)
        ├── llm_logs/            ← LLM I/O for debugging (section 5)
        ├── groups/              ← Per-group drill-down (section 1)
        └── diagnostics/         ← Parse failures, outliers (section 6)
```

---

## 16. Real-Time Progress Monitoring

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

## 17. Fail-Fast Principle

Silent failures are the most expensive bugs in ML. A model that trains for 8 hours
on corrupted data, or inference that runs for 2 hours producing garbage, wastes
both compute and human attention.

### Rules

1. **Never use try/except to silence errors.** Let exceptions propagate. Only catch
   at true system boundaries (e.g., parsing one user input out of thousands) and
   always log the caught exception with a warning.

2. **Never use `.get(key, default)` for required config values.** If a config key
   is missing, that's a bug — let it throw `KeyError` immediately rather than
   silently using a default that produces wrong results.

3. **Validate inputs at entry points.** Check that files exist, datasets are
   non-empty, GPUs are available, and model names resolve before doing any
   expensive work.

4. **Assert invariants, don't check-and-continue.** If your training data should
   have >0 examples, `assert len(train_ds) > 0` is better than
   `if len(train_ds) == 0: print("warning")`.

5. **Fail on the first bad example, not the last.** If parsing fails, raise
   immediately with context (which example, what the raw output was, what was
   expected). Don't silently count failures and report "80 parse failures" at
   the end with no way to debug them.

### Common ML Silent Failures

- `device_map="auto"` silently uses pipeline parallelism (1 GPU idle during
  training). Use explicit device placement or DDP/FSDP.
- `max_length` set too high → extreme padding → 10x slower training with no error.
- Tokenizer `pad_token` defaulting to `eos_token` → model learns to predict EOS
  everywhere. Usually works but can cause subtle generation issues.
- `gradient_accumulation_steps` silently changes effective batch size. Always log
  the effective batch size at training start.
- Data collator silently truncating sequences longer than model's max position
  embeddings — no error, just wrong results.

## Anti-Patterns

- **Flat directory with thousands of files** — use group/entity subdirectories
- **Parquet-only outputs** — humans can't inspect without writing code
- **Discarding raw outputs after parsing** — you will need them for debugging
- **Undocumented runs** — saving metrics without context on what was tested
- **Configs as raw dicts or by reference** — dicts accept typos silently; file references go stale. Use typed dataclasses that serialize resolved values
- **Silent failures** — parse errors that decrement the count but don't tell you what happened
- **Forgetting input prompts** — saving the output but not the input that produced it
- **Optimistic sorting** — showing best results first; you need to see failures first
- **Shared mutable state between pipelines** — pipelines should communicate only through typed inputs/outputs, never global state
- **Method-specific evaluation logic** — if pipeline A computes F1 differently than pipeline B, comparison is meaningless
- **No deduplication** — re-running identical experiments wastes compute and creates confusion about which result is canonical
