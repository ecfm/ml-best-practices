# Pipeline Architecture Patterns

Design patterns for building multi-method ML experiment systems. Read this when
designing a new pipeline or adding a new method — not needed for running experiments.

See also: [experiment-practices.md](experiment-practices.md) for running experiments.

---

## 1. Shared Core, Isolated Pipelines

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

**Avoid:** Shared mutable state between pipelines — pipelines should communicate
only through typed inputs/outputs, never global state.

## 2. Universal Type Contracts

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

**Avoid:** Method-specific evaluation logic — if pipeline A computes F1 differently
than pipeline B, comparison is meaningless.

## 3. Config as Dataclass, Not Dict

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

**Avoid:** Configs as raw dicts or by reference — dicts accept typos silently; file
references go stale. Use typed dataclasses that serialize resolved values.

## 4. Multi-Stage Pipelines with Stage Caching

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
