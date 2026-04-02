# Paper Review Templates

Multi-layer verification system for empirical research papers before submission.
See `research-methodology.md` Section 8 for the full methodology.

## Structure

```
paper-review/
├── phase0/
│   ├── check_numbers.py      # Number reconciliation (provenance → tex → source)
│   └── check_staleness.py    # Output freshness detection (script hash → output hash)
├── prompts/
│   ├── E_execute.md           # Execute & verify (run script, compare outputs)
│   ├── M1_metric_validity.md  # "Does this metric measure what you think?"
│   ├── M2_train_test.md       # Train/test integrity audit
│   ├── M3_statistical.md      # CI/p-value alignment check
│   ├── A1_break_argument.md   # Try to construct counterexample to main claim
│   ├── A2_baseline_fairness.md # Is the comparison systematically unfair?
│   ├── A3_missing_analyses.md # Reviewer simulation: what's absent?
│   ├── naive_reader.md        # First-time reader comprehension test
│   └── citation_verify.md    # Attribution accuracy check
└── launch.py                 # Python orchestrator (replaces bash)
```

## Usage

### 1. Copy to your paper repo
```bash
cp -r templates/paper-review/ my-paper/review/
```

### 2. Customize
- Edit `check_numbers.py`: point `PROVENANCE_PATH` to your provenance file
- Edit `check_staleness.py`: fill in `DEPENDENCIES` with your script→output mappings
- Edit prompts: replace `[PAPER_CLAIM]` placeholders with your paper's specific claims

### 3. Run Phase 0 first (free, instant)
```bash
python review/phase0/check_staleness.py
python review/phase0/check_numbers.py
```

### 4. Run LLM reviews
```bash
python review/launch.py --parallel 2 --layer exec
python review/launch.py --parallel 2 --layer method
python review/launch.py --parallel 2 --layer adversarial
```

## Model Selection

| Track type | Recommended model | Why |
|-----------|-------------------|-----|
| Execute (E) | Codex/GPT-5.4 | Needs tool access for code execution |
| Methodology (M) | Claude Opus | Deep multi-file reasoning |
| Adversarial (A) | Claude Opus | Creative argument construction |
| Naive reader | Any fresh model | Must not have seen the paper before |
| Citation verify | Sonnet + web search | Needs to look up papers |
