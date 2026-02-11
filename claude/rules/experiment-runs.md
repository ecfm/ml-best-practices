# Experiment Runs

- Before any long run, smoke test on a small representative sample that covers key variations (different groups, edge cases, boundary values). The sample and config must be small enough to trigger the full lifecycle quickly — evaluation, metrics logging, checkpointing, output saving — so every behavior is exercised, not just "training starts"
- Complete each run fully (train -> infer -> validate) before starting the next
- Sort worst-first in all summary tables
- After runs: verify outputs exist and have expected count, then check diagnostics and parse failures before looking at metrics. Investigate suspiciously good results — unexpected jumps likely indicate data leaks or bugs, not breakthroughs
