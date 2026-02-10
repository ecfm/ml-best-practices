# Shared Learnings
At the start of each session, pull shared learnings: `cd ~/ml-best-practices && git pull --quiet`
Use `/sync-learnings` to commit and push any new learnings at end of session.

# General Rules
- Always use uv to manage dependencies and run Python
- Fail-fast: no try-catch to silence errors, no default values that mask missing or misconfigured inputs (includes .get with defaults, optional function args that should be required, fillna that hides missing data). Only use try-catch when there is no alternative and warn the user
- Log to files (not just stdout) so long-running jobs can be monitored via `tail -f`. Avoid verbose stdout in Claude Code — redirect to log files and check periodically. Never let a job run more than 5 minutes without checking logs, unless proven stable (check runs.yaml)
- Before any long run, smoke test on a small representative sample that covers key variations (different groups, edge cases, boundary values). The sample and config must be small enough to trigger the full lifecycle quickly — evaluation, metrics logging, checkpointing, output saving — so every behavior is exercised, not just "training starts"
- Complete each run fully (train → infer → validate) before starting the next
- Preserve evidence: save inputs, outputs, and resolved configuration at each pipeline stage. Never overwrite previous results — use unique output directories
- Sort worst-first in all summary tables
- After runs: verify outputs exist and have expected count, then check diagnostics and parse failures before looking at metrics. Investigate suspiciously good results — unexpected jumps likely indicate data leaks or bugs, not breakthroughs
- For study-specific decisions: when the choice is obvious from context or data structure (e.g., grouped split for multi-row-per-unit data, standard normalization), make the choice and state it clearly. When the choice is genuinely ambiguous or could invalidate results (e.g., evaluation metric, binarization threshold, outlier exclusion, comparison baselines), always ask the user before proceeding. When deviating from standard practice, always ask. See experiment-practices.md section 8 for details.

# Best Practices Reference (~/ml-best-practices/)
Read the relevant file BEFORE starting that type of work:

| When you are...                    | Read                         |
|------------------------------------|------------------------------|
| Running or planning ML experiments | experiment-practices.md      |
| Designing a new pipeline or method | pipeline-architecture.md     |
