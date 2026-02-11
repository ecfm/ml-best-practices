# Coding Style

- Always use uv to manage dependencies and run Python
- Fail-fast: no try-catch to silence errors, no default values that mask missing or misconfigured inputs (includes .get with defaults, optional function args that should be required, fillna that hides missing data). Only use try-catch when there is no alternative and warn the user
- Log to files (not just stdout) so long-running jobs can be monitored via `tail -f`. Avoid verbose stdout in Claude Code — redirect to log files and check periodically. Never let a job run more than 5 minutes without checking logs, unless proven stable (check runs.yaml)
- Preserve evidence: save inputs, outputs, and resolved configuration at each pipeline stage. Never overwrite previous results — use unique output directories
