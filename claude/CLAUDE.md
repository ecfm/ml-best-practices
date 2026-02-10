# Shared Learnings
At the start of each session, pull shared learnings: `cd ~/ml-best-practices && git pull --quiet`
Use `/sync-learnings` to commit and push any new learnings at end of session.

# General Rules
- Always use uv to manage dependencies and run Python
- Fail-fast: no try-catch to silence errors, no .get with defaults on required keys. Only use try-catch when there is no alternative and warn the user
- Log to files (not just stdout) so long-running jobs can be monitored via `tail -f`. Avoid verbose stdout in Claude Code — redirect to log files and check periodically
- Never let a job run more than 5 minutes without checking logs — especially after code changes, new experiments, or in a new environment. Only skip monitoring for runs proven stable (check runs.yaml)
- Smoke test (4 examples) before any long run
- After changes, run a small but complete test and debug until it succeeds before running long jobs
- Complete each run fully (train → infer → validate) before starting the next
- Sort worst-first in all summary tables
- Always save LLM inputs (templated prompt) alongside outputs
- After runs: check diagnostics and parse failures BEFORE looking at accuracy numbers
- Review logs, inputs, intermediate results, outputs, metrics to verify all steps follow expectations

# Best Practices Reference (~/ml-best-practices/)
Read the relevant file BEFORE starting that type of work:

| When you are...                    | Read                         |
|------------------------------------|------------------------------|
| Running or planning ML experiments | experiment-practices.md      |
| Designing a new pipeline or method | pipeline-architecture.md     |
