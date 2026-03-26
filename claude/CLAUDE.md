# Shared Learnings
At the start of each session, pull shared learnings: `cd ~/Mao/claude-toolkit && git pull --quiet`
Use `/sync-learnings` to commit and push any new learnings at end of session.

# Best Practices Reference (~/Mao/claude-toolkit/)
Read the relevant file BEFORE starting that type of work:

| When you are...                    | Read                         |
|------------------------------------|------------------------------|
| Running or planning ML experiments | experiment-practices.md      |
| Designing a new pipeline or method | pipeline-architecture.md     |
| Lit review, research planning, or agent workflows | research-methodology.md |
| Designing agents, skills, or hooks | claude/rules/agent-design.md |

# Committing

Use `commit-push` (Haiku agent) for all commits and pushes. Proactively invoke it whenever:
- A feature or task is complete
- Git diff shows 200+ lines changed
- The user says "done", "ship it", "that's it", or similar

Never commit manually with the Bash tool when this agent is available.

**Testing** — most commits need no tests. Two separate agents:

- `test-and-fix`: runs specified tests, fixes **implementation only** (never touches test files). Use when a change could break existing behavior (refactor of core logic, schema change, API contract change).
- `write-tests`: writes new tests per spec, runs them, fixes until passing. Use when a new feature has non-trivial logic worth protecting (not wiring/UI/config).

When invoking either, specify exactly which test files/functions to run — never the full suite.
