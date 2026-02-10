# ML/LLM Best Practices

Shared knowledge base for ML experiment workflows. Synced across servers via Git
and loaded into Claude Code sessions automatically.

## Contents

- **[experiment-practices.md](experiment-practices.md)** — Output structure, logging,
  diagnostics, monitoring, run ordering, early stopping, vLLM optimization. Read this
  before running or planning ML experiments.

- **[pipeline-architecture.md](pipeline-architecture.md)** — Modular pipeline design,
  type contracts, config management, stage caching. Read this when designing a new
  pipeline or adding a new method.

## Setup

```bash
git clone git@github.com:ecfm/ml-best-practices.git ~/ml-best-practices
bash ~/ml-best-practices/install.sh
```

This symlinks Claude Code config files (`~/.claude/CLAUDE.md`, `~/.claude/commands/`)
so every session inherits the shared rules and commands.

## Usage

- Best practices are loaded into Claude Code sessions via the routing table in
  `claude/CLAUDE.md`
- Use `/sync-learnings` at end of session to commit and push new learnings
- `runs.yaml` in each project repo tracks run stability (see experiment-practices.md
  section 11)
