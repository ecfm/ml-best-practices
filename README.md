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

## What Gets Synced

```
claude/
├── CLAUDE.md              → ~/.claude/CLAUDE.md          # Global instructions
├── settings.json          → ~/.claude/settings.json      # Hooks, theme, status line
├── commands/*.md          → ~/.claude/commands/           # Slash commands
└── rules/                 → ~/.claude/rules/             # Auto-loaded rules
    ├── coding-style.md    #   uv, fail-fast, logging, evidence preservation
    ├── experiment-runs.md #   smoke test, complete-each-run, post-run checks
    └── decisions.md       #   when to ask vs decide
```

## Setup

```bash
git clone git@github.com:ecfm/ml-best-practices.git ~/ml-best-practices
bash ~/ml-best-practices/install.sh
```

This symlinks all Claude Code config files so every session inherits shared rules,
commands, hooks, and settings.

### Notifications (optional)

Hooks send push notifications via [ntfy.sh](https://ntfy.sh) when Claude Code
completes a task. This works over tmux/mosh/SSH where desktop notifications don't.

```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc):
export NTFY_TOPIC="your-unique-topic"
```

Then subscribe to `ntfy.sh/your-unique-topic` on your phone or browser.
If `NTFY_TOPIC` is unset, the notification hook is a no-op.

## Usage

- Best practices are loaded into Claude Code sessions via the routing table in
  `claude/CLAUDE.md`
- Rules in `claude/rules/` are auto-loaded by Claude Code from `~/.claude/rules/`
- Use `/sync-learnings` at end of session to commit and push new learnings
- `runs.yaml` in each project repo tracks run stability (see experiment-practices.md
  section 11)
