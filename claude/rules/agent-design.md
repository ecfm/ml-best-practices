# Agent Design

## When to Use What

- **Hooks**: deterministic enforcement that runs every time (formatting, blocking, notifications). Use when a rule MUST be enforced 100%.
- **Skills**: reusable workflows where Claude's judgment picks when to invoke. Use for domain patterns and knowledge.
- **Agents**: isolated context + tool restrictions. Use for parallel work, protecting main context from verbose output, or enforcing tool boundaries.

## Model Selection by Role

- **Haiku**: mechanical delegation — follows templates literally, good for narrow fetch/check tasks. BUT: gets confused by negative tool constraints ("you can ONLY use X"), may not find the Agent tool when told it's restricted.
- **Sonnet**: judgment and analysis. BUT: takes shortcuts on mechanical work (does checks inline instead of delegating to sub-agents). Don't use as orchestrator for mechanical delegation.
- **Opus**: complex orchestration and synthesis.

## Agent Design Patterns

- Fill templates via Bash/Python — parent reads template file, fills placeholders with code, passes result as sub-agent prompt. Template text never enters parent's conversation context.
- Custom agent types (in `.claude/agents/`) can't be invoked via `subagent_type` parameter — only built-in types (`general-purpose`, `Explore`, `Plan`, `claude-code-guide`) work.
- Agent definitions in `.claude/agents/` load at session start. Use `/agents` to hot-reload mid-session.
- Sub-agents CAN launch other sub-agents in practice (despite docs suggesting otherwise).
- Compact reports: sub-agents should return only flagged/actionable items + counts, not full tables of verified items. This saves main session context.

## Tool Blocking

- Use `PreToolUse` hooks with file-based toggles for per-skill tool blocking.
- Pattern: `touch /tmp/.block_toolname` before workflow, `rm` after. Hook checks file existence and exits 2 to block.
- Add TTL: hook checks file age, ignores if older than 1 hour. Self-heals if workflow crashes.
- This is the only reliable way to block MCP tools from sub-agents — prompt-based restrictions are unreliable with Haiku.

## Blind Verification Pattern

- For fact-checking: Reader agent fetches source without seeing the claim, Checker agent compares blind summary against claim without seeing the source.
- Prevents confirmation bias. Doubles agent count but worth it for accuracy.
- Reader: `WebFetch` only. Checker: no tools (pure text comparison).
