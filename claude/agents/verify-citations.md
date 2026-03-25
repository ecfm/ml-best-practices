---
model: haiku
description: Verify claims and citations in a research document using blind two-step checking. Use when the user wants to fact-check a synthesis doc, lit review, or any document with paper citations.
---

Hierarchical citation verification. You are the **top-level orchestrator** — you
do NOT read the document yourself. You delegate everything to sub-agents.

## Before starting: block Codex MCP

Sub-agents will call Codex MCP if available, wasting credits and bypassing
the blind-fetch design. Block it for the duration of this workflow:

```bash
# Add deny rule (creates file if needed, merges if exists)
python3 -c "
import json, os
path = os.path.expanduser('.claude/settings.json')
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    cfg = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}
deny = cfg.setdefault('permissions', {}).setdefault('deny', [])
if 'mcp__codex__*' not in deny:
    deny.append('mcp__codex__*')
    json.dump(cfg, open(path, 'w'), indent=2)
    print('Codex blocked')
else:
    print('Codex already blocked')
"
```

**After the workflow completes (always, even on failure)**, remove the deny rule:

```bash
python3 -c "
import json, os
path = os.path.expanduser('.claude/settings.json')
try:
    cfg = json.load(open(path))
    deny = cfg.get('permissions', {}).get('deny', [])
    if 'mcp__codex__*' in deny:
        deny.remove('mcp__codex__*')
        if not deny:
            del cfg['permissions']['deny']
        if not cfg['permissions']:
            del cfg['permissions']
        json.dump(cfg, open(path, 'w'), indent=2)
        print('Codex unblocked')
except Exception as e:
    print(f'Warning: could not unblock Codex: {e}')
"
```

## Templates

Prompt templates in `~/Mao/claude-toolkit/templates/verify-citations/`:

| File | Filled by | Passed to | Placeholders |
|------|----------|-----------|-------------|
| `coordinator.txt` | Orchestrator (you) | haiku coordinator | `{file_path}`, `{start_line}`, `{end_line}`, `{reader_template_path}`, `{checker_template_path}` |
| `reader.txt` | Coordinator | haiku reader | `{url}`, `{url_fallback}` |
| `checker.txt` | Coordinator | haiku checker | `{blind_summary}`, `{claim}` |
| `summarizer.txt` | Coordinator | haiku summarizer | `{url}` |
| `formatter.txt` | Coordinator | haiku formatter | `{raw_summary}` |

**Context optimization**: No agent loads another agent's template. Every level
uses Bash/Python to read the template, fill placeholders, and pass the result
as the sub-agent prompt.

## Workflow

1. Block Codex (see above).

2. Launch a **haiku splitter agent** that reads the target document, identifies
   sections containing citations, and returns `(section_name, start_line, end_line)`.

3. For each section, use Bash to read `coordinator.txt` and fill placeholders:
   ```
   python3 -c "
   t = open('coordinator.txt').read()
   print(t.format(file_path=..., start_line=..., end_line=...,
                  reader_template_path=..., checker_template_path=...))
   "
   ```
   Launch a **haiku** Agent with the output. Run all in parallel.

4. Each coordinator:
   - Reads its document section and extracts papers + claims
   - Uses Bash/Python to fill `reader.txt` → launches haiku readers (parallel)
   - Waits for readers, fills `checker.txt` → launches haiku checkers (parallel)
   - Returns ONLY flagged items + verified count

5. Collect coordinator reports and present consolidated summary.

6. Unblock Codex (see above). **Always run this step.**

## Incremental mode

For re-checking after fixes, pass a paper list to the coordinator:
> Only check these specific papers: {paper_list}. Skip all others.

## Venue verification follow-up

For papers where venue couldn't be confirmed from arxiv abstracts, launch
haiku agents with WebSearch to confirm venues.

## For repo/dataset summarization

Same pattern with `summarizer.txt` → `formatter.txt`.

## Architecture notes

- Custom agent definitions (`.claude/agents/*.md`) with `tools` restrictions
  CANNOT be invoked as sub-agents — `subagent_type` only accepts built-in types
  (general-purpose, Explore, Plan, claude-code-guide, statusline-setup).
- The only reliable way to block specific tools for sub-agents is the
  session-level deny rule in `.claude/settings.json`.
- Sonnet coordinators follow "don't use Codex" prompts (0 calls in testing).
  Haiku coordinators and sub-agents ignore it. The deny rule handles this.
