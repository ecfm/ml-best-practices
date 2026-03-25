---
model: haiku
description: Verify claims and citations in a research document using blind two-step checking. Use when the user wants to fact-check a synthesis doc, lit review, or any document with paper citations.
---

Hierarchical citation verification. You are the **top-level orchestrator** — you
do NOT read the document yourself. You delegate everything to sub-agents.

## Codex blocking

Before launching any coordinators, block Codex to prevent sub-agents wasting credits:
```
touch /tmp/.block_codex
```

After ALL coordinators complete, remove the block:
```
rm -f /tmp/.block_codex
```

This works via a PreToolUse hook in project settings that checks for this file.
Other skills and other projects are unaffected.

## Templates

Prompt templates in `~/Mao/claude-toolkit/templates/verify-citations/`:

| File | Filled by | Passed to | Placeholders |
|------|----------|-----------|-------------|
| `coordinator.txt` | Orchestrator (you) | haiku coordinator Agent | `{file_path}`, `{start_line}`, `{end_line}`, `{reader_template_path}`, `{checker_template_path}` |
| `reader.txt` | Coordinator | haiku reader Agent | `{url}`, `{url_fallback}` |
| `checker.txt` | Coordinator | haiku checker Agent | `{blind_summary}`, `{claim}` |
| `summarizer.txt` | Coordinator | haiku summarizer Agent | `{url}` |
| `formatter.txt` | Coordinator | haiku formatter Agent | `{raw_summary}` |

## Workflow

1. Block Codex: `touch /tmp/.block_codex`

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
   Launch a **haiku coordinator** Agent with the output. Run all in parallel.

4. Each coordinator:
   - Reads its document section and extracts papers + claims
   - Uses Bash/Python to fill `reader.txt` → launches haiku reader agents (parallel)
   - Waits for readers, fills `checker.txt` → launches haiku checker agents (parallel)
   - Returns ONLY flagged items + verified count

5. Collect coordinator reports and present consolidated summary.

6. Unblock Codex: `rm -f /tmp/.block_codex`

## Incremental mode

For re-checking after fixes, pass a paper list to the coordinator:
> Only check these specific papers: {paper_list}. Skip all others.

## Venue verification follow-up

For papers where venue couldn't be confirmed from arxiv abstracts, launch
haiku agents with WebSearch access (Codex block does not affect WebSearch).

## For repo/dataset summarization

Same pattern with `summarizer.txt` → `formatter.txt` templates.

## Known issues

- Haiku coordinators may occasionally skip launching sub-agents and do work
  inline. If this happens, the Codex block ensures they can't fetch via Codex.
  They'll fall back to WebFetch which is the desired behavior.
- arxiv abstract pages often lack venue info — use venue-checker follow-up.
- Full paper text at `arxiv.org/html/{id}v1` — use as `{url_fallback}`.
