# Memory Tracking

## When recalling a memory

When you read a memory file and use its content to inform your response, disclose this:

> Recalling [memory_name]: [one sentence on how it applies here]

This makes memory influence visible so the user can correct misapplied memories.

## After user feedback

Observe the user's response to your memory-informed recommendation:
- **Confirmed**: user accepts or builds on your recommendation
- **Corrected**: user says you're wrong, adjusts your approach, or overrides
- **Ignored**: user doesn't engage with the memory-informed part

Log the outcome by appending to the access log:

```bash
echo "$(date +%Y-%m-%d) MEMORY_FILENAME STATUS CONTEXT" >> ~/.claude/memory-access.log
```

Where STATUS is `confirmed`, `corrected`, or `ignored`, and CONTEXT is a brief note (under 10 words).

Example:
```
2026-03-26 feedback_research_standards.md confirmed direction_critique_accepted
2026-03-26 reference_external_critique.md corrected codex_model_changed
```

## When a memory causes trouble

If a memory leads to a recommendation the user corrects:
1. Log it as `corrected` in the access log
2. Update the memory file: add a caveat or fix the content
3. Update `last_verified` date in frontmatter

## Periodic review

When the user asks to review memories, or at the start of a long session:
1. Check `~/.claude/memory-access.log` for patterns
2. Memories with multiple `corrected` entries need revision or removal
3. Memories never appearing in the log after 4+ weeks are candidates for archival
