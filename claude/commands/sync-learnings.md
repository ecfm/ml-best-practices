Sync shared learnings across servers. Do the following steps:

1. **Pull latest** from both repos (to avoid conflicts):
   ```
   cd ~/ml-best-practices && git pull --rebase
   ```

2. **Check for local changes** in these locations:
   - `~/ml-best-practices/` (shared best practices)
   - `~/.claude/CLAUDE.md` (global Claude instructions)
   - `~/.claude/projects/*/memory/` (project-specific memory files)

3. **For ~/ml-best-practices/**:
   - Run `git status` and `git diff` to see what changed
   - If there are changes, draft a concise commit message that summarizes WHAT was learned (not "update learnings" — describe the actual content, e.g. "Add early stopping and complete-each-run-fully principles")
   - Stage, commit with the drafted message, and push

4. **For project memory files** that have useful general lessons:
   - Review the project MEMORY.md for any insights that should be promoted to ~/ml-best-practices/
   - If found, copy the generalized version to the appropriate section in ~/ml-best-practices/README.md (remove project-specific details)
   - Commit and push

5. **Report** what was synced and what was already up to date.
