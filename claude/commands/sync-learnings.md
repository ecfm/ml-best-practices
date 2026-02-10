Sync shared learnings across servers. Do the following steps:

1. **Check for local changes** in these locations:
   - `~/ml-best-practices/` (shared best practices)
   - `~/.claude/CLAUDE.md` (global Claude instructions)
   - `~/.claude/projects/*/memory/` (project-specific memory files)

2. **Commit local changes** in `~/ml-best-practices/`:
   - Run `git status` and `git diff` to see what changed
   - If there are changes, draft a concise commit message that summarizes WHAT was learned (not "update learnings" — describe the actual content, e.g. "Add early stopping and complete-each-run-fully principles")
   - Stage and commit with the drafted message

3. **Promote project memory** that has useful general lessons:
   - Review the project MEMORY.md for any insights that should be promoted to ~/ml-best-practices/
   - If found, copy the generalized version to the appropriate file in ~/ml-best-practices/ (experiment-practices.md or pipeline-architecture.md; remove project-specific details)
   - Commit separately with a message describing the promoted content

4. **Pull latest** with rebase (safe now that local work is committed):
   ```
   cd ~/ml-best-practices && git pull --rebase
   ```

5. **Push** to remote:
   ```
   cd ~/ml-best-practices && git push
   ```

6. **Report** what was synced and what was already up to date.
