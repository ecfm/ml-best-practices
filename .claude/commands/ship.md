Stage, commit, and push changes to the remote repository.

1. Run `git status -u` and `git diff --stat` and `git log --oneline -5` in parallel
2. Identify which files are related to the current session's work. Exclude files that appear to be pre-existing uncommitted changes unrelated to this session (ask if unclear)
3. Stage the relevant files by name (never `git add -A` or `git add .`)
4. Draft a commit message:
   - First line: concise summary of what changed (imperative mood)
   - Blank line, then bullet points for key changes if more than a trivial edit
   - Do NOT add Co-Authored-By or any attribution lines
5. Commit and push to the current branch

If there are no changes to commit, say so and stop.
If $ARGUMENTS is provided, use it as guidance for the commit message.
