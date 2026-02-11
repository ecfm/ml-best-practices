Stage, commit, and push changes to the remote repository.

1. Run `git status -u` and `git diff --stat` and `git log --oneline -5` in parallel
2. Identify which files are related to the current session's work. Exclude files that appear to be pre-existing uncommitted changes unrelated to this session (ask if unclear)
3. Present to the user for confirmation:
   - List of files to stage
   - Draft commit message (first line: concise summary in imperative mood; bullet points for key changes if non-trivial)
   - Do NOT add Co-Authored-By or any attribution lines
4. WAIT for user approval before proceeding. Do NOT stage, commit, or push until the user confirms.
5. After approval: stage the listed files by name (never `git add -A` or `git add .`), commit, and push to the current branch.

If there are no changes to commit, say so and stop.
If $ARGUMENTS is provided, use it as guidance for the commit message.
