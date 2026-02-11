#!/bin/bash
# Install Claude Code config files by symlinking from this repo.
# Run once per server after cloning:
#   git clone git@github.com:ecfm/ml-best-practices.git ~/ml-best-practices
#   bash ~/ml-best-practices/install.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing from: ${REPO_DIR}"

# Create ~/.claude directories
mkdir -p ~/.claude/commands
mkdir -p ~/.claude/rules

# Symlink CLAUDE.md (global instructions)
if [ -L ~/.claude/CLAUDE.md ]; then
    echo "~/.claude/CLAUDE.md already symlinked"
elif [ -f ~/.claude/CLAUDE.md ]; then
    echo "~/.claude/CLAUDE.md exists (not a symlink). Backing up to ~/.claude/CLAUDE.md.bak"
    mv ~/.claude/CLAUDE.md ~/.claude/CLAUDE.md.bak
    ln -sf "${REPO_DIR}/claude/CLAUDE.md" ~/.claude/CLAUDE.md
    echo "Symlinked ~/.claude/CLAUDE.md"
else
    ln -sf "${REPO_DIR}/claude/CLAUDE.md" ~/.claude/CLAUDE.md
    echo "Symlinked ~/.claude/CLAUDE.md"
fi

# Symlink settings.json (hooks, theme, statusLine)
if [ -L ~/.claude/settings.json ]; then
    echo "~/.claude/settings.json already symlinked"
elif [ -f ~/.claude/settings.json ]; then
    echo "~/.claude/settings.json exists (not a symlink). Backing up to ~/.claude/settings.json.bak"
    mv ~/.claude/settings.json ~/.claude/settings.json.bak
    ln -sf "${REPO_DIR}/claude/settings.json" ~/.claude/settings.json
    echo "Symlinked ~/.claude/settings.json"
else
    ln -sf "${REPO_DIR}/claude/settings.json" ~/.claude/settings.json
    echo "Symlinked ~/.claude/settings.json"
fi

# Symlink all commands
for cmd in "${REPO_DIR}"/claude/commands/*.md; do
    [ -f "$cmd" ] || continue
    name=$(basename "$cmd")
    ln -sf "$cmd" ~/.claude/commands/"$name"
    echo "Symlinked ~/.claude/commands/${name}"
done

# Symlink all rules
for rule in "${REPO_DIR}"/claude/rules/*.md; do
    [ -f "$rule" ] || continue
    name=$(basename "$rule")
    ln -sf "$rule" ~/.claude/rules/"$name"
    echo "Symlinked ~/.claude/rules/${name}"
done

echo ""
echo "Done. Verify with:"
echo "  ls -la ~/.claude/CLAUDE.md"
echo "  ls -la ~/.claude/settings.json"
echo "  ls -la ~/.claude/commands/"
echo "  ls -la ~/.claude/rules/"
