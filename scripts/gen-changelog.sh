#!/usr/bin/env bash
# Regenerate CHANGELOG.md using git-cliff.
set -euo pipefail

git cliff --output CHANGELOG.md
echo "CHANGELOG.md updated"
