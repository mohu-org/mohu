#!/usr/bin/env bash
# Build a release wheel using maturin.
set -euo pipefail

PROFILE="${1:-dist-release}"

maturin build \
    --release \
    --profile "$PROFILE" \
    --manifest-path crates/mohu-py/Cargo.toml \
    --out dist/

echo "Wheel written to dist/"
