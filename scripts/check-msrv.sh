#!/usr/bin/env bash
# Verify the workspace compiles on the declared MSRV (1.85.0).
set -euo pipefail

MSRV="1.85.0"
rustup toolchain install "$MSRV" --profile minimal
cargo +"$MSRV" check --workspace --all-features
echo "MSRV check passed on $MSRV"
