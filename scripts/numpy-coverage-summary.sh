#!/usr/bin/env bash
# Print NumPy API coverage stats from numpy-api-coverage.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$ROOT/numpy-api-coverage.toml"

if [[ ! -f "$FILE" ]]; then
  echo "missing $FILE" >&2
  exit 1
fi

total=$(grep -c '^status = ' "$FILE" || true)
implemented=$(grep -c '^status = "implemented"' "$FILE" || true)
partial=$(grep -c '^status = "partial"' "$FILE" || true)
missing=$(grep -c '^status = "missing"' "$FILE" || true)

pct=0
if [[ "$total" -gt 0 ]]; then
  pct=$(( (implemented * 100) / total ))
fi

echo "NumPy API coverage (tracked entries)"
echo "  total:       $total"
echo "  implemented: $implemented"
echo "  partial:     $partial"
echo "  missing:     $missing"
echo "  implemented%: ${pct}% (of tracked entries only)"
