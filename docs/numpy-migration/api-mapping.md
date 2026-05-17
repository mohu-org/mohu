# NumPy API mapping

This page is generated from the machine-readable tracker at [`numpy-api-coverage.toml`](../../numpy-api-coverage.toml) in the repository root.

## Coverage summary

Run the helper script to print implemented vs total entries:

```bash
./scripts/numpy-coverage-summary.sh
```

## Status legend

| Status | Meaning |
|--------|---------|
| `implemented` | mohu provides an equivalent with matching intent |
| `partial` | Subset of NumPy behaviour or missing edge cases |
| `missing` | Not yet implemented |

## Tracked APIs (sample)

| NumPy | mohu | Status | Crate |
|-------|------|--------|-------|
| `np.zeros` | `Buffer::zeros` | implemented | mohu-buffer |
| `np.ones` | `Buffer::ones` | implemented | mohu-buffer |
| `np.arange` | `Buffer::arange` | implemented | mohu-buffer |
| `np.reshape` | `Buffer::reshape` | implemented | mohu-buffer |
| `np.sum` | `Buffer::sum_all_f64` / `sum_axis` | partial | mohu-buffer |
| `np.matmul` | — | missing | mohu-ops |
| `np.fft.fft` | — | missing | mohu-fft |

See the TOML file for the full list. Add a new `[[api]]` block when landing NumPy-compatible surface area.

## Future work

- Re-run NumPy's test suite with `import mohu as np` (pytest plugin)
- Publish compatibility percentage as a CI badge
