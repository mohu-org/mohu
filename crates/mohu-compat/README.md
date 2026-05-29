# mohu-compat

NumPy API compatibility tracker for mohu.

## Coverage Summary

`coverage.toml` maps every `numpy.*` function to its mohu equivalent
and tracks implementation status.

### Status values

| Status | Meaning |
|---|---|
| `done` | Implemented and tested |
| `partial` | Exists but missing edge cases |
| `planned` | On the roadmap |
| `not-planned` | Out of scope for mohu |

## How to contribute

If you find a NumPy function missing from `coverage.toml`, open a PR adding it
with `status = "planned"`. If you implement one in mohu, update its status to
`"done"` and link the relevant crate.