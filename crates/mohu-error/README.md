# mohu-error

Unified error handling for the [mohu](https://github.com/mohu-org/mohu) workspace.

This is the foundation crate — it depends on nothing inside mohu, and every other crate
returns its `MohuResult<T>`. Keeping one error enum for the whole workspace means a
shape error raised deep inside a SIMD kernel surfaces to Python as the same typed,
inspectable value it started as.

## What's in it

| Type | Purpose |
|------|---------|
| `MohuError` | The central non-exhaustive error enum — every mohu function returns this |
| `MohuResult<T>` | `Result<T, MohuError>` alias |
| `ErrorCode` | Stable numeric code for programmatic branching across the FFI boundary |
| `ErrorKind` | Coarse category: `Usage`, `Runtime`, `System`, `Internal` |
| `ErrorChain` | Iterator over nested `Context` wrappers |
| `MultiError` | Accumulates multiple errors in a single validation pass |
| `ErrorReporter` | Rich terminal formatting — compact, full, or JSON |
| `ResultExt` | `.context()` / `.with_context()` extension trait |

Errors are organised by domain — shape, dtype, index, buffer, compute, I/O, DLPack,
Arrow, Python — so callers can branch on a domain without matching every variant.

## Usage

```rust
use mohu_error::{MohuError, MohuResult, ResultExt, ensure};

fn safe_divide(a: f64, b: f64) -> MohuResult<f64> {
    ensure!(b != 0.0, MohuError::DivisionByZero);
    Ok(a / b)
}

fn compute() -> MohuResult<f64> {
    safe_divide(1.0, 0.0).context("while normalising the input row")
}
```

The `bail!`, `ensure!`, `assert_shape_eq!`, `assert_axis_valid!`, and
`assert_in_bounds!` macros cover the precondition checks that would otherwise be
repeated in every kernel.

## Features

| Feature | Effect |
|---|---|
| `python` | Enables PyO3 exception conversion — `MohuError` maps onto the matching Python builtin |

## Testing helpers

`mohu_error::test_utils` provides `assert_err()`, `assert_ok()`, `assert_err_code()`,
`assert_err_kind()`, `assert_shape_err()`, and `assert_chain_depth()` for asserting on
error values without hand-rolling match arms in tests.

## License

MIT — see [LICENSE](../../LICENSE).
