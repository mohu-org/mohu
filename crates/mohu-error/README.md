# mohu-error

Unified error handling for the [mohu](https://github.com/mohu-org/mohu) scientific computing library. Every workspace crate returns [`MohuResult<T>`](https://docs.rs/mohu-error/latest/mohu_error/type.MohuResult.html) built on a single [`MohuError`](https://docs.rs/mohu-error/latest/mohu_error/enum.MohuError.html) enum with stable numeric [`ErrorCode`](https://docs.rs/mohu-error/latest/mohu_error/enum.ErrorCode.html) values and a coarse [`ErrorKind`](https://docs.rs/mohu-error/latest/mohu_error/enum.ErrorKind.html) classification.

## Error code domains

| Range      | Domain          |
|------------|-----------------|
| 1000–1999  | Shape           |
| 2000–2999  | DType           |
| 3000–3999  | Index / slice   |
| 4000–4999  | Buffer / memory |
| 5000–5999  | Compute / math  |
| 6000–6999  | I/O             |
| 7000–7999  | DLPack          |
| 8000–8999  | Arrow           |
| 9000–9999  | Python / PyO3   |
| 10000+     | General         |

See [`src/codes.rs`](src/codes.rs) for the full variant list.

## Example

```rust
use mohu_error::{MohuError, MohuResult, bail, ensure};

fn safe_divide(a: f64, b: f64) -> MohuResult<f64> {
    ensure!(b != 0.0, MohuError::DivisionByZero);
    Ok(a / b)
}

fn load_config(path: &str) -> MohuResult<String> {
    std::fs::read_to_string(path)
        .map_err(MohuError::Io)
        .with_context(|| format!("failed to read config at {path}"))
}
```

## Key types

| Type | Purpose |
|------|---------|
| [`MohuError`](https://docs.rs/mohu-error/latest/mohu_error/enum.MohuError.html) | Central error enum for all mohu APIs |
| [`ErrorCode`](https://docs.rs/mohu-error/latest/mohu_error/enum.ErrorCode.html) | Stable numeric code for programmatic branching |
| [`ErrorKind`](https://docs.rs/mohu-error/latest/mohu_error/enum.ErrorKind.html) | Coarse category: usage, runtime, system, internal |
| [`MultiError`](https://docs.rs/mohu-error/latest/mohu_error/struct.MultiError.html) | Collect multiple errors in one pass |
| [`ResultExt`](https://docs.rs/mohu-error/latest/mohu_error/trait.ResultExt.html) | `.context()` / `.with_context()` on `Result` |

## Contributing

See the workspace [CONTRIBUTING.md](../../CONTRIBUTING.md).
