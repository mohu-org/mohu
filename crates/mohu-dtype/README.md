# mohu-dtype

The data type system for [mohu](https://github.com/mohu-org/mohu) — the bridge between
Rust's static type system and the dynamic typing a NumPy-compatible array API requires.

`DType` is a runtime tag; `Scalar` is the compile-time trait hierarchy. The
`dispatch_dtype!` macro family turns the former into the latter, so a runtime dtype
becomes a monomorphised generic call with no vtable and no branch in the inner loop.

## The 15 dtypes

`Bool`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `F16`, `BF16`, `F32`,
`F64`, `C64`, `C128` — exposed as the `DType` enum, plus `ALL_DTYPES` and
`DTYPE_COUNT` for exhaustive iteration.

## What's in it

| Module | Purpose |
|---|---|
| `dtype` | `DType`, `ALL_DTYPES`, `DTYPE_COUNT` |
| `scalar` | `Scalar` (sealed) plus `RealScalar`, `IntScalar`, `SignedScalar`, `UnsignedScalar`, `FloatScalar`, `ComplexScalar` |
| `promote` | `promote()`, `result_type()`, `common_type()`, `can_cast()`, `weak_promote()`, `minimum_scalar_type()`, `CastMode` |
| `cast` | `cast_scalar()`, `cast_slice()`, saturating float→int conversion |
| `finfo` | `FloatInfo` — machine precision metadata, analogous to `numpy.finfo` |
| `iinfo` | `IntInfo` — integer range metadata, analogous to `numpy.iinfo` |
| `dlpack` | DLPack type and device codes for zero-copy interchange |
| `compat` | `ByteOrder` and NumPy-compatible dtype string names |
| `macros` | `dtype_of!`, the `dispatch_*!` family, `for_each_dtype!`, `require_*!` |

## Usage

Promotion follows NumPy's rules, so mixed-dtype arithmetic resolves the way users
already expect:

```rust
use mohu_dtype::{DType, promote, can_cast, CastMode};

assert_eq!(promote(DType::I32, DType::F32), DType::F64);
assert!(can_cast(DType::I32, DType::F64, CastMode::Safe));
assert!(!can_cast(DType::F64, DType::I32, CastMode::Safe));
```

`CastMode` controls how permissive a conversion may be — `Safe` allows only
value-preserving casts, `SameKind` allows narrowing within a kind, and `Unsafe` allows
anything.

## Dispatch

`dispatch_dtype!` takes a runtime `DType` and the name of a macro to invoke with the
corresponding Rust type:

```rust
use mohu_dtype::{DType, dispatch_dtype};

macro_rules! size_of_ty {
    ($t:ty) => { std::mem::size_of::<$t>() };
}

fn size_of_dtype(dt: DType) -> usize {
    dispatch_dtype!(dt, size_of_ty)
}

assert_eq!(size_of_dtype(DType::F64), 8);
assert_eq!(size_of_dtype(DType::C128), 16);
```

It expands to a match over all 15 dtypes, monomorphising the body once per type. The
narrower `dispatch_numeric!`, `dispatch_integer!`, `dispatch_float!`,
`dispatch_real!`, and `dispatch_signed!` variants restrict the match to a subset.

## Features

| Feature | Effect |
|---|---|
| `serde` | `Serialize` / `Deserialize` for `DType` |
| `arrow` | Conversion to and from `arrow::datatypes::DataType` |

## Examples

```bash
cargo run --example dtype_basics
cargo run --example type_promotion
```

## License

MIT — see [LICENSE](../../LICENSE).
