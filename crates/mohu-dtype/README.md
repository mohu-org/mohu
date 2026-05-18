# mohu-dtype

`mohu-dtype` provides the canonical runtime type system used throughout the Mohu workspace. It defines the `DType` enum, scalar type mappings, memory layout metadata, parsing helpers, type conversion utilities, and DLPack interoperability support. The crate is designed as a lightweight foundational dependency shared across array, buffer, compute, and I/O crates.

---

## DType Table

| Variant | Rust type          | Bytes | Alignment |
|---------|--------------------|-------|-----------|
| Bool    | `bool`             |   1   |     1     |
| I8      | `i8`               |   1   |     1     |
| I16     | `i16`              |   2   |     2     |
| I32     | `i32`              |   4   |     4     |
| I64     | `i64`              |   8   |     8     |
| U8      | `u8`               |   1   |     1     |
| U16     | `u16`              |   2   |     2     |
| U32     | `u32`              |   4   |     4     |
| U64     | `u64`              |   8   |     8     |
| F16     | `half::f16`        |   2   |     2     |
| BF16    | `half::bf16`       |   2   |     2     |
| F32     | `f32`              |   4   |     4     |
| F64     | `f64`              |   8   |     8     |
| C64     | `Complex<f32>`     |   8   |     4     |
| C128    | `Complex<f64>`     |  16   |     8     |

The values above match `DType::itemsize()` and `DType::alignment()`.

---

## Example

```rust
use mohu_dtype::dtype::DType;
use mohu_dtype::promote::{can_cast, promote, CastMode};

fn main() {
    // Parse from string
    let dtype = DType::from_str("f32").unwrap();

    // Query size
    assert_eq!(dtype.itemsize(), 4);

    // Classification helper
    assert!(dtype.is_float());

    // Type promotion
    let promoted = promote(DType::I32, DType::F32);
    assert_eq!(promoted, DType::F32);

    // Casting rules
    assert!(can_cast(
        DType::I32,
        DType::F64,
        CastMode::Unsafe
    ));

    assert!(!can_cast(
        DType::F64,
        DType::Bool,
        CastMode::Safe
    ));
}
```


### Implemented Scalar Types

| Rust Type | DType |
|------------|--------|
| `bool` | `Bool` |
| `i8` | `I8` |
| `i16` | `I16` |
| `i32` | `I32` |
| `i64` | `I64` |
| `u8` | `U8` |
| `u16` | `U16` |
| `u32` | `U32` |
| `u64` | `U64` |
| `half::f16` | `F16` |
| `half::bf16` | `BF16` |
| `f32` | `F32` |
| `f64` | `F64` |
| `num_complex::Complex<f32>` | `C64` |
| `num_complex::Complex<f64>` | `C128` |

---

## DLPack DType Mapping

`mohu-dtype` provides DLPack-compatible dtype metadata for tensor interchange.

| DType | DLPack Code | Bits | Lanes |
|--------|--------------|------|--------|
| `Bool` | `kDLBool` | 8 | 1 |
| `I8` | `kDLInt` | 8 | 1 |
| `I16` | `kDLInt` | 16 | 1 |
| `I32` | `kDLInt` | 32 | 1 |
| `I64` | `kDLInt` | 64 | 1 |
| `U8` | `kDLUInt` | 8 | 1 |
| `U16` | `kDLUInt` | 16 | 1 |
| `U32` | `kDLUInt` | 32 | 1 |
| `U64` | `kDLUInt` | 64 | 1 |
| `F16` | `kDLFloat` | 16 | 1 |
| `BF16` | `kDLBfloat` | 16 | 1 |
| `F32` | `kDLFloat` | 32 | 1 |
| `F64` | `kDLFloat` | 64 | 1 |
| `C64` | `kDLComplex` | 64 | 1 |
| `C128` | `kDLComplex` | 128 | 1 |

---

## Contributing

See the workspace contribution guide:

[`CONTRIBUTING.md`](../../CONTRIBUTING.md)