# mohu-dtype

Runtime data types for the [mohu](https://github.com/mohu-org/mohu) scientific computing library. This crate defines [`DType`](https://docs.rs/mohu-dtype/latest/mohu_dtype/enum.DType.html), the element-type tag used by every array and buffer, plus [`Scalar`](https://docs.rs/mohu-dtype/latest/mohu_dtype/trait.Scalar.html) traits and type-promotion rules compatible with NumPy-style casting.

## Supported dtypes

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

## Example

```rust
use mohu_dtype::{can_cast, promote, CastMode, DType};

let dt = DType::from_str("float64")?;
assert_eq!(dt, DType::F64);
assert!(dt.is_float());
assert_eq!(dt.itemsize(), 8);

let promoted = promote(DType::I32, DType::F32);
assert_eq!(promoted, DType::F64);
assert!(can_cast(DType::I32, DType::F64, CastMode::Safe));
```

## Scalar trait

[`Scalar`](https://docs.rs/mohu-dtype/latest/mohu_dtype/trait.Scalar.html) is implemented for all 15 element types (`bool`, integer primitives, `half::f16` / `bf16`, `f32` / `f64`, and `num_complex::Complex`). Sub-traits such as [`FloatScalar`](https://docs.rs/mohu-dtype/latest/mohu_dtype/trait.FloatScalar.html), [`IntScalar`](https://docs.rs/mohu-dtype/latest/mohu_dtype/trait.IntScalar.html), and [`ComplexScalar`](https://docs.rs/mohu-dtype/latest/mohu_dtype/trait.ComplexScalar.html) refine which operations are available at compile time.

## DLPack mapping

Each `DType` maps to a DLPack `DLDataType` `(code, bits, lanes)` via [`to_dlpack`](https://docs.rs/mohu-dtype/latest/mohu_dtype/fn.to_dlpack.html) and [`from_dlpack`](https://docs.rs/mohu-dtype/latest/mohu_dtype/fn.from_dlpack.html). Kind codes follow DLPack v0.8 (`kDLInt`, `kDLUInt`, `kDLFloat`, `kDLBfloat`, `kDLComplex`). See [`src/dlpack.rs`](src/dlpack.rs) for details.

## Contributing

See the workspace [CONTRIBUTING.md](../../CONTRIBUTING.md).
