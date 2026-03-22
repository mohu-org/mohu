# Mohu Crate Map

Structured map of all workspace crates, their dependencies, modules, and public API surface.

> **Legend**: `[stub]` = module declared but source file is empty (not yet implemented).

---

## Foundation Layer

### `mohu-error`
**Description**: Shared error types — zero-dependency foundation used by every crate.
**Dependencies**: `thiserror`; optional: `pyo3` (feature `python`)

| Module | Public Items |
|--------|-------------|
| `error` | `MohuError` (enum) — central error enum with variants: `ShapeMismatch`, `DimensionMismatch`, `DTypeMismatch`, `UnsupportedDType`, `IndexOutOfBounds`, `AxisOutOfRange`, `DivisionByZero`, etc. |
| `codes` | `ErrorCode` (enum) — stable numeric codes for programmatic branching |
| `kind` | `ErrorKind` (enum) — coarse category: `Usage`, `Runtime`, `System`, `Internal` |
| `chain` | `ErrorChain<'a>` (struct) — iterator over nested `Context` wrappers |
| `context` | `ResultExt` (trait) — `.context()` and `.with_context()` extension methods |
| `multi` | `MultiError` (struct) — accumulates multiple errors in a single pass |
| `reporter` | `ErrorReporter<'a>` (struct), `ReportMode` (enum), `Severity` (enum) — rich terminal formatting |
| `test_utils` | `assert_err()`, `assert_ok()`, `assert_err_code()`, `assert_err_kind()`, `assert_shape_err()`, `assert_chain_depth()` |
| `macros` | `bail!`, `ensure!`, `assert_shape_eq!`, `assert_axis_valid!`, `assert_in_bounds!` |
| `python` | PyO3 exception conversions (feature-gated) |
| *lib.rs* | `MohuResult<T>` (type alias) |

---

### `mohu-dtype`
**Description**: DType enum, scalar type traits, and type promotion rules.
**Dependencies**: `mohu-error`, `num-traits`, `num-complex`, `half`; optional: `serde`, `arrow`

| Module | Public Items |
|--------|-------------|
| `dtype` | `DType` (enum: Bool, I8–I64, U8–U64, F16, BF16, F32, F64, C64, C128), `ALL_DTYPES` (const array), `DTYPE_COUNT` (const = 15) |
| `scalar` | `Scalar` (trait — sealed base), `RealScalar`, `IntScalar`, `SignedScalar`, `UnsignedScalar`, `FloatScalar`, `ComplexScalar` |
| `promote` | `CastMode` (enum: Safe, SameKind, Unsafe), `promote()`, `can_cast()`, `result_type()`, `common_type()`, `minimum_scalar_type()`, `weak_promote()` |
| `finfo` | `FloatInfo` (struct) — machine precision metadata (analogous to `numpy.finfo`) |
| `iinfo` | `IntInfo` (struct) — integer range metadata (analogous to `numpy.iinfo`) |
| `cast` | `cast_scalar()`, `cast_scalar_unchecked()`, `cast_slice()`, `f64_to_int_saturating()` |
| `dlpack` | `DLDataTypeCode` (enum), `DLDataType` (struct), `DLDeviceType` (enum), `assert_cpu_device()` |
| `compat` | `ByteOrder` (enum) |
| `macros` | `dtype_of!`, `dispatch_dtype!`, `dispatch_numeric!`, `dispatch_integer!`, `dispatch_float!`, `dispatch_real!`, `dispatch_signed!`, `for_each_dtype!`, `assert_dtype!`, `require_float!`, `require_numeric!`, `require_real!` |

---

### `mohu-buffer`
**Description**: Raw buffer allocation, memory layout, and stride arithmetic.
**Dependencies**: `mohu-error`, `mohu-dtype`, `rayon`, `tracing`, `smallvec`, `half`, `num-complex`, `num-traits`; optional: `memmap2` (feature `mmap`), `libc` (unix)

| Module | Public Items |
|--------|-------------|
| `alloc` | `AllocHandle` (struct), `AllocStats` (struct), `Strategy` (enum: Heap, Mmap), `MmapAdvice` (enum), `SIMD_ALIGN` (64), `CACHE_LINE` (64), `MMAP_THRESHOLD` (1 MiB), `POISON_BYTE` |
| `buffer` | `Buffer` (struct), `RawBuffer` (struct), `BufferFlags` (struct), `DLTensor` (struct), `DLManagedTensor` (struct), `RawDLDataType` (struct), `RawDLDevice` (struct) |
| `layout` | `Layout` (struct), `Order` (enum: C, F), `SliceArg` (struct) |
| `strides` | `ShapeVec` (type alias), `StrideVec` (type alias), `NdIndexIter` (struct), `StridedByteIter` (struct), `c_strides()`, `f_strides()`, `shape_size()`, `contiguous_nbytes()`, `broadcast_strides()`, `unravel_index()`, `ravel_multi_index()`, `byte_offset()`, `validate_strides()` |
| `view` | `BufferView<'buf, T>` (struct), `BufferViewMut<'buf, T>` (struct) |
| `ops` | `fill_raw()`, `fill()`, `fill_zero()`, `fill_one()`, `copy_to_contiguous()`, `cast_copy()`, `parallel_map()`, `parallel_inplace()`, `reduce()`, `fill_sequential()`, `parallel_scan()`, `where_select()`, `clip()`, `gather()`, `scatter()`, `add_scalar_inplace()`, `sub_scalar_inplace()`, `mul_scalar_inplace()`, `div_scalar_inplace()`, `abs_copy()`, `neg_copy()`, `sqrt_copy()`, `ln_copy()`, `exp_copy()`, `flip_axis_copy()`, `sum_all_f64()`, `min_all_f64()`, `max_all_f64()`, `argmin_flat()`, `argmax_flat()`, `fill_nontemporal_f32_buf()`, `parallel_zip()` |
| `pool` | `BufferPool` (struct), `PoolStats` (struct), `TlStats` (struct), `SizeClassStats` (struct), `GLOBAL_POOL` (static) |

---

### `mohu-array`
**Description**: NdArray — the core N-dimensional array type.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `num-traits`, `rayon`
**Status**: Module stubs declared; source files are empty (not yet implemented).

| Module | Status |
|--------|--------|
| `array` | `[stub]` |
| `iter` | `[stub]` |
| `shape` | `[stub]` |
| `slice` | `[stub]` |
| `view` | `[stub]` |

---

### `mohu-core`
**Description**: Re-export facade — convenience crate that bundles the four foundation crates.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `mohu-array`

**Re-exports**: `pub use mohu_array`, `pub use mohu_buffer`, `pub use mohu_dtype`, `pub use mohu_error`

---

## Dispatch & Protocol Layer

### `mohu-simd`
**Description**: AVX2 / AVX-512 / NEON SIMD kernel primitives.
**Dependencies**: `mohu-error`, `mohu-dtype`, `cfg-if`
**Features**: `avx2`, `avx512`, `neon`, `runtime-dispatch`

| Module | Operations |
|--------|-----------|
| `arith` | add, sub, mul, div, neg, abs, min, max |
| `cmp` | eq, ne, lt, le, gt, ge |
| `reduce` | sum, product, min, max, mean (parallel tree) |
| `cast` | SIMD-accelerated element-wise type casts |
| `fill` | broadcast-fill a buffer with a scalar |
| `copy` | SIMD memcpy with non-temporal stores |
| `math` | sqrt, rsqrt, exp, log, sin, cos |
| `fma` | fused multiply-add / multiply-subtract |
| `bitwise` | and, or, xor, not, shl, shr (integer types) |
| `detect` | runtime CPU feature detection |

**Note**: Module source files are stubs; API described by doc comments. Scalar fallbacks when no feature flag set.

---

### `mohu-ufunc`
**Description**: Universal-function protocol — broadcast, reduce, accumulate, outer.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `smallvec`

| Module | Purpose |
|--------|---------|
| `traits` | `Ufunc` trait, `UfuncKind` (Unary/Binary/Generalized), `UfuncMethod` |
| `broadcast` | broadcast engine: shape validation, output shape, parallel iteration |
| `loop_impl` | inner kernel loop over contiguous chunks |
| `dispatch` | dispatch tables for registered ufuncs |
| `resolver` | `TypeResolver` — selects output dtype from input dtypes |
| `reduce` | `reduce` / `accumulate` implementations |
| `methods` | `__call__`, `reduce`, `accumulate`, `outer`, `at` |
| `macros` | `define_ufunc!` — boilerplate generator for common cases |

**Note**: Module source files are stubs.

---

### `mohu-index`
**Description**: Advanced indexing — fancy, boolean mask, take/put, slice objects.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `smallvec`

| Module | Mode |
|--------|------|
| `fancy` | integer array indexing (`a[[0, 2, 4]]`) |
| `boolean` | boolean mask indexing (`a[a > 0]`) |
| `take` | `take` / `put` operations |
| `where_op` | `where` / `nonzero` |
| `slice` | composite slice objects (`s_[1:5:2, None, ...]`) |
| `gather` | scatter / gather (internal) |

**Note**: Module source files are stubs.

---

## Compute Layer

### `mohu-ops`
**Description**: Element-wise arithmetic, comparison, logical, and broadcasting operations.
**Dependencies**: `mohu-core`, `num-traits`, `rayon`, `thiserror`

| Module | Operations |
|--------|-----------|
| `arith` | add, sub, mul, div, mod, pow |
| `cmp` | eq, ne, lt, le, gt, ge |
| `logical` | and, or, xor, not |
| `broadcast` | shape broadcasting utilities |
| `reduce` | sum, prod, min, max, mean, std, var |
| `unary` | abs, neg, sqrt, exp, log, ceil, floor, round |

**Note**: Module source files are stubs.

---

### `mohu-fft`
**Description**: FFT, IFFT, RFFT, and 2-D transforms.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `rustfft`, `num-complex`, `num-traits`
**Features**: `fftw` (optional system FFTW3)

| Module | Public Items |
|--------|-------------|
| `transform` | `fft()`, `ifft()` |
| `real` | `rfft()`, `irfft()` |
| `nd` | `fft2()`, `fftn()` |
| `freq` | `fftfreq()`, `rfftfreq()`, `fftshift()` |
| `norm` | `Norm` (enum: Backward, Ortho, Forward) |
| `plan` | FFT plan caching |
| `helpers` | internal utilities |

**Note**: Most module source files are stubs; `norm.rs` is implemented.

---

### `mohu-random`
**Description**: PRNG engines and statistical distributions.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `rand`, `rand_chacha`, `num-traits`, `num-complex`; optional: `serde`

| Module | Public Items |
|--------|-------------|
| `generator` | `Generator` (trait: `seed()`, `fill_bytes()`, `next_u64()`), `Pcg64` (struct — PCG-64-DXSM), `Philox4x64` (struct — counter-based) |
| `continuous` | uniform, normal, standard_t, gamma, beta, chi2, etc. |
| `discrete` | integers, binomial, poisson, geometric, hypergeometric |
| `multivariate` | multivariate_normal, dirichlet, multinomial |
| `permutation` | shuffle, permutation, choice |
| `entropy` | system entropy source |
| `seeding` | seed utilities |

**Note**: `generator.rs` is implemented; other modules are stubs.

---

### `mohu-special`
**Description**: Special mathematical functions (erf, gamma, beta, Bessel, ...).
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `num-traits`

| Module | Functions |
|--------|----------|
| `erf` | erf, erfc, erfinv, erfcinv |
| `gamma` | gamma, lgamma, digamma, polygamma, rgamma |
| `beta` | beta, lbeta, betainc, betaincinv |
| `bessel` | j0, j1, jn, y0, y1, yn, i0, i1, k0, k1 |
| `expint` | expn, e1, ei |
| `trig` | sinc, sindg, cosdg, cotdg |
| `stats_fn` | ndtr, ndtri, chdtr, fdtr, stdtr, gdtr (CDF/PPF) |
| `misc` | log1p, expm1, logit, expit, xlogy, xlog1py |

**Note**: Module source files are stubs.

---

### `mohu-stats`
**Description**: Descriptive statistics, distributions, random sampling, hypothesis tests.
**Dependencies**: `mohu-core`, `mohu-ops`, `num-traits`, `rayon`, `thiserror`

| Module | Purpose |
|--------|---------|
| `descriptive` | mean, median, std, var, percentile, histogram |
| `distributions` | probability distribution objects |
| `random` | random sampling utilities |
| `sampling` | bootstrap, jackknife, etc. |

**Note**: Module source files are stubs.

---

## Data Structure Extensions

### `mohu-sparse`
**Description**: COO / CSR / CSC sparse matrix formats and operations.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `num-traits`, `num-complex`, `smallvec`, `indexmap`; optional: `serde`

| Module | Purpose |
|--------|---------|
| `coo` | Coordinate format (i, j, value) — incremental construction |
| `csr` | Compressed Sparse Row — row slicing, SpMV |
| `csc` | Compressed Sparse Column — column slicing, SpMM |
| `bsr` | Block Sparse Row — dense block sub-matrices |
| `dia` | Diagonal format — banded / tridiagonal systems |
| `arith` | add, sub, mul (element-wise), scalar multiply |
| `spmv` | sparse matrix x dense vector |
| `spmm` | sparse matrix x dense matrix |
| `convert` | conversions between all format pairs |
| `slice` | row / column slicing |
| `linalg` | triangular solve, norm, condest |

**Note**: Module source files are stubs.

---

### `mohu-masked`
**Description**: Masked arrays — null / invalid value propagation.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `rayon`, `smallvec`

| Module | Purpose |
|--------|---------|
| `array` | `MaskedArray` type, construction, fill_value |
| `arith` | arithmetic with mask propagation |
| `reduce` | sum/mean/min/max/std skipping masked elements |
| `compress` | `compress`, `compressed` — extract non-masked elements |
| `fill` | `filled` — replace masked with fill_value |
| `mask_ops` | `masked_where`, `masked_equal`, `getmask`, `getdata` |
| `io` | serialise/deserialise masked arrays (NPY extension) |

**Note**: Module source files are stubs.

---

## I/O & Interop

### `mohu-io`
**Description**: Array serialization and I/O — .npy/.npz, CSV, Arrow IPC, memory-mapped files.
**Dependencies**: `mohu-core`, `arrow`, `serde`, `memmap2`, `thiserror`

| Module | Format |
|--------|--------|
| `npy` | `.npy` / `.npz` read/write |
| `csv` | CSV import/export |
| `arrow` | Apache Arrow IPC interop |
| `mmap` | memory-mapped file arrays |

**Note**: Module source files are stubs.

---

## Developer Tooling

### `mohu-testing`
**Description**: Test utilities, fixtures, and property-test helpers.
**Dependencies**: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `approx`, `proptest`

| Module | Purpose |
|--------|---------|
| `assert` | `assert_array_eq!`, `assert_allclose!`, numeric checks |
| `approx` | element-wise approximate equality with ULP tolerance |
| `strategies` | `proptest` strategies: random arrays of any dtype |
| `fixtures` | pre-built arrays for test suites |
| `dtype` | dtype-parameterised test helpers |
| `perf` | micro-benchmark helpers: throughput, latency |

**Re-exports**: `approx` crate (as `approx_crate`), `proptest`

**Note**: Module source files are stubs.

---

## Dependency Graph

```
mohu-error (standalone)
  └─► mohu-dtype
        └─► mohu-buffer
              └─► mohu-array [stub]
                    └─► mohu-core (facade: re-exports error + dtype + buffer + array)

mohu-simd ──► mohu-error, mohu-dtype
mohu-ufunc ──► mohu-error, mohu-dtype, mohu-buffer
mohu-index ──► mohu-error, mohu-dtype, mohu-buffer

mohu-ops ──► mohu-core
mohu-fft ──► mohu-error, mohu-dtype, mohu-buffer, rustfft
mohu-random ──► mohu-error, mohu-dtype, mohu-buffer, rand
mohu-special ──► mohu-error, mohu-dtype, mohu-buffer
mohu-stats ──► mohu-core, mohu-ops

mohu-sparse ──► mohu-error, mohu-dtype, mohu-buffer
mohu-masked ──► mohu-error, mohu-dtype, mohu-buffer

mohu-io ──► mohu-core, arrow, serde, memmap2
mohu-testing ──► mohu-error, mohu-dtype, mohu-buffer, approx, proptest
```

## Implementation Status

| Crate | Status |
|-------|--------|
| `mohu-error` | **Fully implemented** — all modules have source code |
| `mohu-dtype` | **Fully implemented** — all modules have source code |
| `mohu-buffer` | **Fully implemented** — all modules have source code |
| `mohu-array` | Stubs only — module files empty |
| `mohu-core` | **Implemented** — re-export facade (5 lines) |
| `mohu-simd` | Stubs only |
| `mohu-ufunc` | Stubs only |
| `mohu-index` | Stubs only |
| `mohu-ops` | Stubs only |
| `mohu-fft` | Mostly stubs; `norm.rs` implemented |
| `mohu-random` | Partially implemented; `generator.rs` has `Pcg64` + `Philox4x64` |
| `mohu-special` | Stubs only |
| `mohu-stats` | Stubs only |
| `mohu-sparse` | Stubs only |
| `mohu-masked` | Stubs only |
| `mohu-io` | Stubs only |
| `mohu-testing` | Stubs only |
