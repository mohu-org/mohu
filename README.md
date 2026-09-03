<div align="center">

# mohu

**Rust-powered arrays for Python. Fast, parallel, and built for the future.**

[![CI](https://github.com/mohu-org/mohu/actions/workflows/ci.yml/badge.svg)](https://github.com/mohu-org/mohu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/rustc-1.85%2B-orange.svg)](rust-toolchain.toml)
[![Edition](https://img.shields.io/badge/edition-2024-lightgrey.svg)](Cargo.toml)
[![good first issues](https://img.shields.io/github/issues/mohu-org/mohu/good%20first%20issue?color=7057ff&label=good%20first%20issues)](https://github.com/mohu-org/mohu/issues?q=is%3Aopen+label%3A%22good+first+issue%22)

[Quickstart](#quickstart) · [Status](#status) · [Architecture](#architecture) · [Contributing](#contributing) · [Roadmap](ROADMAP.md)

</div>

---

mohu is an early-stage NumPy replacement with its core written in Rust. The goal is
simple — take everything Python's scientific stack does and do it without the
bottlenecks that have been accepted for decades.

No GIL. No single-threaded operations. No object overhead. Just arrays.

## Why

NumPy is written in C and hasn't fundamentally changed in 20 years. It's
single-threaded by default, its string arrays are an afterthought, and parallelism
requires reaching for other tools. The Python data ecosystem deserves a better
foundation.

Polars proved you can rewrite the data layer in Rust and win. mohu is that same bet,
one layer down.

## Status

**Pre-alpha. Not usable as a NumPy replacement yet, and not published to crates.io or
PyPI.** The memory foundation is real and tested; the compute layers above it are
scaffolded but largely unimplemented. This table is the honest state of the workspace —
if a crate says *stub*, its implementation modules contain no public implementation.

| Crate | What it does | State |
|---|---|---|
| `mohu-error` | `MohuError`, error codes, context chains, reporter | ✅ implemented |
| `mohu-dtype` | 15 dtypes, scalar traits, promotion, `finfo`/`iinfo`, DLPack codes | ✅ implemented |
| `mohu-buffer` | Aligned/mmap allocation, layout, strides, views, DLPack, pool | ✅ implemented |
| `mohu-random` | PCG64 / Philox PRNG + distributions | 🟡 partial |
| `mohu-array` | `NdArray<T>` — the user-facing array type | ⬜ stub |
| `mohu-core` | Re-export facade over the foundation crates | ⬜ stub |
| `mohu-simd` | AVX2 / AVX-512 / NEON kernels | ⬜ stub |
| `mohu-ufunc` | Universal-function protocol: broadcast, reduce, outer | ⬜ stub |
| `mohu-index` | Fancy / boolean indexing, take & put | ⬜ stub |
| `mohu-ops` | Element-wise arithmetic, comparison, reductions, matmul | ⬜ stub |
| `mohu-fft` | FFT / IFFT / RFFT / 2-D FFT | ⬜ stub |
| `mohu-special` | erf, gamma, beta, Bessel | ⬜ stub |
| `mohu-stats` | Descriptive statistics, hypothesis tests | ⬜ stub |
| `mohu-sparse` | COO / CSR / CSC formats | ⬜ stub |
| `mohu-masked` | Masked arrays, null propagation | ⬜ stub |
| `mohu-io` | `.npy` / `.npz` / CSV / Arrow IPC / HDF5 / Parquet | ⬜ stub |
| `mohu-testing` | Fixtures, property strategies, array comparison | ⬜ stub |

Companion repositories: [`mohu-compute`](https://github.com/mohu-org/mohu-compute)
(SIMD kernels), [`mohu-linalg`](https://github.com/mohu-org/mohu-linalg)
(decompositions and solvers), [`mohu-py`](https://github.com/mohu-org/mohu-py)
(PyO3 bindings). All three are scaffolded and awaiting implementation.

Every stub in the table above is an open issue. If you want to build one, see
[Contributing](#contributing).

## Quickstart

mohu is not on crates.io yet, so build from source:

```bash
git clone https://github.com/mohu-org/mohu.git
cd mohu
cargo build --workspace
cargo test --workspace
```

Today the usable surface is `mohu-buffer` — the strided, dtype-tagged, reference-counted
buffer that every future `NdArray` sits on top of:

```rust
use mohu_buffer::{Buffer, Order};
use mohu_dtype::{CastMode, DType};

// Build a 2x3 buffer from a typed slice. dtype is inferred from T.
let a = Buffer::from_slice_2d::<f32>(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?;

// Views are zero-copy: transpose only rewrites the stride vector.
let t = a.transpose();
assert_eq!(t.shape(), &[3, 2]);
assert_eq!(t.get::<f32>(&[2, 1])?, 6.0);

// Constructors mirror NumPy.
let z = Buffer::zeros(DType::F64, &[3, 4])?;
let r = Buffer::arange(0.0, 10.0, 2.0, DType::I32)?;
assert_eq!(r.to_vec::<i32>()?, vec![0, 2, 4, 6, 8]);

// Casting is explicit and checked against a promotion policy.
let as_f64 = a.cast(DType::F64, CastMode::Safe)?;

// Broadcasting produces a zero-copy view with 0-strides.
let row = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0])?;
let bcast = row.broadcast_to(&[2, 3])?;
assert_eq!(bcast.strides(), &[0, 4]);

// Fortran-order allocation is a first-class option, not an afterthought.
let f = Buffer::alloc(DType::F32, &[2, 3], Order::F)?;
assert!(f.is_f_contiguous());
# Ok::<(), mohu_error::MohuError>(())
```

That snippet is a compiled example, not prose — run it with:

```bash
cargo run --example quickstart
```

Other runnable examples:

```bash
cargo run --example buffer_basics    # construction, views, reshape, slicing
cargo run --example alloc_and_pool   # allocation strategies and the buffer pool
cargo run --example dtype_basics     # the 15 dtypes and their metadata
cargo run --example type_promotion   # NumPy-compatible promotion rules
```

## Architecture

mohu is a Cargo workspace of small, single-responsibility crates layered by dependency
depth. Nothing in a lower layer knows about anything above it.

```
Foundation      mohu-error → mohu-dtype → mohu-buffer → mohu-array → mohu-core
Dispatch        mohu-simd, mohu-ufunc, mohu-index
Compute         mohu-ops, mohu-fft, mohu-random, mohu-special, mohu-stats
Structures      mohu-sparse, mohu-masked
I/O             mohu-io
Tooling         mohu-testing
```

Three design decisions drive the rest:

- **Buffers are dtype-tagged at runtime, monomorphised at the call site.** A `Buffer`
  carries a `DType` enum rather than a generic parameter, and the `dispatch_dtype!`
  macro family expands a runtime tag into a static generic call — dynamic typing at the
  API boundary, zero vtables in the hot loop.
- **Layout is separate from storage.** `Layout` owns shape, strides, and offset;
  `Buffer` owns bytes. Transpose, reshape, slice, broadcast, squeeze, and `expand_dims`
  are all layout rewrites that allocate nothing.
- **Arrow-native memory.** Allocations are SIMD-aligned and use Arrow's validity-bitmap
  conventions, so zero-copy interchange with Polars, DuckDB, and anything speaking
  DLPack or the Arrow C data interface is a pointer handoff.

For the crate-by-crate public API surface, see [CRATE_MAP.md](CRATE_MAP.md). Design
documents live in [`docs/design/`](docs/design/) and accepted proposals in
[`docs/rfcs/`](docs/rfcs/).

## What's Coming

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations by default via Rayon
- First-class string arrays — not `dtype=object`
- Built on Apache Arrow for interoperability with Polars, DuckDB, and the rest of the ecosystem
- Zero-copy Python integration via PyO3
- SIMD-accelerated math operations
- Memory layouts NumPy cannot express

See [ROADMAP.md](ROADMAP.md) for the sequenced plan.

## Contributing

Contributions are welcome, and the stub crates above are the fastest way in.

```bash
make fmt      # rustfmt
make lint     # clippy, warnings denied
make test     # workspace test suite
make bench    # criterion benchmarks
```

Start here:

- [Good first issues](https://github.com/mohu-org/mohu/issues?q=is%3Aopen+label%3A%22good+first+issue%22) — scoped, single-file tasks with acceptance criteria
- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, branch naming, commit conventions, review expectations
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

All commits must be signed off for [DCO](https://developercertificate.org/)
(`git commit -s`), and CI enforces formatting, clippy, MSRV 1.85, semver, docs,
Miri, and `cargo-deny`.

## API Stability

The mohu workspace is under active development, and API stability varies across crates
and features. Nothing here is covered by semver guarantees before 1.0. See
[STABILITY.md](STABILITY.md) for the details, and [SECURITY.md](SECURITY.md) for
vulnerability reporting.

## Built With

- [Rust](https://rust-lang.org)
- [PyO3](https://github.com/PyO3/pyo3) — Python bindings
- [arrow-rs](https://github.com/apache/arrow-rs) — Columnar memory format
- [Rayon](https://github.com/rayon-rs/rayon) — Data parallelism

## License

MIT — see [LICENSE](LICENSE).
