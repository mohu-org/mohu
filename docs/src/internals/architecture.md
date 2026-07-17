# Architecture

mohu is organized as a Cargo workspace of layered crates. Each layer only depends on
the layers below it, which keeps compile times low and makes it possible to use
pieces of mohu (e.g. just the error types, or just the dtype system) without pulling
in the whole library.

┌─────────────────────────────────────────────────────────────┐
│ I/O & tooling        mohu-io · mohu-testing                 │
├─────────────────────────────────────────────────────────────┤
│ Data structures       mohu-sparse · mohu-masked              │
├─────────────────────────────────────────────────────────────┤
│ Compute               mohu-ops · mohu-fft · mohu-random ·    │
│                       mohu-special · mohu-stats              │
├─────────────────────────────────────────────────────────────┤
│ Dispatch & protocol   mohu-simd · mohu-ufunc · mohu-index    │
├─────────────────────────────────────────────────────────────┤
│ Foundation            mohu-error · mohu-dtype · mohu-buffer ·│
│                       mohu-array · mohu-core (facade)         │
└─────────────────────────────────────────────────────────────┘

## Foundation layer

- **`mohu-error`** has zero dependencies on the rest of the workspace and defines
  `MohuError`, `MohuResult<T>`, error codes, and the `bail!`/`ensure!` macros used
  everywhere else. Every other crate depends on it, directly or indirectly.
- **`mohu-dtype`** defines the `DType` enum (bool, signed/unsigned ints, floats,
  complex) and the type-promotion rules used whenever two arrays of different
  dtypes interact.
- **`mohu-buffer`** owns raw allocation, memory layout, and stride arithmetic —
  this is where C-order vs. Fortran-order and DLPack interop live.
- **`mohu-array`** builds `NdArray<T>`, the core N-dimensional array type, on top
  of `mohu-buffer` and `mohu-dtype`.
- **`mohu-core`** is a thin re-export facade over the four crates above, so
  downstream code can depend on one crate instead of four.

## Dispatch & protocol layer

- **`mohu-simd`** provides AVX2/AVX-512/NEON kernel primitives that the compute
  layer dispatches into based on CPU features detected at runtime.
- **`mohu-ufunc`** implements the universal-function protocol — broadcasting,
  reduce, accumulate, and outer — that NumPy-style element-wise operations are
  built on.
- **`mohu-index`** implements advanced indexing: fancy indexing, boolean masks,
  and take/put.

## Compute layer

`mohu-ops`, `mohu-fft`, `mohu-random`, `mohu-special`, and `mohu-stats` each own
one area of numerical functionality (arithmetic/broadcasting, FFTs, PRNGs and
distributions, special functions, and descriptive statistics respectively). They
all sit on top of the dispatch layer rather than calling SIMD or ufunc logic
directly themselves, so a new compute crate only needs to implement its own
math, not its own broadcasting or SIMD dispatch.

## Data structure extensions

- **`mohu-sparse`** adds COO/CSR/CSC sparse matrix formats.
- **`mohu-masked`** adds masked arrays for null/invalid value propagation.

Both extend the dense `NdArray<T>` model rather than replacing it.

## I/O & tooling

- **`mohu-io`** handles `.npy`/`.npz`, CSV, and Arrow IPC, including
  memory-mapped files.
- **`mohu-testing`** provides shared test fixtures, property-test strategies, and
  array-comparison helpers (e.g. `assert_allclose`) used across the workspace's
  test suites.

## Python bindings (`mohu-py`, planned)

Python interop is not yet implemented, but the plan (see `ROADMAP.md`) is a
tiered rollout of standard protocols on top of `mohu-buffer`, so that a `mohu`
array works with existing libraries without those libraries knowing mohu
exists:

1. **Buffer protocol / `__array__`** — zero-copy `np.asarray()`, pandas, PIL.
2. **DLPack** — zero-copy exchange with PyTorch, JAX, CuPy, TensorFlow.
3. **Array API standard / `__array_ufunc__` / `__array_function__`** — native
   dispatch from scikit-learn, scipy, and NumPy's own function calls.

## Where to look next

- [`CRATE_MAP.md`](../../../CRATE_MAP.md) — full per-crate module and public API
  breakdown; the best starting point if you're unsure which crate owns a change.
- [`docs/design/`](../../design/) — architecture decision records for
  cross-cutting design questions (dtype system, memory layout).
- [`docs/rfcs/`](../../rfcs/) — larger proposals, such as the public Array API
  surface.
