
# mohu-simd

> Low-level SIMD kernel primitives for the mohu array library.

`mohu-simd` is the hardware acceleration layer of mohu. It provides
the vectorised kernel functions that [`mohu-ops`](../mohu-ops) dispatches
to after a single runtime CPU feature check at startup.

It targets three ISAs:

- **x86\_64** — AVX2 and AVX-512F
- **aarch64** — NEON
- **Scalar fallback** — portable Rust, works on every platform

There is no user-facing API here. This crate contains only kernels.

---

## Table of Contents

- [Modules](#modules)
- [Safety Requirements](#safety-requirements)
- [Running the Tests](#running-the-tests)
- [Current Status](#current-status)
- [Contributing](#contributing)

---

## Modules

Ten modules live under `src/` and are re-exported from `lib.rs`.

| Module | What it implements |
|------------|-----------------------------------------------------------------------------------|
| `arith` | Element-wise **add, sub, mul, div, neg, abs, min, max** for integer and float types. Includes in-place `_mut` variants. |
| `bitwise` | **AND, OR, XOR, NOT, SHL, SHR** across all integer register widths. |
| `cast` | Type-conversion kernels — widening casts (`f32 → f64`), narrowing with saturation (`i32 → i16`), float-to-integer truncation. |
| `cmp` | Element-wise **eq, ne, lt, le, gt, ge**. Returns a boolean mask aligned to the register width. |
| `copy` | Bulk memcpy with non-temporal stores, strided copy, and gather/scatter for non-contiguous layouts. |
| `detect` | Runtime CPU feature detection. Returns a `CpuFeatures` bitfield (AVX2, AVX-512F, NEON). Called once at startup by `mohu-ops`. |
| `fill` | Broadcast-fill a buffer with a scalar constant. Includes zero-fill and NaN-fill variants for float types. |
| `fma` | Fused multiply-add and fused multiply-subtract — `a * b + c` and `a * b - c` in a single rounded instruction. |
| `math` | Transcendental functions — `sqrt`, `rsqrt`, `exp`, `ln`, `log2`, `log10`, `sin`, `cos`. |
| `reduce` | Horizontal reductions — sum, product, min, max, mean — as a parallel tree reduction. |

---

## Safety Requirements

`mohu-simd` is `unsafe` throughout. Every function that uses vector intrinsics
must follow the two rules below without exception.

> PRs that omit either rule will be rejected at review.

<br>

### Rule 1 — Every `unsafe` block needs a `// SAFETY:` comment

The comment must explain **exactly** which preconditions make the block sound.
It must name the pointer or reference involved and state why alignment, length,
and validity are guaranteed at the call site.

```rust
// SAFETY: `ptr` is non-null, 32-byte aligned, and valid for at least
// 8 consecutive `f32` values. The caller guarantees `len` is a multiple
// of 8 before entering this function.
let v = unsafe { _mm256_loadu_ps(ptr) };
```

A comment that says only `// safe` or `// SAFETY: ok` is **not** acceptable.

<br>

### Rule 2 — Every AVX2 function needs `#[target_feature(enable = "avx2")]`

```rust
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    // _mm256_* intrinsics are safe to call inside this function
}
```

- **AVX-512** functions must use `#[target_feature(enable = "avx512f")]`.
  Add further extension strings (e.g. `"avx512bw"`) when the instruction
  requires them.

- **NEON** functions on aarch64 do **not** need `#[target_feature]` because
  NEON is mandatory in ARMv8-A — but they must still be `unsafe` and carry
  `// SAFETY:` comments.

> Calling a `#[target_feature]`-gated function on a CPU that does not support
> that feature causes `SIGILL`. The `detect` module exists to prevent this.
> Always guard dispatch behind a feature check.

---

## Running the Tests

Because SIMD kernels use CPU instructions that may not exist on every machine,
you must pass the target feature to the compiler when running tests.

<br>

**AVX2** — most x86\_64 machines from 2013 onwards:

```bash
RUSTFLAGS="-C target-feature=+avx2" cargo test -p mohu-simd
```

**AVX-512F** — Skylake-X, Ice Lake, and newer:

```bash
RUSTFLAGS="-C target-feature=+avx512f" cargo test -p mohu-simd
```

**Scalar fallback only** — works on any machine:

```bash
cargo test -p mohu-simd
```

> ⚠️ **Warning:** Running an AVX2 or AVX-512 build on a CPU that does not
> support those features will crash the process with `SIGILL`.
> If you are unsure about your CPU, run the scalar variant first.

<br>

### Full CI suite before pushing

```bash
# Check formatting
cargo fmt --all

# Lint — warnings are treated as errors
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run SIMD tests
RUSTFLAGS="-C target-feature=+avx2" cargo test -p mohu-simd

# Build docs — doc warnings are treated as errors
RUSTDOCFLAGS="-D warnings" cargo doc --package mohu-simd --no-deps
```

---

## Current Status

> **Kernels are stubs. Implementation is in progress.**

The module layout, public API surface, and `Cargo.toml` feature flags
(`avx2`, `avx512`, `neon`, `runtime-dispatch`) are all in place so that
`mohu-ops` can already import and reference this crate. Every kernel
currently returns `todo!()` or delegates to a scalar loop.

Implementation is landing in the following priority order:

| # | Module | Why first |
|---|--------|-----------|
| 1 | `arith` — f32/f64 add, mul | Unblocks `mohu-ops` benchmarks |
| 2 | `reduce` — f32 horizontal sum | Unblocks `mohu-stats` |
| 3 | `fma` — f32/f64 fused multiply-add | Unblocks ML workloads |
| 4 | `math` — sqrt, exp, log | Unblocks `mohu-special` |
| 5 | Remaining modules | General compute coverage |

To pick up a kernel, find an open issue labelled
[`crate: mohu-simd`](https://github.com/mohu-org/mohu/issues?q=label%3A%22crate%3A+mohu-simd%22+is%3Aopen),
leave a comment to claim it, and open a draft PR when you have something
runnable.

---

## Contributing

Please read the project-wide [CONTRIBUTING.md](../../CONTRIBUTING.md) before
opening a PR.
