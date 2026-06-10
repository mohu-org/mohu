# mohu-simd

`mohu-simd` provides low-level SIMD kernel primitives that power performance-critical paths in `mohu-ops`, exposing architecture-specific building blocks for vectorized numeric operations while keeping higher-level operator logic in the ops layer.

## Supported Architectures

- `x86_64`: AVX2, AVX-512
- `aarch64`: NEON

## SIMD Modules

| Module | Purpose |
| --- | --- |
| `arith` | Element-wise arithmetic kernels (for example add/sub/mul/div style vector math) used by higher-level ops dispatch. |
| `bitwise` | Bit-level vector kernels for logical operations such as AND/OR/XOR and related mask-oriented primitives. |
| `cast` | SIMD-accelerated type conversion and reinterpretation helpers for numeric cast paths. |
| `cmp` | Vector comparison kernels producing per-lane comparison results for equality/ordering style operations. |
| `copy` | Fast SIMD copy/move primitives for contiguous memory regions. |
| `detect` | CPU feature detection helpers used to select safe/optimal SIMD implementations at runtime. |
| `fill` | Vectorized fill/set kernels for writing repeated values into buffers. |
| `fma` | Fused multiply-add kernels and related multiply-accumulate SIMD primitives. |
| `math` | Core unary/binary math kernels (for example abs/sqrt/round-style primitives) used by math ops layers. |
| `reduce` | Horizontal reduction kernels (for example sum/min/max style accumulations) over SIMD lanes. |

## Safety Requirements

- Every `unsafe` block must include a clear `SAFETY:` comment explaining why the operation is valid.
- AVX2 implementations must be guarded with appropriate `#[target_feature(enable = "avx2")]` annotations.
- SIMD intrinsics must execute only when runtime or compile-time feature detection guarantees support on the current CPU.
- Follow the existing project safety conventions and review patterns used across the `mohu` crates.

## Running Tests

```bash
RUSTFLAGS="-C target-feature=+avx2" cargo test -p mohu-simd
```
