# mohu-simd

Low-level SIMD kernels with portable fallbacks for the MOHU ecosystem.

## Implemented

- Runtime CPU feature detection (`avx2`, `avx512f`, SSE4.1/SSE4.2, NEON).
- AVX2-accelerated typed fill and copy for the supported types.
- Portable fill/copy fallbacks on other CPUs and architectures.

AVX-512 and NEON kernels, plus arithmetic, reduction, comparison, cast,
bitwise, FMA, and math kernels remain planned.

## Safety

The fill and copy functions are unsafe raw-pointer APIs. Callers must provide
valid ranges for the requested length, exclusive writable access for fills, and
non-overlapping source/destination ranges for copies. Zero-length calls do not
access their pointers. AVX2 helpers use unaligned operations and are entered
only when both the `avx2` Cargo feature is enabled and runtime detection reports
hardware support.

## Testing

```bash
cargo test -p mohu-simd --all-features
```
