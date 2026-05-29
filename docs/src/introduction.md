# Introduction

mohu is a Rust-powered N-dimensional array library for Python. It is designed as a modern, parallel-by-default replacement for NumPy, built from the ground up in Rust.

## Why mohu?

NumPy has served the Python ecosystem well for two decades, but its C-based architecture has fundamental limitations:

- **Single-threaded by default** — parallelism requires manual orchestration
- **No first-class string arrays** — strings are stored as Python objects (`dtype=object`)
- **GIL-bound** — Python's Global Interpreter Lock limits concurrent operations
- **Fixed memory model** — no support for custom strides or exotic layouts

mohu addresses these by building on modern foundations: Apache Arrow for memory, Rayon for parallelism, and PyO3 for zero-copy Python interop.

## Architecture

The library is organized as a Rust workspace of 17 crates in strict dependency layers:

| Layer | Key Crates | Responsibility |
|-------|-----------|----------------|
| Foundation | error, dtype, buffer, array, core | Shared types, memory allocation, array representation |
| Dispatch | simd, ufunc, index | SIMD kernels, universal functions, indexing |
| Compute | ops, fft, random, special, stats | Numerical operations |
| Extensions | sparse, masked | Specialized array types |
| IO/Tooling | io, testing | Serialization, development tooling |

See [CRATE_MAP.md](../CRATE_MAP.md) for the full API surface.

## Next Steps

- [Getting Started](./getting-started.md) — build and run mohu
- [Architecture](../design/) — design documents and ADRs
- [Contributing](../CONTRIBUTING.md) — how to get involved
