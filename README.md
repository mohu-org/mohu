# mohu

Rust-powered arrays for Python. Fast, parallel, and built for the future.

mohu is an early-stage NumPy replacement with its core written in Rust. The goal is simple—take everything Python's scientific stack does and do it without the bottlenecks that have been accepted for decades.

No GIL. No single-threaded operations. No object overhead. Just arrays.

## Why

NumPy is written in C and hasn't fundamentally changed in 20 years. It's single-threaded by default, its string arrays are an afterthought, and parallelism requires reaching for other tools. The Python data ecosystem deserves a better foundation.

Polars proved you can rewrite the data layer in Rust and win. mohu is that same bet, one layer down.

## What's Coming

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations by default via Rayon
- First-class string arrays—not `dtype=object`
- Built on Apache Arrow for interoperability with Polars, DuckDB, and the rest of the ecosystem
- Zero-copy Python integration via PyO3
- SIMD-accelerated math operations
- Memory layouts NumPy cannot express

## Status

Early. The foundation is being laid. If you believe the Python numerical stack deserves a rewrite, watch this repository or contribute.

## API Stability

The Mohu workspace is under active development, and API stability varies across crates and features.

For details about the project's API stability guarantees, supported compatibility expectations, and guidance on using public APIs, see [STABILITY.md](STABILITY.md).

## Built With

- [Rust](https://rust-lang.org)
- [PyO3](https://github.com/PyO3/pyo3) — Python bindings
- [arrow-rs](https://github.com/apache/arrow-rs) — Columnar memory format
- [Rayon](https://github.com/rayon-rs/rayon) — Data parallelism

## License

MIT
