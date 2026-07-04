# mohu

Rust-powered arrays for Python. Fast, parallel, and built for the future.

mohu is an early-stage NumPy replacement with its core written in Rust. The goal is simple — take everything Python's scientific stack does and do it without the bottlenecks that have been accepted for decades.

No GIL. No single-threaded ops. No object overhead. Just arrays.

## why

NumPy is written in C and hasn't fundamentally changed in 20 years. It's single-threaded by default, its string arrays are an afterthought, and parallelism requires reaching for other tools. The Python data ecosystem deserves a better foundation.

Polars proved you can rewrite the data layer in Rust and win. mohu is that same bet, one layer down.

## what's coming

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations by default via Rayon
- First-class string arrays — not `dtype=object`
- Built on Apache Arrow — interop with Polars, DuckDB, and the rest of the ecosystem out of the box
- Zero-copy Python integration via PyO3
- SIMD-accelerated math ops
- Memory layouts NumPy can't express

  ## project architecture

this repo is organized as a cargo workspace. here's a quick map of where things live:
- **`crates/`** — the actual library code, split into individual crates (core arrays, dispatch, compute kernels, I/O, etc). this is where most contributions happen.
- **`tests/`** — integration tests that exercise the public API across crates, separate from the unit tests inside each crate.
- **`examples/`** — small standalone programs showing how to use mohu, meant to be read and run directly.
- **`docs/`** — longer-form documentation and design notes that don't fit in the README.
- **`benches/`** — performance benchmarks used to track regressions and validate the "parallel by default" and zero-copy goals.
- **`scripts/`** — developer tooling and automation scripts (setup, checks, etc) that aren't part of the library itself.

- mohu/
├── crates/       # library code, one crate per component
├── tests/        # cross-crate integration tests
├── examples/     # runnable usage examples
├── docs/         # extended documentation
├── benches/      # performance benchmarks
└── scripts/      # dev tooling and automation

## status

Early. The foundation is being laid. If you believe the Python numerical stack deserves a rewrite, watch this repo or contribute.

## built with

- [Rust](https://rust-lang.org)
- [PyO3](https://github.com/PyO3/pyo3) — Python bindings
- [arrow-rs](https://github.com/apache/arrow-rs) — columnar memory format
- [Rayon](https://github.com/rayon-rs/rayon) — data parallelism

## license

MIT
